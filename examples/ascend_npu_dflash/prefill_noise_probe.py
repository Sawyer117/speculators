#!/usr/bin/env python3
"""纯 prefill 的可复现性 —— 量本底噪声,顺便给语料溯源一个可解释的尺子。

WHY
---
2026-09-23。我们一直用 `corpus_provenance_check.py --repeat R` 量确定性:发一个 prompt、
贪心生成 32 个 token、看三次之间「完全一致前缀」有多长。这个指标有两个毛病:

  1. **decode 把一次翻转放大成一整段。** 第 5 位翻一次,后面 27 位全部继承,于是读数从
     32/32 掉到 5/32。放大倍数未知 ⟹ 那个数只能比大小,没有物理意义。
  2. **它取的是 min。** 3 行 × 3 次 ≈ 9 次两两比较,取最差那次。一个单次复现率 80% 的
     配置,9 次全过的概率只有 0.8^9 = 13% —— 八成可复现会被读成「不确定」。实测:同一
     套配置两次测量,2048 那格分别是 6 和 20。

而我们真正要守的不变量根本不是 decode:

  * **训练侧** —— HS 是把整条语料一次 prefill 抓出来的;
  * **部署侧** —— 投机解码里主模型验证的是 `[上次接受的 token + γ 个草稿 token]`,
    一次 forward 过 γ+1 个,**是 prefill 形状,不是单 token decode**。

decode 只出现在语料生成那一次(176 老栈),而那份语料现在是固定输入,不在训推回路里。
⟹ 该量的是 **prefill ↔ prefill**,而且要量数值,不是量 argmax 翻没翻。

量什么
------
同一串 token,只做 prefill(`max_tokens=1`),重复 R 次,比较每个位置的 `prompt_logprobs`:

  A. **argmax 翻转率** —— 多少比例的位置,两次 prefill 的 top-1 不是同一个 token。
     这是本底噪声,而且**位置之间互不污染**(每个位置都拿语料自己的前缀做条件),
     没有 decode 那种反馈放大。
  B. **|Δlogprob|** —— 同一个 token 在两次之间的 logprob 差。这才是物理量;
     argmax 翻转只是它过了个阈值。
  C. **翻转率 vs margin(top1−top2)** —— 如果翻转只发生在接近平局的位置,那是数值
     抖动;如果笃定的位置也翻,那是别的东西坏了。
  D. **语料一致率** —— top-1 是不是就等于语料里那个 token。这正是历史上那个
     「response 段 mismatch 64%」的量,而 A 给了它一把尺子:

       A ≈ 60%  ⟹ 64% 是引擎本底,语料清白;
       A ≈  1%  ⟹ 64% 是语料本身 —— 那批 response 不是这个目标模型会产生的,
                   而所有 DSpark 训练都跑在这份 Arrow 上。

⚠ 两个前提,不满足这脚本的数就是假的
------------------------------------
  * **serve 必须带 `--no-enable-prefix-caching`。** 否则第二次 prefill 直接命中 KV 缓存,
    结果按构造就是一致的,量出来的「完全可复现」毫无意义。我们的 serve 脚本默认带了。
  * `prompt_logprobs` 要被这版 vLLM 认。不认时本脚本自动退回 `echo=True, logprobs=k`,
    并在表头写明用的是哪条路径。

USAGE
-----
  ENDPOINT=http://localhost:7000/v1 ARROW=/path/to/arrow \
    python prefill_noise_probe.py --n 8 --repeat 3 --seq-len 2048

  # 同一条命令加背景负载,回答「输出会不会随同批里还有谁而变」
  ... --bg 4 --bg-len 512
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

# 探针是 HTTP 客户端 + Arrow 读取,不需要 NPU。不关掉的话 `import torch`(datasets 的某些
# 路径会拉起来)会自动加载 torch_npu,在没 source CANN 的进程里直接报
# "Failed to load the backend extension: torch_npu"。而且就算能加载也不该加载 —— 探针
# 占卡会和正在被测的 serve 抢资源。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

MARGIN_BUCKETS = [
    (0.0, 0.01, "≤0.01  近乎平局"),
    (0.01, 0.1, "0.01–0.1"),
    (0.1, 0.5, "0.1–0.5"),
    (0.5, 2.0, "0.5–2"),
    (2.0, float("inf"), ">2     笃定"),
]


def _flat(e: BaseException, n: int = 200) -> str:
    """异常信息压成一行 —— 服务返回 HTML 错误页时 str(e) 是多行的,原样打会把表冲乱。"""
    return " ".join(str(e).split())[:n]


def _pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{100.0 * num / den:5.2f}%"


def _wilson(num: int, den: int) -> str:
    """二项比例的 95% 区间(Wilson)。取 min 的老指标没有误差棒,这里补上。"""
    if den == 0:
        return ""
    z = 1.96
    p = num / den
    d = 1 + z * z / den
    c = (p + z * z / (2 * den)) / d
    h = z * ((p * (1 - p) / den + z * z / (4 * den * den)) ** 0.5) / d
    return f" [{100 * max(0.0, c - h):.2f}–{100 * min(1.0, c + h):.2f}]"


def parse_positions(plp, seq: list[int]) -> list[dict | None]:
    """把 vLLM 的 prompt_logprobs 拍平成每个位置一条记录。

    vLLM 的结构:长度 == prompt token 数,第 0 项是 None(第一个 token 没有条件分布),
    第 i 项是 {token_id_str: {"logprob":…, "rank":…, "decoded_token":…}},**总是包含
    实际的 token_i**(不管它排第几)。所以 top1(i) 和 actual(i) 都能直接取到。
    """
    out: list[dict | None] = []
    for i, ent in enumerate(plp or []):
        if not ent or i >= len(seq):
            out.append(None)
            continue
        items = []
        for k, v in ent.items():
            lp = v.get("logprob") if isinstance(v, dict) else None
            rk = v.get("rank") if isinstance(v, dict) else None
            if lp is None:
                continue
            try:
                items.append((int(k), float(lp), rk))
            except (TypeError, ValueError):
                continue
        if not items:
            out.append(None)
            continue
        items.sort(key=lambda t: -t[1])
        top1, top1_lp, _ = items[0]
        top2_lp = items[1][1] if len(items) > 1 else None
        actual = seq[i]
        actual_lp = next((lp for tid, lp, _ in items if tid == actual), None)
        out.append({
            "top1": top1,
            "top1_lp": top1_lp,
            "margin": None if top2_lp is None else top1_lp - top2_lp,
            "actual": actual,
            "actual_lp": actual_lp,
        })
    return out


def one_prefill(cli, model: str, seq: list[int], topk: int, mode: str, req_id: str):
    """返回 (positions, mode_used)。mode: 'prompt_logprobs' | 'echo'。"""
    if mode == "prompt_logprobs":
        r = cli.completions.create(
            model=model, prompt=seq, max_tokens=1, temperature=0,
            extra_headers={"X-Request-Id": req_id},
            extra_body={"prompt_logprobs": topk}, timeout=1800,
        )
        d = r.choices[0].model_dump()
        plp = d.get("prompt_logprobs")
        if plp is None:
            raise RuntimeError("服务接受了 prompt_logprobs 但没回 —— 换 echo 路径")
        return parse_positions(plp, seq), "prompt_logprobs"

    r = cli.completions.create(
        model=model, prompt=seq, max_tokens=1, temperature=0, echo=True, logprobs=topk,
        extra_headers={"X-Request-Id": req_id}, timeout=1800,
    )
    d = r.choices[0].model_dump()
    lp = d.get("logprobs") or {}
    tops = lp.get("top_logprobs") or []
    # echo 路径给的是 token 文本而不是 id,只能按文本比。够用来算翻转率,但拿不到
    # 语料一致率(要 id)。表头会写明。
    out: list[dict | None] = []
    for i, ent in enumerate(tops):
        if not ent:
            out.append(None)
            continue
        items = sorted(((k, float(v)) for k, v in ent.items()), key=lambda t: -t[1])
        top2_lp = items[1][1] if len(items) > 1 else None
        out.append({
            "top1": items[0][0],                      # 文本,不是 id
            "top1_lp": items[0][1],
            "margin": None if top2_lp is None else items[0][1] - top2_lp,
            "actual": None,
            "actual_lp": None,
        })
    return out, "echo"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:7000/v1"))
    ap.add_argument("--arrow", default=os.environ.get("ARROW"))
    ap.add_argument("--n", type=int, default=8, help="测几行")
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=2048, help="每行取前多少个 token 做 prefill")
    ap.add_argument("--repeat", type=int, default=3, help="同一串 token 重复 prefill 几次")
    ap.add_argument("--topk", type=int, default=5, help="每个位置取 top-k(要 ≥2 才有 margin)")
    ap.add_argument("--bg", type=int, default=0, metavar="N",
                    help="测量期间后台持续打 N 路无关请求,改变批次组成")
    ap.add_argument("--bg-len", type=int, default=512)
    ap.add_argument("--csv", default="", help="逐位置明细写到这个 CSV(可选)")
    ap.add_argument("--label", default="", help="打在表头,便于多轮对照")
    args = ap.parse_args()

    if not args.arrow:
        raise SystemExit("!! 需要 --arrow(或 ARROW=)")
    if args.repeat < 2:
        raise SystemExit("!! --repeat 至少 2,否则没有「两次之间」可言")

    import openai  # noqa: PLC0415
    from datasets import load_from_disk  # noqa: PLC0415

    ds = load_from_disk(args.arrow)
    if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
        ds = ds[next(iter(ds.keys()))]
    try:
        ds = ds.with_format(None)
    except Exception:  # noqa: BLE001
        pass
    has_mask = "loss_mask" in ds.column_names

    cli = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    model = cli.models.list().data[0].id

    bg_stop = threading.Event()

    def _bg_worker(seed: int) -> None:
        c = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
        toks = [(seed * 7919 + i * 31) % 100000 + 1000 for i in range(args.bg_len)]
        while not bg_stop.is_set():
            try:
                c.completions.create(model=model, prompt=toks, max_tokens=24,
                                     temperature=0, timeout=300)
            except Exception:  # noqa: BLE001,S110
                pass

    print("=" * 78)
    print(f"  纯 prefill 可复现性   {args.label}")
    print(f"  serve={model}   arrow={args.arrow}  ({len(ds)} 行)")
    print(f"  {args.n} 行 × 每行前 {args.seq_len} token × 重复 {args.repeat} 次   "
          f"top-{args.topk}   背景负载 {args.bg} 路")
    print("  ⚠ serve 必须带 --no-enable-prefix-caching,否则第二次直接命中 KV 缓存,")
    print("    「完全一致」是构造出来的,不是测出来的。")
    print("=" * 78)

    # 先探一次,确定走哪条 API 路径,免得每行都试一遍
    probe_seq = [int(t) for t in list(ds[args.start_row]["input_ids"])[:64]]
    mode = "prompt_logprobs"
    try:
        _, mode = one_prefill(cli, model, probe_seq, args.topk, mode, "prefill_probe_0")
    except Exception as e:  # noqa: BLE001
        print(f"  prompt_logprobs 不可用({type(e).__name__}: {_flat(e, 160)}),退回 echo 路径")
        try:
            _, mode = one_prefill(cli, model, probe_seq, args.topk, "echo", "prefill_probe_1")
        except Exception as e2:  # noqa: BLE001
            print(f"!! echo 路径也不行:{type(e2).__name__}: {_flat(e2, 300)}")
            print("   ⟹ 这版 vLLM 两条都不给 prompt 侧 logprob,这个探针量不了,先别信任何数。")
            return 2
    print(f"  API 路径:{mode}"
          + ("" if mode == "prompt_logprobs" else "   ⚠ echo 只给 token 文本,拿不到语料一致率"))
    print()

    if args.bg > 0:
        for k in range(args.bg):
            threading.Thread(target=_bg_worker, args=(k + 1,), daemon=True).start()
        time.sleep(5)
        print(f"★ 背景负载已稳定:{args.bg} 路无关请求(prompt {args.bg_len} token)\n")

    flip_n = flip_d = 0
    corp_n = corp_d = 0                     # 全部位置上的语料一致
    corp_resp_n = corp_resp_d = 0           # 只看 loss_mask==1(受监督)的位置
    dlp: list[float] = []
    bucket_flip = [[0, 0] for _ in MARGIN_BUCKETS]
    rows_used = 0
    csv_f = open(args.csv, "w", encoding="utf-8") if args.csv else None
    if csv_f:
        csv_f.write("row,pos,supervised,top1_run0,flipped,margin,max_abs_dlogprob,corpus_match\n")

    t0 = time.time()
    for i in range(args.n):
        row = ds[args.start_row + i]
        ids = [int(t) for t in row["input_ids"]][:args.seq_len]
        if len(ids) < 32:
            print(f"  [{i}] 太短({len(ids)}),跳过")
            continue
        mask = [int(v) for v in row["loss_mask"]][:len(ids)] if has_mask else [0] * len(ids)

        runs = []
        ok = True
        for rep in range(args.repeat):
            try:
                pos, _ = one_prefill(cli, model, ids, args.topk, mode,
                                     f"prefill_{args.start_row + i}_{rep}")
                runs.append(pos)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] 第 {rep} 次 prefill 失败:{type(e).__name__}: {_flat(e)}")
                ok = False
                break
        if not ok or len(runs) < 2:
            continue
        rows_used += 1

        n_pos = min(len(r) for r in runs)
        r_flip = r_cmp = 0
        for p in range(n_pos):
            cells = [r[p] for r in runs]
            if any(c is None for c in cells):
                continue
            r_cmp += 1
            flip_d += 1
            tops = {c["top1"] for c in cells}
            flipped = len(tops) > 1
            if flipped:
                flip_n += 1
                r_flip += 1

            lps = [c["top1_lp"] for c in cells]
            d = max(lps) - min(lps)
            dlp.append(d)

            m = cells[0]["margin"]
            if m is not None:
                for bi, (lo, hi, _) in enumerate(MARGIN_BUCKETS):
                    if lo <= m < hi:
                        bucket_flip[bi][1] += 1
                        if flipped:
                            bucket_flip[bi][0] += 1
                        break

            match = None
            if mode == "prompt_logprobs":
                match = cells[0]["top1"] == cells[0]["actual"]
                corp_d += 1
                corp_n += 1 if match else 0
                if has_mask and p < len(mask) and mask[p]:
                    corp_resp_d += 1
                    corp_resp_n += 1 if match else 0
            if csv_f:
                # top1 在 echo 路径下是 token 文本,可能带逗号/引号 —— 用 JSON 转义,
                # 否则这份 CSV 解析出来是错位的。
                t1 = json.dumps(cells[0]["top1"], ensure_ascii=False)
                csv_f.write(f"{args.start_row + i},{p},{int(bool(mask[p]) if p < len(mask) else 0)},"
                            f"{t1},{int(flipped)},"
                            f"{'' if m is None else f'{m:.6f}'},{d:.6f},"
                            f"{'' if match is None else int(match)}\n")
        print(f"  [{i}] {len(ids):>5} token   可比位置 {r_cmp:>5}   argmax 翻转 "
              f"{r_flip:>5} = {_pct(r_flip, r_cmp)}")

    if csv_f:
        csv_f.close()
    bg_stop.set()

    print()
    print("=" * 78)
    print(f"  结果   {args.label}    用时 {time.time() - t0:.0f}s   有效行 {rows_used}")
    print("=" * 78)
    if flip_d == 0:
        print("!! 一个可比位置都没有 —— 服务没返回 prompt 侧 logprob。数不可用。")
        return 2

    print(f"A. 本底噪声 —— 两次 prefill 的 argmax 翻转率")
    print(f"     {flip_n} / {flip_d} = {_pct(flip_n, flip_d)}{_wilson(flip_n, flip_d)}   (95% CI)")
    print()
    print(f"B. |Δlogprob|(同一位置 top-1 在各次之间的极差)")
    if dlp:
        s = sorted(dlp)
        def q(f: float) -> float:
            return s[min(len(s) - 1, int(f * len(s)))]
        print(f"     中位 {statistics.median(s):.3e}   p90 {q(0.90):.3e}   "
              f"p99 {q(0.99):.3e}   最大 {s[-1]:.3e}")
        print(f"     完全逐位相同(Δ==0)的位置:{_pct(sum(1 for x in s if x == 0.0), len(s))}")
    print()
    print("C. 翻转率 vs margin(top1−top2)—— 只在近乎平局处翻 = 纯数值抖动")
    for (lo, hi, name), (fn, fd) in zip(MARGIN_BUCKETS, bucket_flip):
        if fd:
            print(f"     {name:<16} {fn:>6} / {fd:>6} = {_pct(fn, fd)}{_wilson(fn, fd)}")
    print()
    if mode == "prompt_logprobs":
        print("D. 语料一致率 —— top-1 是不是就等于语料里那个 token")
        print(f"     全部位置    {corp_n} / {corp_d} = {_pct(corp_n, corp_d)}")
        if corp_resp_d:
            print(f"     response 段 {corp_resp_n} / {corp_resp_d} = {_pct(corp_resp_n, corp_resp_d)}"
                  f"   (loss_mask==1,历史上那个 mismatch 64% 量的就是这个)")
        print()
        fr = flip_n / flip_d
        mm = 1.0 - (corp_resp_n / corp_resp_d if corp_resp_d else (corp_n / max(corp_d, 1)))
        print("★ 判读")
        print(f"     本底翻转率 {100 * fr:.2f}%   vs   语料 mismatch {100 * mm:.2f}%")
        if mm <= max(0.02, 3 * fr):
            print("     ⟹ mismatch 和本底同量级 —— 语料没问题,历史上那个 64% 是引擎本底。")
        elif fr < 0.05 and mm > 0.30:
            print("     ⟹ ★★ 本底很干净而 mismatch 很大:**那批 response 不是这个目标模型")
            print("        会产生的**。所有 DSpark 训练都跑在这份 Arrow 上,这件事的影响")
            print("        远大于任何算子/栈的问题。先停批量 HS 生产,查语料来源。")
        else:
            print("     ⟹ 两者都不小,分不开。加大 --n / --repeat,或先把本底压下去再测语料。")
    print()
    print("对照怎么做:同一条命令跑 --bg 0 和 --bg 4。")
    print("  两者 A 差不多  ⟹ 输出不受同批其他请求影响,批次组成不是机制。")
    print("  --bg 4 明显更差 ⟹ **输出取决于同一批里还有谁** —— 那是正确性 bug,而且能同时")
    print("                     解释「上下文越长越差」和「隔天数字不一样」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
