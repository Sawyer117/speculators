#!/usr/bin/env python3
"""HS 的终极自检:拿 serve 自己吐的 token 当基准,不依赖任何语料来源。

WHY
---
2026-09-20 的困境:dump 出来的 HS 在 `arrow_0730_77w_dedup` 上 response 段
(loss_mask==1)mismatch 64.46%,而 token_ids 和 Arrow 逐个一致、切片/norm/shift 三个
维度也都验过没错。于是卡在两种解释之间,而它们要的下一步完全相反:

  (a) **语料不是这套栈贪心生成的** —— dump 是好的,可以放行批量生产;
  (b) **dump 的值本身不对** —— 批量生产免谈,先修捕获。

用外部语料永远分不开这两者。但有一个**定义上必然成立**的关系可以用:

    serve 在某一步吐出的 token  ==  argmax( lm_head( 它那一步的 final hidden ) )

温度 0 时这是恒等式 —— serve 就是这么算出来的。所以:让 serve 自己生成,再看 dump 的
hidden 能不能重现它自己刚吐的 token。**对得上 ⟹ dump 是对的,问题在语料;对不上 ⟹
dump 捕获的不是 lm_head 实际吃的那个张量。** 没有第三种可能,也不需要相信任何数据集。

两段检查
--------
  A 段(便宜,1 次请求):``max_tokens=1``。dump 的**最后一行** hidden 过 lm_head,
    argmax 必须 == serve 返回的那个 token。只查 1 个位置,但它是恒等式,极硬。
  B 段(强,2 次请求):先 ``max_tokens=G`` 贪心生成 G 个 token,再把
    ``prompt + 生成的 G 个`` 整条回灌做一次 prefill dump,检查那 G 个位置。
    这段等价于"教师强制自己刚生成的文本",mismatch 必须 ~0%。

USAGE
-----
    ENDPOINT=http://localhost:7000/v1 \
    ARROW=/home/canada_group_folder/dataset/arrow_0730_77w_dedup \
    HS_DIR=/home/canada_group_folder/dataset/dsv4_hs_dump \
      python hs_self_oracle.py --model-dir /home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16

    ... --n 4 --gen 16 --prompt-len 256      # 更快
需要:一台开了 DSPARK_HS_DUMP=1 的 serve;本机有 lm_head(只读它,~1.8 GB)。
⚠ id 段默认 950000+,落在数据集 772,684 行之外 —— 不会污染生产 HS,也不会被训练侧误用。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_HEAD_KEYS = ("lm_head.weight", "head.weight")


def load_lm_head(model_dir: str):
    from safetensors import safe_open  # noqa: PLC0415

    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        wmap = json.load(open(idx))["weight_map"]
        for k in _HEAD_KEYS:
            if k in wmap:
                with safe_open(os.path.join(model_dir, wmap[k]), framework="pt") as fh:
                    return k, fh.get_tensor(k).float()
    for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(p, framework="pt") as fh:
            for k in _HEAD_KEYS:
                if k in set(fh.keys()):
                    return k, fh.get_tensor(k).float()
    raise SystemExit(f"!! 在 {model_dir} 里找不到 lm_head")


def wait_file(path: str, timeout: float = 120.0):
    dl = time.time() + timeout
    while time.time() < dl:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            time.sleep(0.3)          # 让 writer 收尾
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://localhost:7000/v1"))
    ap.add_argument("--arrow", default=os.environ.get("ARROW"))
    ap.add_argument("--hs-dir", default=os.environ.get("HS_DIR",
                    "/home/canada_group_folder/dataset/dsv4_hs_dump"))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=int, default=4, help="试几条")
    ap.add_argument("--gen", type=int, default=16, help="B 段贪心生成几个 token")
    ap.add_argument("--prompt-len", type=int, default=256, help="prompt 截到这么长(快)")
    ap.add_argument("--id-base", type=int, default=950000,
                    help="★ 必须 > 数据集行数,否则会污染生产 HS 目录")
    ap.add_argument("--rows-full", type=int, default=772684)
    ap.add_argument("--start-row", type=int, default=0)
    args = ap.parse_args()

    if args.id_base <= args.rows_full:
        raise SystemExit(f"!! --id-base {args.id_base} 落在数据集行号内(共 {args.rows_full} 行),"
                         "会把生产 HS 目录写脏。用 > 772684 的值。")
    if not args.arrow:
        raise SystemExit("!! 需要 --arrow(或 ARROW=)")

    # openai 客户端只认 no_proxy —— 不设会被公司代理劫走,回来一张 HTML 错误页
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

    import openai  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from datasets import load_from_disk  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    ds = load_from_disk(args.arrow)
    if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
        ds = ds[next(iter(ds.keys()))]
    try:
        ds = ds.with_format(None)
    except Exception:  # noqa: BLE001
        pass

    key, W = load_lm_head(args.model_dir)
    cli = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    model = cli.models.list().data[0].id
    print(f"serve={model}  lm_head='{key}' {tuple(W.shape)}  样本 {args.n} 条  "
          f"prompt≤{args.prompt_len}  生成 {args.gen}\n")

    def argmax_at(hs, pos: int) -> int:
        return int((hs[pos, -1, :].float() @ W.T).argmax())

    def fire(ids, tag_id: int, max_tokens: int):
        """打一条,返回(生成的 token id 列表, dump 文件路径)。"""
        r = cli.completions.create(
            model=model, prompt=ids, max_tokens=max_tokens, temperature=0,
            extra_headers={"X-Request-Id": f"hs_{tag_id}"},
            extra_body={"return_token_ids": True}, timeout=600,
        )
        out = getattr(r.choices[0], "token_ids", None)
        if out is None:                      # 兜底:不同版本字段名不一样
            d = r.choices[0].model_dump()
            out = d.get("token_ids") or (d.get("logprobs") or {}).get("tokens")
        f = os.path.join(args.hs_dir, f"hs_{tag_id}.safetensors")
        return (list(out) if out else None), f

    okA = badA = okB = badB = 0
    tot_resp = mism_resp = 0
    for i in range(args.n):
        src = list(ds[args.start_row + i]["input_ids"])[: args.prompt_len]
        tid = args.id_base + i * 2

        # ── A 段:1 个 token,查恒等式 ──────────────────────────────────────
        gen, f = fire(src, tid, 1)
        if not wait_file(f):
            print(f"  [{i}] A 段:等不到 {f} —— serve 没开 DSPARK_HS_DUMP?"); continue
        hs = load_file(f)["hidden_states"]
        pred = argmax_at(hs, hs.shape[0] - 1)
        if gen and len(gen) >= 1:
            got = int(gen[-1])
            hit = pred == got
            okA += hit; badA += (not hit)
            print(f"  [{i}] A 段  serve 吐 {got:>6}   dump argmax {pred:>6}   "
                  f"{'✅ 一致' if hit else '❌ 不一致'}   (T={hs.shape[0]})")
        else:
            print(f"  [{i}] A 段:serve 没返回 token_ids,跳过(试试别的 vllm 版本字段)")

        # ── B 段:生成 G 个再回灌 ─────────────────────────────────────────
        gen, _ = fire(src, tid + 1000000, args.gen)     # 这次的 dump 不用
        if not gen or len(gen) < 2:
            print(f"  [{i}] B 段:拿不到生成的 token,跳过"); continue
        full = src + [int(t) for t in gen]
        _, f2 = fire(full, tid + 1, 1)
        if not wait_file(f2):
            print(f"  [{i}] B 段:等不到 {f2}"); continue
        hs2 = load_file(f2)["hidden_states"]
        # 生成段在 full 里的下标是 [len(src), len(full)) ;预测位置 i 对应目标 i+1
        bad = tot = 0
        for p in range(len(src) - 1, len(full) - 1):
            if p + 1 >= hs2.shape[0]:
                break
            tot += 1
            bad += int(argmax_at(hs2, p) != int(full[p + 1]))
        okB += (tot - bad); badB += bad
        tot_resp += tot; mism_resp += bad
        print(f"  [{i}] B 段  回灌自己生成的 {tot} 个位置 → mismatch "
              f"{100.0 * bad / max(tot, 1):.2f}%")

    print("\n" + "=" * 70)
    if okA + badA:
        print(f"A 段(恒等式):{okA}/{okA + badA} 条一致")
    if tot_resp:
        r = 100.0 * mism_resp / tot_resp
        print(f"B 段(回灌自己生成的文本):mismatch {r:.2f}%  over {tot_resp} 个位置")
    print("=" * 70)
    if (okA and badA == 0) and (tot_resp and 100.0 * mism_resp / tot_resp < 5):
        print("✅ **dump 的值是对的。** serve 自己吐的 token 能被 dump 的 hidden 完全重现。")
        print("   ⟹ 之前在 arrow_0730_77w_dedup 上的 64% 是【语料问题】:那批 response")
        print("      不是这套栈贪心生成的(原始 SFT 语料?别的模型?别的采样温度?)。")
        print("      查 dsv4-dspark-article 的 rollout-data.md 确认这份 Arrow 的来源。")
        print("   ⟹ 批量 dump 可以放行(HS 的正确性与语料的来源无关)。")
    elif badA or (tot_resp and 100.0 * mism_resp / tot_resp >= 5):
        print("❌ **dump 的值不对。** 连 serve 自己刚吐的 token 都重现不了 ——")
        print("   而这是个恒等式,不该失败。捕获点拿到的不是 lm_head 实际吃的那个张量。")
        print("   下一步:核对 model_runner_v1 里 execute_model 的捕获点相对")
        print("   `_all_gather_hidden_states_and_aux`(flashcomm v1 的 all-gather)的位置,")
        print("   以及 dumper 的 per-request 切片在 chunked prefill 下对不对。")
        print("   ⚠ 在修好之前【绝对不要】开始批量 dump。")
    else:
        print("⚠️ 数据不够下结论(serve 没返回 token_ids,或 dump 文件没出现)。先看上面的逐条输出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
