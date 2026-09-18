#!/usr/bin/env python3
"""两份 checkpoint 之间,routed 专家的【下标】有没有被打乱。

WHY
---
EP 训练把 256 个专家按 `Shard(0)` 切在 N 个 rank 上,保存时要 gather 回来。gather 如果
把顺序拼错,产出的 checkpoint **格式完好、张量齐全、校验全过**,但 router 选第 i 个专家
却拿到第 j 个的权重。症状是服务端整个输出分布垮掉,而**训练侧曲线完全正常**(内存里的
模型没错,只有落盘的产物错了),`verify_dspark_conversion.py` 也抓不到 —— 它的契约是
「转换忠实于 trainer 的那份文件」,不是「那份文件本身对」。

2026-09-19 就撞上这个形状:同一个 run 的 1.0ep checkpoint 服务 accept_len 4.933,
2.0ep 和 3.0ep 塌到 1.258 / 更低,而训练日志 80,631 步全程健康。

怎么测
------
正常训练下**专家 e 的权重是缓慢变化的**,所以 B 的专家 e 应该和 A 的专家 **e** 最像。
被置换过的话,B 的 e 会和 A 的**另一个下标**最像。于是:

  1. 每个专家取 w1 的前 ``--rows`` 行当指纹(8×4096 bf16 ≈ 64 KB,不读全量)
  2. 行归一化后算 256×256 的余弦相似度 ``B @ A.T``
  3. 每行 argmax 落在对角线上 = 没置换

⚠ **可分性判据必须挂在「最佳匹配的质量」上,不能挂在对角线上。** 被置换过的 ckpt 对角线
本来就低 —— 拿对角线当可分性,会把唯一要抓的那种情况判成 inconclusive。合成用例上实测
翻过车(块内倒序置换被误判),所以这里用的是:每行的最佳匹配够高(中位 > 0.30)**且**
明显高于次佳(中位差 > 0.10)。三档结论:

  OK            最佳匹配可分,且全部落在对角线 -> 下标对齐
  MISALIGNED    最佳匹配可分,但不在对角线   -> 换位了;并报告是块内还是跨块
  INCONCLUSIVE  最佳匹配本身就不像           -> 内容被写坏 / 两份相隔太远,不给结论

⚠ 这个测法查的是**下标对齐**。专家内容被整个写坏(而不是换位)会落进 INCONCLUSIVE 并
提示"更像是内容被写坏" —— 不会被误判成"正常",但也不要当成通过。

USAGE
-----
    python check_ckpt_expert_alignment.py --a <好的那份> --b <可疑的那份>
    ... --layers 0 1 2       # 默认全查
    ... --rows 16            # 指纹行数,默认 8;可分性不足时加大
    ... --experts 256        # 专家数,默认自动数

两个目录都是**转换后**的 mtp.* 布局(``mtp.{L}.ffn.experts.{e}.w1.weight``)。按训练量
命名的那种,例如 ``dsv4_dspark_blk15_ep1p0_vllm-77w``。纯 CPU,只读文件头 + 每专家几十 KB。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEY = "mtp.{L}.ffn.experts.{e}.w1.weight"


def _open(d: Path):
    from safetensors import safe_open  # noqa: PLC0415

    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        # 分片:按 weight_map 找到每个键在哪个分片里,按需开
        wm = json.loads(idx.read_text())["weight_map"]
        handles: dict[str, object] = {}

        def get_rows(key: str, rows: int):
            shard = wm[key]
            if shard not in handles:
                handles[shard] = safe_open(str(d / shard), framework="pt").__enter__()
            return handles[shard].get_slice(key)[0:rows, :]

        def has(key: str) -> bool:
            return key in wm

        return get_rows, has
    f = safe_open(str(d / "model.safetensors"), framework="pt").__enter__()
    ks = set(f.keys())
    return (lambda key, rows: f.get_slice(key)[0:rows, :]), (lambda key: key in ks)


def count_experts(has, L: int) -> int:
    n = 0
    while has(KEY.format(L=L, e=n)):
        n += 1
    return n


def fingerprints(get_rows, L: int, n_exp: int, rows: int):
    import torch  # noqa: PLC0415

    fp = torch.stack([get_rows(KEY.format(L=L, e=e), rows).float().reshape(-1)
                      for e in range(n_exp)])
    # 行归一化 -> 余弦相似度就是一次矩阵乘
    return fp / fp.norm(dim=1, keepdim=True).clamp_min(1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="参照(已知好的那份)")
    ap.add_argument("--b", required=True, help="待查")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="默认 0 1 2")
    ap.add_argument("--rows", type=int, default=8, help="每个专家取 w1 的前几行当指纹")
    ap.add_argument("--experts", type=int, default=None, help="专家数(默认自动数)")
    ap.add_argument("--show", type=int, default=12, help="最多列出几个错位的专家")
    args = ap.parse_args()

    import torch  # noqa: PLC0415

    ga, ha = _open(Path(args.a))
    gb, _ = _open(Path(args.b))
    layers = args.layers if args.layers is not None else [
        L for L in range(16) if ha(KEY.format(L=L, e=0))]
    if not layers:
        print("!! 两份里都找不到 mtp.{L}.ffn.experts.0.w1.weight —— 不是 mtp.* 布局?",
              file=sys.stderr)
        return 2

    print(f"A(参照) {args.a}")
    print(f"B(待查) {args.b}")
    print(f"层 {layers}   指纹 = w1 前 {args.rows} 行\n")

    verdicts = []
    for L in layers:
        n_exp = args.experts or count_experts(ha, L)
        A = fingerprints(ga, L, n_exp, args.rows)
        B = fingerprints(gb, L, n_exp, args.rows)
        sim = B @ A.T                                   # [n, n]
        best_sim, best = sim.max(dim=1)
        hit = int((best == torch.arange(n_exp)).sum())

        # ★ 判据必须挂在【最佳匹配的质量】上,不能挂在对角线上 —— 被置换过的 ckpt 对角线
        #   本来就低,拿对角线当「可分性」会把唯一要抓的那种情况判成 inconclusive。
        #   (合成用例上实测过:块内倒序置换曾被误判,就是因为这个。)
        #   可分 = 每行都有一个明显胜出的匹配:best 够高,且明显高于「非 best」的整体水平。
        s2 = sim.clone()
        s2.scatter_(1, best.unsqueeze(1), float("-inf"))
        runner = s2.max(dim=1).values
        med_best = float(best_sim.median())
        contrast = float((best_sim - runner).median())
        discriminative = med_best > 0.30 and contrast > 0.10

        n_uniq = int(torch.unique(best).numel())
        bijection = n_uniq == n_exp

        if not discriminative:
            v = "INCONCLUSIVE"
        elif hit == n_exp:
            v = "OK"
        else:
            v = "MISALIGNED"
        verdicts.append(v)

        print(f"L{L}  专家 {n_exp}   对角命中 {hit}/{n_exp}   "
              f"最佳匹配相似度 中位 {med_best:.4f}   与次佳的差 {contrast:+.4f}   "
              f"最佳匹配是双射 {'是' if bijection else f'否({n_uniq} 个不同)'}   → {v}")
        if v == "MISALIGNED":
            mis = [(e, int(best[e])) for e in range(n_exp) if int(best[e]) != e]
            same_shard = sum(1 for e, b in mis if e // 32 == b // 32)
            print(f"      {len(mis)} 个错位;其中 {same_shard} 个仍落在同一个 32 专家的 rank 块内"
                  f"({'块内换位' if same_shard == len(mis) else '跨块错位'})")
            for e, b in mis[: args.show]:
                print(f"      B 的专家 {e:>3} ← A 的 {b:>3}"
                      f"(自身 {float(sim[e, e]):.4f} / 最佳 {float(sim[e, b]):.4f})")
            if len(mis) > args.show:
                print(f"      ... 共 {len(mis)} 个")
        elif v == "INCONCLUSIVE":
            print(f"      最佳匹配本身就不像(中位 {med_best:.4f})"
                  f"{' —— 更像是专家内容被写坏,而不是换位' if med_best < 0.30 else ''}")

    print()
    if all(v == "OK" for v in verdicts):
        print("✅ 每一层的专家下标都对齐 —— 保存时的 EP gather 没有打乱顺序。")
        print("   (只证明了下标对齐;内容本身写坏查不出来,但那种情况对角相似度会塌,"
              "会落进 INCONCLUSIVE。)")
        return 0
    if any(v == "MISALIGNED" for v in verdicts):
        print("❌ 有层的专家下标错位 —— 保存时的 EP gather 拼错了顺序。")
        print("   这份 checkpoint 不能用,且同一条 run 里其它存档都要逐个查。")
        return 1
    print("⚠️ INCONCLUSIVE:指纹分不开(对角没有明显高于非对角)。")
    print("   要么两份相隔太远、权重已经认不出,要么专家内容本身被写坏了。")
    print("   先加大 --rows 再试;还是分不开,就换一对相邻的存档(间隔越小越好)。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
