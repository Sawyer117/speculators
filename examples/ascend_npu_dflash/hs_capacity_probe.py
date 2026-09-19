#!/usr/bin/env python3
"""预存 HS 到底要多少盘 —— 一条命令给出算术,不靠估。

WHY
---
在线 HS(serve 现产现删)的唯一理由是「存不下」,而「存不下」这个判断一直是拍脑袋的。
它却决定了一件大事:HS 若能预存,两台 serve 机就能腾出来参训,DP 从 8 卡变 24 卡,
**每个 epoch 的墙钟缩到 1/3**。(注意收益来自「机器变多」,不是「HS 变快」——
训练日志自己说 fetch_frac 中位 0.010,HS 从来不是单步的瓶颈。)

体积 = 行数 × 平均 token 数 × 每 token 字节数。三个因子:

  行数          已知(77W = 775,965)
  平均 token    ★ 唯一需要量的,而 Arrow 就在本地,采样几千行几秒钟
  字节/token    [seq_len, n_layers, hidden] × 2B。默认 4×4096×2 = 32,768,
                但只要给了 --hs-dir 且里面有真文件,就【实测】而不是用默认值

⚠ 别用「平均生成 token 数」去推。rollout 的 657 tok/s ÷ 1.15 行/s 只是**生成部分**,
prompt 不算在内 —— 用它算出来的体积会低一半以上。这个脚本量的是 Arrow 里
`input_ids` 的真实长度,也就是 HS 实际要覆盖的 token 数。

USAGE
-----
    python hs_capacity_probe.py                       # 自己找 Arrow、自己找盘
    python hs_capacity_probe.py --arrow <dir>         # 指定数据集
    python hs_capacity_probe.py --hs-dir <dir>        # 有真 HS 文件时:实测字节/行 + 压缩比
    python hs_capacity_probe.py --budget-tb 6         # 自定预算(默认按目标盘剩余的一半)

纯 CPU,只读几千行 + 最多一个 HS 文件。不碰 NPU,serve 跑着也能跑。
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import random
import shutil
import sys
from pathlib import Path

# Arrow 数据集的候选位置。scp 过来的可能少一层 open_perfectblend.dsv4_rollout/,所以
# 两种深度都找。
ARROW_ROOTS = (
    "/home/canada_group_folder/dataset",
    "/share/canada_group_folder/dataset",
    "/mnt/nfs/canada_group_folder/dataset",
)
HS_ROOTS = (
    "/share/canada_group_folder/dataset/dsv4_hs_dump",
    "/home/canada_group_folder/dataset/dsv4_hs_dump",
    "/mnt/nfs/canada_group_folder/dataset/dsv4_hs_dump",
)
FULL_ROWS = 775_965          # 77W dedup,registry §3.1 记的确数


def find_arrow(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for root in ARROW_ROOTS:
        for pat in (f"{root}/arrow*", f"{root}/*/arrow*"):
            for d in sorted(glob.glob(pat)):
                if os.path.isfile(os.path.join(d, "dataset_info.json")):
                    return Path(d)
    return None


def _st_header(path: str) -> dict:
    """safetensors 的头:前 8 字节小端 u64 = 头长,紧跟着那么长一段 JSON。只读几 KB。"""
    import json  # noqa: PLC0415

    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(n))


def measure_hs(hs_dir: str | None) -> dict | None:
    """从真 HS 文件量事实,而不是靠和 Arrow 交叉推算。

    ★ 每 token 字节数直接从 **文件头的 shape** 来(``hidden_states`` = [seq, L, H]),
      这是精确值;拿「平均字节/行 ÷ Arrow 平均长度」去反推会把两个不同的抽样混在一起
      (HS 文件常是连续一段行,Arrow 均值是全局的),得到一个看着合理其实错位的数。
    ★ 压缩比取**多个文件的中位数**,不是一个文件 —— 单文件可能极不典型
      (自测时用零填充的 fixture 给出过 970×,真 bf16 HS 的上限在 1.4× 附近)。
    """
    cands = [hs_dir] if hs_dir else list(HS_ROOTS)
    for d in cands:
        if not d or not os.path.isdir(d):
            continue
        fs = sorted(glob.glob(f"{d}/hs_*.safetensors"))[:2000]
        if not fs:
            continue
        sizes = [os.path.getsize(f) for f in fs]
        hdr = _st_header(fs[0])
        hs = hdr.get("hidden_states")
        per_tok = seq0 = None
        if hs and len(hs.get("shape", [])) == 3:
            seq0, nl, hid = hs["shape"]
            width = 2 if "16" in str(hs.get("dtype", "BF16")) else 4
            per_tok = nl * hid * width
        # 多个文件取中位压缩比
        ratios = []
        for f in fs[: min(3, len(fs))]:
            blob = Path(f).read_bytes()
            ratios.append(len(blob) / max(len(gzip.compress(blob, 6)), 1))
        ratios.sort()
        ratio = ratios[len(ratios) // 2]
        # 每个文件的 token 数 = 文件大小 / 每 token 字节(头里的 shape 是精确的)
        seqs = [s_ / per_tok for s_ in sizes] if per_tok else []
        print(f">>> 实测 HS:{d}  {len(fs)} 个文件,压缩比取 {len(ratios)} 个的中位")
        return {"mean_bytes": sum(sizes) / len(sizes), "ratio": ratio, "n": len(fs),
                "per_tok": per_tok, "mean_seq": (sum(seqs) / len(seqs)) if seqs else None,
                "shape0": hs.get("shape") if hs else None, "seq0": seq0}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrow", help="Arrow 数据集目录(省略=自动找)")
    ap.add_argument("--hs-dir", help="真 HS 目录(有文件就实测字节/行与压缩比)")
    ap.add_argument("--sample", type=int, default=3000, help="采样多少行量长度")
    ap.add_argument("--col", default="input_ids")
    ap.add_argument("--layers", type=int, default=4, help="dump 的层数(3 aux + 1 final)")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--seq-cap", type=int, default=3072,
                    help="训练侧 --total-seq-len;额外给一版按它截断后的体积")
    ap.add_argument("--target", help="要存到哪个盘(省略=Arrow 所在的盘)")
    ap.add_argument("--budget-tb", type=float, default=None,
                    help="可用预算 TB(省略=目标盘剩余的一半,因为那是共享盘)")
    args = ap.parse_args()

    arrow = find_arrow(args.arrow)
    if arrow is None or not arrow.is_dir():
        print(f"!! 找不到 Arrow 数据集。找过:{', '.join(ARROW_ROOTS)}\n"
              f"   用 --arrow <dir> 指一下。", file=sys.stderr)
        return 2

    from datasets import load_from_disk  # noqa: PLC0415 — 只有 box 上有

    ds = load_from_disk(str(arrow))
    if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
        ds = ds[next(iter(ds.keys()))]
    try:
        ds = ds.with_format(None)
    except Exception:  # noqa: BLE001
        pass
    n = len(ds)
    if args.col not in ds.column_names:
        print(f"!! 数据集里没有列 {args.col!r};有的是 {ds.column_names}", file=sys.stderr)
        return 2

    print(f"数据集  {arrow}")
    print(f"        {n:,} 行   列 {ds.column_names}")

    random.seed(0)
    idx = random.sample(range(n), min(args.sample, n))
    lens = sorted(len(ds[i][args.col]) for i in idx)
    m = len(lens)

    def pct(q: float) -> float:
        return lens[min(m - 1, int(q / 100 * m))]

    mean_len = sum(lens) / m
    capped = sum(min(x, args.seq_cap) for x in lens) / m
    print(f"\n采样 {m} 行的 {args.col} 长度:")
    print("        " + "   ".join(f"p{q}={pct(q):,.0f}" for q in (50, 80, 90, 95, 99)))
    print(f"        均值 {mean_len:,.1f}   最大 {lens[-1]:,}")

    # ── 每 token 字节数:能实测就实测 ────────────────────────────────────────────
    per_tok = args.layers * args.hidden * 2
    ratio = None
    meas = measure_hs(args.hs_dir)
    if meas:
        if meas["per_tok"]:
            per_tok = meas["per_tok"]
            print(f"        头里的 shape {meas['shape0']} ⟹ {per_tok:,} B/token"
                  f"(理论 {args.layers}×{args.hidden}×2 = {args.layers*args.hidden*2:,})")
        print(f"        实测 {meas['mean_bytes'] / 2**20:,.1f} MiB/行", end="")
        if meas["mean_seq"]:
            print(f"  ⟹ {meas['mean_seq']:,.0f} token/行(Arrow 采样是 {mean_len:,.0f})")
            if abs(meas["mean_seq"] - mean_len) > 0.25 * mean_len:
                print("        ⚠ 两者差 >25% —— HS 文件多半只覆盖了连续一段行,"
                      "不是随机样本;体积按 Arrow 的均值算更稳")
        else:
            print()
        ratio = meas["ratio"]
        print(f"        gzip -6 压缩比中位 {ratio:.2f}×", end="")
        if ratio > 3:
            print("   ⚠ 高得不合理:bf16 浮点的无损上限在 1.4× 附近"
                  "(尾数是随机比特)。这批文件多半不是真数据,别拿这个比值做决策。")
            ratio = None
        else:
            print()
    else:
        print(f"        (没找到真 HS 文件,字节/token 用理论值 "
              f"{args.layers}×{args.hidden}×2 = {per_tok:,})")

    # ── 盘 ────────────────────────────────────────────────────────────────────
    target = args.target or str(arrow)
    du = shutil.disk_usage(target)
    free_tb = du.free / 1e12
    budget = args.budget_tb if args.budget_tb is not None else free_tb / 2
    print(f"\n目标盘  {target}")
    print(f"        总 {du.total / 1e12:,.1f} TB   已用 {du.used / 1e12:,.1f} TB "
          f"({du.used / du.total * 100:.0f}%)   剩 {free_tb:,.1f} TB")
    if args.budget_tb is None:
        print(f"        ⚠ 这多半是共享盘 —— 预算按【剩余的一半】{budget:,.1f} TB 算。"
              f"填满共享盘先死的是自己的训练和 serve。用 --budget-tb 覆盖。")

    # ── 表 ────────────────────────────────────────────────────────────────────
    # 候选行数:标准档位里不超过数据集大小的那些;数据集本身很小(测试/子集)时退化成
    # 它自己的几个分数 —— 否则表会是空的,而空表看起来像"没算",不像"没档位"。
    cands = [r for r in (FULL_ROWS, 600_000, 400_000, 300_000, 200_000, 100_000, 50_000)
             if r <= n]
    if not cands:
        cands = [n, n // 2, n // 4, n // 10]
    cands = sorted({c for c in cands if c > 0}, reverse=True)

    print(f"\n{'行数':>10} {'占数据集':>8}  {'bf16':>9}"
          + (f"  {'+gzip %.2fx' % ratio:>11}" if ratio else "")
          + f"  判决(预算 {budget:.1f} TB)")
    print("-" * (48 + (13 if ratio else 0)))
    for rows in cands:
        tb = rows * mean_len * per_tok / 1e12
        cell = f"{rows:>10,} {rows / n * 100:7.0f}%  {tb:8.2f} TB"
        best = tb
        if ratio:
            cell += f"  {tb / ratio:8.2f} TB"
            best = tb / ratio
        print(f"{cell}  {'✅ 放得下' if best <= budget else '❌'}")

    if capped < mean_len * 0.995:
        print(f"\n若 dump 也截断到 --total-seq-len {args.seq_cap}:均值 "
              f"{mean_len:,.0f} → {capped:,.0f} token,全量 "
              f"{FULL_ROWS * capped * per_tok / 1e12:,.1f} TB")

    # ── 能存多少行 ────────────────────────────────────────────────────────────
    fit = int(budget * 1e12 / (mean_len * per_tok))
    eff = fit * (ratio or 1.0)
    print(f"\n⟹ 预算 {budget:,.1f} TB 下能存 **{min(eff, n):,.0f} 行**"
          f"(本数据集的 {min(eff, n) / n * 100:.0f}%"
          + (f",bf16 原样则 {fit:,} 行)" if ratio else ")"))
    print("   ⚠ 行数砍了不等于数据量砍了 —— 拉长 schedule 多跑几个 epoch 可以补回样本总数。"
          "\n     这是个确定性教师的蒸馏任务,重复见同一条序列的害处远小于预训练。")
    print("\n下一个数:dump 一次要多久。需要一台 HS-dump serve 活着,用 "
          "dsv4_fire_hs_dumps.py 打几百行计时。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
