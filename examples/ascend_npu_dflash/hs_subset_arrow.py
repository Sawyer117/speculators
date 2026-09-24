#!/usr/bin/env python3
"""从训练 Arrow 里按固定种子随机抽一部分行,另存成一个新 Arrow —— HS 预存用。

WHY
---
盘放不下全量 HS(77W 行 × 平均 776 token × 32 KB ≈ 19.6 TB),只能存一部分。
取「前一半」不行:Arrow 可能按来源排好了序,前一半就是另一种数据分布。所以按固定
种子随机抽,**另存成新 Arrow**,dump 和训练都用它 —— 训练侧按 `hs_<行号>.safetensors`
找文件,行号必须是新 Arrow 自己的行号,不能是原 Arrow 的。

原行号写进新目录里的 `orig_rows.txt`(一行一个),不加成 Arrow 的列:训练侧对多出来的
列怎么处理没验证过,不冒这个险。`SUBSET.txt` 记来源、种子、行数和体积估算。

写完会重新读回来,抽几行和原 Arrow 逐 token 比对,对不上直接报错。

USAGE
-----
    python hs_subset_arrow.py --arrow <src> --out <dst> --fraction 0.5
    python hs_subset_arrow.py --arrow <src> --out <dst> --rows 300000 --seed 1
    python hs_subset_arrow.py --arrow <src> --out <dst> --budget-tb 6 --ratio 1.40

纯 CPU,不碰 NPU。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

BYTES_PER_TOKEN = 4 * 4096 * 2      # 3 aux + 1 final,bf16
MEASURED_RATIO = 1.40               # 真 HS 文件实测:字节拆分 + gzip -6(尾数那一字节压不动)


def _load(path: str):
    from datasets import load_from_disk  # noqa: PLC0415 — box dep

    ds = load_from_disk(path)
    if hasattr(ds, "keys") and not hasattr(ds, "num_rows"):
        ds = ds[next(iter(ds.keys()))]
    return ds.with_format(None)


def _lengths(ds, idx) -> list[int]:
    if "seq_len" in ds.column_names:
        return [int(x) for x in ds.select(idx)["seq_len"]]
    return [len(r["input_ids"]) for r in ds.select(idx)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arrow", required=True, help="源 Arrow 目录")
    ap.add_argument("--out", required=True, help="新 Arrow 目录(必须不存在)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fraction", type=float, help="抽多大比例,如 0.5")
    g.add_argument("--rows", type=int, help="抽多少行")
    g.add_argument("--budget-tb", type=float, help="按盘预算反推行数(TB,十进制)")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help=f"--budget-tb 用的压缩比。1.0 = 原样 bf16;实测无损上限 {MEASURED_RATIO}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-len", type=int, default=5000, help="估平均长度时抽多少行")
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"!! {args.out} 已存在。换个目录,或确认不要了再删 —— 不覆盖。")
        return 2

    import numpy as np  # noqa: PLC0415

    t0 = time.time()
    ds = _load(args.arrow)
    n_total = len(ds)
    rng = np.random.default_rng(args.seed)
    probe = np.sort(rng.choice(n_total, min(args.sample_len, n_total), replace=False))
    mean_len = float(np.mean(_lengths(ds, probe)))
    row_bytes = mean_len * BYTES_PER_TOKEN

    if args.fraction is not None:
        n = int(round(n_total * args.fraction))
    elif args.rows is not None:
        n = args.rows
    else:
        n = int(args.budget_tb * 1e12 * args.ratio / row_bytes)
    if not 0 < n <= n_total:
        print(f"!! 行数 {n} 不在 (0, {n_total}] 里")
        return 2

    # 抽行用独立的 rng(种子相同),这样 --sample-len 改了也不影响抽到哪些行。
    idx = np.sort(np.random.default_rng(args.seed).choice(n_total, n, replace=False))
    sub = ds.select(idx.tolist())
    sub.save_to_disk(args.out)
    with open(os.path.join(args.out, "orig_rows.txt"), "w") as fh:
        fh.write("\n".join(str(int(i)) for i in idx) + "\n")

    # ── 读回校验:行数 + 抽几行逐 token 比 ────────────────────────────────────
    back = _load(args.out)
    if len(back) != n:
        print(f"!! 读回来 {len(back)} 行,应为 {n}")
        return 1
    chk = np.random.default_rng(args.seed + 1).choice(n, min(8, n), replace=False)
    for j in chk:
        a = back[int(j)]["input_ids"]
        b = ds[int(idx[j])]["input_ids"]
        if list(a) != list(b):
            print(f"!! 新行 {j} ≠ 原行 {idx[j]} —— 写出的数据不对,别用")
            return 1

    sub_len = float(np.mean(_lengths(back, np.sort(
        np.random.default_rng(args.seed + 2).choice(n, min(args.sample_len, n), replace=False)))))
    raw_tb = n * sub_len * BYTES_PER_TOKEN / 1e12
    free_tb = shutil.disk_usage(args.out).free / 1e12
    lines = [
        f"source      {os.path.abspath(args.arrow)}",
        f"rows        {n:,} / {n_total:,} ({100 * n / n_total:.1f}%)   seed {args.seed}",
        f"mean_len    {sub_len:.1f} token   (源 Arrow 抽样 {mean_len:.1f})",
        f"hs_raw      {raw_tb:.2f} TB  (bf16 原样,{BYTES_PER_TOKEN:,} B/token)",
        f"hs_packed   {raw_tb / MEASURED_RATIO:.2f} TB  (无损字节拆分,实测 {MEASURED_RATIO}×)",
        f"disk_free   {free_tb:.2f} TB  ({os.path.dirname(os.path.abspath(args.out))})",
        f"created     {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "orig_rows   orig_rows.txt(第 k 行 = 新行号 k 对应的原行号)",
    ]
    with open(os.path.join(args.out, "SUBSET.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("=" * 72)
    print(f"  新 Arrow → {args.out}   ({time.time() - t0:.0f}s,已读回校验)")
    print("=" * 72)
    for ln in lines:
        print("  " + ln)
    if raw_tb > free_tb:
        print(f"\n  ⚠ 原样 {raw_tb:.2f} TB 放不下(剩 {free_tb:.2f} TB)—— 要么压缩,要么少抽。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
