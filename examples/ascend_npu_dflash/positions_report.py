#!/usr/bin/env python3
"""把 prefill_noise_probe 的逐位置 CSV 切开:失效是随【序列长度】来的,还是随【位置】来的?

WHY
---
2026-09-23 第一次跑 prefill_noise_sweep 出的汇总数自相矛盾:整体翻转率 78.8%,而
margin>2 nat 的「笃定」档也翻了 61.8% —— 末位数值抖动翻不动那种位置。同时 bg=4
(加了背景负载)反而比 bg=0 好一倍。两条都说明汇总层面读不出真相。

但逐行那一列是自洽的:262/276/330 token 的行翻转 ~2%,而 2010–2048 token 的行翻转
82–99.7%。所以问题跟长度强相关。这个脚本把 CSV 按两个轴切开,定位它到底跟什么走:

  * 按【行】—— 每行的长度 vs 翻转率 vs 语料一致率。长度单调 ⟹ 长序列路径的问题。
  * 按【位置】—— 位置分桶。如果翻转集中在某个位置之后(比如 512、或某个分块边界),
    那就不是「长序列整体更脏」,而是**越过某条线之后开始算错**,那条线就是线索。
  * 按【margin】—— 笃定档还翻多少。这是判断整组数可不可用的开关。

USAGE
  python positions_report.py ~/prefill_noise/positions_bg0.csv ~/prefill_noise/positions_bg4.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

POS_BUCKETS = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 1 << 30)]
MARGIN_BUCKETS = [(0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 2.0), (2.0, float("inf"))]


def pct(n: int, d: int) -> str:
    return "  n/a " if d == 0 else f"{100.0 * n / d:6.2f}%"


def report(path: str) -> None:
    rows: dict[int, list[int]] = {}          # row -> [flip, cmp, match, match_cmp, maxpos]
    posb = [[0, 0, 0, 0] for _ in POS_BUCKETS]   # flip, cmp, match, match_cmp
    marb = [[0, 0] for _ in MARGIN_BUCKETS]
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                row = int(r["row"]); p = int(r["pos"]); fl = int(r["flipped"])
            except (KeyError, ValueError):
                continue
            st = rows.setdefault(row, [0, 0, 0, 0, 0])
            st[0] += fl; st[1] += 1; st[4] = max(st[4], p)
            cm = r.get("corpus_match", "")
            has_cm = cm not in ("", None)
            if has_cm:
                st[2] += int(cm); st[3] += 1
            for bi, (lo, hi) in enumerate(POS_BUCKETS):
                if lo <= p < hi:
                    posb[bi][0] += fl; posb[bi][1] += 1
                    if has_cm:
                        posb[bi][2] += int(cm); posb[bi][3] += 1
                    break
            m = r.get("max_abs_dlogprob")  # 占位,保持列顺序可读
            mg = r.get("margin", "")
            if mg not in ("", None):
                try:
                    mv = float(mg)
                except ValueError:
                    mv = None
                if mv is not None:
                    for bi, (lo, hi) in enumerate(MARGIN_BUCKETS):
                        if lo <= mv < hi:
                            marb[bi][0] += fl; marb[bi][1] += 1
                            break
            del m

    print("=" * 84)
    print(f"  {os.path.basename(path)}")
    print("=" * 84)
    print("\n按【行】—— 长度是不是决定因素")
    print(f"  {'row':>8} {'长度':>7} {'翻转率':>9} {'语料一致':>9}")
    for row in sorted(rows, key=lambda r: rows[r][4]):
        f, c, m, mc, mx = rows[row]
        print(f"  {row:>8} {mx + 1:>7} {pct(f, c):>9} {pct(m, mc):>9}")

    print("\n按【位置】—— 是整条都脏,还是越过某条线之后才脏")
    print(f"  {'位置区间':>14} {'样本':>8} {'翻转率':>9} {'语料一致':>9}")
    for (lo, hi), (f, c, m, mc) in zip(POS_BUCKETS, posb):
        if c:
            hi_s = "∞" if hi > 1 << 20 else str(hi)
            print(f"  {lo:>6}–{hi_s:<7} {c:>8} {pct(f, c):>9} {pct(m, mc):>9}")

    print("\n按【margin】—— 笃定档还翻多少(>2 nat 本不该翻)")
    names = ["≤0.01", "0.01–0.1", "0.1–0.5", "0.5–2", ">2 笃定"]
    for nm, (f, c) in zip(names, marb):
        if c:
            print(f"  {nm:>10} {c:>8} {pct(f, c):>9}")
    conf_f, conf_c = marb[-1]
    if conf_c and conf_f / conf_c > 0.05:
        print(f"\n  ★ 笃定档翻转 {pct(conf_f, conf_c).strip()} —— 这组数不能用来判语料。")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    args = ap.parse_args()
    for p in args.csv:
        if not os.path.isfile(p):
            print(f"!! 找不到 {p}")
            continue
        report(p)
    print("读法")
    print("  翻转率随长度单调上升、但每个位置桶内部差不多  ⟹ 问题是【整条序列有多长】。")
    print("  翻转集中在某个位置之后(桶之间有断崖)          ⟹ 越过那条线才开始算错,")
    print("                                                    那条线就是根因的坐标。")
    print("  笃定档(>2 nat)翻转 >5%                        ⟹ 不是数值抖动,先别碰语料结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
