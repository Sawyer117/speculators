#!/usr/bin/env python3
"""真 HS 文件上量压缩率和吞吐 —— 「一边 roll 一边 tar,用时再 untar」值不值。

WHY
---
预存 HS 的唯一障碍是盘。「tar 一下」是最自然的想法,但**通用无损压缩在浮点数据上有
硬上限**:bf16 = 1 符号 + 8 指数 + 7 尾数,指数集中所以能压,**尾数是均匀随机的 7 个
比特,信息论上压不动**,而尾数占一半体积。合成数据上实测 gzip 1.25×、字节拆分 1.40×。

但合成数据不算数 —— 真 hidden state 有离群通道、有稀疏性、逐层分布也不同。这个脚本在
**真文件**上量,并且同时量**解压吞吐**:如果打算训练时在线 untar,解压速度必须跟得上
消费速率,否则就是把一个不是瓶颈的东西(HS fetch 占单步 1.0%)变成瓶颈。

方法里特意包含「字节平面拆分」:把每个 bf16 的高字节(符号+指数)和低字节(尾数)分成
两个平面各自压。高字节熵低能压 2× 以上,低字节压不动 —— 拆开压比混在一起压好,而且这个
对比本身就直观说明了「剩下那一半为什么压不动」。

USAGE
-----
    python hs_compress_bench.py --hs-dir <dir>            # 默认取 3 个文件
    python hs_compress_bench.py --hs-dir <dir> --files 8 --with-xz

纯 CPU。xz 很慢(单线程几 MB/s),默认不跑,用 --with-xz 打开。
"""

from __future__ import annotations

import argparse
import glob
import gzip
import lzma
import os
import sys
import time
import zlib
from pathlib import Path


def _mbps(nbytes: int, secs: float) -> float:
    return nbytes / 2**20 / max(secs, 1e-9)


def byte_split(buf: bytes) -> tuple[bytes, bytes]:
    """bf16 的高字节平面 / 低字节平面。小端:每 2 字节里 [低, 高]。"""
    mv = memoryview(buf)
    return bytes(mv[1::2]), bytes(mv[0::2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hs-dir", required=True)
    ap.add_argument("--files", type=int, default=3, help="量几个文件(取中位)")
    ap.add_argument("--with-xz", action="store_true", help="也试 xz(很慢)")
    args = ap.parse_args()

    fs = sorted(glob.glob(f"{args.hs_dir}/hs_*.safetensors"))
    if not fs:
        print(f"!! {args.hs_dir} 里没有 hs_*.safetensors", file=sys.stderr)
        return 2
    # 取中间大小的几个,避开最小/最大那种不典型的
    fs.sort(key=os.path.getsize)
    mid = len(fs) // 2
    pick = fs[max(0, mid - args.files // 2): max(0, mid - args.files // 2) + args.files]

    methods: list[tuple[str, object, object]] = [
        ("gzip -1", lambda b: gzip.compress(b, 1), gzip.decompress),
        ("gzip -6", lambda b: gzip.compress(b, 6), gzip.decompress),
        ("zlib -6", lambda b: zlib.compress(b, 6), zlib.decompress),
    ]
    try:
        import zstandard as zstd  # noqa: PLC0415

        for lvl in (1, 3):
            methods.append((f"zstd -{lvl}",
                            (lambda L: lambda b: zstd.ZstdCompressor(level=L).compress(b))(lvl),
                            lambda b: zstd.ZstdDecompressor().decompress(b)))
    except ImportError:
        print("(没装 zstandard —— 只测 gzip/zlib。zstd 通常同压缩比下快 3-5 倍,"
              "真要上线值得装:pip install zstandard)\n")
    if args.with_xz:
        methods.append(("xz -1", lambda b: lzma.compress(b, preset=1), lzma.decompress))

    print(f"样本 {len(pick)} 个文件,大小 "
          + ", ".join(f"{os.path.getsize(f)/2**20:.0f} MiB" for f in pick) + "\n")
    print(f"{'方法':<22} {'压缩比':>7}  {'压 MiB/s':>9}  {'解压 MiB/s':>11}")
    print("-" * 56)

    results = {}
    for name, comp, decomp in methods:
        ratios, cms, dms = [], [], []
        for f in pick:
            raw = Path(f).read_bytes()
            t0 = time.time(); out = comp(raw); tc = time.time() - t0
            t0 = time.time(); back = decomp(out); td = time.time() - t0
            assert back == raw, f"{name} 解压回来对不上 —— 这个方法不能用"
            ratios.append(len(raw) / len(out))
            cms.append(_mbps(len(raw), tc)); dms.append(_mbps(len(raw), td))
        ratios.sort(); cms.sort(); dms.sort()
        r = ratios[len(ratios) // 2]
        results[name] = r
        print(f"{name:<22} {r:7.2f}×  {cms[len(cms)//2]:9.0f}  {dms[len(dms)//2]:11.0f}")

    # ── 字节平面拆分 ──────────────────────────────────────────────────────────
    ratios, cms = [], []
    hi_r = lo_r = 0.0
    for f in pick:
        raw = Path(f).read_bytes()
        t0 = time.time()
        hi, lo = byte_split(raw)
        ch, cl = gzip.compress(hi, 6), gzip.compress(lo, 6)
        tc = time.time() - t0
        ratios.append(len(raw) / (len(ch) + len(cl)))
        cms.append(_mbps(len(raw), tc))
        hi_r, lo_r = len(hi) / len(ch), len(lo) / len(cl)
    ratios.sort(); cms.sort()
    r = ratios[len(ratios) // 2]
    results["byte-split + gzip -6"] = r
    print(f"{'byte-split + gzip -6':<22} {r:7.2f}×  {cms[len(cms)//2]:9.0f}  "
          f"{'—':>11}")
    print(f"{'':<22} 高字节(符号+指数) {hi_r:.2f}×   低字节(尾数) {lo_r:.2f}×")
    if lo_r < 1.05:
        print(f"{'':<22} ⟹ 尾数确实压不动。剩下的路只有【量化】,不是换压缩算法。")

    best = max(results.items(), key=lambda kv: kv[1])
    print(f"\n最好 {best[0]} = {best[1]:.2f}×")
    print("换算:全量 77W bf16 ≈ 19.6 TB(772,684 行 × 776 token × 32 KB)"
          f" → {19.6 / best[1]:.1f} TB")
    print("\n⚠ 在线 untar 的前提是解压吞吐跟得上消费速率。训练侧 HS fetch 只占单步的"
          " ~1.0%,\n   压缩把它变成瓶颈的话就得不偿失 —— 上面那列解压 MiB/s 就是判据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
