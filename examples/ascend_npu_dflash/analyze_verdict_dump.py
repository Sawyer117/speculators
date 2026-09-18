#!/usr/bin/env python3
"""Compare two drafts' per-slot accept/reject dumps: where does OURS lose that theirs wins?

Input = the ``verdict_<tag>_<pid>.bin`` streams written by vllm-ascend's ``DsparkVerdictDumper``
(``DSPARK_VERDICT_DUMP=1``), one tag per draft, both from the SAME eval on the SAME stack.

WHAT THIS ANSWERS THAT /metrics CANNOT. The serve's per-position accept rate says slot k
survives x% of the time. It cannot say *on which tokens*, and — the part that decides what to
do next — whether the TARGET was even sure there. Those two cases demand opposite moves:

    target confident (p>0.9), draft missed  ->  the draft is weak. Trainable. Worth money.
    target unsure    (p<0.5), draft missed  ->  nobody predicts that position. Chasing it is
                                                chasing noise.

★ JOINING THE TWO RUNS. Their request ids are freshly generated per run and do not match, and
their block boundaries never line up either (released is block-5 at ns=5, ours is block-15 at
ns=15). What IS identical is the OUTPUT TOKEN SEQUENCE — greedy verification reproduces the
target's own autoregressive output exactly, so both runs emit the same tokens for the same
prompt. So: rebuild each request's output stream from its rows, hash it, and match runs on
that hash. Positions then align by ``out_idx``, which is the only alignment that survives two
different block sizes.

⚠️⚠️ THE TAIL OF EVERY STREAM DIFFERS AND THAT IS NORMAL. A step emits up to K+1 tokens, so
when generation stops mid-block the tokens past the stop point are emitted by the sampler,
recorded here, and only then discarded by the scheduler. Different block sizes overshoot by
different amounts, so the last token or two never agree. Measured on synthetic runs built from
IDENTICAL streams: 36/40 requests differed, every one of them ONLY in the final token.
Therefore: the join key is a PREFIX hash (``--key-len``), and the comparison drops a whole
block's worth off the end of the shorter stream. Matching on the full stream silently rejects
~90% of requests and leaves a confident-looking analysis of the 10% that happened to line up.

⚠️ Requests that still fail to match are REPORTED, not silently dropped -- a large unmatched
fraction means the two runs did not see the same work (different dataset, concurrency-truncated
run, or a temperature that was not 0), and every number below would then be meaningless.

USAGE
    python analyze_verdict_dump.py <dir> --ours ours-blk15-ep5 --theirs released-blk5
    python analyze_verdict_dump.py <dir> --ours <tag>                  # single-run summary
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from collections import Counter

import numpy as np

# Confidence bands for the target's own top-1 probability. The boundaries are reporting
# choices, not thresholds anything keys off -- the point is only to separate "the target knew"
# from "the target did not", and anything in between is explicitly the grey zone.
BANDS = ((0.9, 1.01, "目标很确定 >0.9"), (0.5, 0.9, "中间 0.5-0.9"), (-0.01, 0.5, "目标也不确定 <0.5"))


# ⚠ 必须与 vllm_ascend/dspark_verdict_dumper.py 的 _REC 逐字段一致(名字、顺序、宽度)。
# 这是定长二进制流,没有自描述头部 —— 对不上不会报错,只会读出一堆看似合理的垃圾。
_REC = np.dtype([
    ("step", "<i4"), ("req", "<i4"), ("out_idx", "<i4"),
    ("draft_tok", "<i4"), ("target_tok", "<i4"), ("bonus_tok", "<i4"),
    ("slot", "i1"), ("accepted", "?"),
    ("target_top1_p", "<f2"), ("target_p_draft", "<f2"),
])


def load(dirpath: str, tag: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(dirpath, f"verdict_{tag}_*.bin")))
    if not files:
        raise SystemExit(f"!! 没找到 verdict_{tag}_*.bin in {dirpath}\n"
                         f"   (旧版写的是 .npz 分片 —— 那批 dump 的 out_idx 不可用,重采)")
    parts: dict[str, list] = {}
    # `req` 是每个 writer 进程各自从 0 开始编号的行号,所以按文件累加偏移,否则两个进程的
    # 请求 0 会合成一条不存在的流。偏移量取 reqs_*.txt 的行数 —— 用它而不是 req.max()+1,
    # 因为最后登记的请求可能一行都还没轮到发 token。
    off = 0
    for f in files:
        a = np.fromfile(f, dtype=_REC)   # 末尾不足一条记录的残片会被自动丢弃
        rq = f.replace("verdict_", "reqs_")[: -len(".bin")] + ".txt"
        n_ids = sum(1 for _ in open(rq)) if os.path.exists(rq) else \
            (int(a["req"].max()) + 1 if len(a) else 0)
        for k in _REC.names:
            v = a[k].astype(np.int64) + off if k == "req" else a[k]
            parts.setdefault(k, []).append(v)
        off += n_ids
    d = {k: np.concatenate(v) for k, v in parts.items()}
    print(f">>> {tag}: {len(files)} 个流, {len(d['step']):,} 行, {len(np.unique(d['req'])):,} 个请求")
    return d


def rebuild_streams(d: dict[str, np.ndarray]) -> dict[int, list[int]]:
    """request -> the token sequence it emitted. See the module docstring for why this is the key."""
    streams: dict[int, list[int]] = {}
    order = np.lexsort((d["slot"], d["step"], d["req"]))
    req, step, slot = d["req"][order], d["step"][order], d["slot"][order]
    dk, tk, bk, ac = d["draft_tok"][order], d["target_tok"][order], d["bonus_tok"][order], d["accepted"][order]
    # One pass over the sorted rows; group boundaries are where (req, step) changes.
    bounds = np.flatnonzero(np.diff(req) | np.diff(step)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(req)]):
        n = int(ac[lo:hi].sum())
        out = streams.setdefault(int(req[lo]), [])
        out.extend(dk[lo:hi][:n].tolist())
        out.append(int(tk[lo + n]) if n < hi - lo else int(bk[lo]))
    return streams


def band_of(p: np.ndarray) -> np.ndarray:
    b = np.full(len(p), len(BANDS) - 1, dtype=np.int8)
    for i, (lo, hi, _) in enumerate(BANDS):
        b[(p >= lo) & (p < hi)] = i
    return b


def per_run_summary(d: dict[str, np.ndarray], tag: str) -> None:
    acc, slot, p1 = d["accepted"], d["slot"], d["target_top1_p"].astype(np.float32)
    print(f"\n── {tag} ──")
    print(f"  总体逐 slot 接受率 = {100 * acc.mean():.2f}%   (行 = 被提出的草稿 slot)")
    print(f"  {'slot':>5}{'提出':>10}{'接受率':>9}   按目标置信度分:" +
          "".join(f"{n:>16}" for *_, n in BANDS))
    bands = band_of(p1)
    for s in range(int(slot.max()) + 1):
        m = slot == s
        row = f"  {s:>5}{int(m.sum()):>10,}{100 * acc[m].mean():>8.1f}%   "
        for i in range(len(BANDS)):
            mb = m & (bands == i)
            row += f"{(f'{100*acc[mb].mean():.0f}% / n={int(mb.sum()):,}' if mb.sum() else '—'):>16}"
        print(row)


def cross_tab(ours, theirs, args) -> None:
    so, st = rebuild_streams(ours), rebuild_streams(theirs)

    def key(v):
        # Prefix only, and never the final token: the last one is the overshoot that differs
        # between block sizes. Short streams fall back to everything-but-the-last.
        n = max(1, min(args.key_len, len(v) - 1))
        return hashlib.blake2b(np.asarray(v[:n], np.int64).tobytes(), digest_size=8).digest()

    ho = {key(v): k for k, v in so.items()}
    ht = {key(v): k for k, v in st.items()}
    shared = set(ho) & set(ht)
    print(f"\n>>> 输出流匹配: {len(shared):,} / 我们 {len(so):,} / 对方 {len(st):,}")
    if not shared:
        raise SystemExit("!! 一个都没匹配上 —— 两次 run 不是同一份工作(数据集/温度/截断不同?)")
    frac = len(shared) / max(len(so), len(st))
    if frac < 0.9:
        print(f"    ⚠️ 只匹配上 {100*frac:.0f}% —— 低于 90%,下面的数要打折看。"
              "常见原因:两次跑的数据集不同、并发尾部截断、或温度不是 0。")

    def index(d, keep):
        m = np.isin(d["req"], list(keep))
        return {k: v[m] for k, v in d.items()}

    o = index(ours, {ho[h] for h in shared}); t = index(theirs, {ht[h] for h in shared})
    # Re-key both sides onto the SHARED stream hash so (request, out_idx) means the same thing.
    o_key = np.array([ho_inv[r] for r in o["req"]]) if (ho_inv := {v: k for k, v in ho.items()}) else None
    t_key = np.array([ht_inv[r] for r in t["req"]]) if (ht_inv := {v: k for k, v in ht.items()}) else None

    # ★★ KEEP ONLY THE ROWS WHOSE PREFIX WAS CORRECT. An output position is drafted MORE THAN
    # ONCE: a step starting at position p covers p..p+K, and if only 3 slots are accepted the
    # next step starts at p+3 and drafts that region again. So position p+8 is proposed once at
    # slot 8 -- conditioned on five tokens the draft itself invented, which the target already
    # rejected -- and again at slot 5 of the next step, conditioned on the true prefix. Those
    # are different questions, and only the second one is "can the draft predict this token".
    # The rows with a correct prefix are exactly: accepted, or the step's FIRST rejection.
    # (Taking the first occurrence instead scores mostly wrong-prefix rows and collapses the
    # measured accept rate -- 43 vs the ~69% the per-slot summary reports on the same data.)
    def valid_prefix(d):
        order = np.lexsort((d["slot"], d["step"], d["req"]))
        ok = np.zeros(len(order), dtype=bool)
        req, step, acc = d["req"][order], d["step"][order], d["accepted"][order]
        bounds = np.flatnonzero(np.diff(req) | np.diff(step)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(order)]):
            n = int(acc[lo:hi].sum())            # accepted prefix length
            ok[lo:lo + min(n + 1, hi - lo)] = True   # the accepted run, plus the first rejection
        m = np.zeros(len(order), dtype=bool); m[order] = ok
        return m

    def first_by_pos(keys, d):
        seen, idx = {}, []
        for i, (k, oi) in enumerate(zip(keys, d["out_idx"])):
            kk = (k, int(oi))
            if kk not in seen:
                seen[kk] = i; idx.append(i)
        return {c: v[idx] for c, v in d.items()}, [k for k in seen]

    # Drop a full block off the end of each stream before comparing: those positions exist in
    # one run and not the other (or carry the discarded overshoot), and including them would
    # score the block-size difference as a draft-quality difference.
    kmax = int(max(o["slot"].max(), t["slot"].max())) + 1
    lim_o = {ho_inv[r]: len(so[r]) - 1 - kmax for r in np.unique(o["req"])}
    lim_t = {ht_inv[r]: len(st[r]) - 1 - kmax for r in np.unique(t["req"])}
    lim = {k: min(lim_o.get(k, 0), lim_t.get(k, 0)) for k in set(lim_o) & set(lim_t)}
    keep_o = np.array([lim.get(k, -1) > oi for k, oi in zip(o_key, o["out_idx"])])
    keep_t = np.array([lim.get(k, -1) > oi for k, oi in zip(t_key, t["out_idx"])])
    dropped = int((~keep_o).sum())
    o = {c: v[keep_o] for c, v in o.items()}; o_key = o_key[keep_o]
    t = {c: v[keep_t] for c, v in t.items()}; t_key = t_key[keep_t]
    print(f"    尾部剔除: 每条流末尾 {kmax + 1} 个位置不参与比较(块越界),我们这边少了 {dropped:,} 行")

    vo, vt_ = valid_prefix(o), valid_prefix(t)
    print(f"    前缀过滤: 只留前缀正确的行 —— 我们 {int(vo.sum()):,}/{len(vo):,}, "
          f"对方 {int(vt_.sum()):,}/{len(vt_):,}(其余是在草稿自己编错的前缀上提出的,不可比)")
    o = {c: v[vo] for c, v in o.items()}; o_key = o_key[vo]
    t = {c: v[vt_] for c, v in t.items()}; t_key = t_key[vt_]

    od, okeys = first_by_pos(o_key, o); td, tkeys = first_by_pos(t_key, t)
    omap = {k: i for i, k in enumerate(okeys)}
    common = [(omap[k], j) for j, k in enumerate(tkeys) if k in omap]
    if not common:
        raise SystemExit("!! 没有共同的 (流, out_idx) 位置")
    oi = np.array([a for a, _ in common]); ti = np.array([b for _, b in common])
    oa, ta = od["accepted"][oi], td["accepted"][ti]
    p1 = od["target_top1_p"][oi].astype(np.float32)   # 目标是同一个模型,两边一致,取一边即可
    bands = band_of(p1)

    print(f"\n>>> 逐位置交叉表(共同位置 {len(oi):,} 个)")
    print(f"  {'':<14}{'对方接受':>12}{'对方拒绝':>12}")
    print(f"  {'我们接受':<14}{int((oa&ta).sum()):>12,}{int((oa&~ta).sum()):>12,}")
    print(f"  {'我们拒绝':<14}{int((~oa&ta).sum()):>12,}  ★{int((~oa&~ta).sum()):>10,}")
    print(f"  ★ C 格(对方能、我们不能)= {int((~oa&ta).sum()):,} 个位置 = 改进清单")

    print(f"\n>>> C 格按目标置信度拆开 —— 决定哪些值得追")
    C = ~oa & ta
    for i, (*_, name) in enumerate(BANDS):
        m = C & (bands == i)
        tot = int((bands == i).sum())
        note = "  ← 可修,训练能买到" if i == 0 else ("  ← 别追,是噪声" if i == 2 else "")
        print(f"  {name:<18} {int(m.sum()):>9,} / {tot:>9,} 该带位置 = {100*m.sum()/max(tot,1):>5.2f}%{note}")

    hi = C & (bands == 0)
    if hi.sum():
        toks = Counter(od["target_tok"][oi][hi].tolist()).most_common(args.top)
        print(f"\n>>> ★ C ∩ 高置信 的目标 token top-{args.top}(id: 次数) —— 我们最该学会的东西")
        print("    " + "  ".join(f"{t}:{c}" for t, c in toks))
        print("    (拿 tokenizer 解码一下就知道是数字、标点、换行还是别的)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--theirs", default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--key-len", type=int, default=64,
                    help="用输出流的前 N 个 token 做 join key(默认 64)。不能用整条流:尾部必然不同")
    args = ap.parse_args()

    ours = load(args.dir, args.ours)
    per_run_summary(ours, args.ours)
    if args.theirs:
        theirs = load(args.dir, args.theirs)
        per_run_summary(theirs, args.theirs)
        cross_tab(ours, theirs, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
