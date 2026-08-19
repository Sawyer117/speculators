#!/usr/bin/env python3
"""Join a serve-side DSPARK_TOPK_DUMP into the one number that decides path selection.

    python topk_headroom_join.py /tmp/dspark_topk            # a dir of topk_rank*.pt
    python topk_headroom_join.py /tmp/dspark_topk/topk_rank0.pt --k 8

THE QUESTION
------------
Our ``num_spec=7`` result put every token we lose to the released draft at the block
positions training never covered, and we read that as a block-width mismatch. inco.ai's
DFlash2 post reports a second, orthogonal gap on the same symptom: on their drafter,
GSM8K position-6 recall is 72.9% at top-1 but **87.8% at top-16** — the right token is
usually still among the candidates, and ``argmax`` is what throws it away. Their number
is on their model at temperature 1.0 against a DSpark baseline that scores below Qwen's
native MTP on their own 27B table, so none of it transfers. Whether OUR draft carries
that headroom is a property of our weights, and this measures it on the serving stack.

WHAT IT REPORTS
---------------
The proposer never sees the target's logits, so per-position recall@k is not
computable at serve. What IS computable — and is the quantity a selector would have to
act on — is the **first-miss recall**:

    P( the target's token at the FIRST REJECTED position was in the draft's top-k there )

  high -> ``argmax`` is discarding a token the draft had already ranked. A selector has
          something real to recover; cost out path selection.
  low  -> the draft does not know that token at all. Nothing to select from, and the
          block-width retrain stays the only lever.

Also printed: the **rank distribution** of the target token when it was recovered (how
deep a selector must look — if the mass sits at ranks 1-3, k=4 buys nearly all of it and
the per-step top-k cost drops), and the first-miss position histogram (does the failure
concentrate at the tail, as our per-position conditional acceptance says it should?).

WHAT IT CANNOT TELL YOU
-----------------------
Only the FIRST miss. vLLM stops at the first mismatch and never reveals the target's
token further into the block, so the full ``oracle_accept_len`` is not reconstructible
from a serve dump. Recovering one miss extends the block by **at least** one token, so
the accept-length gain implied here is a lower bound; the upper bound is the
teacher-forced ``oracle_accept_len_16`` from ``recall_headroom_probe.py``. The truth is
between them, and the two are worth reading together.

⚠ The join offset is the one thing that can silently produce a plausible wrong answer:
a round's correction judges the block drafted the round BEFORE. The dump carries
``verify["describes_round"]`` precisely so this script never has to infer it — it
matches that field to ``cand["round"]`` and refuses to guess.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

import torch


def load_dumps(path: str) -> list[dict]:
    files = (
        sorted(glob.glob(os.path.join(path, "topk_rank*.pt")))
        if os.path.isdir(path)
        else [path]
    )
    if not files:
        sys.exit(f"no topk_rank*.pt under {path}")
    out = []
    for f in files:
        d = torch.load(f, weights_only=False)
        d["_file"] = f
        out.append(d)
    return out


def analyse(dump: dict, k_use: int | None) -> dict:
    block_size = int(dump["block_size"])
    k_dumped = int(dump["k"])
    k = min(k_use or k_dumped, k_dumped)

    by_round = {int(c["round"]): c for c in dump["cand"]}
    stats = {
        "block_size": block_size,
        "k_dumped": k_dumped,
        "k_used": k,
        "blocks": 0,  # blocks with at least one rejected token
        "full_accept": 0,  # blocks the target took whole (no first miss to study)
        "hit": 0,  # first-miss token was inside top-k
        "rank_hist": Counter(),  # rank of the target token when recovered (0-based)
        "pos_hist": Counter(),  # which position the first miss happened at
        "pos_hit": Counter(),  # ... and how often it was recoverable there
        "unmatched": 0,
    }

    for ver in dump["verify"]:
        r = ver.get("describes_round")
        if r is None:  # a dump from before the field existed: refuse to guess
            sys.exit(
                "dump lacks verify['describes_round'] — it predates the explicit join "
                "key. Re-dump with the current proposer; inferring the offset here "
                "risks a silent off-by-one."
            )
        cand = by_round.get(int(r))
        if cand is None:
            stats["unmatched"] += 1
            continue

        rejected = ver["rejected"]
        nxt = ver["next_token_ids"]
        n = min(int(cand["num_reqs"]), int(ver["num_reqs"]))
        if rejected is None:
            stats["unmatched"] += 1
            continue

        for i in range(n):
            rej = int(rejected[i])
            if rej <= 0:
                stats["full_accept"] += 1
                continue
            first_miss = block_size - rej
            if not (0 <= first_miss < block_size):
                stats["unmatched"] += 1
                continue

            target_tok = int(nxt[i])
            cands = cand["ids"][i, first_miss, :k]
            stats["blocks"] += 1
            stats["pos_hist"][first_miss] += 1

            where = (cands == target_tok).nonzero()
            if where.numel():
                rank = int(where[0])
                stats["hit"] += 1
                stats["rank_hist"][rank] += 1
                stats["pos_hit"][first_miss] += 1

    return stats


def report(stats: dict, label: str) -> None:
    n = stats["blocks"]
    print(f"\n=== {label} ===")
    print(
        f"blocks with a rejection: {n}   |   fully-accepted blocks: {stats['full_accept']}"
        f"   |   unmatched: {stats['unmatched']}"
    )
    if not n:
        print("  no rejected blocks — nothing to measure (serve longer, or the draft is perfect)")
        return

    rate = stats["hit"] / n
    print(f"block_size={stats['block_size']}  k={stats['k_used']} (dumped {stats['k_dumped']})")
    print(f"\n★ FIRST-MISS RECALL@{stats['k_used']} = {stats['hit']}/{n} = {rate:.1%}")
    print(
        "   = when argmax lost the block, how often the target's token was still among "
        "the candidates"
    )

    print("\n-- rank of the target token when recovered (how deep a selector must look) --")
    cum = 0
    for r in sorted(stats["rank_hist"]):
        cum += stats["rank_hist"][r]
        print(f"   rank {r + 1:>2}: {stats['rank_hist'][r]:>6}   cumulative {cum / n:6.1%} of all misses")

    print("\n-- where the first miss happens (compare to per-position conditional acceptance) --")
    for p in sorted(stats["pos_hist"]):
        tot, hit = stats["pos_hist"][p], stats["pos_hit"][p]
        print(f"   pos {p}: {tot:>6} misses   recoverable {hit / tot:6.1%}")

    print("\n-- read it --")
    if rate >= 0.50:
        print(f"   {rate:.0%} of lost blocks were recoverable from candidates the draft already")
        print("   produced. The headroom is real; cost out path selection against the added")
        print("   host-side latency (_sample_sequential is eager and outside the ACLGraph).")
    elif rate <= 0.20:
        print(f"   only {rate:.0%} recoverable — the draft does not know the token, it is not")
        print("   merely mis-ranking it. Path selection is dead for us; the block-width")
        print("   retrain stays the lever. This is the decisive negative; stop here.")
    else:
        print(f"   {rate:.0%} is the ambiguous middle. Check the rank histogram: mass at ranks")
        print("   1-3 still makes a cheap k=4 selector worth pricing; mass spread to rank 16")
        print("   means an expensive selector for a modest, uncertain return.")
    print("\n   Lower bound only: recovering one miss extends a block by >=1 token. The upper")
    print("   bound is oracle_accept_len_16 from recall_headroom_probe.py (teacher-forced).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="dir holding topk_rank*.pt, or one .pt file")
    ap.add_argument("--k", type=int, default=None, help="evaluate at a smaller k than was dumped")
    args = ap.parse_args()

    for dump in load_dumps(args.path):
        report(analyse(dump, args.k), os.path.basename(dump["_file"]))


if __name__ == "__main__":
    main()
