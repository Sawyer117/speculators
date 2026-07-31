#!/usr/bin/env python3
"""Decide train/serve MISMATCH vs draft-QUALITY ceiling — apples-to-apples.

The trap this untangles: the training console logs TWO accept lengths (see
``dspark/metrics.py``):
  * ``accept_len``       — SOFT, E[len] under sampling acceptance ``sum_v min(p,q)``.
                           Systematically OPTIMISTIC (this is the ~3.57 number).
  * ``hard_accept_len``  — HARD greedy: longest block prefix where argmax(draft)
                           == argmax(target). This is EXACTLY what vllm-ascend
                           spec-decode reports at serve (temp=0).
Comparing the SOFT train number (3.57) against the HARD serve number (2.582) is
apples-to-oranges and manufactures a fake ~1.0 gap. The honest comparison is
**train ``hard_accept_len`` vs serve accept_len, on the SAME distribution** (the
trainsample eval == the training rollout distribution).

Verdict:
  * train_hard ≈ serve            -> NO train/serve mismatch. The 3.57 was just the
                                     soft metric being optimistic. The ceiling is
                                     draft QUALITY -> lever = data / recipe / capacity.
  * train_hard >> serve (gap>~0.2) -> REAL train/serve gap on the hard metric ->
                                     a per-slot forward inconsistency -> localize with
                                     the per-slot parity harness
                                     (dsv4_dspark_serve_forward_parity.py).

Per-position localization: prints train ``position_k_acc`` (teacher-forced marginal
per-slot accuracy) next to the serve MARGINAL accept (S_k / S_{k-1}). NOTE these two
are not the identical conditional (train marginal is unconditional over blocks; serve
marginal is conditioned on the accepted prefix) — read the SHAPE, not the exact
equality. The scalar hard_accept_len comparison is the load-bearing verdict.

Nothing here is imported by training/serving — standalone stdlib only, no NPU/torch.

Usage:
  python compare_train_vs_serve.py \
      --train-log /home/n84449292/dsv4_run/faithful_ep_<TS>.log \
      --serve-eval ~/eval_trainsample_1turn.txt
  # average the last N logged train steps (default 20) to smooth step noise:
  python compare_train_vs_serve.py --train-log <log> --serve-eval <txt> --last 20
  # or feed the train numbers directly if you already grepped them:
  python compare_train_vs_serve.py --serve-eval <txt> \
      --train-hard 2.61 --train-soft 3.57 --train-pos 0.70,0.62,0.60,0.58,0.57
"""
import argparse
import re

# Metric keys emitted by dspark/metrics.py after reduce (bare, no _sum/_total).
_BLOCK = 5  # dspark_block_size; position_0..position_{BLOCK-1}


_NUM = r"(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)"


def _one(pat: str, line: str):
    """First numeric value for a key on a line. Word-boundary-anchored so `accept_len`
    does NOT match inside `hard_accept_len` (the `_` before it IS a word char)."""
    m = re.search(r"(?<![\w])" + pat + r"['\"]?\s*[:=]\s*" + _NUM, line)
    return float(m.group(1)) if m else None


def parse_train_log(path: str, last: int, around: int | None, window: int) -> dict:
    """Line-oriented: each metric_logger.info call is ONE line holding global_step +
    all the reduced keys. Collect per-step records so we can window by global_step
    (the served ckpt's step) instead of blindly averaging the over-trained tail."""
    keys = ["hard_accept_len", "accept_len", "accept_rate"] + [f"position_{k}_acc" for k in range(_BLOCK)]
    records: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "hard_accept_len" not in line:
                continue
            rec = {k: _one(k, line) for k in keys}
            rec["global_step"] = _one("global_step", line)
            records.append(rec)

    if not records:
        return {"hard_accept_len": None, "accept_len": None, "accept_rate": None,
                "pos": [None] * _BLOCK, "_n": 0, "_span": None}

    if around is not None:
        sel = [r for r in records if r["global_step"] is not None
               and abs(r["global_step"] - around) <= window]
        note = f"±{window} around global_step {around}"
    else:
        sel = records[-last:] if last > 0 else records
        note = f"last {len(sel)} logged steps (END of log)"
    if not sel:
        sel = records[-last:] if last > 0 else records

    def _avg(key: str):
        vals = [r[key] for r in sel if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    steps = [r["global_step"] for r in sel if r["global_step"] is not None]
    return {
        "hard_accept_len": _avg("hard_accept_len"),
        "accept_len": _avg("accept_len"),
        "accept_rate": _avg("accept_rate"),
        "pos": [_avg(f"position_{k}_acc") for k in range(_BLOCK)],
        "_n": len(sel),
        "_note": note,
        "_span": (min(steps), max(steps)) if steps else None,
    }


def parse_serve_eval(path: str) -> dict:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    out: dict = {}
    m = re.search(r"Accept length:\s*([0-9.]+)", text)
    out["accept_len"] = float(m.group(1)) if m else None
    m = re.search(r"Accept rate:\s*([0-9.]+)%", text)
    out["accept_rate"] = float(m.group(1)) / 100.0 if m else None
    # cumulative per-position accept: "pos 0: 70.37%"
    cum = []
    for k in range(_BLOCK):
        m = re.search(rf"pos\s+{k}\s*:\s*([0-9.]+)%", text)
        cum.append(float(m.group(1)) / 100.0 if m else None)
    out["cum"] = cum
    return out


def marginals(cum: list) -> list:
    """S_k / S_{k-1}; slot 0's marginal == S_0."""
    out = []
    prev = 1.0
    for s in cum:
        if s is None:
            out.append(None)
            prev = None
            continue
        out.append(s / prev if prev else None)
        prev = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", help="training console log (grep hard_accept_len etc.)")
    ap.add_argument("--serve-eval", required=True, help="eval_trainsample_*.txt")
    ap.add_argument("--last", type=int, default=20, help="avg the last N logged steps (0=all)")
    ap.add_argument("--around", type=int, help="global_step of the SERVED ckpt; window the log "
                    "around it instead of the (over-trained) tail — the apples-to-apples point")
    ap.add_argument("--window", type=int, default=200, help="± steps for --around (default 200)")
    ap.add_argument("--train-hard", type=float, help="override: train hard_accept_len")
    ap.add_argument("--train-soft", type=float, help="override: train soft accept_len")
    ap.add_argument("--train-pos", help="override: comma per-pos acc, e.g. 0.70,0.62,0.60,0.58,0.57")
    ap.add_argument("--gap-eps", type=float, default=0.20, help="hard-gap that flags a mismatch")
    args = ap.parse_args()

    serve = parse_serve_eval(args.serve_eval)

    if args.train_log:
        tr = parse_train_log(args.train_log, args.last, args.around, args.window)
    else:
        tr = {"hard_accept_len": None, "accept_len": None, "accept_rate": None,
              "pos": [None] * _BLOCK, "_n": 0, "_note": None, "_span": None}
    if args.train_hard is not None:
        tr["hard_accept_len"] = args.train_hard
    if args.train_soft is not None:
        tr["accept_len"] = args.train_soft
    if args.train_pos:
        tr["pos"] = [float(x) for x in args.train_pos.split(",")]

    smarg = marginals(serve["cum"])

    def f(x, p="{:.3f}"):
        return p.format(x) if isinstance(x, (int, float)) else "  ?  "

    print("=" * 72)
    print("TRAIN (teacher-forced, rollout dist)   vs   SERVE (spec-decode, same dist)")
    if tr.get("_n"):
        span = f" (global_step {tr['_span'][0]:.0f}..{tr['_span'][1]:.0f})" if tr.get("_span") else ""
        print(f"  [train = mean of {tr['_n']} steps: {tr.get('_note', '')}{span}]")
    print("-" * 72)
    print(f"  soft accept_len  (train, OPTIMISTIC)      : {f(tr['accept_len'])}")
    print(f"  hard accept_len  (train, serve-equivalent): {f(tr['hard_accept_len'])}")
    print(f"  accept_len       (SERVE, hard/temp=0)     : {f(serve['accept_len'])}")
    print(f"  accept_rate      train {f(tr['accept_rate'],'{:.3f}')}  |  serve {f(serve['accept_rate'],'{:.3f}')}")
    print("-" * 72)
    print("  per-position   train pos_acc (marginal)   serve marginal (S_k/S_k-1)   serve cum")
    for k in range(_BLOCK):
        print(f"    pos {k}:      {f(tr['pos'][k]):>10}                {f(smarg[k]):>10}            {f(serve['cum'][k])}")
    print("=" * 72)

    th, sv = tr["hard_accept_len"], serve["accept_len"]
    if th is None or sv is None:
        print("VERDICT: missing train hard_accept_len or serve accept_len — "
              "pass --train-hard / check the log has 'hard_accept_len'.")
        return
    gap = th - sv
    print(f"HARD gap (train_hard - serve) = {gap:+.3f}")
    if abs(gap) <= args.gap_eps:
        print("  => NO train/serve mismatch. The 3.57 was the SOFT metric being")
        print("     optimistic; hard train ≈ serve. Ceiling = draft QUALITY.")
        print("     LEVER: data scale / recipe (LR anneal, epochs) / capacity — NOT a serve bug.")
        if tr["accept_len"] and th:
            print(f"     (soft-vs-hard train gap alone = {tr['accept_len'] - th:+.3f}, explains the illusion.)")
    elif gap > args.gap_eps:
        print("  => REAL train/serve gap on the HARD metric. A per-slot forward")
        print("     inconsistency (train forward != serve forward). LOCALIZE with the")
        print("     per-slot parity: dsv4_dspark_serve_forward_parity.py on one real block;")
        print("     prime suspects = norm convention (double-norm's win points here), RoPE")
        print("     per-slot offset (#72), block-internal attn/KV base.")
        print("     Read the per-position table: the slot where train pos_acc stays high but")
        print("     serve marginal collapses is where the forwards diverge.")
    else:
        print("  => serve HARDER than train-hard (unexpected). Re-check same distribution / draft.")


if __name__ == "__main__":
    main()
