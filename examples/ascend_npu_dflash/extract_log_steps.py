#!/usr/bin/env python3
"""Cut a huge training log down to WHOLE step records, so it can be shared in one push.

WHY. A two-day run log is ~181 MB. ``archive_log_push.sh`` can ship that, but it splits
into ~100 parts and therefore ~100 commits, because the gateway caps a single push. Most
questions -- "what is accept_len at step 2000", "did grad_norm move between two arms" --
need a few hundred records, not the whole file.

WHY NOT ``grep global_step=``. The rich logger wraps ONE step across ~26 physical lines:

    [02:53:22] INFO     train/confidence_loss=0.377,        trainer.py:799
                        train/loss=5.255, train/ce_loss=37.000,
                        ...
                        lr/AdamW=1.53e-06, global_step=38

Only the LAST line carries ``global_step``. Grepping it keeps that one line and silently
drops train/loss, accept_len, step_ms and everything else -- measured once at
124,480 records -> 0 usable. So this reads RECORDS: a record starts at a line beginning
with ``[`` and runs to just before the next one, and a record is kept or dropped whole.

Streams with a bounded buffer, so the 181 MB file costs no memory.

USAGE
    extract_log_steps.py <log> <out> --steps 1900-2004
    extract_log_steps.py <log> <out> --every 100            # every 100th step
    extract_log_steps.py <log> <out> --last 200             # the final 200 steps
    extract_log_steps.py <log> <out> --every 100 --steps 1900-2004 --head 60

``--head N`` also keeps the first N lines (launcher banner, resolved env, versions),
which is usually the context that makes the numbers readable. Selections are a UNION.

⚠ The output still contains absolute paths with the box account id. Before pushing to
the PUBLIC fork, run it through ``redact_log.py`` -- ``archive_log_push.sh pack``
refuses an unredacted file, but a one-off ``git add`` of this output would not.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ruff: noqa: T201 - a CLI whose whole job is to report what it kept

STEP_RE = re.compile(rb"global_step=(\d+)")


def _parse_steps(spec: str) -> tuple[int, int]:
    lo, _, hi = spec.partition("-")
    return (int(lo), int(hi or lo))


def _iter_records(fh):
    """Yield (lines, step_or_None); a record starts at a line beginning with '['."""
    buf: list[bytes] = []
    for raw in fh:
        if raw.startswith(b"[") and buf:
            m = STEP_RE.search(b"".join(buf))
            yield buf, (int(m.group(1)) if m else None)
            buf = []
        buf.append(raw)
    if buf:
        m = STEP_RE.search(b"".join(buf))
        yield buf, (int(m.group(1)) if m else None)


def _selector(every: int, ranges: list[tuple[int, int]]):
    def keep(step: int) -> bool:
        if every and step % every == 0:
            return True
        return any(lo <= step <= hi for lo, hi in ranges)

    return keep


def _write(log: Path, out_path: Path, head: int, keep) -> tuple[int, int]:
    kept = seen = 0
    with log.open("rb") as fh, out_path.open("wb") as out:
        if head:
            for i, raw in enumerate(fh):
                if i >= head:
                    out.write(b"\n... [head cut] ...\n\n")
                    break
                out.write(raw)
            fh.seek(0)
        for lines, step in _iter_records(fh):
            if step is None:
                continue
            seen += 1
            if keep(step):
                kept += 1
                out.writelines(lines)
    return kept, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", action="append", default=[],
                    help="inclusive range 'A-B' (repeatable)")
    ap.add_argument("--every", type=int, default=0, help="keep every Nth step")
    ap.add_argument("--last", type=int, default=0, help="keep the final N steps")
    ap.add_argument("--head", type=int, default=60,
                    help="also keep the first N lines (banner/env); 0 to disable")
    args = ap.parse_args()

    if not (args.steps or args.every or args.last):
        ap.error("nothing selected: pass at least one of --steps / --every / --last")

    ranges = [_parse_steps(s) for s in args.steps]

    # --last needs the highest step, which is only known after a pass. One extra scan of
    # a 181 MB file is a few seconds and costs no memory; guessing from the tail is not
    # safe because the final record can be a warning rather than a step.
    max_step = 0
    if args.last:
        with args.log.open("rb") as fh:
            for _, step in _iter_records(fh):
                if step is not None and step > max_step:
                    max_step = step
        ranges.append((max(0, max_step - args.last + 1), max_step))

    keep = _selector(args.every, ranges)
    kept, seen = _write(args.log, args.out, args.head, keep)

    src_mb = args.log.stat().st_size / 2**20
    dst_mb = args.out.stat().st_size / 2**20
    print(f"records with a step : {seen}")
    print(f"kept                : {kept}")
    if args.last:
        print(f"highest step seen   : {max_step}")
    ratio = src_mb / max(dst_mb, 1e-9)
    print(f"{src_mb:.1f} MB -> {dst_mb:.3f} MB  ({ratio:.0f}x smaller)")
    print("\n⚠ still unredacted -- redact_log.py before pushing to the public fork.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
