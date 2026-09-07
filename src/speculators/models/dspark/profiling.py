"""DSPARK_PROFILE_FWD=1 — an EXHAUSTIVE breakdown of the draft forward.

WHY IT IS BUILT THIS WAY. The first version printed one line per instrumented call and
nothing else, so the only way to know whether the points covered the forward was to add up
the prints by hand and compare against `profile/fwd_ms`. Twice they did not: four in-layer
points caught 630 ms of 6,900, and nine points caught 730 ms of 7,100. Each round of "add a
few more points" is a guess about where the time is, and a guess that lands on the wrong
place looks exactly like a guess that lands on the right one until the arithmetic is done.

So the design is now: the `TOP.*` points **partition the whole forward by construction**,
and `end()` prints the total, the parts, and **UNACCOUNTED = total - sum(parts)** on one
line. If UNACCOUNTED is large, the partition is wrong and the line says so immediately
instead of after another run. The finer points (`MLA.*`, `MoE.*`, `OUT.*`) are reported on a
second line against `TOP.backbone`, with their own UNACCOUNTED.

Nesting is handled by keeping the two tiers in separate buckets, so an inner point inside
`TOP.backbone` is never double-counted against the total.

⚠ Under activation checkpointing the layer forwards run AGAIN in backward, and those calls
land in the accumulator after `end()` has printed. `begin()` clears, so they are discarded
at the next forward rather than polluting it.

ENV
  DSPARK_PROFILE_FWD=1       turn it on (no-op, no syncs, when off)
  DSPARK_PROFILE_FWD_MS=<n>  ALSO print each call over n ms; 0 (default) = summary only
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist

# ruff: noqa: T201 - a profiler whose entire output is prints

ON = os.environ.get("DSPARK_PROFILE_FWD") == "1"
_PER_CALL_MS = float(os.environ.get("DSPARK_PROFILE_FWD_MS", "0") or 0)
_ACC: dict[str, float] = defaultdict(float)
_N: dict[str, int] = defaultdict(int)
# Reserved-memory delta per bucket. Time alone cannot see an allocator that is
# unmapping and remapping segments -- and `mem/reserved_gb` was observed swinging
# 55 -> 18.9 -> 55 GB between logged steps at block_size=15/240 anchors, with
# max_reserved at 59.9 of 64. That churn goes through the driver and is slow, so a
# region whose RESERVED jumps is the region paying for it.
_RES: dict[str, float] = defaultdict(float)
# Below this the reserved delta is allocator noise, not a signal worth a column.
_MEM_NOISE_GB = 0.05


def _reserved_gb() -> float:
    if hasattr(torch, "npu"):
        return torch.npu.memory_reserved() / 2**30
    return 0.0


def _sync() -> None:
    """Make the timing mean device time, not launch time."""
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif hasattr(torch, "accelerator"):
        torch.accelerator.synchronize()


def _rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def prof(tag: str, fn):
    """Time ``fn()`` into the ``tag`` bucket. Transparent: returns what fn returns."""
    if not ON:
        return fn()
    _sync()
    t0, r0 = time.perf_counter(), _reserved_gb()
    out = fn()
    _sync()
    dt = (time.perf_counter() - t0) * 1000.0
    _ACC[tag] += dt
    _RES[tag] += _reserved_gb() - r0
    _N[tag] += 1
    if _PER_CALL_MS and dt > _PER_CALL_MS:
        print(f"[FWD_PROF] {tag}: {dt:.0f} ms", flush=True)
    return out


@contextlib.contextmanager
def region(tag: str):
    """Time a multi-statement region into ``tag``. The ``TOP.*`` regions are meant to
    partition the whole forward, which is what makes UNACCOUNTED meaningful."""
    if not ON:
        yield
        return
    _sync()
    t0, r0 = time.perf_counter(), _reserved_gb()
    try:
        yield
    finally:
        _sync()
        dt = (time.perf_counter() - t0) * 1000.0
        _ACC[tag] += dt
        _RES[tag] += _reserved_gb() - r0
        _N[tag] += 1
        if _PER_CALL_MS and dt > _PER_CALL_MS:
            print(f"[FWD_PROF] {tag}: {dt:.0f} ms", flush=True)


def begin():
    """Start a forward. Clears the accumulator (see the note on recompute above)."""
    if not ON:
        return None
    _ACC.clear()
    _N.clear()
    _RES.clear()
    _sync()
    return time.perf_counter()


def _fmt(d: dict[str, float], prefix: str = "") -> str:
    """``prefix`` restores the accumulator key when the display name was stripped."""
    out = []
    for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
        r = _RES[prefix + k]
        # Only show the memory column when the region actually moved reserved memory;
        # a steady-state region moves none and the noise would bury the signal.
        mem = f"/{r:+.1f}GB" if abs(r) >= _MEM_NOISE_GB else ""
        out.append(f"{k}={v:.0f}{mem}({_N[prefix + k]}x)")
    return "  ".join(out)


def end(t0) -> None:
    """Close the forward and print the exhaustive breakdown (rank 0 only)."""
    if not ON or t0 is None:
        return
    _sync()
    total = (time.perf_counter() - t0) * 1000.0
    if not _rank0():
        return
    top = {k[4:]: v for k, v in _ACC.items() if k.startswith("TOP.")}
    inner = {k: v for k, v in _ACC.items() if not k.startswith("TOP.")}
    s_top = sum(top.values())
    print(
        f"[FWD_PROF] TOTAL={total:.0f}ms reserved={_reserved_gb():.1f}GB | "
        f"{_fmt(top, 'TOP.')} | "
        f"UNACCOUNTED={total - s_top:.0f}ms "
        f"({100 * (total - s_top) / max(total, 1e-9):.0f}%)",
        flush=True,
    )
    if inner:
        bb = top.get("backbone", 0.0)
        s_in = sum(inner.values())
        print(
            f"[FWD_PROF]   within backbone={bb:.0f}ms | {_fmt(inner)} | "
            f"UNACCOUNTED={bb - s_in:.0f}ms",
            flush=True,
        )
