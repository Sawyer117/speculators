"""The forward profiler must ACCOUNT for the time, not just report some of it.

WHY. Two rounds of instrumentation reported 630 ms of a 6,900 ms forward and then 730 ms
of a 7,100 ms one, and neither round said so — the only way to notice was to add the
prints up by hand. The profiler now prints total, parts and UNACCOUNTED, and this pins the
property that makes that trustworthy: when a region is missed, UNACCOUNTED must show it.

Loaded by path so the test needs neither torch_npu nor the model package.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

PROF_PY = (
    Path(__file__).resolve().parents[3]
    / "src" / "speculators" / "models" / "dspark" / "profiling.py"
)


def _load(capsys):
    os.environ["DSPARK_PROFILE_FWD"] = "1"
    spec = importlib.util.spec_from_file_location("_prof_under_test", PROF_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.ON, "DSPARK_PROFILE_FWD=1 must switch it on"
    return mod


def _burn(ms: float) -> None:
    t = time.perf_counter()
    while (time.perf_counter() - t) * 1000 < ms:
        pass


def _unaccounted(line: str) -> float:
    return float(line.split("UNACCOUNTED=")[1].split("ms")[0])


def test_full_partition_accounts_for_everything(capsys):
    p = _load(capsys)
    t0 = p.begin()
    with p.region("TOP.backbone"):
        p.prof("MLA.attn", lambda: _burn(30))
    with p.region("TOP.loss"):
        _burn(60)
    p.end(t0)
    top = [ln for ln in capsys.readouterr().out.splitlines() if "TOTAL=" in ln][0]
    assert _unaccounted(top) < 15, (
        f"a full partition should account for the time: {top}"
    )


def test_a_missed_region_shows_up_as_unaccounted(capsys):
    """The failure mode that went unnoticed twice: work outside every region."""
    p = _load(capsys)
    t0 = p.begin()
    with p.region("TOP.backbone"):
        _burn(20)
    _burn(120)  # instrumented by nothing
    p.end(t0)
    top = [ln for ln in capsys.readouterr().out.splitlines() if "TOTAL=" in ln][0]
    assert _unaccounted(top) > 100, f"a 120 ms gap must be reported, got: {top}"


def test_off_by_default_is_a_no_op(capsys):
    os.environ.pop("DSPARK_PROFILE_FWD", None)
    spec = importlib.util.spec_from_file_location("_prof_off", PROF_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert not mod.ON
    assert mod.begin() is None
    assert mod.prof("x", lambda: 42) == 42       # transparent
    with mod.region("TOP.y"):
        pass
    mod.end(None)
    assert capsys.readouterr().out == "", "off must print nothing and cost nothing"
