#!/usr/bin/env bash
# check_ep_env.sh — READ-ONLY readiness probe for the pr/moe-ep-training EP-MoE tests.
#
# SAFE TO RUN DURING TRAINING: it only imports torch in this (separate) process and
# NEVER allocates accelerator memory — no NPU tensors, no device grab, no npu-smi.
# It just reports whether THIS conda env can run the CPU-gloo EP tests, and whether
# the NPU stack is present for the optional NPU run, against the A2 deployment spec.
#
# Usage (run INSIDE the conda env you'll test in, e.g. the A2 training/compute env):
#   bash check_ep_env.sh [/path/to/speculators/checkout]
# The repo path is optional: given it, the probe also checks the branch/files and
# runs a tiny CPU-only MoE forward. Without it, only the base env is checked.
set -u
REPO="${1:-${REPO:-}}"
# auto-detect: if no repo arg and cwd looks like a speculators checkout, use it.
if [ -z "$REPO" ] && [ -f "pyproject.toml" ] && grep -qi '^name = "speculators"' pyproject.toml 2>/dev/null; then
  REPO="$(pwd)"
fi

python3 - "$REPO" <<'PY'
import os, sys, subprocess

REPO = sys.argv[1] if len(sys.argv) > 1 else ""
ok = warn = fail = 0
def L(status, name, detail=""):
    global ok, warn, fail
    mark = {"PASS":"\033[32m✓\033[0m","WARN":"\033[33m⚠\033[0m",
            "FAIL":"\033[31m✗\033[0m","INFO":"·"}[status]
    if status == "PASS": ok += 1
    elif status == "WARN": warn += 1
    elif status == "FAIL": fail += 1
    print(f"  {mark} {name:<36} {detail}")

def header(t):
    print(f"\n\033[1m{t}\033[0m")

print("="*72)
print(" EP-MoE test-env readiness probe  (READ-ONLY — safe during training)")
print("="*72)
print(f"  host    : {os.uname().nodename}")
print(f"  conda   : {os.environ.get('CONDA_DEFAULT_ENV','(none)')}")
print(f"  python  : {sys.version.split()[0]}  ({sys.executable})")

# ---------------------------------------------------------------- core (CPU tests)
header("Core — required for the CPU-gloo EP tests")
v = sys.version_info
L("PASS" if v >= (3,10) else "FAIL", "python >= 3.10", f"{v.major}.{v.minor}")

torch = None
try:
    import torch
    L("PASS", "torch import", torch.__version__)
except Exception as e:
    L("FAIL", "torch import", repr(e))

if torch is not None:
    L("PASS" if hasattr(torch, "accelerator") else "FAIL",
      "torch.accelerator", "present" if hasattr(torch,"accelerator") else "MISSING (need torch>=2.6)")
    import torch.distributed as dist
    L("PASS" if dist.is_available() else "FAIL", "torch.distributed", "available" if dist.is_available() else "MISSING")
    try:
        g = dist.is_gloo_available()
        L("PASS" if g else "FAIL", "gloo backend (CPU tests)", "yes" if g else "NO")
    except Exception as e:
        L("WARN", "gloo backend (CPU tests)", repr(e))
    L("PASS" if hasattr(dist, "get_default_backend_for_device") else "FAIL",
      "get_default_backend_for_device", "present" if hasattr(dist,"get_default_backend_for_device") else "MISSING (need torch>=2.5)")
    try:
        from torch.distributed.tensor import DTensor, Shard  # noqa: F401
        L("PASS", "torch.distributed.tensor (DTensor/Shard)", "ok")
    except Exception as e:
        L("FAIL", "torch.distributed.tensor (DTensor/Shard)", repr(e))
    try:
        from torch.distributed.device_mesh import init_device_mesh  # noqa: F401
        L("PASS", "init_device_mesh", "ok")
    except Exception as e:
        L("FAIL", "init_device_mesh", repr(e))
    L("INFO", "torch._grouped_mm (opt-in fast path)",
      "present" if hasattr(torch,"_grouped_mm") else "absent (default loop path used; fine)")

try:
    import pytest  # noqa: F401
    L("PASS", "pytest", pytest.__version__)
except Exception as e:
    L("FAIL", "pytest", repr(e))

# --------------------------------------------------------- A2 version spec (warn)
header("A2 deployment version spec (warn-only — the CPU tests may not need exact)")
try:
    import numpy as np
    exp = "2.3.5"
    L("PASS" if np.__version__ == exp else "WARN", "numpy == 2.3.5 (SSOT)",
      f"{np.__version__}" + ("" if np.__version__==exp else f"  (A2 pins {exp})"))
except Exception as e:
    L("WARN", "numpy", repr(e))
try:
    import transformers
    tv = transformers.__version__
    parts = tuple(int(x) for x in tv.split(".")[:3] if x.isdigit())
    inrange = (4,56,1) <= parts < (5,14,0)
    L("PASS" if inrange else "WARN", "transformers in [4.56.1, 5.14.0)",
      tv + ("" if inrange else "  (out of A2 range)"))
except Exception as e:
    L("WARN", "transformers", repr(e) + "  (speculators import will fail without it)")

# ---------------------------------------------------- NPU stack (optional NPU run)
header("NPU stack — only needed for the optional NPU run (not the CPU tests)")
try:
    import torch_npu  # noqa: F401
    L("PASS", "torch_npu import", getattr(torch_npu, "__version__", "?"))
    try:
        n = torch.npu.device_count()          # read-only: queries driver, no alloc
        L("INFO", "torch.npu.device_count()", str(n))
    except Exception as e:
        L("WARN", "torch.npu.device_count()", repr(e))
except Exception:
    L("WARN", "torch_npu import", "absent (CPU-gloo tests still run; NPU run unavailable)")
if torch is not None and hasattr(torch, "accelerator"):
    try:
        acc = torch.accelerator.current_accelerator()   # read-only, no alloc
        L("INFO", "current_accelerator", str(acc))
    except Exception as e:
        L("INFO", "current_accelerator", repr(e))
cann = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
L("INFO" if cann else "WARN", "CANN set_env sourced", cann or "ASCEND_TOOLKIT_HOME unset (source set_env.sh for NPU)")

# --------------------------------------------------------------- repo / branch
header("Branch & code (needs the pr/moe-ep-training checkout)")
if not REPO:
    L("WARN", "speculators checkout", "no repo path given -> skipping branch/file/smoke checks")
else:
    def git(*a):
        return subprocess.run(["git","-C",REPO,*a], capture_output=True, text=True).stdout.strip()
    br = git("rev-parse","--abbrev-ref","HEAD")
    L("PASS" if br=="pr/moe-ep-training" else "WARN", "git branch",
      br + ("" if br=="pr/moe-ep-training" else "  (expected pr/moe-ep-training)"))
    need = ["src/speculators/models/moe/layer.py",
            "src/speculators/models/moe/dispatch_ep.py",
            "src/speculators/train/expert_parallel.py",
            "tests/unit/models/test_moe.py",
            "tests/e2e/smoke/test_expert_parallel.py"]
    miss = [f for f in need if not os.path.exists(os.path.join(REPO,f))]
    L("PASS" if not miss else "FAIL", "EP files present",
      "all 5" if not miss else f"MISSING: {miss}")

# ------------------------------------------------------- import + CPU-only smoke
header("Import + CPU-only functional smoke (no NPU touched)")
sp = None
try:
    import speculators
    sp = os.path.dirname(os.path.dirname(speculators.__file__))
    in_repo = (REPO and os.path.realpath(sp)==os.path.realpath(REPO))
    L("PASS", "import speculators", speculators.__file__ + ("  (editable=this repo)" if in_repo else ""))
except Exception as e:
    L("FAIL", "import speculators", repr(e))
try:
    from speculators.models.moe import MoE, MoEConfig
    L("PASS", "import speculators.models.moe", "MoE, MoEConfig ok")
    if torch is not None:
        cfg = MoEConfig(hidden_size=16, moe_inter_dim=32, n_routed_experts=4, n_activated_experts=2)
        m = MoE(cfg)                                   # CPU (default device)
        y = m(torch.randn(2, 3, 16))
        good = tuple(y.shape)==(2,3,16) and bool(torch.isfinite(y).all())
        L("PASS" if good else "FAIL", "MoE forward on CPU", "shape+finite ok" if good else "BAD OUTPUT")
except Exception as e:
    L("FAIL", "import speculators.models.moe / smoke", repr(e))

# ------------------------------------------------------------------- verdict
print("\n" + "="*72)
verdict = "READY" if fail==0 else "NOT READY"
color = "\033[32m" if fail==0 else "\033[31m"
print(f"  {color}{verdict}\033[0m   pass={ok}  warn={warn}  fail={fail}")
if fail==0:
    print("  → run:  python -m pytest tests/unit/models/test_moe.py tests/e2e/smoke/test_expert_parallel.py -v")
else:
    print("  → fix the ✗ items above (warns are ok for the CPU-gloo run).")
print("="*72)
sys.exit(1 if fail else 0)
PY
