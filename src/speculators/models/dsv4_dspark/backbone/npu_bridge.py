"""Ascend NPU accelerator bridge for the DSV4 DSpark backbone.

**This module is the only place in the package that may import ``torch_npu``, and it is the
piece that would NOT be upstreamed.** Everything else -- the registry in :mod:`.kernels`, the
pure-torch references, the model, and the expert-parallel all-to-all in :mod:`.moe_ep` --
is device-agnostic and correct with zero accelerated kernels registered.

WHY THE SPLIT IS SHAPED THIS WAY
--------------------------------
Upstream speculators has ruled on this twice, and the two rulings point the same way:

* **PR #775** ("opt-in ``transfer_to_npu`` + robust device_id", +17/-1) was **closed
  unmerged after two days**. It put a vendor shim in ``src/``.
* **PR #589** ("selectable attention backend (sdpa/eager) with dense mask", +234/-14) was
  **merged**. It solved the same class of problem -- flex attention is unavailable on Ascend
  -- by adding a *portable option*, not a vendor branch.

Upstream's own device handling is consistent with that: ``torch.accelerator.current_accelerator()``
throughout, and not one direct ``torch_npu`` call or ``is_cuda`` branch anywhere in ``src/``.

So the contract is: the shareable core carries a registry and a torch reference; an
accelerator package registers faster implementations under the same op keys from outside.

DEPLOYMENT — AND ITS HONEST COST
--------------------------------
An Ascend user has to install this bridge separately. That is the same arrangement vLLM
already has with ``vllm-ascend``: vLLM core ships no Ascend kernels and the vendor plugin
supplies them.

The soft spot is not the extra install, it is that **an unbridged install is correct but
quietly slower** -- the torch reference runs and nothing says so. :func:`active_backend_report`
exists to make that visible, and :func:`speculators...kernels.discover_plugins` lets an
installed bridge register itself through a Python entry point so the user never writes an
import at all.

USAGE
-----
    from speculators.models.dsv4_dspark.backbone import npu_bridge
    npu_bridge.install()          # no-op off NPU; idempotent; never raises

or, once the bridge ships as its own distribution, declare it and let discovery do it:

    [project.entry-points."speculators.kernels"]
    ascend = "speculators_ascend.bridge"
"""

from __future__ import annotations

import logging
import os

from .kernels import (
    TORCH_BACKEND,
    get_active_backend,
    has_kernel,
    registered_ops,
    set_active_backend,
)

logger = logging.getLogger(__name__)

NPU_BACKEND = "npu"

_INSTALLED = False


def npu_available() -> bool:
    """True when torch reports an Ascend accelerator AND ``torch_npu`` imports.

    Deliberately not ``torch.cuda.is_available()``: ``torch_npu``'s ``transfer_to_npu`` shim
    monkeypatches CUDA predicates to True on NPU, so anything phrased in CUDA terms answers
    yes on the wrong hardware. This asks torch's device-agnostic API instead, matching what
    upstream already does.
    """
    try:
        import torch  # noqa: PLC0415

        acc = torch.accelerator.current_accelerator()
        if acc is None or acc.type != "npu":
            return False
        import torch_npu  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001 - probing must never break an import
        return False


def install(*, activate: bool = True) -> bool:
    """Register the NPU implementations and (by default) make them the active backend.

    Returns whether anything was registered. Safe and silent off NPU, safe to call twice, and
    never raises: a bridge that fails to load must degrade to the torch reference rather than
    take down training.

    ``SPECULATORS_DISABLE_NPU_KERNELS=1`` forces the torch reference even on NPU -- the switch
    to reach for when an accelerated kernel is suspected of being wrong, since the reference
    is also the parity oracle.
    """
    global _INSTALLED  # noqa: PLW0603

    if _INSTALLED:
        return True
    if os.environ.get("SPECULATORS_DISABLE_NPU_KERNELS") == "1":
        logger.info("npu_bridge: disabled by SPECULATORS_DISABLE_NPU_KERNELS=1")
        return False
    if not npu_available():
        return False

    # Importing these registers ("moe_dispatch", "npu") etc. as a side effect. They are
    # imported HERE rather than at package import so that a CPU/CUDA install never pulls in
    # torch_npu -- which is the whole point of the split.
    try:
        from . import moe_ep, moe_grouped_gemm  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("npu_bridge: kernel import failed (%s); staying on torch", exc)
        return False

    if activate:
        set_active_backend(NPU_BACKEND)
    _INSTALLED = True
    logger.info("npu_bridge: registered %d op(s); active backend=%s",
                len(registered_ops()), get_active_backend())
    return True


def active_backend_report(ops: tuple[str, ...] = ()) -> str:
    """One line saying which backend each op will actually run on.

    Worth printing at startup: the failure mode this whole design accepts is a correct run
    that is silently on the torch reference, and the only cure is saying so out loud.
    """
    active = get_active_backend()
    if not ops:
        ops = tuple(sorted({op for op, _ in registered_ops()}))
    parts = [
        f"{op}={'npu' if (active != TORCH_BACKEND and has_kernel(op, active)) else 'torch'}"
        for op in ops
    ]
    tail = "" if active != TORCH_BACKEND else "  (no accelerator bridge installed)"
    return f"kernels: active={active}  " + " ".join(parts) + tail
