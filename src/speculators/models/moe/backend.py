"""Opt-in accelerator-kernel dispatch for the MoE draft layer.

Every heavy op in the MoE FFN ships a **pure-torch reference** that is always
present and runs on CPU or any accelerator torch supports (CUDA, Ascend NPU,
...). An external bridge may register a faster, hardware-specific implementation
under the same op key at import time; the layer resolves ``(op, backend)`` at
call time and silently falls back to the torch reference when no accelerated
impl is registered.

Design goals (the shareable draft must be self-contained and CPU/GPU-runnable):

* **No accelerator import here.** This module never imports a device-specific
  kernel package. A hardware bridge lives *outside* the shareable core and calls
  :func:`register_kernel` from its own conditional-import module, so installing
  the draft on a plain box pulls in nothing hardware-specific.
* **Torch reference is the source of truth.** Registered kernels are validated
  against it; the layer is correct with zero kernels registered. This is what
  makes fast kernels "re-insertable" rather than load-bearing.
* **Backend selection is explicit and global-with-override.** The active backend
  defaults to ``"torch"``; a process may switch it (e.g. a bridge sets its own
  backend after registering), and any call site may force a backend via the
  ``backend=`` argument.
"""

from __future__ import annotations

from collections.abc import Callable

TORCH_BACKEND = "torch"

_REGISTRY: dict[tuple[str, str], Callable] = {}
_ACTIVE_BACKEND = TORCH_BACKEND


def register_kernel(op: str, backend: str, fn: Callable) -> None:
    """Register an implementation ``fn`` for ``op`` under ``backend``.

    Called by the pure-torch references at import (``backend="torch"``) and by an
    external accelerator bridge for its own backend. Re-registering the same key
    overwrites — a bridge may override the torch default deliberately.
    """
    _REGISTRY[(op, backend)] = fn


def torch_kernel(op: str) -> Callable[[Callable], Callable]:
    """Decorator registering a function as the torch reference for ``op``."""

    def _wrap(fn: Callable) -> Callable:
        register_kernel(op, TORCH_BACKEND, fn)
        return fn

    return _wrap


def set_active_backend(backend: str) -> str:
    """Set the process-wide default backend; returns the previous value."""
    global _ACTIVE_BACKEND
    prev, _ACTIVE_BACKEND = _ACTIVE_BACKEND, backend
    return prev


def get_active_backend() -> str:
    return _ACTIVE_BACKEND


def get_kernel(op: str, backend: str | None = None) -> Callable:
    """Resolve the implementation for ``op``.

    Resolution order: the requested ``backend`` (or the active default) → the
    torch reference. Raises if not even a torch reference is registered (a
    programming error — every op must register its reference at import).
    """
    backend = backend or _ACTIVE_BACKEND
    fn = _REGISTRY.get((op, backend))
    if fn is not None:
        return fn
    fn = _REGISTRY.get((op, TORCH_BACKEND))
    if fn is None:
        raise KeyError(
            f"No implementation for op '{op}' (backend '{backend}' and no torch "
            "reference). Import the MoE modules so their torch references "
            "register, or register a kernel via register_kernel()."
        )
    return fn


def has_kernel(op: str, backend: str) -> bool:
    return (op, backend) in _REGISTRY


def registered_ops() -> list[tuple[str, str]]:
    return sorted(_REGISTRY.keys())
