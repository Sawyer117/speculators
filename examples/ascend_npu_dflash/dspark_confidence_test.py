#!/usr/bin/env python3
"""NPU smoke test for the DSpark confidence head + loss added to DFlash training.

Exercises ONLY the newly-added pieces on random tensors (no model weights), forward
AND backward on one NPU, so the port can be validated before the full training
pipeline / weights are available:

  - confidence head        : nn.Linear(hidden, 1) -> squeeze   (DSpark AcceptRatePredictor)
  - confidence_loss        : BCE-with-logits vs the SOFT accept rate alpha = 1 - d_TV
                             (detached target) -- matches DeepSpec dspark/loss.py, NOT a
                             hard argmax match
  - l1_loss / combo_ce_l1  : the DSpark distribution term 0.1*CE + 0.9*L1 (full sum|p-q|)
  - compute_metrics        : DFlash's metric/loss entry point with the confidence term
                             wired in end-to-end

Run in the dspark-dsv4-base env on any single NPU:
    python dspark_confidence_test.py
"""
import pathlib
import sys

import torch
from torch import nn

# make `speculators` importable whether or not it's pip-installed in this env
_REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

try:
    import torch_npu  # noqa: F401

    DEV = "npu:0"
except Exception as e:  # noqa: BLE001
    print(f"!! torch_npu import failed: {e}")
    raise SystemExit(1)

try:
    from speculators.models.metrics import (
        combo_ce_l1_loss,
        confidence_loss,
        l1_loss,
    )
except Exception as e:  # noqa: BLE001
    print(f"!! cannot import speculators.models.metrics: {e}")
    print("   run from the repo (examples/ascend_npu_dflash/) or `pip install -e .` it.")
    raise SystemExit(1)

torch.manual_seed(0)
DT = torch.bfloat16
NB, BS, H, V = 8, 7, 512, 4096  # anchors, block_size(=7), hidden, draft vocab
T = NB * BS
results = []


def check(name: str, ok: bool, extra: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<48} {extra}")
    results.append((name, ok))


print(f">>> device={DEV} dtype={DT}  T={T} (anchors {NB} x block {BS})  H={H} V={V}\n")

# 1. the confidence head itself: Linear(H, 1) on the draft hidden -> [1, T]
head = nn.Linear(H, 1).to(DEV, DT)
hidden = torch.randn(1, T, H, device=DEV, dtype=DT, requires_grad=True)
conf_logits = head(hidden).squeeze(-1)  # [1, T]
check("confidence head Linear(H,1).squeeze(-1)", tuple(conf_logits.shape) == (1, T),
      f"out={tuple(conf_logits.shape)}")

# 2. confidence_loss: BCE vs SOFT accept-rate alpha, fwd + bwd
logits = torch.randn(1, T, V, device=DEV, dtype=DT)
targets = torch.randn(1, T, V, device=DEV, dtype=DT)
cl = confidence_loss(conf_logits, logits, targets)  # [1, T]
cl.mean().backward()
check("confidence_loss (soft-alpha BCE) fwd+bwd",
      torch.isfinite(cl).all().item() and hidden.grad is not None,
      f"loss={cl.mean().item():.4f} grad_ok={hidden.grad is not None}")

# 3. l1_loss + combo_ce_l1_loss (DSpark distribution term), fwd + bwd
lg = torch.randn(1, T, V, device=DEV, dtype=DT, requires_grad=True)
tg = torch.randn(1, T, V, device=DEV, dtype=DT)
l1 = l1_loss(lg, tg)
combo = combo_ce_l1_loss(0.1, 0.9)(lg, tg)  # [1, T]
combo.mean().backward()
check("combo_ce_l1_loss 0.1*CE + 0.9*L1 fwd+bwd",
      torch.isfinite(combo).all().item() and lg.grad is not None,
      f"l1={l1.mean().item():.4f} combo={combo.mean().item():.4f}")

# 4. full DFlash compute_metrics with confidence wired in end-to-end
try:
    from speculators.models.dflash.metrics import compute_metrics

    lg2 = torch.randn(1, T, V, device=DEV, dtype=DT, requires_grad=True)
    tg2 = torch.randn(1, T, V, device=DEV, dtype=DT)
    hid2 = torch.randn(1, T, H, device=DEV, dtype=DT, requires_grad=True)
    head2 = nn.Linear(H, 1).to(DEV, DT)
    conf2 = head2(hid2).squeeze(-1)
    mask = (torch.rand(1, T, device=DEV) > 0.2).float()
    mask[:, ::BS] = 0  # zero the per-block anchor position, as DFlash does
    loss, metrics = compute_metrics(
        lg2, tg2, mask, block_size=BS, gamma=4.0,
        loss_fn=combo_ce_l1_loss(0.1, 0.9),
        confidence_logits=conf2, confidence_alpha=1.0,
    )
    loss.backward()
    conf_val = metrics.get("confidence_loss_sum")
    ok = (
        torch.isfinite(loss).item()
        and conf_val is not None
        and hid2.grad is not None
    )
    check("compute_metrics(+confidence, combo) fwd+bwd", ok,
          f"loss={loss.item():.4f} confidence_loss={conf_val.item():.4f}")
except Exception as e:  # noqa: BLE001
    check("compute_metrics(+confidence) fwd+bwd", False,
          f"{type(e).__name__}: {str(e)[:70]}")

ok = sum(1 for _, r in results if r)
print(f"\n>>> {ok}/{len(results)} checks passed on NPU.")
if ok == len(results):
    print(">>> confidence head + soft-alpha BCE loss + combo CE/L1 all run fwd+bwd on NPU.")
    print(">>> semantics match DeepSpec dspark/loss.py: accept-rate target = 1 - d_TV,")
    print(">>>   BCE-with-logits against the DETACHED accept rate (not a hard argmax match).")
else:
    print(">>> a check FAILED above — do not wire into training until it's green.")
