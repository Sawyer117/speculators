#!/usr/bin/env python3
"""Measure the SELECTION headroom of a trained draft: recall@k and oracle accept_len.

WHY
---
Our ``num_spec=7`` result showed every token we lose to the released draft sits at the
positions the block never trained on (pos5/pos6), and we read that as "block width does
not match the serving configuration -> retrain at block_size 8". inco.ai's DFlash2 post
reports a second, ORTHOGONAL gap on the same symptom: on a Qwen3-4B DFlash drafter,
GSM8K recall at draft position 6 is **72.9% at top-1 but 87.8% at top-16**, and accept
length goes **4.27 (argmax) -> 6.79 (oracle over top-16)**. That is +2.5 tokens available
from *choosing better among candidates the draft already produces*, with no retrain and
no change to the model.

Their number is on their model, at temperature 1.0, against a DSpark baseline whose
quality we cannot verify (it scores BELOW Qwen's native MTP on their own 27B table).
None of that transfers. But the question we actually need answered — *does OUR draft
carry the same headroom* — is a property of our own weights, and it is measurable from
the tensors ``compute_metrics`` already receives. This script measures it.

DECISION RULE (fix it before looking at the number)
--------------------------------------------------
  * pos>=4 ``recall@16`` exceeds ``recall@1`` by **>=10 points**  -> the headroom is real;
    path selection is worth costing out (serve-side, no retrain).
  * the gap is **<3 points**, or ``oracle_accept_len`` beats ``hard_accept_len`` by **<1
    token** -> our draft genuinely does not know the tail token, it is not merely picking
    the wrong one. Path selection is dead for us. Stop here; the block-8 retrain remains
    the only lever.

This is deliberately a LARGE-SIGNAL test. A "build the selector and see" experiment would
land near the +-0.025 noise floor this ledger established and could not be read; a 85%
vs 99.5% recall gap can.

HOW IT WORKS
------------
``DSparkDraftModel.forward`` returns ``(None, loss, metrics)`` — the logits never leave
it. But ``compute_metrics`` receives ``(logits, targets, confidence_logits, loss_mask,
block_size, ..., sample_from_anchor=...)``, which is everything needed. So this script
**wraps that function at runtime** and appends the new metrics to the dict it returns; it
does NOT edit ``metrics.py`` (same pattern as ``eval_trainsample.py``, which imports the
proven ``Evaluator.py`` unmodified and injects a dataset).

The added keys follow the established ``*_sum`` / ``*_total`` convention, so
``speculators/train/utils.py`` folds them into ratios and the trainer all-reduces and logs
them next to ``position_k_acc`` with no further wiring:

    position_{k}_recall{K}      per-position recall@K          (compare vs position_{k}_acc = recall@1)
    oracle_accept_len_{K}       accept_len if selection were perfect within top-K
    (metrics.py already logs ``hard_accept_len`` = the argmax/greedy one -> the pair is
     the headroom, on the SAME batch, in the SAME log line)

RUNNING IT
----------
The draft is ~19.8B with 256 experts; it only loads under the launcher's FSDP2+EP setup,
so this script does not rebuild any of that — it patches, then hands off to
``scripts/train.py:main()`` unchanged. Point the launcher at this file instead of
``scripts/train.py`` and set **LR=0**::

    # on the training box, serve up (HS is online), from the repo root:
    DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
    DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3 \
    LR=0 MAX_ANCHORS=512 \
    SAVE_PATH=/home/a00652497/dspark_austin/run/ckpt_faithful_ep_20260804_165215 \
    TRAIN_PY=examples/ascend_npu_dflash/recall_headroom_probe.py \
      bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful

``train_dsv4_dspark.sh`` hardcodes ``scripts/train.py`` (line ~274), so either add a
``TRAIN_PY`` override there or torchrun this file directly with the same arguments the
launcher prints.

**LR=0 is what makes this safe**: AdamW's decoupled weight decay is also scaled by lr, so
at lr=0 no parameter moves and the resumed checkpoint cannot be damaged. Read the first
few metric lines and kill it — a few hundred steps is far more than this measurement
needs (one step at ``max_anchors=512`` already yields ~512 blocks per rank), and nothing
is written back.

⚠ Two honest limits on the number this produces:
  1. It is measured on the TRAINING distribution (open_perfectblend rollouts), not gsm8k.
     The *existence* of a large gap is robust to that; the exact figure is not. The gsm8k
     number needs a serve-side top-k dump, which is a day of work — only worth it if this
     probe comes back green.
  2. ``topk`` runs in the logits' native dtype (bf16). bf16 ties can reorder candidates
     near the cut, which perturbs recall@k by a hair. Irrelevant at the effect size the
     decision rule keys on; do not quote these to three decimals.
"""

from __future__ import annotations

import os
import sys

import torch

# Read before the wrapper is installed so the banner can echo it.
_KS = tuple(
    int(k) for k in os.environ.get("RECALL_KS", "2,4,8,16").split(",") if k.strip()
)
_CHUNK = int(os.environ.get("RECALL_CHUNK", "512"))


def _topk_hit(
    logits: torch.Tensor,  # [1, T, V]
    target_ids: torch.Tensor,  # [1, T]
    kmax: int,
) -> torch.Tensor:
    """[1, T, kmax] bool — whether the target id is the j-th candidate at each position.

    Chunked over the sequence: the full logits tensor is [1, ~3072, 129280] and this run
    already peaks at 86% of a 64 GB card, so the topk workspace is taken in slices rather
    than over the whole thing at once.
    """
    hits = []
    seq_len = logits.shape[1]
    for i in range(0, seq_len, _CHUNK):
        sl = slice(i, min(i + _CHUNK, seq_len))
        idx = logits[:, sl, :].topk(kmax, dim=-1).indices  # [1, c, kmax]
        hits.append(idx == target_ids[:, sl].unsqueeze(-1))
    return torch.cat(hits, dim=1)


def recall_metrics(
    logits: torch.Tensor,  # [1, T, V]
    targets: torch.Tensor,  # [1, T, V]
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    sample_from_anchor: bool,
    ks: tuple[int, ...] = _KS,
) -> dict[str, torch.Tensor]:
    """Per-position recall@k plus the oracle accept length, as ``*_sum``/``*_total`` pairs.

    Aggregation mirrors ``metrics.py`` exactly so the numbers are directly comparable to
    the ``position_k_acc`` / ``hard_accept_len`` printed on the same line:
      * ``start_pos`` = 0 under ``sample_from_anchor`` (DSpark drafts slot 0 too), else 1;
      * the oracle length counts the same slots ``hard_accept_len`` counts, and adds the
        same +1 for the always-emitted verifier token.
    """
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        seq_len = logits.shape[1]
        num_blocks = seq_len // block_size
        start_pos = 0 if sample_from_anchor else 1
        kmax = max(ks)

        target_ids = targets.argmax(dim=-1)  # [1, T]
        hit_at = _topk_hit(logits, target_ids, kmax)  # [1, T, kmax]

        acc_dtype = torch.float32
        mask_b = loss_mask.to(acc_dtype).view(num_blocks, block_size)
        hslice = slice(None) if sample_from_anchor else slice(1, None)
        hmask = mask_b[:, hslice]
        block_valid = (hmask.sum(dim=-1) > 0).to(acc_dtype)

        for k in ks:
            hit_k = hit_at[..., :k].any(dim=-1)  # [1, T]
            hit_b = hit_k.to(acc_dtype).view(num_blocks, block_size)

            # Per-position recall@k. position_{p}_acc in metrics.py is exactly this with k=1.
            correct = (hit_b * mask_b).sum(dim=0)  # [block_size]
            total = mask_b.sum(dim=0)  # [block_size]
            for pos in range(start_pos, block_size):
                out[f"position_{pos}_recall{k}_sum"] = correct[pos]
                out[f"position_{pos}_recall{k}_total"] = total[pos]
            out[f"full_recall{k}_sum"] = correct[start_pos:].sum()
            out[f"full_recall{k}_total"] = total[start_pos:].sum()

            # Oracle accept length: the longest prefix whose target is inside top-k.
            # Same shape as hard_accept_len -> the difference IS the selection headroom.
            oracle = (hit_b[:, hslice] * hmask).cumprod(dim=-1).sum(dim=-1) + 1.0
            out[f"oracle_accept_len_{k}_sum"] = (oracle * block_valid).sum()
            out[f"oracle_accept_len_{k}_total"] = block_valid.sum().clamp_min(1.0)

    return out


def install() -> None:
    """Wrap ``compute_metrics`` in place. Idempotent."""
    import speculators.models.dspark.core as dspark_core

    if getattr(dspark_core.compute_metrics, "_recall_probe", False):
        return

    original = dspark_core.compute_metrics

    # core.py calls it with 5 positional args and the rest by keyword (core.py:196).
    def wrapped(logits, targets, confidence_logits, loss_mask, block_size, *args, **kw):
        loss, metrics = original(
            logits, targets, confidence_logits, loss_mask, block_size, *args, **kw
        )
        try:
            metrics.update(
                recall_metrics(
                    logits,
                    targets,
                    loss_mask,
                    block_size,
                    kw.get("sample_from_anchor", True),
                )
            )
        except Exception as exc:  # a probe must never take down a training run
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[recall-probe] skipped this step: {exc!r}", flush=True)
        return loss, metrics

    wrapped._recall_probe = True  # type: ignore[attr-defined]
    dspark_core.compute_metrics = wrapped


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo_root, "scripts"))

    install()

    import train as train_script  # scripts/train.py

    args = train_script.parse_args()

    if int(os.environ.get("RANK", "0")) == 0:
        lr = getattr(args, "lr", None)
        print("=" * 78, flush=True)
        print(f">>> RECALL HEADROOM PROBE — recall@{_KS} + oracle accept_len", flush=True)
        print(">>> compute_metrics wrapped at runtime; metrics.py untouched", flush=True)
        print(f">>> lr={lr}" + ("" if lr == 0 else "   ⚠ NOT 0 — this run WILL train"), flush=True)
        print(">>> read position_{k}_recall16 vs position_{k}_acc, and", flush=True)
        print(">>>      oracle_accept_len_16 vs hard_accept_len, then kill it", flush=True)
        print("=" * 78, flush=True)

    train_script.main(args)


if __name__ == "__main__":
    main()
