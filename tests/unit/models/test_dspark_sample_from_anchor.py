"""Regression tests for the DSpark ``sample_from_anchor`` block task.

Root cause of the epoch-1 draft's serve collapse (2026-07-17): the fork had DELETED
upstream's ``sample_from_anchor`` switch and hardcoded the FALSE branch (targets rolled
+1, slot-0 loss masked, per-position reported from slot 1). DSpark serving (vllm-ascend)
samples EVERY block slot, i.e. requires ``sample_from_anchor=True`` (no roll, slot 0
trained, reported from slot 0). Training with False => off-by-one targets + an untrained
slot 0 that collapse at serve. These tests pin the True/False behaviour so it can't
silently regress again.
"""

import torch

from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import dflash_loss_decay, resolve_loss_config

_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestSampleFromAnchorConfig:
    def test_dspark_defaults_true_dflash_defaults_false(self):
        # DSpark serving samples every slot -> config default MUST be True; the DFlash
        # base keeps the anchor-as-bonus default (False).
        assert DSparkSpeculatorConfig.model_fields["sample_from_anchor"].default is True
        assert DFlashSpeculatorConfig.model_fields["sample_from_anchor"].default is False


class TestSampleFromAnchorMetrics:
    # block_size=2, two blocks. Every slot supervised (loss_mask all ones) so both the
    # True (slots 0,1) and False (slot 1 only) reporting ranges are exercised.
    def _inputs(self):
        ids = torch.tensor([[5, 1, 5, 2]])
        logits = _ids_to_logits(ids, 8)
        return logits, logits.clone(), torch.ones((1, 4), dtype=torch.float32)

    def test_true_reports_position_0_false_skips_it(self):
        logits, targets, loss_mask = self._inputs()
        _, m_true = compute_metrics(
            logits, targets, None, loss_mask, 2,
            gamma=4.0, loss_config=_DEFAULT_LOSS, sample_from_anchor=True,
        )
        _, m_false = compute_metrics(
            logits, targets, None, loss_mask, 2,
            gamma=4.0, loss_config=_DEFAULT_LOSS, sample_from_anchor=False,
        )
        # True: slot 0 is a real prediction -> position_0 accuracy reported.
        assert "position_0_acc_sum" in m_true
        assert "position_1_acc_sum" in m_true
        # False: slot 0 is the given anchor -> reporting starts at position 1.
        assert "position_0_acc_sum" not in m_false
        assert "position_1_acc_sum" in m_false

    def test_default_samples_from_anchor(self):
        # No explicit arg -> compute_metrics defaults to True (DSpark) -> slot 0 reported.
        logits, targets, loss_mask = self._inputs()
        _, m = compute_metrics(
            logits, targets, None, loss_mask, 2, gamma=4.0, loss_config=_DEFAULT_LOSS,
        )
        assert "position_0_acc_sum" in m

    def test_true_hard_accept_len_counts_all_slots(self):
        # Perfect draft (draft==target on every slot). True counts both block slots + the
        # verifier bonus -> hard_accept_len == block_size + 1 == 3; False counts slot 1
        # only + bonus -> 2.
        logits, targets, loss_mask = self._inputs()
        _, m_true = compute_metrics(
            logits, targets, None, loss_mask, 2,
            gamma=4.0, loss_config=_DEFAULT_LOSS, sample_from_anchor=True,
        )
        _, m_false = compute_metrics(
            logits, targets, None, loss_mask, 2,
            gamma=4.0, loss_config=_DEFAULT_LOSS, sample_from_anchor=False,
        )
        hal_true = float(m_true["hard_accept_len_sum"] / m_true["hard_accept_len_total"])
        hal_false = float(m_false["hard_accept_len_sum"] / m_false["hard_accept_len_total"])
        assert abs(hal_true - 3.0) < 1e-4
        assert abs(hal_false - 2.0) < 1e-4


class TestLossDecaySlot0:
    """The per-position loss weight must NOT zero slot 0 under sample_from_anchor=True.

    Root cause of the sample_from_anchor=True retrain's stuck slot-0 (position_0_acc~0.03,
    zero gradient): the roll + loss-mask were gated, but dflash_loss_decay still multiplied
    position 0 by (pos_idx != 0) -> weight 0 -> the first draft token got no gradient.
    """

    def test_true_weights_slot0_highest_false_zeros_it(self):
        pos = torch.tensor([0, 1, 2, 3, 4])
        w_true = dflash_loss_decay(pos, gamma=4.0, sample_from_anchor=True)
        w_false = dflash_loss_decay(pos, gamma=4.0, sample_from_anchor=False)
        # True: slot 0 gets the HIGHEST weight (1.0), then decays -> first token IS trained.
        assert abs(float(w_true[0]) - 1.0) < 1e-6
        assert float(w_true[1]) < float(w_true[0])
        assert float(w_true[4]) < float(w_true[1])
        # False (DFlash): slot 0 is the anchor -> weight 0 (would starve gradient under True).
        assert float(w_false[0]) == 0.0
        assert abs(float(w_false[1]) - 1.0) < 1e-6
