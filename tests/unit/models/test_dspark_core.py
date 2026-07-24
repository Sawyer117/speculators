"""Focused unit tests for DSpark correction integration helpers."""

from types import SimpleNamespace

import torch
from torch import nn

from speculators.models.dspark.core import DSparkDraftModel


class _RecordingCorrectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.previous_embeddings: list[torch.Tensor] = []
        self.block_positions: list[torch.Tensor] = []

    def forward(
        self,
        previous_embeddings,
        dflash_hidden,
        block_positions,
        cache=None,
        *,
        use_cache=False,
    ):
        del cache
        self.previous_embeddings.append(previous_embeddings.detach().clone())
        self.block_positions.append(block_positions.detach().clone())
        states = dflash_hidden.new_zeros(*dflash_hidden.shape[:-1], 4)
        delta_hidden = torch.nn.functional.one_hot(
            block_positions, num_classes=dflash_hidden.shape[-1]
        ).to(dflash_hidden.dtype) * self.scale
        next_cache = [] if use_cache else None
        return delta_hidden, states, next_cache


class _RolloutHarness:
    _generated_feedback_correction = (
        DSparkDraftModel._generated_feedback_correction
    )
    rollout_correction = DSparkDraftModel.rollout_correction


class _GeneratedFeedbackHarness:
    _generated_feedback_correction = (
        DSparkDraftModel._generated_feedback_correction
    )


class _CountingLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(value)


def test_rollout_uses_generated_token_as_next_feedback():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(sample_from_anchor=True)
    harness.correction_head = _RecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.d2t = None
    with torch.no_grad():
        harness.embed_tokens.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 4)
        )
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[1, 0] = 10.0
        harness.lm_head.weight[2, 1] = 10.0
        harness.lm_head.weight[3, 2] = 10.0

    chosen_ids = torch.tensor([[1, 2, 3]])
    hidden = torch.zeros(1, 3, 4)
    tokens, _ = harness.rollout_correction(
        hidden,
        anchor_token_ids=torch.tensor([7]),
    )

    assert torch.equal(tokens, chosen_ids)
    feedback = torch.cat(harness.correction_head.previous_embeddings, dim=1)
    assert torch.equal(feedback[:, :, 0], torch.tensor([[7.0, 1.0, 2.0]]))
    positions = torch.cat(harness.correction_head.block_positions, dim=1)
    assert torch.equal(positions, torch.tensor([[0, 1, 2]]))
    assert harness.lm_head.calls == harness.block_size


def test_generated_feedback_training_retains_gradients_and_generated_tokens():
    harness = _GeneratedFeedbackHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_hidden_size=4,
    )
    harness.correction_head = _RecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.d2t = None
    with torch.no_grad():
        harness.embed_tokens.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 4)
        )
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[1, 0] = 10.0
        harness.lm_head.weight[2, 1] = 10.0
        harness.lm_head.weight[3, 2] = 10.0

    hidden = torch.zeros(1, 3, 4, requires_grad=True)
    tokens, logits, states = harness._generated_feedback_correction(
        hidden,
        anchor_token_ids=torch.tensor([7]),
    )
    logits.sum().backward()

    assert torch.equal(tokens, torch.tensor([[1, 2, 3]]))
    feedback = torch.cat(harness.correction_head.previous_embeddings, dim=1)
    assert torch.equal(feedback[:, :, 0], torch.tensor([[7.0, 1.0, 2.0]]))
    assert logits.shape == (1, 3, 8)
    assert states.shape == (1, 3, 4)
    assert harness.correction_head.scale.grad is not None
    assert hidden.grad is not None
    assert harness.lm_head.calls == harness.block_size


def test_confidence_feature_detach_switch_controls_both_inputs():
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    sequential = torch.randn(2, 3, 2, requires_grad=True)

    coupled = DSparkDraftModel._confidence_features(
        hidden, sequential, detach=False
    )
    coupled.sum().backward()
    assert hidden.grad is not None
    assert sequential.grad is not None

    detached = DSparkDraftModel._confidence_features(hidden, sequential, detach=True)
    assert not detached.requires_grad
    assert torch.equal(detached[..., :4], hidden.detach())
    assert torch.equal(detached[..., 4:], sequential.detach())
