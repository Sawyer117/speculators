"""Focused unit tests for DSpark correction integration helpers."""

from types import SimpleNamespace

import torch
from torch import nn

from speculators.models.dspark.core import DSparkDraftModel


class _RecordingCorrectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.previous_embeddings: list[torch.Tensor] = []

    def forward(
        self,
        previous_embeddings,
        dflash_hidden,
        logit_context,
        logit_stats,
        cache=None,
        *,
        use_cache=False,
    ):
        del logit_context, logit_stats, cache
        self.previous_embeddings.append(previous_embeddings.detach().clone())
        states = dflash_hidden.new_zeros(*dflash_hidden.shape[:-1], 4)
        next_cache = [] if use_cache else None
        return torch.zeros_like(dflash_hidden), states, next_cache


class _RolloutHarness:
    _summarize_base_logits = DSparkDraftModel._summarize_base_logits
    rollout_correction = DSparkDraftModel.rollout_correction


def test_logit_summary_handles_top_k_one():
    harness = SimpleNamespace(
        config=SimpleNamespace(correction_top_k=1),
        lm_head=nn.Linear(3, 4, bias=False),
    )
    with torch.no_grad():
        harness.lm_head.weight.copy_(torch.arange(12).view(4, 3))
    logits = torch.tensor([[[1.0, 4.0, 2.0, 0.0], [3.0, 1.0, 0.0, 2.0]]])

    context, stats = DSparkDraftModel._summarize_base_logits(harness, logits)

    expected_ids = logits.argmax(dim=-1)
    assert context.shape == (1, 2, 3)
    assert stats.shape == (1, 2, 3)
    assert torch.equal(context, harness.lm_head.weight[expected_ids])
    assert torch.allclose(stats[..., 0], stats[..., 1])
    assert torch.allclose(stats[..., 0], stats[..., 2])


def test_rollout_uses_generated_token_as_next_feedback():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(sample_from_anchor=True, correction_top_k=1)
    harness.correction_head = _RecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = nn.Linear(4, 8, bias=False)
    harness.d2t = None
    with torch.no_grad():
        harness.embed_tokens.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 4)
        )
        harness.lm_head.weight.zero_()

    chosen_ids = torch.tensor([[1, 2, 3]])
    base_logits = torch.zeros(1, 3, 8)
    base_logits.scatter_(-1, chosen_ids.unsqueeze(-1), 10.0)
    hidden = torch.randn(1, 3, 4)
    tokens, _ = harness.rollout_correction(
        base_logits,
        hidden,
        anchor_token_ids=torch.tensor([7]),
    )

    assert torch.equal(tokens, chosen_ids)
    feedback = torch.cat(harness.correction_head.previous_embeddings, dim=1)
    assert torch.equal(feedback[:, :, 0], torch.tensor([[7.0, 1.0, 2.0]]))


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
