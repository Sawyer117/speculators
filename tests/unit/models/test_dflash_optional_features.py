"""Tests for opt-in DFlash backbone features and baseline parity."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _make_model(**feature_flags) -> DFlashDraftModel:
    transformer_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention"],
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=2)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
        **feature_flags,
    )
    return DFlashDraftModel(config).eval()


def test_optional_features_default_off_preserves_original_helpers():
    torch.manual_seed(0)
    model = _make_model()
    hidden = torch.randn(1, 5, 32)
    expected = model.hidden_norm(model.fc(hidden))
    actual = model._fuse_target_hidden(hidden)
    assert torch.equal(actual, expected)

    noise = torch.randn(1, 6, 16)
    document_ids = torch.zeros(1, 5, dtype=torch.long)
    conditioned = model._condition_noise_embedding(
        noise,
        actual,
        torch.tensor([2, 4]),
        document_ids,
    )
    assert conditioned is noise
    assert model.layer_fusion_norms is None
    assert model.layer_fusion_score is None
    assert model.context_hidden_proj is None
    assert model.verifier_final_hidden_proj is None
    assert model.block_position_embedding is None
    optional_prefixes = (
        "layer_fusion_",
        "context_hidden_",
        "verifier_final_hidden_",
        "block_position_embedding",
    )
    assert not any(
        key.startswith(optional_prefixes) for key in model.state_dict()
    )


def test_gated_layer_fusion_returns_draft_hidden_shape():
    torch.manual_seed(1)
    model = _make_model(dflash_gated_layer_fusion=True)
    hidden = torch.randn(1, 5, 32)
    fused = model._fuse_target_hidden(hidden)
    baseline = model.hidden_norm(model.fc(hidden))
    assert fused.shape == (1, 5, 16)
    assert torch.isfinite(fused).all()
    assert torch.equal(fused, baseline)
    assert model.fc.in_features == 32

    assert model.layer_fusion_gate is not None
    with torch.no_grad():
        model.layer_fusion_gate.fill_(1.0)
    assert not torch.equal(model._fuse_target_hidden(hidden), baseline)


def test_context_and_slot_residuals_start_at_exact_zero():
    torch.manual_seed(2)
    model = _make_model(
        dflash_context_residual=True,
        dflash_block_position_embedding=True,
    )
    noise = torch.randn(1, 6, 16)
    fused = torch.randn(1, 5, 16)
    document_ids = torch.zeros(1, 5, dtype=torch.long)
    initial = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2, 4]),
        document_ids,
    )
    assert torch.equal(initial, noise)

    assert model.context_hidden_proj is not None
    assert model.context_hidden_gate is not None
    assert model.block_position_embedding is not None
    with torch.no_grad():
        model.context_hidden_proj.weight.copy_(torch.eye(16))
        model.context_hidden_gate.fill_(1.0)
        model.block_position_embedding.weight[1].fill_(0.5)
    conditioned = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2, 4]),
        document_ids,
    )
    assert not torch.equal(conditioned, noise)


def test_context_residual_does_not_cross_document_boundary():
    model = _make_model(dflash_context_residual=True)
    assert model.context_hidden_proj is not None
    assert model.context_hidden_gate is not None
    with torch.no_grad():
        model.context_hidden_proj.weight.copy_(torch.eye(16))
        model.context_hidden_gate.fill_(1.0)

    noise = torch.zeros(1, 3, 16)
    fused = torch.ones(1, 4, 16)
    document_ids = torch.tensor([[0, 0, 1, 1]])
    conditioned = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2]),
        document_ids,
    )
    assert torch.equal(conditioned, noise)


def test_verifier_final_residual_uses_pre_lm_context_and_starts_at_zero():
    model = _make_model(dflash_verifier_final_residual=True)
    assert model.verifier_final_hidden_proj is not None
    assert model.verifier_final_hidden_gate is not None

    noise = torch.randn(1, 3, 16)
    fused = torch.randn(1, 4, 16)
    pre_lm = torch.ones(1, 4, 16)
    document_ids = torch.zeros(1, 4, dtype=torch.long)
    initial = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2]),
        document_ids,
        verifier_pre_lm_hidden=pre_lm,
    )
    assert torch.equal(initial, noise)

    with torch.no_grad():
        model.verifier_final_hidden_proj.weight.copy_(torch.eye(16))
        model.verifier_final_hidden_gate.fill_(1.0)
    conditioned = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2]),
        document_ids,
        verifier_pre_lm_hidden=pre_lm,
    )
    assert not torch.equal(conditioned, noise)


def test_verifier_final_residual_does_not_cross_document_boundary():
    model = _make_model(dflash_verifier_final_residual=True)
    assert model.verifier_final_hidden_proj is not None
    assert model.verifier_final_hidden_gate is not None
    with torch.no_grad():
        model.verifier_final_hidden_proj.weight.copy_(torch.eye(16))
        model.verifier_final_hidden_gate.fill_(1.0)

    noise = torch.zeros(1, 3, 16)
    fused = torch.zeros(1, 4, 16)
    pre_lm = torch.ones(1, 4, 16)
    document_ids = torch.tensor([[0, 0, 1, 1]])
    conditioned = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2]),
        document_ids,
        verifier_pre_lm_hidden=pre_lm,
    )
    assert torch.equal(conditioned, noise)
