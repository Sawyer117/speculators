"""Tests for opt-in DFlash backbone features and baseline parity."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _make_model(*, num_draft_layers: int = 1, **feature_flags) -> DFlashDraftModel:
    transformer_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_draft_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention"] * num_draft_layers,
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
    assert model.dfly_layer_fusion_logits is None
    assert model.layers[0].self_attn.target_k_proj is None
    assert model.layers[0].self_attn.target_v_proj is None
    assert model.context_hidden_proj is None
    assert model.verifier_final_hidden_proj is None
    assert model.block_position_embedding is None
    optional_prefixes = (
        "layer_fusion_",
        "dfly_layer_fusion_",
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


def test_dfly_layer_residual_requires_current_gated_fusion():
    with pytest.raises(ValueError, match="requires dflash_gated_layer_fusion"):
        _make_model(dflash_dfly_layer_residual=True)


def test_dfly_layer_residual_adds_distinct_per_draft_layer_views():
    torch.manual_seed(2)
    model = _make_model(
        num_draft_layers=2,
        dflash_gated_layer_fusion=True,
        dflash_dfly_layer_residual=True,
    )
    hidden = torch.randn(1, 5, 32)
    shared_projection, target_layer_states = model._prepare_target_hidden(hidden)
    assert target_layer_states is not None
    assert model.dfly_layer_fusion_logits is not None
    assert model.dfly_layer_residual_gate is not None

    initial_0 = model._add_dfly_layer_residual(
        shared_projection, target_layer_states, 0
    )
    initial_1 = model._add_dfly_layer_residual(
        shared_projection, target_layer_states, 1
    )
    baseline = model.hidden_norm(shared_projection)
    assert torch.equal(initial_0, baseline)
    assert torch.equal(initial_0, initial_1)

    with torch.no_grad():
        model.dfly_layer_residual_gate.fill_(1.0)
        model.dfly_layer_fusion_logits[0].copy_(torch.tensor([8.0, -8.0]))
        model.dfly_layer_fusion_logits[1].copy_(torch.tensor([-8.0, 8.0]))

    fused_0 = model._add_dfly_layer_residual(
        shared_projection, target_layer_states, 0
    )
    fused_1 = model._add_dfly_layer_residual(
        shared_projection, target_layer_states, 1
    )
    assert fused_0.shape == (1, 5, 16)
    assert fused_1.shape == (1, 5, 16)
    assert torch.isfinite(fused_0).all()
    assert torch.isfinite(fused_1).all()
    assert not torch.equal(fused_0, fused_1)


def test_heterogeneous_kv_projections_are_separate_and_used_for_context():
    torch.manual_seed(3)
    model = _make_model(dflash_heterogeneous_kv_projections=True)
    attention = model.layers[0].self_attn
    assert attention.target_k_proj is not None
    assert attention.target_v_proj is not None
    assert attention.target_k_proj is not attention.k_proj
    assert attention.target_v_proj is not attention.v_proj

    attention.config._attn_implementation = "eager"  # noqa: SLF001
    draft_hidden = torch.randn(1, 3, 16)
    target_hidden = torch.randn(1, 5, 16)
    cos = torch.ones(1, 8, 4)
    sin = torch.zeros(1, 8, 4)
    output_before = attention(draft_hidden, target_hidden, (cos, sin), None)[0]
    with torch.no_grad():
        attention.target_v_proj.weight.zero_()
    output_after = attention(draft_hidden, target_hidden, (cos, sin), None)[0]
    assert output_before.shape == draft_hidden.shape
    assert torch.isfinite(output_before).all()
    assert not torch.equal(output_before, output_after)


def test_missing_heterogeneous_kv_copies_shared_projections():
    model = _make_model(dflash_heterogeneous_kv_projections=True)
    attention = model.layers[0].self_attn
    assert attention.target_k_proj is not None
    assert attention.target_v_proj is not None
    with torch.no_grad():
        attention.k_proj.weight.fill_(1.0)
        attention.v_proj.weight.fill_(2.0)
        attention.target_k_proj.weight.zero_()
        attention.target_v_proj.weight.zero_()

    model._prepare_missing_checkpoint_weights(
        {
            "missing_keys": [
                "layers.0.self_attn.target_k_proj.weight",
                "layers.0.self_attn.target_v_proj.weight",
            ]
        }
    )

    assert torch.equal(attention.target_k_proj.weight, attention.k_proj.weight)
    assert torch.equal(attention.target_v_proj.weight, attention.v_proj.weight)


def test_legacy_dfly_checkpoint_preserves_ungated_residual():
    model = _make_model(
        dflash_gated_layer_fusion=True,
        dflash_dfly_layer_residual=True,
    )
    assert model.dfly_layer_residual_gate is not None
    assert model.dfly_layer_residual_gate.item() == 0.0

    model._prepare_missing_checkpoint_weights(
        {"missing_keys": ["dfly_layer_residual_gate"]}
    )

    assert model.dfly_layer_residual_gate.item() == 1.0


def test_dfly_rejects_checkpoint_without_trained_feature_weights():
    model = _make_model(
        dflash_gated_layer_fusion=True,
        dflash_dfly_layer_residual=True,
    )

    with pytest.raises(RuntimeError, match="does not contain their trained weights"):
        model._prepare_missing_checkpoint_weights(
            {"missing_keys": ["dfly_layer_fusion_logits"]}
        )


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
