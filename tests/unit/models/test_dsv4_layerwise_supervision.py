"""Small CPU tests for the public training-only supervision path."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dsv4_dspark.core import DSV4DSparkConfig, DSV4DSparkDraftModel
from speculators.models.dsv4_dspark.inter_heads import (
    IntermediateHead,
    chunked_ce,
    settings_from_env,
    weight_at,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):

    for name in list(os.environ):
        if name.startswith("DSPARK_"):
            monkeypatch.delenv(name)


def tiny():
    layer = LlamaConfig(
        hidden_size=32,
        vocab_size=47,
        num_hidden_layers=3,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        num_attention_heads=2,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3,
    )
    cfg = DSV4DSparkConfig(
        transformer_layer_config=layer,
        num_heads=2,
        head_dim=16,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        window_size=8,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_inter_dim=16,
        hc_mult=2,
        markov_rank=8,
        draft_vocab_size=47,
        block_size=3,
        mask_token_id=5,
        sample_from_anchor=True,
        sliding_window_non_causal=True,
        aux_hidden_state_layer_ids=[0, 1, 2],
        speculators_config=SpeculatorsConfig(
            algorithm="dsv4_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig.from_config(layer, name_or_path=None),
        ),
    )
    model = DSV4DSparkDraftModel(cfg)
    # The tiny verifier tables are normally loaded from a target checkpoint.
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    return model


def inputs():
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(1, 24, 96, generator=generator),
        torch.randint(6, 47, (1, 24), generator=generator),
        torch.ones(1, 24),
        torch.randn(1, 24, 32, generator=generator),
        torch.zeros(1, 24, dtype=torch.long),
    )


def run(model):
    torch.manual_seed(19)
    return model(*inputs(), max_anchors=3, gamma=3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_head_init_rng_and_chunked_gradient(dtype):
    before = torch.get_rng_state().clone()
    head = IntermediateHead(16, 8, 31, 1e-6, 0).to(dtype)
    assert torch.equal(before, torch.get_rng_state())
    x = torch.randn(1, 9, 2, 16, dtype=dtype, requires_grad=True)
    z = head(x)
    labels = torch.arange(9)
    chunked = chunked_ce(z, head.codebook, labels, chunk=3)
    dense = torch.nn.functional.cross_entropy(
        (z @ head.codebook.t()).float(), labels, reduction="none"
    )
    torch.testing.assert_close(chunked, dense)
    left = torch.autograd.grad(chunked.sum(), (x, head.codebook), retain_graph=True)
    right = torch.autograd.grad(dense.sum(), (x, head.codebook))
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(
            a,
            b,
            rtol=0.03 if dtype == torch.bfloat16 else 1e-5,
            atol=0.01 if dtype == torch.bfloat16 else 1e-6,
        )


def pair(monkeypatch):
    base = tiny()
    monkeypatch.setenv("DSPARK_INTER_HEADS", "aux")
    monkeypatch.setenv("DSPARK_INTER_RANK", "8")
    monkeypatch.setenv("DSPARK_INTER_AUX_SCHEDULE", "step")
    model = tiny()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(k.startswith("inter_heads.") for k in missing)
    return base, model


def test_auxiliary_loss_and_main_metrics(monkeypatch):
    base, model = pair(monkeypatch)
    base.train()
    model.train()
    _, original, metrics = run(base)
    _, combined, aux = run(model)
    torch.testing.assert_close(combined, original + 0.2 * aux["inter_aux_loss_sum"])
    for key in metrics:
        torch.testing.assert_close(metrics[key], aux[key])
    combined.backward()
    assert model.inter_heads[0].down.grad.abs().sum() > 0
    assert model.layers[0].attn.wq_a.weight.grad is not None
    assert model.config.intermediate_supervision["rank"] == 8


def test_eval_never_executes_auxiliary_heads(monkeypatch):
    base, model = pair(monkeypatch)
    base.eval()
    model.eval()

    def forbidden(*_args):
        raise AssertionError("Training-only head was executed in eval")

    for head in model.inter_heads:
        monkeypatch.setattr(head, "forward", forbidden)
    with torch.no_grad():
        _, a, am = run(base)
        _, b, bm = run(model)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert am.keys() == bm.keys()


def test_schedule_boundary_and_saved_settings(monkeypatch):
    monkeypatch.setenv("DSPARK_INTER_HEADS", "aux")
    monkeypatch.setenv("DSPARK_INTER_AUX_SCHEDULE", "step")
    s = settings_from_env()
    assert weight_at(s, 6117) == 0.2
    assert weight_at(s, 6118) == 0.08
    monkeypatch.delenv("DSPARK_INTER_HEADS")
    assert settings_from_env(s) == s
    monkeypatch.setenv("DSPARK_INTER_HEADS", "inject")
    with pytest.raises(ValueError):
        settings_from_env(s)


def test_export_removes_only_training_heads(monkeypatch):
    base, model = pair(monkeypatch)
    path = Path(__file__).resolve().parents[3] / "scripts/convert_dspark_to_vllm.py"
    spec = importlib.util.spec_from_file_location("public_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original, _, _ = module.convert(base.state_dict(), 3)
    exported, skipped, _ = module.convert(model.state_dict(), 3)
    assert original.keys() == exported.keys()
    assert any(k.startswith("inter_heads.") for k in skipped)
    for key in original:
        assert torch.equal(original[key], exported[key])
