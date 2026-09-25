"""DSPARK_BLOCK_INPUT: the E1 measurement arm is a no-op when off, leak-free when on.

* Off (unset or ``mask``): the block's non-anchor slots are mask tokens, so the
  backbone's logits cannot depend on any non-anchor token id. Unset and ``mask`` are
  the same model.
* On (``true``): slot k reads x[p+k]. Its own target x[p+k+1] sits in slot k+1's
  input, so under the required causal block mask, changing x[p+k+1] must leave slots
  0..k untouched.
* The model refuses ``true`` whenever any block mask would be non-causal.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from speculators.config import SpeculatorsConfig, VerifierConfig  # noqa: E402
from speculators.models.dsv4_dspark.core import (  # noqa: E402
    DSV4DSparkConfig,
    DSV4DSparkDraftModel,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig  # noqa: E402

N_LAYERS = 2
BLOCK = 3
HIDDEN = 64
VOCAB = 97
SEQ = 40
ANCHORS = 4


def tiny_model(*, non_causal: bool) -> DSV4DSparkDraftModel:
    layer_config = transformers.LlamaConfig(
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        num_hidden_layers=N_LAYERS,
        rms_norm_eps=1e-6,
        intermediate_size=64,
        num_attention_heads=2,
        sliding_window=8,
        layer_types=["sliding_attention"] * N_LAYERS,
    )
    config = DSV4DSparkConfig(
        transformer_layer_config=layer_config,
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
        moe_inter_dim=32,
        hc_mult=2,
        markov_rank=8,
        block_size=BLOCK,
        mask_token_id=5,
        sample_from_anchor=True,
        sliding_window_non_causal=non_causal,
        aux_hidden_state_layer_ids=list(range(N_LAYERS)),
        speculators_config=SpeculatorsConfig(
            algorithm="dsv4_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=BLOCK)],
            default_proposal_method="greedy",
            verifier=VerifierConfig.from_config(layer_config, name_or_path=None),
        ),
    )
    model = DSV4DSparkDraftModel(config)
    # Seed AFTER construction: building the model draws from the global RNG a
    # different number of times on the first call in a process, so a
    # pre-construction seed does not make two builds identical.
    torch.manual_seed(0)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.5)
    return model.eval()


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, SEQ, N_LAYERS * HIDDEN, generator=g)
    input_ids = torch.randint(6, VOCAB, (1, SEQ), generator=g)
    loss_mask = torch.ones(1, SEQ, dtype=torch.long)
    last = torch.randn(1, SEQ, HIDDEN, generator=g)
    doc = torch.zeros(1, SEQ, dtype=torch.long)
    return hidden, input_ids, loss_mask, last, doc


def _block_logits(model, input_ids, hidden, loss_mask, last, doc):
    torch.manual_seed(123)  # anchor sampling is random; pin it so two calls compare
    with torch.no_grad():
        _, logits, _, _, idx = model._backbone_forward(
            hidden, input_ids, loss_mask, last, doc, max_anchors=ANCHORS
        )
    return logits[0].view(ANCHORS, BLOCK, -1), idx.view(ANCHORS, BLOCK)


def _perturb(input_ids, pos):
    out = input_ids.clone()
    out[0, pos] = 6 + (int(out[0, pos]) - 6 + 1) % (VOCAB - 6)
    return out


def test_default_is_unset_and_mask_is_identical(monkeypatch):
    hidden, ids, lm, last, doc = _inputs()
    monkeypatch.delenv("DSPARK_BLOCK_INPUT", raising=False)
    a, _ = _block_logits(tiny_model(non_causal=True), ids, hidden, lm, last, doc)
    monkeypatch.setenv("DSPARK_BLOCK_INPUT", "mask")
    b, _ = _block_logits(tiny_model(non_causal=True), ids, hidden, lm, last, doc)
    assert torch.equal(a, b)


def test_off_ignores_non_anchor_tokens(monkeypatch):
    monkeypatch.delenv("DSPARK_BLOCK_INPUT", raising=False)
    model = tiny_model(non_causal=True)
    hidden, ids, lm, last, doc = _inputs()
    base, idx = _block_logits(model, ids, hidden, lm, last, doc)
    anchors = set(idx[:, 0].tolist())
    non_anchor = next(int(p) for p in idx[:, 1:].flatten() if int(p) not in anchors)
    after, _ = _block_logits(model, _perturb(ids, non_anchor), hidden, lm, last, doc)
    assert torch.equal(base, after)


def test_true_is_causal_and_leak_free(monkeypatch):
    monkeypatch.setenv("DSPARK_BLOCK_INPUT", "true")
    model = tiny_model(non_causal=False)
    hidden, ids, lm, last, doc = _inputs()
    base, idx = _block_logits(model, ids, hidden, lm, last, doc)
    for b in range(ANCHORS):
        for k in range(1, BLOCK):
            pos = int(idx[b, k])  # slot k's input = slot k-1's target
            if pos in set(idx[:, 0].tolist()):
                continue  # also an anchor elsewhere; skip to keep the check local
            after, _ = _block_logits(model, _perturb(ids, pos), hidden, lm, last, doc)
            # Slots before k never see x[pos]: that is the leak check. Not bit-exact:
            # the changed token can route to a different expert, which changes that
            # expert's batch shape and so the float summation order for every token
            # it serves.
            leak = (after[b, :k] - base[b, :k]).abs().max().item()
            live = (after[b, k] - base[b, k]).abs().max().item()
            assert leak < 1e-4, (b, k, leak)
            # Slot k reads it directly, so the arm is really live.
            assert live > 1e-2, (b, k, live)


def test_true_refuses_non_causal(monkeypatch):
    monkeypatch.setenv("DSPARK_BLOCK_INPUT", "true")
    with pytest.raises(ValueError, match="CAUSAL"):
        tiny_model(non_causal=True)


def test_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("DSPARK_BLOCK_INPUT", "truth")
    with pytest.raises(ValueError, match="mask' or 'true"):
        tiny_model(non_causal=False)
