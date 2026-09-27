"""DSPARK_INTER_HEADS (E11): off builds nothing; aux adds a loss without touching the
backbone; inject is an exact no-op at a zero gate and never reads a non-anchor token.

* Off (unset / ``off``): no ``inter_`` parameters exist and the forward is the old one.
* ``aux``: the backbone logits and the main loss are bit-identical to off with the same
  weights; the total loss grows by weight x the heads' CE, and per-head metrics appear.
* ``inject``: at gate 0 the logits equal off; at a non-zero gate they change. The
  injected ids are the heads' own guesses, so even under the non-causal block mask the
  logits cannot depend on any non-anchor input token.
* Building a head does not draw from the global RNG, so turning the heads on does not
  move the anchor sampling or hidden-state noise that follow model construction.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from speculators.config import SpeculatorsConfig, VerifierConfig  # noqa: E402
from speculators.models.dsv4_dspark.core import (  # noqa: E402
    DSV4DSparkConfig,
    DSV4DSparkDraftModel,
)
from speculators.models.dsv4_dspark.inter_heads import (  # noqa: E402
    IntermediateHead,
    chunked_ce,
    shift_within_blocks,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig  # noqa: E402

N_LAYERS = 3
BLOCK = 3
HIDDEN = 64
VOCAB = 97
SEQ = 40
ANCHORS = 4


def tiny_model(*, reinit: bool = True) -> DSV4DSparkDraftModel:
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
        draft_vocab_size=VOCAB,  # DSV4 drafts over the full vocab; inject relies on it
        block_size=BLOCK,
        mask_token_id=5,
        sample_from_anchor=True,
        sliding_window_non_causal=True,
        aux_hidden_state_layer_ids=list(range(N_LAYERS)),
        speculators_config=SpeculatorsConfig(
            algorithm="dsv4_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=BLOCK)],
            default_proposal_method="greedy",
            verifier=VerifierConfig.from_config(layer_config, name_or_path=None),
        ),
    )
    model = DSV4DSparkDraftModel(config)
    if reinit:
        torch.manual_seed(0)  # after construction; see test_dsv4_block_input.py
        for param in model.parameters():
            torch.nn.init.normal_(param, std=0.5)
    return model.eval()


def _with_heads(
    monkeypatch, mode: str, base: DSV4DSparkDraftModel
) -> DSV4DSparkDraftModel:
    """A model with heads whose shared weights are ``base``'s, bit for bit."""
    monkeypatch.setenv("DSPARK_INTER_HEADS", mode)
    model = tiny_model()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(k.startswith("inter_") for k in missing)
    return model


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


def _full(model, inputs):
    torch.manual_seed(123)
    hidden, ids, lm, last, doc = inputs
    _, loss, metrics = model(
        hidden, ids, lm, last, doc, max_anchors=ANCHORS, gamma=BLOCK
    )
    return loss, metrics


def _perturb(input_ids, pos):
    out = input_ids.clone()
    out[0, pos] = 6 + (int(out[0, pos]) - 6 + 1) % (VOCAB - 6)
    return out


def test_off_builds_nothing(monkeypatch):
    for value in (None, "", "off", "OFF"):
        if value is None:
            monkeypatch.delenv("DSPARK_INTER_HEADS", raising=False)
        else:
            monkeypatch.setenv("DSPARK_INTER_HEADS", value)
        model = tiny_model()
        assert model.inter_heads is None
        assert model.inter_gates is None
        assert not [n for n, _ in model.named_parameters() if "inter_" in n]


def test_aux_keeps_backbone_and_adds_loss(monkeypatch):
    monkeypatch.delenv("DSPARK_INTER_HEADS", raising=False)
    off = tiny_model()
    aux = _with_heads(monkeypatch, "aux", off)
    inputs = _inputs()
    hidden, ids, lm, last, doc = inputs
    assert torch.equal(
        _block_logits(off, ids, hidden, lm, last, doc)[0],
        _block_logits(aux, ids, hidden, lm, last, doc)[0],
    )
    with torch.no_grad():
        loss_off, m_off = _full(off, inputs)
        loss_aux, m_aux = _full(aux, inputs)
    assert torch.equal(m_off["loss_sum"], m_aux["loss_sum"])  # main loss untouched
    assert torch.equal(m_off["hard_accept_len_sum"], m_aux["hard_accept_len_sum"])
    extra = aux.inter_aux_weight * m_aux["inter_aux_loss_sum"]
    torch.testing.assert_close(loss_aux, loss_off + extra)
    assert m_aux["inter_aux_loss_sum"] > 0
    for i in range(1, N_LAYERS):
        for key in (
            "ce",
            "acc",
            "hard_accept_len",
            *(f"position_{k}_acc" for k in range(BLOCK)),
        ):
            assert f"inter{i}_{key}_sum" in m_aux
            assert f"inter{i}_{key}_total" in m_aux
    assert f"inter{N_LAYERS}_ce_sum" not in m_aux  # no head after the last layer


def test_aux_trains_heads_and_backbone(monkeypatch):
    monkeypatch.setenv("DSPARK_INTER_HEADS", "aux")
    monkeypatch.setenv("DSPARK_INTER_AUX_WEIGHT", "1.0")
    model = tiny_model().train()
    # Backprop the auxiliary term alone (weight 1 minus weight 0, same anchors): it must
    # reach the heads and the draft layers below them.
    with_aux, _ = _full(model, _inputs())
    model.inter_aux_weight = 0.0
    without, _ = _full(model, _inputs())
    (with_aux - without).backward()
    head = model.inter_heads[0]
    assert head.codebook.grad is not None
    assert head.codebook.grad.abs().sum() > 0
    assert head.down.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.layers[0].parameters()
    )


def test_inject_zero_gate_is_identical_and_gate_is_live(monkeypatch):
    monkeypatch.delenv("DSPARK_INTER_HEADS", raising=False)
    off = tiny_model()
    inj = _with_heads(monkeypatch, "inject", off)
    hidden, ids, lm, last, doc = _inputs()
    base = _block_logits(off, ids, hidden, lm, last, doc)[0]
    with torch.no_grad():
        for gate in inj.inter_gates:
            gate.zero_()
    assert torch.equal(base, _block_logits(inj, ids, hidden, lm, last, doc)[0])
    with torch.no_grad():
        for gate in inj.inter_gates:
            gate.fill_(1.0)
    live = (_block_logits(inj, ids, hidden, lm, last, doc)[0] - base).abs().max().item()
    assert live > 1e-2


def test_inject_ignores_non_anchor_tokens(monkeypatch):
    monkeypatch.setenv("DSPARK_INTER_HEADS", "inject")
    model = tiny_model()  # gates re-initialized to N(0, 0.5): injection is on
    hidden, ids, lm, last, doc = _inputs()
    base, idx = _block_logits(model, ids, hidden, lm, last, doc)
    anchors = set(idx[:, 0].tolist())
    for pos in {int(p) for p in idx[:, 1:].flatten()} - anchors:
        after, _ = _block_logits(model, _perturb(ids, pos), hidden, lm, last, doc)
        assert torch.equal(base, after), pos


def test_heads_do_not_consume_global_rng():
    # Checked on the head itself: the model's pre-existing _init_backbone_params
    # re-draws only the params whose torch.empty memory happens to be NaN or zero,
    # so the whole model's draw count is not repeatable from build to build.
    torch.manual_seed(7)
    want = torch.rand(4)
    torch.manual_seed(7)
    a = IntermediateHead(HIDDEN, 8, VOCAB, 1e-6, index=0)
    assert torch.equal(want, torch.rand(4))
    # Same index -> same weights on every rank; different layers get different heads.
    b = IntermediateHead(HIDDEN, 8, VOCAB, 1e-6, index=0)
    c = IntermediateHead(HIDDEN, 8, VOCAB, 1e-6, index=1)
    assert torch.equal(a.codebook, b.codebook)
    assert torch.equal(a.down, b.down)
    assert not torch.equal(a.codebook, c.codebook)


def test_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("DSPARK_INTER_HEADS", "on")
    with pytest.raises(ValueError, match="off, aux or inject"):
        tiny_model()


def test_chunked_ce_matches_direct():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(11, 6, generator=g, requires_grad=True)
    cb = torch.randn(13, 6, generator=g, requires_grad=True)
    labels = torch.randint(0, 13, (11,), generator=g)
    got = chunked_ce(z, cb, labels, chunk=4)
    want = torch.nn.functional.cross_entropy(z @ cb.t(), labels, reduction="none")
    torch.testing.assert_close(got, want)
    gz, gc = torch.autograd.grad(got.sum(), (z, cb))
    wz, wc = torch.autograd.grad(want.sum(), (z, cb))
    torch.testing.assert_close(gz, wz)
    torch.testing.assert_close(gc, wc)


def test_shift_within_blocks():
    ids = torch.tensor([10, 11, 12, 20, 21, 22])
    shifted, valid = shift_within_blocks(ids, 3)
    assert shifted.tolist() == [0, 10, 11, 0, 20, 21]
    assert valid.tolist() == [False, True, True, False, True, True]


def test_empty_knobs_mean_defaults(monkeypatch):
    # train_dsv4_dspark.sh passes unset knobs through as "".
    monkeypatch.setenv("DSPARK_INTER_HEADS", "aux")
    monkeypatch.setenv("DSPARK_INTER_AUX_WEIGHT", "")
    monkeypatch.setenv("DSPARK_INTER_RANK", "")
    model = tiny_model()
    assert model.inter_aux_weight == pytest.approx(0.2)
    assert model.inter_heads[0].codebook.shape == (VOCAB, 256)
