#!/usr/bin/env python3
"""Compare six block DECODERS offline, on the training box, with no eval server.

    TRAIN_PY=examples/ascend_npu_dflash/decoder_ablation_probe.py \
    DECODER_K=4 LR=0 EPOCHS=6 CKPT_FREQ=99 MAX_ANCHORS=128 ... \
      bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful

WHY THIS EXISTS
---------------
`recall_headroom_probe.py` measured the *headroom*: at the last block position the target
token sits inside the draft's top-16 84.6% of the time but argmax picks it only 58.1% --
26.4 points, and hard 3.845 vs oracle@16 5.309 (+1.46 tokens). An oracle, however, is not
a decoder. The question that actually decides whether to build a selector is: **how much of
that headroom does a real decoding rule recover?**

Answering it does NOT need an eval server. The parallel backbone's ``base_logits`` are
computed once per block and do not depend on the path taken through it; the Markov bias is
just ``markov_w2(markov_w1(prev))``. Both are in hand during training, so any decoder can
be replayed exactly as the serve would run it -- feeding each decoder its OWN pick as the
next predecessor rather than the true token -- and scored against the target.

THE DECODERS
------------
  today      argmax over the FULL vocabulary of ``base + bias(prev)``, committing each
             step. This is what the serve does now (deepseek_v4_dspark_proposer.py's
             ``_sample_sequential``) and is therefore THE baseline.
  restrict   the same rule, but candidates first restricted to top-k of ``base``.
             ⚠ This is a strict handicap on ``today``, never an improvement: same commit
             point, smaller search set. It is here as a CONTROL -- it prices the candidate
             restriction that viterbi and decay also pay, so their net gain can be split
             into (joint decoding) − (restriction cost). It is NOT "DFlash2's decoder":
             their greedy walk runs over *learned* A/B/H scores, and without training those
             there is no reproduction of anything.
  viterbi    exact max-total-chain-score path over the k-candidate lattice.
  decay      viterbi with the training loss's own position weights, w_t = exp(-t/gamma),
             gamma=4. Prefix acceptance pays for E[accepted] = Σ_t P(0..t all correct), not
             for the block's total score, so down-weighting late positions is the cheap way
             to point the decoder at the right objective.
  viterbiN   viterbi, but each position's scores are log_softmax'd before being summed.
  decayN     decay, likewise normalised.
             ★ The N pair exists because summing RAW logits across positions weights each
             position by its own logit scale, which is arbitrary. Sequence decoders sum
             log-probabilities, and then the chain total is log P(path). Today's decoder is
             invariant to this (per-position argmax), joint decoding is not -- so an
             unnormalised loss could be a units artefact rather than objective mismatch, and
             only running both tells the two apart. Sharing one lattice makes the extra pair
             a DP pass, not another sweep over the vocabulary.

WHAT IT REPORTS (all as *_sum/*_total, so the trainer's aggregation folds them into ratios)
    dec_{name}_accept_len     hard accept length under that decoder
    dec_{name}_pos{p}_acc     per-position accuracy
    dec_{name}_win / _loss    fraction of blocks where it beats / trails ``today``
    dec_{name}_gain           mean accept-length delta vs ``today``  ← the headline
    dec_restrict_cost         today − restrict, i.e. what top-k candidate pruning costs
    dec_{viterbiN,decayN}_*   the same, with per-position normalisation

⚠ TEACHER-FORCED CONTEXT. The backbone hidden states here were produced with the true
prefix, as in training. The decoder simulation itself is faithful (each decoder feeds its
own pick forward), but the block's starting context is the true one, where at serve it is
whatever the target model verified. Same caveat as the recall probe: an upper bound,
decisive when negative.

⚠ vanilla Markov head only (``markov_head_type=vanilla``, our config). The gated/rnn
variants make the bias depend on hidden state or recurrent state, so ``bias(a)[b]`` is no
longer a pure function of the predecessor and the replay below would not match the serve.
The probe refuses to run on those rather than reporting a wrong number.
"""

from __future__ import annotations

import os
import sys

import torch

_K = int(os.environ.get("DECODER_K", "4"))
_GAMMA = float(os.environ.get("DECODER_GAMMA", "4.0"))
_CHUNK = int(os.environ.get("DECODER_CHUNK", "256"))

_STASH: dict = {}


def _lattice(base, head, seed_ids, k):
    """Build the candidate lattice ONCE: unary values, ids, and all pairwise transitions.

    Shared by every lattice decoder, so adding a scoring variant costs a DP pass rather
    than another O(T*N*k*V) sweep through markov_bias -- which is what makes it affordable
    to measure normalised and unnormalised scoring in the same run.
    """
    N, T, _ = base.shape
    cand_val, cand_id = base.topk(k, dim=-1)  # [N, T, k]
    bias_of = lambda ids: head.markov_w2(head.markov_w1(ids.long()))  # noqa: E731
    trans = []  # trans[t][n, i, j] = bias(cand[t-1][i])[cand[t][j]], t >= 1
    b0 = bias_of(seed_ids).to(base.dtype).gather(1, cand_id[:, 0])  # [N, k]
    for t in range(1, T):
        b = bias_of(cand_id[:, t - 1].reshape(-1)).to(base.dtype)
        trans.append(
            b.gather(1, cand_id[:, t].unsqueeze(1).expand(N, k, k).reshape(N * k, k)).view(N, k, k)
        )
    return cand_val, cand_id, b0, trans


def _dp(cand_val, cand_id, b0, trans, w, normalise):
    """Viterbi over the cached lattice. ``w`` weights positions; ``normalise`` turns the
    raw logits into per-position log-probabilities first.

    ⚠ WHY normalise MATTERS. ``base_logits`` are unnormalised. Summing them across
    positions weights each position by its own logit scale, which is arbitrary -- a
    position whose logits happen to span a wider range dominates which path is chosen.
    Sequence decoders sum LOG-PROBABILITIES for exactly this reason (so does beam search),
    and then the total is log P(path) and Viterbi maximises P(the whole block correct).
    Today's decoder is unaffected either way, since a per-position argmax is invariant to
    normalisation -- but joint decoding is not, so the two must be measured separately.
    """
    N, T, k = cand_val.shape

    def unary(t):
        v = cand_val[:, t]
        return v - v.logsumexp(dim=-1, keepdim=True) if normalise else v

    def trans_at(t):
        m = trans[t - 1]
        return m - m.logsumexp(dim=-1, keepdim=True) if normalise else m

    score = (unary(0) + (b0 - b0.logsumexp(-1, keepdim=True) if normalise else b0)) * w[0]
    back = torch.zeros((N, T, k), dtype=torch.long, device=cand_val.device)
    for t in range(1, T):
        total = score.unsqueeze(2) + (unary(t).unsqueeze(1) + trans_at(t)) * w[t]
        score, arg = total.max(dim=1)
        back[:, t] = arg
    out = torch.empty((N, T), dtype=torch.long, device=cand_val.device)
    j = score.argmax(dim=-1)
    for t in range(T - 1, -1, -1):
        out[:, t] = cand_id[:, t].gather(1, j.unsqueeze(1)).squeeze(1)
        j = back[:, t].gather(1, j.unsqueeze(1)).squeeze(1)
    return out


def _stepwise(base, head, seed_ids, k, restrict):
    """Commit-per-step decoding: the serve's current rule (``restrict=False``) and the same
    rule with candidates pruned to top-k of the unary term (``restrict=True``)."""
    N, T, _ = base.shape
    bias_of = lambda ids: head.markov_w2(head.markov_w1(ids.long()))  # noqa: E731
    prev = seed_ids
    out = torch.empty((N, T), dtype=torch.long, device=base.device)
    for t in range(T):
        scores = base[:, t] + bias_of(prev).to(base.dtype)
        if restrict:
            cid = base[:, t].topk(k, dim=-1).indices
            pick = cid.gather(1, scores.gather(1, cid).argmax(dim=-1, keepdim=True)).squeeze(1)
        else:
            pick = scores.argmax(dim=-1)
        out[:, t] = pick
        prev = pick
    return out


def _accept_len(ids, target, mask):
    """1 + longest all-correct prefix, matching metrics.py's hard_accept_len exactly."""
    m = (ids == target).to(torch.float32) * mask
    return m.cumprod(dim=-1).sum(dim=-1) + 1.0


def decoder_metrics(logits, targets, loss_mask, block_size, head, prev_token_ids, markov_bias):
    """Replay every decoder on this batch and return *_sum/*_total metrics."""
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        N = logits.shape[1] // block_size
        V = logits.shape[-1]
        # `logits` arrives with the Markov bias already folded in (dspark/core.py); undo it
        # to recover the unary term the decoders must start from.
        base = (logits.view(N, block_size, V) - markov_bias.view(N, block_size, V)).float()
        target = targets.view(N, block_size, V).argmax(dim=-1)
        mask = loss_mask.to(torch.float32).view(N, block_size)
        # The block's predecessor chain starts at the anchor: slot 0's prev token.
        seed = prev_token_ids.view(N, block_size)[:, 0]
        valid = (mask.sum(dim=-1) > 0).to(torch.float32)
        nvalid = valid.sum().clamp_min(1.0)

        acc_len = {}
        cv, ci, b0, tr = _lattice(base, head, seed, _K)
        T = block_size
        w1 = [1.0] * T
        wd = [float(torch.exp(torch.tensor(-t / _GAMMA))) for t in range(T)]
        runs = [
            ("today", lambda: _stepwise(base, head, seed, _K, False)),
            ("restrict", lambda: _stepwise(base, head, seed, _K, True)),
            ("viterbi", lambda: _dp(cv, ci, b0, tr, w1, False)),
            ("decay", lambda: _dp(cv, ci, b0, tr, wd, False)),
            ("viterbiN", lambda: _dp(cv, ci, b0, tr, w1, True)),
            ("decayN", lambda: _dp(cv, ci, b0, tr, wd, True)),
        ]
        for name, fn in runs:
            ids = fn()
            al = _accept_len(ids, target, mask)
            acc_len[name] = al
            out[f"dec_{name}_accept_len_sum"] = (al * valid).sum()
            out[f"dec_{name}_accept_len_total"] = nvalid.clone()
            hit = (ids == target).to(torch.float32) * mask
            for p in range(block_size):
                out[f"dec_{name}_pos{p}_acc_sum"] = hit[:, p].sum()
                out[f"dec_{name}_pos{p}_acc_total"] = mask[:, p].sum().clamp_min(1.0)

        for name in ("restrict", "viterbi", "decay", "viterbiN", "decayN"):
            d = (acc_len[name] - acc_len["today"]) * valid
            out[f"dec_{name}_gain_sum"] = d.sum()
            out[f"dec_{name}_gain_total"] = nvalid.clone()
            out[f"dec_{name}_win_sum"] = ((d > 0).to(torch.float32) * valid).sum()
            out[f"dec_{name}_win_total"] = nvalid.clone()
            out[f"dec_{name}_loss_sum"] = ((d < 0).to(torch.float32) * valid).sum()
            out[f"dec_{name}_loss_total"] = nvalid.clone()
        # What the top-k candidate pruning costs on its own (today − restrict >= 0).
        out["dec_restrict_cost_sum"] = ((acc_len["today"] - acc_len["restrict"]) * valid).sum()
        out["dec_restrict_cost_total"] = nvalid.clone()
    return out


def install() -> None:
    """Stash the Markov head + its bias, then append decoder metrics in compute_metrics."""
    import speculators.models.dspark.core as core
    from speculators.models.dspark.model_definitions import MarkovHead

    if getattr(core.compute_metrics, "_decoder_probe", False):
        return

    orig_bias = MarkovHead.block_bias

    def bias_spy(self, *, prev_token_ids, hidden_states, prev_emb=None):
        b = orig_bias(self, prev_token_ids=prev_token_ids, hidden_states=hidden_states, prev_emb=prev_emb)
        if self.head_type != "vanilla":
            raise RuntimeError(
                f"decoder_ablation_probe: markov_head_type={self.head_type!r}. The replay assumes "
                "bias(a)[b] is a pure function of the predecessor, which only holds for 'vanilla'."
            )
        _STASH["head"], _STASH["bias"], _STASH["prev"] = self, b.detach(), prev_token_ids.detach()
        return b

    MarkovHead.block_bias = bias_spy

    original = core.compute_metrics

    def wrapped(logits, targets, confidence_logits, loss_mask, block_size, *a, **kw):
        loss, metrics = original(logits, targets, confidence_logits, loss_mask, block_size, *a, **kw)
        try:
            if "head" in _STASH:
                metrics.update(
                    decoder_metrics(logits, targets, loss_mask, block_size,
                                    _STASH["head"], _STASH["prev"], _STASH["bias"])
                )
        except Exception as exc:  # a probe must never take down a run
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[decoder-probe] skipped this step: {exc!r}", flush=True)
        return loss, metrics

    wrapped._decoder_probe = True  # type: ignore[attr-defined]
    core.compute_metrics = wrapped


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    install()
    import train as train_script

    args = train_script.parse_args()
    if int(os.environ.get("RANK", "0")) == 0:
        lr = getattr(args, "lr", None)
        print("=" * 78, flush=True)
        print(f">>> DECODER ABLATION PROBE — today/restrict/viterbi/decay(+N)   k={_K} gamma={_GAMMA}", flush=True)
        print(">>> replays each decoder on the SAME blocks; 'today' == the serve's current rule", flush=True)
        print(f">>> lr={lr}" + ("" if lr == 0 else "   ⚠ NOT 0 — this run WILL train"), flush=True)
        print(">>> read dec_*_gain (vs today) and dec_restrict_cost, then kill it", flush=True)
        print("=" * 78, flush=True)
    train_script.main(args)


if __name__ == "__main__":
    main()
