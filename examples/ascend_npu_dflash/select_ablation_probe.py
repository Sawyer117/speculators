#!/usr/bin/env python3
"""What is the SELECTION TERM alone worth? -- an in-run, perfectly paired ablation.

In-run (add to any training launch):
    TRAIN_PY=examples/ascend_npu_dflash/select_ablation_probe.py  <normal env> \
      bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful

After the fact, on a checkpoint, with LR=0 so no weight can move:
    TRAIN_PY=examples/ascend_npu_dflash/select_ablation_probe.py \
    SELECT_RANK=256 LR=0 EPOCHS=6 CKPT_FREQ=99 MAX_ANCHORS=128 \
    SAVE_PATH=$RUN/ckpt_faithful_ep_<TS> \
      bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful

WHY THIS EXISTS
---------------
The paired run-vs-baseline comparison answers "is the SELECT arm ahead of ROPEFIX", which is
the primary question but not the only one. It cannot answer "how much of that is the
selection term itself", because the two arms train different backbones and different Markov
heads -- everything moved at once.

Because SelectHead is ADDITIVE, that second question is answerable exactly and for free:

    logits            = base + markov_bias + select_bias      <- what the model computes
    logits - select_bias                                      <- the same model, term removed

Same weights, same batch, same step. The difference between their accept lengths is the
selection term's own contribution, with no confound whatsoever. It is also precisely what
the vllm-ascend patch would buy: a serve that has not been taught about SelectHead computes
the `select_off` row, and one that has computes `select_on`.

⚠ TEACHER-FORCED, like every training-side metric here: the Markov and select heads both see
the TRUE predecessor, where at serve they see the draft's own pick. Measured exposure bias
on this model is 0.014 tokens (see the improvement-experiments worklog §7.1), so this is a
mild caveat rather than a load-bearing one -- but the number is still an upper bound.

WHAT IT REPORTS (as *_sum/*_total, so the trainer's aggregation folds them into ratios)
    sel_{on,sel_off,mk_off,both_off}_accept_len   the four corners of (markov) x (select)
    sel_gain     on - sel_off  <- THE headline: what the vllm-ascend patch buys
    mk_gain      on - mk_off      the Markov term's own contribution, for comparison
    both_gain    on - both_off    both terms together, over the bare backbone
    sel_{corner}_pos{p}_acc       per-position accuracy at each corner
    sel_bias_rms / mk_bias_rms    how far each term has grown from its init -- distinguishes
                                  "trained but useless" from "not trained yet"

SELECT_PROBE_EVERY=N runs it every N steps if the extra argmaxes ever cost too much.
"""

from __future__ import annotations

import os
import sys

import torch

_EVERY = int(os.environ.get("SELECT_PROBE_EVERY", "1"))
_STASH: dict = {}
_STEP = [0]


def _accept_len(ids, target, mask):
    """1 + longest all-correct prefix, matching metrics.py's hard_accept_len exactly."""
    m = (ids == target).to(torch.float32) * mask
    return m.cumprod(dim=-1).sum(dim=-1) + 1.0


_ARMS = (
    # name        markov?  select?      what it is
    ("on",        True,  True),   # what the model computes
    ("sel_off",   True,  False),  # ← EXACTLY what an unpatched vllm-ascend computes
    ("mk_off",    False, True),   # selection alone
    ("both_off",  False, False),  # the bare backbone
)


def select_metrics(logits, targets, loss_mask, block_size, markov_bias, select_bias, chunk=64):
    """Accept length at all FOUR corners of (markov on/off) x (select on/off).

    Two corners answer the two questions that "what is selection worth" actually splits into:

        on - sel_off   what the vllm-ascend patch buys ON THIS CHECKPOINT. Exact: same
                       weights, same batch, one additive term removed -- not a re-run.
        sel_off vs mk_off   how the model DIVIDED the work. If dropping select costs about
                       what dropping markov costs, the two terms carry comparable load and
                       the backbone has genuinely leaned on the new one; if select is nearly
                       free to drop, it never took on any.

    That second reading is why two arms are not enough. `sel_off` is NOT "the vanilla model":
    it is this model with a limb removed, and the backbone trained knowing the limb was
    there. A large sel_gain can therefore mean "the model learned to depend on selection"
    rather than "selection made it better" -- only the paired SELECT-vs-ROPEFIX run
    comparison settles that, and the four corners say which story is more likely.

    ⚠ CHUNKED ON PURPOSE. At MAX_ANCHORS=512 a single [512, 5, 129280] tensor is 1.3 GB in
    fp32, and a naive float subtraction would allocate several beside a run already using
    activation checkpointing to fit. Nothing is cast, argmax needs no copy, and the
    accept-length arithmetic runs on the small [chunk, block] results.
    """
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        N = logits.shape[1] // block_size
        V = logits.shape[-1]
        lg = logits.view(N, block_size, V)
        mb = markov_bias.view(N, block_size, V)
        sb = select_bias.view(N, block_size, V)
        target = targets.view(N, block_size, V).argmax(dim=-1)
        mask = loss_mask.to(torch.float32).view(N, block_size)
        valid = (mask.sum(dim=-1) > 0).to(torch.float32)
        nvalid = valid.sum().clamp_min(1.0)

        al = {n: [] for n, _, _ in _ARMS}
        hits = {n: [] for n, _, _ in _ARMS}
        sq_s = torch.zeros((), dtype=torch.float32, device=lg.device)
        sq_m = torch.zeros((), dtype=torch.float32, device=lg.device)
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            m, b = mb[i:j], sb[i:j]
            sq_s += b.float().pow(2).sum()
            sq_m += m.float().pow(2).sum()
            for name, keep_m, keep_s in _ARMS:
                z = lg[i:j]
                if not keep_m:
                    z = z - m
                if not keep_s:
                    z = z - b
                ids = z.argmax(dim=-1)
                al[name].append(_accept_len(ids, target[i:j], mask[i:j]))
                hits[name].append((ids == target[i:j]).to(torch.float32) * mask[i:j])
        al = {k: torch.cat(v) for k, v in al.items()}

        for name, _, _ in _ARMS:
            out[f"sel_{name}_accept_len_sum"] = (al[name] * valid).sum()
            out[f"sel_{name}_accept_len_total"] = nvalid.clone()
            hit = torch.cat(hits[name])
            for p in range(block_size):
                out[f"sel_{name}_pos{p}_acc_sum"] = hit[:, p].sum()
                out[f"sel_{name}_pos{p}_acc_total"] = mask[:, p].sum().clamp_min(1.0)

        # gains are always measured against `on`, the model as it actually is
        for name in ("sel_off", "mk_off", "both_off"):
            d = (al["on"] - al[name]) * valid
            tag = {"sel_off": "sel_gain", "mk_off": "mk_gain", "both_off": "both_gain"}[name]
            out[f"{tag}_sum"] = d.sum()
            out[f"{tag}_total"] = nvalid.clone()
            if name == "sel_off":
                out["sel_win_sum"] = ((d > 0).to(torch.float32) * valid).sum()
                out["sel_win_total"] = nvalid.clone()
                out["sel_loss_sum"] = ((d < 0).to(torch.float32) * valid).sum()
                out["sel_loss_total"] = nvalid.clone()
        # How far each term has grown. A near-zero gain means opposite things depending on
        # whether the term is still at its zero init or has grown and simply does nothing.
        out["sel_bias_rms_sum"] = (sq_s / sb.numel()).sqrt()
        out["sel_bias_rms_total"] = torch.ones((), device=sb.device)
        out["mk_bias_rms_sum"] = (sq_m / mb.numel()).sqrt()
        out["mk_bias_rms_total"] = torch.ones((), device=mb.device)
    return out


def install() -> None:
    """Stash the selection bias, then append the on/off comparison in compute_metrics."""
    import speculators.models.dspark.core as core
    from speculators.models.dspark.model_definitions import MarkovHead, SelectHead

    if getattr(core.compute_metrics, "_select_probe", False):
        return

    orig_sel = SelectHead.block_bias

    def sel_spy(self, *, prev_token_ids, hidden_states):
        b = orig_sel(self, prev_token_ids=prev_token_ids, hidden_states=hidden_states)
        _STASH["sel"] = b.detach()
        return b

    SelectHead.block_bias = sel_spy

    orig_mk = MarkovHead.block_bias

    def mk_spy(self, *, prev_token_ids, hidden_states, prev_emb=None):
        b = orig_mk(self, prev_token_ids=prev_token_ids, hidden_states=hidden_states,
                    prev_emb=prev_emb)
        _STASH["mk"] = b.detach()
        return b

    MarkovHead.block_bias = mk_spy

    original = core.compute_metrics

    def wrapped(logits, targets, confidence_logits, loss_mask, block_size, *a, **kw):
        loss, metrics = original(logits, targets, confidence_logits, loss_mask, block_size, *a, **kw)
        _STEP[0] += 1
        try:
            if "sel" in _STASH and "mk" in _STASH and _STEP[0] % _EVERY == 0:
                metrics.update(
                    select_metrics(logits, targets, loss_mask, block_size,
                                   _STASH["mk"], _STASH["sel"])
                )
        except Exception as exc:  # a probe must never take down a run
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[select-probe] skipped this step: {exc!r}", flush=True)
        return loss, metrics

    wrapped._select_probe = True  # type: ignore[attr-defined]
    core.compute_metrics = wrapped


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    install()
    import train as train_script

    args = train_script.parse_args()
    if int(os.environ.get("RANK", "0")) == 0:
        sr, lr = getattr(args, "select_rank", 0), getattr(args, "lr", None)
        print("=" * 78, flush=True)
        print(">>> SELECT ABLATION PROBE — accept_len with vs without the selection term", flush=True)
        print(f">>> select_rank={sr}" + ("   ⚠ 0 — SelectHead ABSENT, this probe reports nothing" if not sr else ""), flush=True)
        print(f">>> lr={lr}" + ("" if lr == 0 else "   (nonzero: this run trains, the probe just rides along)"), flush=True)
        print(f">>> four corners: on / sel_off / mk_off / both_off   every {_EVERY} step(s)", flush=True)
        print(">>> read sel_gain (on − sel_off) alongside sel_bias_rms and mk_gain", flush=True)
        print("=" * 78, flush=True)
    train_script.main(args)


if __name__ == "__main__":
    main()
