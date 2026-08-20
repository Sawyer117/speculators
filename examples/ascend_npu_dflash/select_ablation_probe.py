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
    sel_on_accept_len     hard accept length as the model computes it
    sel_off_accept_len    the same model with the selection term subtracted out
    sel_gain              on - off   <- THE headline: the selection term's own contribution
    sel_{on,off}_pos{p}_acc   per-position accuracy for both
    sel_bias_rms          RMS of the selection bias, i.e. how far it has grown from its
                          zero init -- distinguishes "no effect" from "not trained yet"
"""

from __future__ import annotations

import os
import sys

import torch

_STASH: dict = {}


def _accept_len(ids, target, mask):
    """1 + longest all-correct prefix, matching metrics.py's hard_accept_len exactly."""
    m = (ids == target).to(torch.float32) * mask
    return m.cumprod(dim=-1).sum(dim=-1) + 1.0


def select_metrics(logits, targets, loss_mask, block_size, select_bias, chunk=64):
    """Compare accept_len with and without the selection term, chunked over blocks.

    ⚠ CHUNKED ON PURPOSE. At MAX_ANCHORS=512 a single [512, 5, 129280] tensor is 1.3 GB in
    fp32, and a naive `logits.float() - bias.float()` would allocate three of them next to a
    run already using activation checkpointing to fit. Everything below stays in the logits'
    own dtype and materialises at most `chunk` blocks at a time; argmax needs no cast, and
    the accept-length arithmetic is done on the small [chunk, block] results.
    """
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        N = logits.shape[1] // block_size
        V = logits.shape[-1]
        lg = logits.view(N, block_size, V)
        sb = select_bias.view(N, block_size, V)
        target = targets.view(N, block_size, V).argmax(dim=-1)
        mask = loss_mask.to(torch.float32).view(N, block_size)
        valid = (mask.sum(dim=-1) > 0).to(torch.float32)
        nvalid = valid.sum().clamp_min(1.0)

        al = {"on": [], "off": []}
        hits = {"on": [], "off": []}
        sq = torch.zeros((), dtype=torch.float32, device=lg.device)
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            b = sb[i:j]
            sq += b.float().pow(2).sum()
            for name, ids in (("on", lg[i:j].argmax(dim=-1)),
                              ("off", (lg[i:j] - b).argmax(dim=-1))):
                al[name].append(_accept_len(ids, target[i:j], mask[i:j]))
                hits[name].append((ids == target[i:j]).to(torch.float32) * mask[i:j])
        al = {k: torch.cat(v) for k, v in al.items()}

        for name in ("on", "off"):
            out[f"sel_{name}_accept_len_sum"] = (al[name] * valid).sum()
            out[f"sel_{name}_accept_len_total"] = nvalid.clone()
            hit = torch.cat(hits[name])
            for p in range(block_size):
                out[f"sel_{name}_pos{p}_acc_sum"] = hit[:, p].sum()
                out[f"sel_{name}_pos{p}_acc_total"] = mask[:, p].sum().clamp_min(1.0)

        d = (al["on"] - al["off"]) * valid
        out["sel_gain_sum"] = d.sum()
        out["sel_gain_total"] = nvalid.clone()
        out["sel_win_sum"] = ((d > 0).to(torch.float32) * valid).sum()
        out["sel_win_total"] = nvalid.clone()
        out["sel_loss_sum"] = ((d < 0).to(torch.float32) * valid).sum()
        out["sel_loss_total"] = nvalid.clone()
        # How far the term has grown from its zero init -- separates "trained but useless"
        # from "still ~zero", which look identical in the gain alone.
        out["sel_bias_rms_sum"] = (sq / sb.numel()).sqrt()
        out["sel_bias_rms_total"] = torch.ones((), device=sb.device)
    return out


def install() -> None:
    """Stash the selection bias, then append the on/off comparison in compute_metrics."""
    import speculators.models.dspark.core as core
    from speculators.models.dspark.model_definitions import SelectHead

    if getattr(core.compute_metrics, "_select_probe", False):
        return

    orig = SelectHead.block_bias

    def spy(self, *, prev_token_ids, hidden_states):
        b = orig(self, prev_token_ids=prev_token_ids, hidden_states=hidden_states)
        _STASH["bias"] = b.detach()
        return b

    SelectHead.block_bias = spy

    original = core.compute_metrics

    def wrapped(logits, targets, confidence_logits, loss_mask, block_size, *a, **kw):
        loss, metrics = original(logits, targets, confidence_logits, loss_mask, block_size, *a, **kw)
        try:
            if "bias" in _STASH:
                metrics.update(
                    select_metrics(logits, targets, loss_mask, block_size, _STASH["bias"])
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
        print(">>> read sel_gain (on − off) and sel_bias_rms together", flush=True)
        print("=" * 78, flush=True)
    train_script.main(args)


if __name__ == "__main__":
    main()
