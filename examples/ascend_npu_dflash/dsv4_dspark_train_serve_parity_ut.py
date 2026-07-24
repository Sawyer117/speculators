#!/usr/bin/env python3
"""DSV4-Flash DSpark TRAIN↔SERVE precision-parity UT (run on the box).

Each test compares a TRAIN reference (readable torch, clean-room backbone) against
the SERVE behaviour (vllm-ascend fused NPU op or the actual serve math), on IDENTICAL
inputs/weights, and prints max|Δ| + a PASS/FAIL. Born from the 4-subsystem audit
(2026-07-24). Ordered by actionability:

  T1  double-norm teacher      PURE TORCH (no NPU) — the ONE non-common-mode TRAIN bug.
                               Quantifies how far our distillation target is from the
                               real verifier distribution, and validates the 1-line fix.
  T2  MoE gating scale/renorm  serve op vs train Router — confirm/refute routed 2.25×
                               (double-applied 1.5) and dropped norm_topk_prob.
  T3  mHC hc_pre/hc_post        serve npu_hc_* vs train _hyper_connection_torch (Sinkhorn).
  T4  YaRN draft RoPE           serve factor=16 vs train YaRN-off — angular/logit divergence
                               at the draft's REAL operating positions (is it common-mode noise?).
  T5  sink block attention      serve npu_sparse_attn_sharedkv vs train _sink_block_attention_torch.

USAGE (box, env dspark-dsv4-compile for T1/train refs; NPU ops T2-5 need the serve env
+ torch_npu):
  # T1 only (pure torch, quantify the teacher bug + validate the fix):
  VERIFIER=/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
  HS_FILE=/mnt/nfs/canada_group_folder/dsv4_hs_dump/hs_<somerow>.safetensors \
  python examples/ascend_npu_dflash/dsv4_dspark_train_serve_parity_ut.py --tests t1

  # all (needs torch_npu + a vllm-ascend serve install on PYTHONPATH):
  python examples/ascend_npu_dflash/dsv4_dspark_train_serve_parity_ut.py --tests all
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys

import torch


def _hdr(name):
    print("\n" + "=" * 78 + f"\n{name}\n" + "=" * 78)


def _report(label, a, b, atol):
    """Print max/mean |Δ| + argmax-flip and PASS/FAIL vs atol on logit-like tensors."""
    a, b = a.float(), b.float()
    d = (a - b).abs()
    flip = (a.argmax(-1) != b.argmax(-1)).float().mean().item() if a.dim() >= 2 else float("nan")
    print(f"  {label}: max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}  "
          f"argmax-flip={flip:.4%}")
    ok = d.max().item() <= atol
    print(f"    -> {'PASS' if ok else 'FAIL'} (atol={atol:.1e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# T1 — DOUBLE-NORM TEACHER (the actionable, non-common-mode TRAIN bug). PURE TORCH.
#
# Training builds the distillation target as:
#     targets = verifier_lm_head( verifier_norm( verifier_last_hidden_states ) )   [core.py:515]
# but the HS dumper writes verifier_last_hidden_states = the FINAL POST-NORM hidden
# (self.norm(...) output — hs-dumper-planB.md:37/105/184). So verifier_norm is applied
# a SECOND time => norm(norm(h)). The CORRECT teacher = verifier_lm_head(h) directly.
# This test loads the verifier's final norm + lm_head and a real dumped post-norm hidden,
# and measures how far the BUGGY (double-norm) teacher is from the CORRECT (single) one.
# A large argmax-flip => the draft is being distilled toward the wrong distribution and
# even the train accept-metric (argmax vs argmax(target)) is measuring the wrong task.
# ─────────────────────────────────────────────────────────────────────────────
def test_t1_double_norm():
    _hdr("T1  DOUBLE-NORM TEACHER  (train bug; correct teacher = single norm)")
    import json
    import glob
    from safetensors.torch import load_file
    from safetensors import safe_open

    verifier = os.environ.get("VERIFIER", "/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16")
    hs_file = os.environ.get("HS_FILE", "")

    # --- load the verifier final RMSNorm weight (`model.norm.weight`) + lm_head (`lm_head.weight`)
    idx = os.path.join(verifier, "model.safetensors.index.json")
    if not os.path.exists(idx):
        print(f"  SKIP: no index at {idx}; set VERIFIER=<DeepSeek-V4-Flash-bf16 dir>")
        return
    wmap = json.load(open(idx))["weight_map"]
    def _get(key):
        shard = os.path.join(verifier, wmap[key])
        with safe_open(shard, framework="pt") as f:
            return f.get_tensor(key)
    norm_w = None
    head_w = None
    for k in wmap:
        if k in ("model.norm.weight", "norm.weight"):
            norm_w = _get(k)
        if k in ("lm_head.weight", "head.weight"):
            head_w = _get(k)
    if norm_w is None or head_w is None:
        print(f"  SKIP: could not find model.norm.weight / lm_head.weight in {idx}")
        return
    rms_eps = 1e-6
    try:
        cfg = json.load(open(os.path.join(verifier, "config.json")))
        rms_eps = cfg.get("rms_norm_eps", 1e-6)
    except Exception:
        pass
    H = norm_w.numel()
    print(f"  verifier: H={H}  vocab={head_w.shape[0]}  rms_eps={rms_eps}")

    def rmsnorm(x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + rms_eps)).to(x.dtype) * norm_w

    # --- get a real POST-NORM hidden (what the dumper writes as verifier_last)
    if hs_file and os.path.exists(hs_file):
        d = load_file(hs_file)
        # dumped layout: hidden_states[..., last] = the post-norm final hidden [T, H] (data.py:507)
        hs = d.get("hidden_states")
        if hs is not None and hs.dim() == 3:      # [T, n_layers, H] -> last layer
            post_norm_h = hs[:, -1, :]
        elif hs is not None and hs.dim() == 2 and hs.shape[-1] == H:
            post_norm_h = hs
        else:                                     # fall back to any [*,H] tensor
            post_norm_h = next(v for v in d.values() if v.dim() >= 2 and v.shape[-1] == H).reshape(-1, H)
        post_norm_h = post_norm_h.float()[:512]
        print(f"  hidden: real dumped post-norm, {tuple(post_norm_h.shape)} from {os.path.basename(hs_file)}")
    else:
        # synthesize a plausible post-norm hidden: rmsnorm of random residual (same statistics)
        torch.manual_seed(0)
        post_norm_h = rmsnorm(torch.randn(512, H) * 6.0).float()
        print("  hidden: SYNTHETIC (set HS_FILE=<hs_*.safetensors> for a real one)")

    head = head_w.float()
    logits_correct = post_norm_h @ head.T                      # verifier_lm_head(h)          — TRUE teacher
    logits_buggy = rmsnorm(post_norm_h).float() @ head.T       # verifier_lm_head(norm(h))    — what training does

    p_c = torch.softmax(logits_correct, -1)
    p_b = torch.softmax(logits_buggy, -1)
    tv = 0.5 * (p_c - p_b).abs().sum(-1).mean().item()
    kl = torch.nn.functional.kl_div(torch.log_softmax(logits_buggy, -1), p_c, reduction="batchmean").item()
    flip = (logits_correct.argmax(-1) != logits_buggy.argmax(-1)).float().mean().item()
    print(f"  TV(correct‖buggy)={tv:.4f}   KL={kl:.4f}   argmax-flip={flip:.4%}")
    print("  READ: flip% = fraction of positions where the DOUBLE-NORM teacher's top token")
    print("        differs from the real verifier's top token. Non-trivial flip => the TV(1.8)")
    print("        distillation + the train accept-metric are pointed at a distorted teacher.")
    print("  FIX : dsv4_dspark/core.py:515 -> verifier_lm_head(verifier_last_hidden_states)  (drop verifier_norm)")
    print(f"  VERDICT: {'material distortion (fix + retrain)' if flip > 0.01 or tv > 0.02 else 'small (norm weights ~1) — fix anyway for correctness'}")


# ─────────────────────────────────────────────────────────────────────────────
# T2 — MoE gating: routed_scaling_factor + norm_topk_prob renorm. serve op vs train.
# Confirms/refutes the audit's "routed 2.25× (1.5 applied twice)" + "renorm dropped".
# ─────────────────────────────────────────────────────────────────────────────
def test_t2_moe_gating():
    _hdr("T2  MoE gating  (routed_scaling 1.5 double? norm_topk_prob dropped?)")
    try:
        import torch_npu  # noqa: F401
    except Exception as e:
        print(f"  SKIP: no torch_npu ({e}); run in the serve env on an NPU box.")
        return
    torch.manual_seed(0)
    T, E, K, R = 64, 256, 6, 1.5
    logits = torch.randn(T, E).npu().float()          # router logits (fp32, as serve computes)
    bias = torch.zeros(E).npu().float()               # e_score_correction_bias (0 for the base draft)

    # --- TRAIN reference (backbone/moe.py:57-67): sqrt(softplus) -> +bias for topk ->
    #     weights gathered from PRE-bias scores -> renorm to sum 1 -> *1.5
    scores = torch.nn.functional.softplus(logits).sqrt()
    sel = (scores + bias).topk(K, dim=-1).indices
    w = scores.gather(-1, sel)
    w = w / w.sum(-1, keepdim=True)                    # norm_topk_prob=True
    w_train = w * R                                    # routed scale 1.5 (ONCE)
    print(f"  TRAIN topk-weight row0 sum={w_train.sum(-1)[0].item():.4f}  (== {R} if renorm+scale once)")

    # --- SERVE op (signature per audit; adapt names to your build if the schema differs)
    try:
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts  # or the hash op directly
        w_serve, ids_serve = select_experts(
            hidden_states=None, router_logits=logits, top_k=K,
            use_grouped_topk=False, renormalize=True,          # <- training intent
            e_score_correction_bias=bias, routed_scaling_factor=R,
            scoring_func="sqrtsoftplus",
        )[:2]
        print(f"  SERVE topk-weight row0 sum={w_serve.float().sum(-1)[0].item():.4f}")
        print("  READ: TRAIN row-sum == 1.5. If SERVE row-sum != 1.5:")
        print("        ~2.25 => routed_scaling applied TWICE (kernel + combine).")
        print("        != 1.5 and not renormalized (weights are raw sqrt(softplus)) => norm_topk_prob dropped.")
        print("  Also compare the COMBINED routed output incl. deepseek_v4.py:498 muls_add_triton(...,1.5).")
    except Exception as e:
        print(f"  SERVE op call failed / schema differs: {e}")
        print("  -> call the actual op: torch.ops._C_ascend.moe_gating_top_k_hash(router_logits, k=6,")
        print("     bias=..., routed_scaling_factor=1.5, renorm=0, norm_type=2, ...) and compare row-sum.")


# ─────────────────────────────────────────────────────────────────────────────
# T3 — mHC hc_pre/hc_post (Sinkhorn + residual mixing) — CANNOT-DETERMINE from source.
# ─────────────────────────────────────────────────────────────────────────────
def test_t3_mhc():
    _hdr("T3  mHC hc_pre/hc_post  (Sinkhorn iters/order + residual place)")
    try:
        import torch_npu  # noqa: F401
        from vllm_ascend.models.deepseek_v4 import npu_hc_pre_v2  # or torch.ops._C_ascend.npu_hc_pre_v2
    except Exception as e:
        print(f"  SKIP: serve op unavailable ({e}). On the box, call:")
        print("    torch.ops._C_ascend.npu_hc_pre_v2(x[N,hc,H], hc_fn, hc_scale[3], hc_base,")
        print("       hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6) -> (collapsed, post, comb)")
        print("    torch.ops._C_ascend.npu_hc_post(out[1,N,H], residual[1,N,hc,H], post, comb) -> streams")
        print("  and diff vs speculators backbone/hyper.py: _hyper_connection_torch + place().")
        return
    try:
        from speculators.models.dsv4_dspark.backbone.hyper import _hyper_connection_torch  # noqa: F401
        print("  TODO: build matched (x, hc_fn, hc_scale, hc_base) from a real draft layer's HC params,")
        print("        run npu_hc_pre_v2 vs _hyper_connection_torch, _report('hc_pre.collapsed', ...).")
        print("  Focus: Sinkhorn column-first vs row-first pass + '+eps' placement (the silent divergence).")
    except Exception as e:
        print(f"  train ref import failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# T4 — YaRN draft RoPE: serve applies factor=16, train+reference disable YaRN.
# Quantify the ACTUAL rotation/logit divergence at the draft's real positions.
# (Likely common-mode + near-identity at pos << orig_max=65536 — measure, don't assume.)
# ─────────────────────────────────────────────────────────────────────────────
def test_t4_yarn_rope():
    _hdr("T4  YaRN draft RoPE  (serve factor=16 vs train YaRN-off) — divergence vs position")
    theta, dim, factor, orig = 10000.0, 64, 16.0, 65536
    beta_fast, beta_slow = 32.0, 1.0

    def inv_freq_yarn_off():
        return 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    def inv_freq_yarn_on():
        # NTK-by-parts (YaRN): interpolate low-freq dims by 1/factor, ramp between beta_slow..beta_fast
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        def dim_for(nrot):
            return dim * torch.log(torch.tensor(orig / (nrot * 2 * torch.pi))) / (2 * torch.log(torch.tensor(theta)))
        lo, hi = dim_for(beta_fast), dim_for(beta_slow)
        ramp = ((torch.arange(dim // 2).float() - lo) / max((hi - lo).item(), 1e-3)).clamp(0, 1)
        mask = 1 - ramp
        return freqs / factor * (1 - mask) + freqs * mask

    off, on = inv_freq_yarn_off(), inv_freq_yarn_on()
    for pos in (64, 256, 1024, 2048):
        ang_off = pos * off
        ang_on = pos * on
        dtheta = (ang_off - ang_on).abs()
        print(f"  pos={pos:5d}: max|Δangle|={dtheta.max().item():.4f} rad  "
              f"mean={dtheta.mean().item():.4f}  (>~0.1 rad starts to matter for q·k)")
    print("  READ: if Δangle is tiny at the gen positions we actually eval (<=2048), the YaRN")
    print("        mismatch is near-identity there => common-mode + small, NOT our tail's cause.")
    print("  serve fix (if it matters): make the DSpark draft rope YaRN-off (original_seq_len=0),")
    print("        matching train (dsv4_dspark/core.py:151) + official reference.")


# ─────────────────────────────────────────────────────────────────────────────
# T5 — sink block attention: serve fused op vs train torch reference.
# ─────────────────────────────────────────────────────────────────────────────
def test_t5_sink_attention():
    _hdr("T5  sink block attention  (serve npu_sparse_attn_sharedkv vs train ref)")
    try:
        from speculators.models.dsv4_dspark.backbone.attention import _sink_block_attention_torch  # noqa: F401
    except Exception as e:
        print(f"  train ref import failed: {e}")
        return
    print("  Build matched q[N,Sq,Hh,D], k=v[N,Sk,D] (shared KV head), sink[Hh], scale=D**-0.5,")
    print("  a window+block NON-CAUSAL additive bias, then:")
    print("    ref  = _sink_block_attention_torch(q,k,v,sink,scale,attn_bias)   (backbone/attention.py)")
    print("    serve= torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, ori_kv=..., ori_sparse_indices=...,")
    print("             sinks=sink, softmax_scale=D**-0.5, ori_mask_mode=4, ori_win_left=127, ori_win_right=0)[0]")
    print("  ★ Drive RoPE BOTH ways (YaRN-off and factor=16) to isolate T4 from the op itself.")
    print("  Focus: the per-head sink denominator term e^{sink-m} and the window-edge (127 vs 128).")


TESTS = {"t1": test_t1_double_norm, "t2": test_t2_moe_gating, "t3": test_t3_mhc,
         "t4": test_t4_yarn_rope, "t5": test_t5_sink_attention}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default="all", help="comma list: t1,t2,t3,t4,t5 or 'all'")
    args = ap.parse_args()
    sel = list(TESTS) if args.tests == "all" else [t.strip() for t in args.tests.split(",")]
    for t in sel:
        if t not in TESTS:
            print(f"unknown test {t}; choices: {list(TESTS)}"); sys.exit(2)
        try:
            TESTS[t]()
        except Exception as e:
            print(f"  {t} ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
