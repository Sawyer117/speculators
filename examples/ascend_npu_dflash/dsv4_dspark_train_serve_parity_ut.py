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

    # --- get a real dumped verifier_last hidden (+ token_ids for the DECISIVE check)
    tok_ids = None
    if hs_file and os.path.exists(hs_file):
        d = load_file(hs_file)
        # dumped layout: hidden_states[..., last] = verifier_last [T, H] (data.py:507);
        # token_ids = the rollout tokens (data.py:506).
        hs = d.get("hidden_states")
        if hs is not None and hs.dim() == 3:      # [T, n_layers, H] -> last layer
            vlast = hs[:, -1, :]
        elif hs is not None and hs.dim() == 2 and hs.shape[-1] == H:
            vlast = hs
        else:                                     # fall back to any [*,H] tensor
            vlast = next(v for v in d.values() if v.dim() >= 2 and v.shape[-1] == H).reshape(-1, H)
        vlast = vlast.float()
        tok_ids = d.get("token_ids")
        if tok_ids is not None:
            tok_ids = tok_ids.reshape(-1).long()
        print(f"  hidden: real dumped verifier_last, {tuple(vlast.shape)} from {os.path.basename(hs_file)}")
        print(f"  dumped-hidden RMS={vlast.pow(2).mean().sqrt().item():.3f}  "
              f"(post-norm ~ O(RMS of norm.weight)={norm_w.float().pow(2).mean().sqrt().item():.3f}; "
              f"a pre-norm residual would be MUCH larger)")
    else:
        torch.manual_seed(0)
        vlast = rmsnorm(torch.randn(512, H) * 6.0).float()   # synthetic post-norm-ish
        print("  hidden: SYNTHETIC (set HS_FILE=<hs_*.safetensors> for the decisive check)")

    head = head_w.float()
    logits_A = vlast @ head.T                        # lm_head(vlast)        — correct IF vlast is post-norm
    logits_B = rmsnorm(vlast).float() @ head.T       # lm_head(norm(vlast))  — what training DOES (F1 = drop this)

    # magnitude of the distortion between the two candidate teachers
    p_a, p_b = torch.softmax(logits_A, -1), torch.softmax(logits_B, -1)
    tv = 0.5 * (p_a - p_b).abs().sum(-1).mean().item()
    flip = (logits_A.argmax(-1) != logits_B.argmax(-1)).float().mean().item()
    print(f"  A=lm_head(vlast)  vs  B=lm_head(norm(vlast)):  TV={tv:.4f}   argmax-flip={flip:.4%}")

    # ── DECISIVE: which teacher's argmax matches the rollout's NEXT token? (rollout is greedy temp0,
    #    so the REAL verifier's argmax at pos i == token_ids[i+1]. Whichever of A/B agrees is the real
    #    verifier distribution — settling pre/post-norm and whether F1 is a real bug, no assumptions.)
    if tok_ids is not None and tok_ids.numel() >= vlast.shape[0] and vlast.shape[0] >= 2:
        n = min(vlast.shape[0], tok_ids.numel()) - 1
        nxt = tok_ids[1:n + 1]
        agr_A = (logits_A[:n].argmax(-1) == nxt).float().mean().item()
        agr_B = (logits_B[:n].argmax(-1) == nxt).float().mean().item()
        print(f"  DECISIVE — next-token agreement (rollout greedy):  A={agr_A:.2%}   B={agr_B:.2%}")
        if agr_A > agr_B + 0.05:
            verdict = ("✅ vlast IS post-norm → training's B (lm_head(norm(vlast))) DOUBLE-NORMS → "
                       "F1 is a REAL bug. Apply the fix + retrain.")
        elif agr_B > agr_A + 0.05:
            verdict = ("❌ vlast is PRE-norm → training's B is CORRECT → F1 is NOT a bug. DO NOT kill/retrain "
                       "for this reason (Claude's double-norm diagnosis would be wrong here).")
        else:
            verdict = ("⚠ inconclusive (A≈B, both low?) — check the rollout was greedy + the file's "
                       "token_ids align with hidden_states; don't act until this is green.")
        print(f"  VERDICT: {verdict}")
    else:
        print("  VERDICT: need a real HS_FILE with token_ids to decide. The magnitude above only shows")
        print("           A and B DIFFER, not which is correct. Do NOT kill/retrain on magnitude alone.")
    print("  FIX (if A wins): dsv4_dspark/core.py -> verifier_lm_head(verifier_last_hidden_states) (already welded).")


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

    # --- SERVE op. This serving build (386530d12) exposes it as torch.ops.npu.npu_moe_gating_top_k
    #     (NOT the hsdump build's _C_ascend.moe_gating_top_k_hash — different build, different op).
    #     Call with the training INTENT (renorm on, norm_type=2 sqrtsoftplus, scale 1.5): row-sum should be 1.5.
    gop = getattr(getattr(torch.ops, "npu", None), "npu_moe_gating_top_k", None)
    if gop is None:
        print("  SKIP: torch.ops.npu.npu_moe_gating_top_k not found on this build."); return
    for desc, kwargs in [
        ("kw", dict(k=K, bias=bias, k_group=1, group_count=1, group_select_mode=0,
                    renorm=1, norm_type=2, out_flag=False, routed_scaling_factor=R, eps=1e-20)),
        ("pos", None),
    ]:
        try:
            out = gop(logits, **kwargs) if kwargs is not None else \
                gop(logits, K, bias, 1, 1, 0, 1, 2, False, R, 1e-20)   # ⚠ positional arg order best-effort
            w_serve = out[0].float()
            rs = w_serve.sum(-1)[0].item()
            print(f"  SERVE npu_moe_gating_top_k (renorm=1,norm_type=2,scale={R}) row0 sum={rs:.4f}  [{desc}]")
            print(f"  READ vs TRAIN 1.5:  ~1.5 => renorm+scale once = MATCH (op is faithful when asked);")
            print(f"        ~2.25 => scaled twice = F2;   != 1.5 & unnormalized => renorm not applied = F3.")
            break
        except Exception as e:
            print(f"  [{desc}] call failed: {e}")
    else:
        print("  ⚠ both call forms failed — inspect the signature: print(torch.ops.npu.npu_moe_gating_top_k._schemas)")


# ─────────────────────────────────────────────────────────────────────────────
# T3 — mHC hc_pre/hc_post (Sinkhorn + residual mixing) — CANNOT-DETERMINE from source.
# ─────────────────────────────────────────────────────────────────────────────
def test_t3_mhc():
    _hdr("T3  mHC hc_pre  (Sinkhorn iters/order + '+eps' placement — the CANNOT-DETERMINE op)")
    try:
        import torch_npu  # noqa: F401
    except Exception as e:
        print(f"  SKIP: no torch_npu ({e})"); return
    # serving build (386530d12) exposes it as torch.ops.npu.npu_mhc_pre; hsdump build as _C_ascend.npu_hc_pre_v2
    op = getattr(getattr(torch.ops, "npu", None), "npu_mhc_pre", None) or \
        getattr(getattr(torch.ops, "_C_ascend", None), "npu_hc_pre_v2", None)
    if op is None:
        print("  SKIP: neither torch.ops.npu.npu_mhc_pre nor _C_ascend.npu_hc_pre_v2 registered."); return
    try:
        from speculators.models.dsv4_dspark.backbone.hyper import HyperConnection, _hyper_connection_torch
    except Exception as e:
        print(f"  SKIP: `import speculators` failed ({e}) — run `pip install -e .` in this env (needed for the ref).")
        return
    import types
    torch.manual_seed(0)
    H, hc, N = 256, 4, 8
    cfg = types.SimpleNamespace(hc_mult=hc, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6, hidden_size=H)
    mod = HyperConnection(cfg)
    with torch.no_grad():
        mod.fn.normal_(0, 0.02); mod.base.zero_(); mod.scale.fill_(1.0)
    streams = torch.randn(1, N, hc, H)                     # [B, S, hc, D]
    post_t, comb_t, collapsed_t = _hyper_connection_torch(mod, streams)   # TRAIN ref -> (post, comb, collapsed)
    try:
        x = streams[0].npu().float()                       # serve x: [N, hc, H]
        collapsed_s, _post_s, _comb_s = op(                # serve op -> (collapsed, post, comb)
            x, mod.fn.detach().npu().float(), mod.scale.detach().npu().float(),
            mod.base.detach().npu().float(), hc, 20, 1e-6, 1e-6)
        d = (collapsed_t[0].float() - collapsed_s.cpu().float()).abs()
        ok = d.max().item() <= 2e-2
        print(f"  hc_pre.collapsed:  max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}  "
              f"-> {'PASS' if ok else 'FAIL'} (atol=2e-2)")
        print("  READ: PASS => the compiled Sinkhorn/mixing MATCHES the torch ref (F5 clean).")
        print("        FAIL => the .so's Sinkhorn (iteration order / '+eps' placement) diverges — the silent mHC gap.")
    except Exception as e:
        print(f"  ⚠ serve op call failed ({e}) — arg order/shape/dtype best-effort. On your build inspect it:")
        print("    print(torch.ops._C_ascend.npu_hc_pre_v2)   # match (x, fn, scale, base, hc_mult, iters, norm_eps, hc_eps)")


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
    _hdr("T5  sink block attention  (train ref sanity — serve op needs the engine, see note)")
    try:
        from speculators.models.dsv4_dspark.backbone.attention import _sink_block_attention_torch
    except Exception as e:
        print(f"  SKIP: `import speculators` failed ({e}) — `pip install -e .` in this env."); return
    torch.manual_seed(0)
    N, Sq, Sk, Hh, D = 1, 5, 133, 8, 64                    # block γ=5, ctx window 128 + block 5, shared KV head
    q = torch.randn(N, Sq, Hh, D); k = torch.randn(N, Sk, D); v = torch.randn(N, Sk, D)
    sink = torch.randn(Hh)
    try:
        out = _sink_block_attention_torch(q, k, v, sink, D ** -0.5)
        print(f"  TRAIN ref ran ✓  out={tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")
    except Exception as e:
        print(f"  TRAIN ref failed: {e}")
    print("  ⚠ SERVE side NOT run standalone: `npu_sparse_attn_sharedkv` consumes the engine's PAGED KV cache")
    print("     + block tables + sas_metadata — it does not execute cleanly outside a running serve. Its")
    print("     train<->serve parity is best measured INSIDE the serve = the T6 harness")
    print("     (dsv4_dspark_serve_forward_parity.py): a late-slot BASE-logit divergence there IS the")
    print("     sink/attention (or RoPE) mismatch. Also: dspark_attn_ref_bench.py already validated the")
    print("     non-causal einsum vs the serve GOLD, so the attention math is corroborated.")


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
