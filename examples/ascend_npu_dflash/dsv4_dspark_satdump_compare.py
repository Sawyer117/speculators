#!/usr/bin/env python3
"""Bisect train↔serve DSpark block forward layer-by-layer to the FIRST diverging stage.

Pairs two SATURATED per-stage dumps of the SAME block:
  * serve_sat.pt  — from the vLLM-Ascend model.forward (DSPARK_SATDUMP=1 on the serve)
  * train_sat.pt  — from our _backbone_forward (DSPARK_SATDUMP=1 while running the parity
                    script on the matching serve_block_0.pt)

Both record: embed → streams_in (post hc-repeat) → each draft-layer output → hc_head_out.
We already proved (statically) run_train_block faithfully reproduces training and that the
train↔serve BOUNDARY ops match (fc/hidden_norm, norm/lm_head, QuaRot no-op, hc_head). So a
uniform, precision-invariant ~13 logit gap must be born INSIDE some layer. This walks the
stages in order and prints the FIRST one whose max|Δ| jumps above bf16 noise — that stage
(a layer's attention / mHC / MoE) is where train and serve compute different math.

USAGE:
  python dsv4_dspark_satdump_compare.py --serve /tmp/dspark_sat/serve_sat.pt \
                                        --train ~/dspark_sat/train_sat.pt
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys


def _load(path):
    import torch
    return torch.load(path, map_location="cpu")


def _align(t):
    """serve tensors are [S, ...]; train tensors are [1, S, ...] — drop the batch dim so the
    two line up. Returns a contiguous float tensor."""
    import torch
    if not isinstance(t, torch.Tensor):
        return t
    if t.dim() >= 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t.float().contiguous()


def _cmp(name, a, b, atol, out):
    import torch
    a, b = _align(a), _align(b)
    if tuple(a.shape) != tuple(b.shape):
        out.append((name, None, None, f"SHAPE MISMATCH serve={tuple(a.shape)} train={tuple(b.shape)}"))
        return
    d = (a - b).abs()
    maxd = d.max().item()
    meand = d.mean().item()
    cos = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    flag = "DIVERGES" if maxd > atol else "ok"
    out.append((name, maxd, cos, f"mean|Δ|={meand:.3e} cos={cos:.5f}  {flag}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", required=True, help="serve_sat.pt from the vLLM-Ascend forward")
    ap.add_argument("--train", required=True, help="train_sat.pt from our _backbone_forward")
    ap.add_argument("--atol", type=float, default=5e-2, help="max|Δ| above this = real divergence (fp32 dumps)")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("need torch")

    s, t = _load(args.serve), _load(args.train)
    ns, nt = len(s.get("layers", [])), len(t.get("layers", []))
    ss, ts = s.get("substages") or [], t.get("substages") or []
    print(f">>> serve: embed + {ns} layers ({len(ss)} sub-staged) + hc_head | "
          f"train: embed + {nt} layers ({len(ts)} sub-staged) + hc_head")
    if ns != nt:
        print(f"  ⚠ layer count differs (serve {ns} vs train {nt}) — comparing min({ns},{nt})")

    # fine sub-stage walk within each layer; falls back to coarse layer-boundary if no substages
    _SUBORDER = ["hc_pre_attn", "attn_norm", "attn_out", "post_attn",
                 "hc_pre_ffn", "ffn_norm", "moe_out", "layer_out"]
    rows = []
    _cmp("embed", s.get("embed"), t.get("embed"), args.atol, rows)
    _cmp("streams_in", s.get("streams_in"), t.get("streams_in"), args.atol, rows)
    for i in range(min(ns, nt)):
        if i < len(ss) and i < len(ts):
            for key in _SUBORDER:
                _cmp(f"L{i}.{key}", ss[i].get(key), ts[i].get(key), args.atol, rows)
        else:
            _cmp(f"layer[{i}]", s["layers"][i], t["layers"][i], args.atol, rows)
    _cmp("hc_head_out", s.get("hc_head_out"), t.get("hc_head_out"), args.atol, rows)

    print("\n  stage           max|Δ|      detail")
    first = None
    for name, maxd, _cos, detail in rows:
        md = "   n/a   " if maxd is None else f"{maxd:.3e}"
        print(f"  {name:<14}  {md}   {detail}")
        if first is None and (maxd is None or maxd > args.atol):
            first = name

    print("\n" + "=" * 74)
    if first is None:
        print("VERDICT: ✅ every stage matches — train ≡ serve block forward. The gap is NOT here.")
        print("  ⇒ re-examine the CONTEXT-KV path (precompute_and_store_context_kv) or the dump pairing.")
        sys.exit(0)
    print(f"VERDICT: 🎯 first divergence at → {first}")
    _sub = first.split(".", 1)[1] if "." in first else first
    _READ = {
        "embed": "before any layer: token ids / embed weights / QuaRot / hc-repeat differ.",
        "streams_in": "before any layer: the hc-stream repeat or stream ORDER differs.",
        "hc_pre_attn": "the ATTN mHC hyper-connection (HyperConnection pre/collapse) — hc_attn weights/scale/base or input_norm.",
        "attn_norm": "the pre-attn RMSNorm (input_layernorm vs attn_norm) — eps or weight.",
        "attn_out": "★ the SINK ATTENTION — sink term / scale / rope / MLA proj / KV assembly differ. Top suspect.",
        "post_attn": "the ATTN mHC placement (place: post/comb, Sinkhorn) — hc_post vs place.",
        "hc_pre_ffn": "the FFN mHC hyper-connection (collapse) — hc_ffn weights/scale/base.",
        "ffn_norm": "the pre-MoE RMSNorm (post_attention_layernorm vs ffn_norm).",
        "moe_out": "★ the MoE — routing (noaux_tc bias / topk / renorm / routed_scaling) or expert GEMM differ.",
        "layer_out": "the FFN mHC placement (place after MoE).",
        "hc_head_out": "layers match; the final mHC collapse differs (hc_head weights/scale).",
    }
    print(f"  ⇒ {_READ.get(_sub, 'inspect this op on both sides.')}")
    print("  (attn_norm/ffn_norm should be identical RMSNorm; a jump THERE with clean inputs = eps/weight mismatch.)")
    sys.exit(0)


if __name__ == "__main__":
    main()
