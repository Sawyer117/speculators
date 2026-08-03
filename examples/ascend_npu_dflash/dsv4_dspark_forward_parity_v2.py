#!/usr/bin/env python3
"""T6-v2 — train↔serve DSpark DRAFT-FORWARD parity, STATISTICAL + margin-bucketed.

Supersedes ``dsv4_dspark_serve_forward_parity.py`` (which compared ONE block). This
version answers the audit's ask: is the ASSEMBLED training draft-forward numerically
identical to the vLLM-Ascend serve proposer forward, PER POSITION, over MANY blocks,
and are any argmax disagreements bf16-noise (low serve margin) or a real bug (high
margin)?

Why this is the decisive test: our draft's serve accept is measured on the SERVE
forward; the released draft proves the SERVE forward is correct (4.658). The only
un-closed question is whether OUR TRAINING forward reproduces the serve's per-slot
logits — if it diverges at pos2+, we optimized the wrong objective there. (You can run
it with OUR ckpt: forward parity is weight-agnostic — a match for our weights is a
match for released too, same architecture.)

────────────────────────────────────────────────────────────────────────────────────
PIECE 1 (SERVE) — paste this N-block dump into the vLLM-Ascend proposer
  ``vllm_ascend/spec_decode/deepseek_v4_dspark_proposer.py``, inside
  ``_sample_sequential`` right BEFORE ``return self._draft_buffer[:num_reqs]``.
  Env-gated + capped; side-effect-free unless DSPARK_PARITY_DUMP=1. Run the serve with
  DSPARK_PARITY_DUMP=1 (+ DSPARK_PARITY_N=32, DSPARK_PARITY_DIR=/tmp/dspark_parity) and
  send ONE request (batch_size==1) that generates ≥N tokens; each spec-decode step
  writes serve_block_<i>.pt. ⚠ verify the attribute names on your build.

    import os as _os
    if _os.environ.get("DSPARK_PARITY_DUMP") == "1":
        _cnt = getattr(self, "_parity_count", 0)
        _N = int(_os.environ.get("DSPARK_PARITY_N", "32"))
        if _cnt < _N and int(getattr(self, "_dflash_num_context", 0)) > 0:
            self._parity_count = _cnt + 1
            import torch as _torch
            _dir = _os.environ.get("DSPARK_PARITY_DIR", "/tmp/dspark_parity"); _os.makedirs(_dir, exist_ok=True)
            _nctx = int(self._dflash_num_context)
            _seed = self._seed_buffer[0]; _cfg = self.model.config
            _final, _prev = [], _seed
            for _s in range(self.block_size):
                _lg = base_logits[0, _s] + self.model.markov_bias(self.model.markov_embed(_prev.view(1)))[0]
                _final.append(_lg.detach().float().cpu()); _prev = self._draft_buffer[0, _s]
            _torch.save({
                "aux": self._dflash_hidden_states[:_nctx].detach().float().cpu(),
                "ctx_positions": self._context_positions_buffer[:_nctx].detach().cpu(),
                "anchor_token": int(_seed.item()),
                "draft_positions": self.positions[:self.block_size].detach().cpu(),
                "serve_base_logits": base_logits[0].detach().float().cpu(),
                "serve_final_logits": _torch.stack(_final),
                "drafted": self._draft_buffer[0, :self.block_size].detach().cpu(),
                "block_size": int(self.block_size),
                "dspark_noise_token_id": int(self.parallel_drafting_token_id),
            }, _os.path.join(_dir, f"serve_block_{_cnt}.pt"))
            if _cnt + 1 == _N:
                print(f">>> [DSPARK_PARITY_DUMP] wrote {_N} blocks to {_dir}", flush=True)

PIECE 2 (TRAIN, this script) — CPU is fine (train backbone is plain torch):
  python dsv4_dspark_forward_parity_v2.py \
      --dumps /tmp/dspark_parity \
      --ckpt  /home/.../ckpt_faithful_ep_XX/0   (or the released draft in TRAIN format) \
      --verifier /home/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
      --atol 2e-2
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import os
import sys

# serve-margin buckets (top1-top2 logit gap): low margin = a near-tie where bf16 noise
# can legitimately flip argmax; high margin = a confident slot that must NOT flip.
_BUCKETS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, float("inf"))]


def _cosine(a, b):
    import torch
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def _topk_overlap(a, b, k):
    ta = set(a.topk(k).indices.tolist())
    tb = set(b.topk(k).indices.tolist())
    return len(ta & tb) / k


def compare_block(our, serve, block, topk):
    """Per-slot records for one block. our/serve: [block, vocab] float tensors."""
    recs = []
    for s in range(block):
        o, v = our[s], serve[s]
        top2 = v.topk(2).values
        recs.append({
            "slot": s,
            "maxabs": (o - v).abs().max().item(),
            "cosine": _cosine(o, v),
            "our_top1": int(o.argmax()),
            "serve_top1": int(v.argmax()),
            "match": int(o.argmax()) == int(v.argmax()),
            "serve_margin": (top2[0] - top2[1]).item(),
            "topk_overlap": _topk_overlap(o, v, topk),
        })
    return recs


def _mean(xs):
    xs = [x for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def per_position(records, block):
    """Aggregate across blocks, grouped by slot position (pos0..block-1)."""
    rows = []
    for s in range(block):
        rs = [r for r in records if r["slot"] == s]
        rows.append({
            "pos": s,
            "n": len(rs),
            "maxabs": _mean([r["maxabs"] for r in rs]),
            "cosine": _mean([r["cosine"] for r in rs]),
            "top1_agree": _mean([1.0 if r["match"] else 0.0 for r in rs]),
            "topk_overlap": _mean([r["topk_overlap"] for r in rs]),
        })
    return rows


def margin_table(records):
    """top-1 disagreement rate bucketed by serve margin — the audit's key discriminator."""
    rows = []
    for lo, hi in _BUCKETS:
        rs = [r for r in records if lo <= r["serve_margin"] < hi]
        dis = [r for r in rs if not r["match"]]
        rows.append({"lo": lo, "hi": hi, "n": len(rs), "disagree": len(dis),
                     "rate": (len(dis) / len(rs)) if rs else float("nan")})
    return rows


def run_train_block(model, d, dev, dtype):
    """Rebuild the SAME single-block input from a serve dump and run OUR forward.
    Returns (our_base [block,vocab], our_final [block,vocab] or None). Mirrors the v1
    alignment (see dsv4_dspark_serve_forward_parity.py for the WHY)."""
    import torch
    aux = d["aux"].to(dev, dtype)
    ctx_pos = d["ctx_positions"].to(dev, torch.long)
    anchor_token = int(d["anchor_token"])
    draft_pos = d["draft_positions"].to(dev, torch.long)
    drafted = d["drafted"].to(dev, torch.long).view(-1)
    block = int(d["block_size"])
    ctx_len = aux.shape[0]
    H3 = aux.shape[1]
    H = H3 // 3
    a = ctx_len
    total = ctx_len + 1 + block
    hidden_states = torch.zeros(1, total, H3, device=dev, dtype=dtype)
    hidden_states[0, :ctx_len] = aux
    input_ids = torch.zeros(1, total, dtype=torch.long, device=dev)
    input_ids[0, a] = anchor_token
    position_ids = torch.zeros(1, total, dtype=torch.long, device=dev)
    position_ids[0, :ctx_len] = ctx_pos
    position_ids[0, a] = draft_pos[0]
    position_ids[0, a + 1:] = draft_pos[0] + 1 + torch.arange(block, device=dev)
    loss_mask = torch.zeros(1, total, dtype=torch.long, device=dev)
    loss_mask[0, a] = 1
    document_ids = torch.zeros(1, total, dtype=torch.long, device=dev)
    verifier_last = torch.zeros(1, total, H, device=dev, dtype=dtype)
    with torch.no_grad():
        hidden, base_logits, _t, _alm, _abi = model._backbone_forward(
            hidden_states, input_ids, loss_mask, verifier_last, document_ids, position_ids, max_anchors=1,
        )
        our_base = base_logits.view(block, -1).float()
        our_final = None
        if model.markov_head is not None:
            prev = torch.empty(1, block, dtype=torch.long, device=dev)
            prev[0, 0] = anchor_token
            if block > 1:
                prev[0, 1:] = drafted[: block - 1]
            hb = hidden.view(1, block, -1)
            pe = model.markov_head.prev_embeddings(prev)
            mbias = model.markov_head.block_bias(prev_token_ids=prev, hidden_states=hb, prev_emb=pe)
            our_final = (base_logits.view(1, block, -1) + mbias).view(block, -1).float()
    return our_base, our_final


def _report(name, records, block, atol):
    print(f"\n{'=' * 78}\n== {name} ==   (aggregated over {len(records) // block} blocks)")
    print("  per-position:")
    print("   pos    n   mean|Δ|     cosine    top1-agree   top5-overlap")
    for r in per_position(records, block):
        print(f"    {r['pos']}   {r['n']:>4}  {r['maxabs']:.3e}   {r['cosine']:.5f}    "
              f"{r['top1_agree']:.2%}       {r['topk_overlap']:.2%}")
    print("  top-1 disagreement by SERVE margin (top1-top2 logit gap):")
    print("   margin bucket        n     disagree   rate      read")
    for r in margin_table(records):
        hi = "inf" if r["hi"] == float("inf") else f"{r['hi']:.2f}"
        read = "" if r["n"] == 0 else ("bf16-noise OK" if r["lo"] < 0.05 else "⚠ REAL if >0")
        rate = "  —  " if r["n"] == 0 else f"{r['rate']:.2%}"
        print(f"   [{r['lo']:.2f},{hi:>4})   {r['n']:>6}   {r['disagree']:>6}    {rate:>6}   {read}")
    worst = max((r["maxabs"] for r in records), default=0.0)
    hi_margin_flip = [r for r in records if not r["match"] and r["serve_margin"] >= 0.1]
    ok = worst <= atol and not hi_margin_flip
    print(f"  worst max|Δ|={worst:.3e} (atol={atol:.0e}) | high-margin(≥0.1) flips={len(hi_margin_flip)}"
          f"  → {'PASS' if ok else 'FAIL'}")
    return ok, hi_margin_flip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="dir of serve_block_*.pt (or a single .pt file)")
    ap.add_argument("--ckpt", required=True, help="draft ckpt dir (SAME weights the serve ran; TRAIN format)")
    ap.add_argument("--verifier", required=True, help="verifier dir (embed/lm_head/verifier_norm reloaded from here)")
    ap.add_argument("--atol", type=float, default=2e-2, help="max|Δ| PASS tolerance (bf16 ~2e-2)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--limit", type=int, default=0, help="cap #blocks (0=all)")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("need torch — run in the training env (or any torch env).")
    from speculators.models.dsv4_dspark.core import DSV4DSparkDraftModel

    files = ([args.dumps] if args.dumps.endswith(".pt")
             else sorted(glob.glob(os.path.join(args.dumps, "serve_block_*.pt")),
                         key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0])))
    if not files:
        sys.exit(f"no serve_block_*.pt in {args.dumps}")
    if args.limit:
        files = files[: args.limit]

    dtype = getattr(torch, args.dtype)
    dev = torch.device(args.device)
    d0 = torch.load(files[0], map_location="cpu")
    block = int(d0["block_size"])
    print(f">>> {len(files)} blocks | block_size={block} | ckpt={args.ckpt}")

    cfg = DSV4DSparkDraftModel.config_class.from_pretrained(args.ckpt)
    cfg.transformer_layer_config._attn_implementation = "sdpa"
    # low_cpu_mem_usage=False avoids transformers' meta-init: with the default meta
    # device-context, __init__'s precompute_freqs_cis buffer is built on meta and
    # `freqs_cis.to(device)` throws "Cannot copy out of meta tensor". False → normal CPU
    # init (the draft + verifier embed/lm_head fit in RAM), so the buffer has real data.
    model = DSV4DSparkDraftModel.from_pretrained(
        args.ckpt, config=cfg, verifier=args.verifier, low_cpu_mem_usage=False,
    )
    model = model.to(dev, dtype).eval()
    assert model.block_size == block, f"ckpt block_size {model.block_size} != dump {block}"

    base_recs, final_recs = [], []
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        our_base, our_final = run_train_block(model, d, dev, dtype)
        base_recs += compare_block(our_base, d["serve_base_logits"].to(dev).float(), block, args.topk)
        sf = d.get("serve_final_logits")
        if our_final is not None and sf is not None:
            final_recs += compare_block(our_final, sf.to(dev).float(), block, args.topk)
        if (i + 1) % 8 == 0:
            print(f"  ... {i + 1}/{len(files)} blocks")

    ok_base, hmf_base = _report("BASE logits (pre-markov) = the CLEAN forward parity", base_recs, block, args.atol)
    ok_final = True
    if final_recs:
        ok_final, _ = _report("FINAL logits (base+markov, serve prev replayed)", final_recs, block, args.atol)

    print("\n" + "=" * 78)
    if ok_base and ok_final:
        print("VERDICT: ✅ PASS — training draft-forward ≡ vLLM-Ascend serve forward (per-position).")
        print("  ⇒ the forward loop is CLOSED. The pos2-4 gap vs released is DATA/RECIPE, not a bug.")
    else:
        print("VERDICT: ❌ FAIL — forward diverges beyond bf16 noise.")
        print("  Read the per-position table + margin buckets:")
        print("   - divergence starts at pos2+, high-margin flips there  → attention/RoPE/mask (per-slot).")
        print("   - all positions off ~uniformly                         → MoE (routed_scaling/renorm) / RMSNorm / mHC.")
        print("   - disagreements ONLY in the [0,0.05) margin buckets     → just bf16 kernel noise, NOT a bug.")
        if hmf_base:
            r = hmf_base[0]
            print(f"   first high-margin flip: pos{r['slot']} serve_margin={r['serve_margin']:.3f} "
                  f"our={r['our_top1']} serve={r['serve_top1']}")
    sys.exit(0 if (ok_base and ok_final) else 1)


if __name__ == "__main__":
    main()
