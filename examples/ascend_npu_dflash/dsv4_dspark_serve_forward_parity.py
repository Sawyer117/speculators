#!/usr/bin/env python3
"""T6 — END-TO-END train↔vLLM-Ascend DSpark DRAFT-FORWARD parity.

Proves (or refutes) that OUR training draft-forward produces the SAME per-slot draft
logits as the vLLM-Ascend SERVE draft-forward, on IDENTICAL inputs + weights. This is
the definitive "are we aligned with vllm-ascend" test — it catches ANY residual
forward mismatch the per-op audit might miss (F2 MoE routed_scaling 2.25×, F3
norm_topk_prob renorm, F4 YaRN factor=16, F5 mHC Sinkhorn, or a norm-stage slip),
because it compares the WHOLE forward end-to-end.

TWO pieces (this file = piece 2):
  Piece 1 (SERVE): a tiny env-gated dump patched into the vllm-ascend proposer
      `deepseek_v4_dspark_proposer.py::_sample_sequential` (see the snippet in the
      T6 report). With DSPARK_PARITY_DUMP=1 it writes ONE file (first spec-decode
      step of the first request) with: aux, anchor_token, positions, per-slot serve
      base/final logits, drafted tokens, config knobs.
  Piece 2 (TRAIN, this script): rebuilds the SAME single-block input, runs OUR
      `_backbone_forward` + markov, and diffs per slot.

WHY the two forwards line up (mapping, verified from source):
  * SERVE drafts ONE block per request: slot0 token = `next_token_ids` (the anchor/
    bonus token), slots 1..γ-1 = `parallel_drafting_token_id` (=dspark_noise_token_id
    128799); block positions = `target_positions[valid_end-1] + 1 + arange`
    (proposer :404/:408-409). The target aux (target_hidden_states, the [40,41,42]
    concat) feeds the draft attention KV via main_proj. base_logits =
    `compute_logits(hidden) = lm_head(norm(hidden))` (:614); per slot the markov bias
    `markov_bias(markov_embed(prev))` is added then greedy-argmax (:630/:648), where
    prev = seed(anchor) for slot0 then the just-drafted token (:617/:650).
  * TRAIN `_backbone_forward` (dsv4_dspark/core.py:468) drafts a block per sampled
    ANCHOR: slot0 token = `input_ids[anchor]`, slots = mask_token_id; block positions
    = `get_base_indices(position_ids[anchor], γ)` = [pos_anchor, +1, …]. aux
    (`hidden_states`) → `fc`(main_proj) → `hidden_norm`(main_norm) → attn KV. logits =
    `lm_head(norm(hc_head(streams)))`. markov added in dspark/core.py:163-172.
  * ALIGNMENT: set our anchor at sequence index `a` with position VALUE = serve's
    draft_positions[0] (= target_positions[valid_end-1]+1), and input_ids[a] =
    anchor_token. Then our block positions == serve draft_positions, our attended
    context == serve context ([<a], sliding-window 128), and slot0 token matches.
    (Train's "anchor a" == serve's "p+1"; the label off-by-one nets out to identical
    positions — see the audit.) The context tokens (input_ids[<a]) are NOT embedded by
    the draft (only the aux feeds KV), so they can be zeros.
  * BASE logits (pre-markov) depend ONLY on aux+anchor+positions+noise — NO drafted
    tokens — so they are EXPOSURE-FREE and must match exactly. THIS is the clean
    forward test (all of F2..F5 live here). FINAL logits additionally depend on the
    markov `prev`; we replay the SERVE's exact prev (seed + dumped drafted tokens) so
    the final comparison isolates the (already-audited) markov add, not exposure.

USAGE (box; CPU is fine — the train backbone is plain torch, no NPU needed):
  python examples/ascend_npu_dflash/dsv4_dspark_serve_forward_parity.py \
      --dump /tmp/dspark_parity/serve_step0.pt \
      --ckpt /home/n84449292/dsv4_run/ckpt_faithful_ep_XX:/0 \
      --verifier /mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
      --atol 2e-2
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys


def _load_dump(path):
    """Load the serve dump (torch.save dict). Returns a dict of CPU tensors/scalars."""
    import torch
    obj = torch.load(path, map_location="cpu")
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="serve dump file (from the DSPARK_PARITY_DUMP=1 proposer patch)")
    ap.add_argument("--ckpt", required=True, help="draft ckpt dir (the SAME weights the serve ran): <N>/ with config.json+model.safetensors")
    ap.add_argument("--verifier", required=True, help="verifier dir — embed/lm_head/verifier_norm are reloaded from it (excluded from the draft ckpt)")
    ap.add_argument("--atol", type=float, default=2e-2, help="max|Δ| tolerance on logits for PASS (bf16 ~2e-2)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="compute dtype for the replay")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("need torch — run this in the training env (dspark-dsv4-compile) or any env with torch.")

    from speculators.models.dsv4_dspark.core import DSV4DSparkDraftModel

    dtype = getattr(torch, args.dtype)
    dev = torch.device(args.device)
    d = _load_dump(args.dump)

    # ---- unpack the serve dump ----
    aux = d["aux"].to(dev, dtype)                       # [ctx_len, 3H]   raw [40,41,42] concat (pre main_proj)
    ctx_pos = d["ctx_positions"].to(dev, torch.long)    # [ctx_len]       position VALUES of the context tokens
    anchor_token = int(d["anchor_token"])               # scalar          serve next_token (slot0)
    draft_pos = d["draft_positions"].to(dev, torch.long)  # [block]       serve block position VALUES
    serve_base = d["serve_base_logits"].to(dev, torch.float32)   # [block, vocab]
    serve_final = d.get("serve_final_logits")
    if serve_final is not None:
        serve_final = serve_final.to(dev, torch.float32)
    drafted = d["drafted"].to(dev, torch.long).view(-1)  # [block]  serve-drafted tokens (for the markov prev)
    block = int(d["block_size"])
    noise_tok = int(d.get("dspark_noise_token_id", 128799))
    ctx_len = aux.shape[0]
    H3 = aux.shape[1]
    print(f">>> dump: ctx_len={ctx_len}  3H={H3}  block={block}  anchor_token={anchor_token}  "
          f"draft_pos0={int(draft_pos[0])}  noise_tok={noise_tok}")

    # ---- load OUR model (draft weights from ckpt + embed/head/verifier_norm from verifier) ----
    cfg = DSV4DSparkDraftModel.config_class.from_pretrained(args.ckpt)
    # _attn_implementation isn't serialized; the training run used --draft-attn-impl sdpa.
    cfg.transformer_layer_config._attn_implementation = "sdpa"
    model = DSV4DSparkDraftModel.from_pretrained(args.ckpt, config=cfg, verifier=args.verifier)  # reloads verifier weights
    model = model.to(dev, dtype).eval()
    assert model.block_size == block, f"ckpt block_size {model.block_size} != dump {block}"
    if int(getattr(model, "mask_token_id", noise_tok)) != noise_tok:
        print(f"!! WARN mask_token_id {model.mask_token_id} != dump noise {noise_tok}")

    # ---- rebuild the SAME single-block input (see WHY, above) ----
    # anchor at sequence index a = ctx_len; pad `block` extra positions so `a` isn't in the
    # last block_size positions (select_anchors excludes those). position_ids[a] = draft_pos[0]
    # so our block positions == serve draft_positions. input_ids[a] = anchor_token; the rest
    # (context tokens) are unused by the draft (only aux feeds KV), so 0.
    a = ctx_len
    total = ctx_len + 1 + block                          # ctx + anchor + block-padding
    H = H3 // 3
    hidden_states = torch.zeros(1, total, H3, device=dev, dtype=dtype)
    hidden_states[0, :ctx_len] = aux                     # positions <a get the real aux (KV context)
    input_ids = torch.zeros(1, total, dtype=torch.long, device=dev)
    input_ids[0, a] = anchor_token                       # slot0 token
    position_ids = torch.zeros(1, total, dtype=torch.long, device=dev)
    position_ids[0, :ctx_len] = ctx_pos
    position_ids[0, a] = draft_pos[0]                    # ⇒ block_positions == serve draft_positions
    position_ids[0, a + 1:] = draft_pos[0] + 1 + torch.arange(block, device=dev)  # padding (irrelevant)
    loss_mask = torch.zeros(1, total, dtype=torch.long, device=dev)
    loss_mask[0, a] = 1                                   # a = the SOLE anchor
    document_ids = torch.zeros(1, total, dtype=torch.long, device=dev)  # single doc → block attends its context
    verifier_last = torch.zeros(1, total, H, device=dev, dtype=dtype)   # ⚠ only feeds `targets` (teacher), NOT draft logits

    with torch.no_grad():
        # _backbone_forward returns (hidden, logits, targets, aligned_loss_mask, anchored_block_indices);
        # `logits` = PRE-markov per-slot draft logits [1, num_anchors*block, vocab].
        hidden, base_logits, _targets, _alm, anchored_block_indices = model._backbone_forward(
            hidden_states, input_ids, loss_mask, verifier_last, document_ids, position_ids, max_anchors=1,
        )
        our_base = base_logits.view(block, -1).float()   # [block, vocab]

        # markov: replay the SERVE's exact prev per slot (seed=anchor for slot0, then drafted[k-1]).
        our_final = None
        if model.markov_head is not None:
            prev = torch.empty(1, block, dtype=torch.long, device=dev)
            prev[0, 0] = anchor_token
            if block > 1:
                prev[0, 1:] = drafted[: block - 1]
            hidden_blocks = hidden.view(1, block, -1)
            prev_emb = model.markov_head.prev_embeddings(prev)
            mbias = model.markov_head.block_bias(prev_token_ids=prev, hidden_states=hidden_blocks, prev_emb=prev_emb)
            our_final = (base_logits.view(1, block, -1) + mbias).view(block, -1).float()

    # ---- compare ----
    def _cmp(name, ours, serve):
        dif = (ours - serve).abs()
        flip = (ours.argmax(-1) != serve.argmax(-1))
        print(f"\n== {name} ==   max|Δ|={dif.max().item():.3e}  mean|Δ|={dif.mean().item():.3e}  "
              f"argmax-agree={ (~flip).float().mean().item():.2%}")
        print("  slot  max|Δ|     our_argmax  serve_argmax  match")
        for k in range(block):
            print(f"   {k}    {dif[k].max().item():.3e}   {int(ours[k].argmax()):>10}  "
                  f"{int(serve[k].argmax()):>11}   {'✓' if not bool(flip[k]) else '✗ <-- diverges here'}")
        ok = dif.max().item() <= args.atol and not bool(flip.any())
        return ok

    print("\n" + "=" * 74)
    ok_base = _cmp("BASE logits (pre-markov)  = the clean FORWARD parity test", our_base, serve_base)
    ok_final = True
    if our_final is not None and serve_final is not None:
        ok_final = _cmp("FINAL logits (base+markov, serve prev replayed)", our_final, serve_final)
    elif our_final is not None:
        print("\n(serve_final_logits not in dump — skipped FINAL; BASE is the decisive forward test anyway)")

    print("\n" + "=" * 74)
    print(f"VERDICT: {'✅ PASS — train↔vllm-ascend draft-forward ALIGNED' if (ok_base and ok_final) else '❌ FAIL — forward diverges'}"
          f"   (atol={args.atol:.0e})")
    if not ok_base:
        print("  ▶ BASE diverges = a real forward mismatch. Read the per-slot table:")
        print("    - LATE slots (pos2+) diverge, early match  → attention/RoPE (F4 YaRN, sink, window edge).")
        print("    - ALL slots diverge ~uniformly            → MoE (F2 2.25× / F3 renorm) or an RMSNorm/mHC (F5) slip.")
        print("    Bisect: dump per-layer streams on both sides and compare layer-by-layer.")
    sys.exit(0 if (ok_base and ok_final) else 1)


if __name__ == "__main__":
    main()
