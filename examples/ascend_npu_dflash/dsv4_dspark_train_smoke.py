#!/usr/bin/env python3
"""Training-path smoke for DSV4DSparkDraftModel (build + forward + backward).

Shakes out the subclass end-to-end on a tiny config: build the model (backbone
swapped in), feed a fake flat batch in the trainer's exact contract, run the
inherited DSpark forward (anchor sampling -> our sparse stack -> Markov +
confidence + compound loss), and backprop. This is the box gate for the
_backbone_forward decoder-stack adaptation — it exercises the eager mask ->
sink attn_bias reconciliation, the DSV4 RoPE position indexing, the mHC streams,
and the freshly-built backbone param init.

CPU, fp32, no verifier download (verifier name_or_path=None -> load_verifier_weights
is a no-op; the frozen embed/lm_head are filled with random values here). Run::

    python dsv4_dspark_train_smoke.py
"""
from __future__ import annotations

import sys

import torch


def build_model():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.dsv4_dspark.core import (
        DSV4DSparkConfig,
        DSV4DSparkDraftModel,
    )
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    H, V, block = 128, 256, 5
    tl = Qwen3Config(
        hidden_size=H, vocab_size=V, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=1, head_dim=32, intermediate_size=128,
        rms_norm_eps=1e-6, max_position_embeddings=512,
    )
    cfg = DSV4DSparkConfig(
        transformer_layer_config=tl,
        draft_vocab_size=V,  # == vocab -> no vocab mapping
        block_size=block,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=V - 1,
        speculators_config=SpeculatorsConfig(
            algorithm="dsv4_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
        markov_rank=32,
        # small DSV4 backbone
        num_heads=4, head_dim=32, rope_head_dim=8, q_lora_rank=64, o_lora_rank=64,
        o_groups=2, window_size=16, n_routed_experts=8, n_shared_experts=1,
        n_activated_experts=2, moe_inter_dim=128, hc_mult=2, hc_sinkhorn_iters=2,
    )
    model = DSV4DSparkDraftModel(cfg).float()
    model.load_vocab_mappings(None, None)
    # verifier not loaded -> fill the NaN-init frozen weights for the smoke.
    with torch.no_grad():
        for p in [model.embed_tokens.weight, model.lm_head.weight,
                  model.verifier_lm_head.weight, model.verifier_norm.weight]:
            if torch.isnan(p).any():
                p.normal_(std=0.02) if p.dim() > 1 else p.fill_(1.0)
    return model, cfg


def main() -> int:
    torch.manual_seed(0)
    print(f"torch {torch.__version__}\n[1/3] building DSV4DSparkDraftModel ...", flush=True)
    try:
        model, cfg = build_model()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nFAIL at build: {type(e).__name__}: {e}")
        return 1
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      built. trainable params: {n_train:,}  layers: {len(model.layers)}")

    H = cfg.transformer_layer_config.hidden_size
    V = cfg.transformer_layer_config.vocab_size
    n_aux = len(cfg.aux_hidden_state_layer_ids)
    T = 48
    batch = {
        "hidden_states": torch.randn(1, T, n_aux * H),
        "input_ids": torch.randint(0, V, (1, T)),
        "loss_mask": torch.ones(1, T),
        "verifier_last_hidden_states": torch.randn(1, T, H),
        "document_ids": torch.zeros(1, T, dtype=torch.long),
        "position_ids": torch.arange(T).unsqueeze(0),
    }
    from speculators.models.metrics import kl_div_loss

    fwd_kwargs = {
        "loss_config": {"kl_div": (kl_div_loss, 1.0)},
        "gamma": 4.0,
        "max_anchors": 8,
        "confidence_head_alpha": 1.0,
        "per_position_loss_weight": "fixed-exp-decay",
        "dpace_alpha": 0.5,
    }

    print("[2/3] forward ...", flush=True)
    try:
        draft_tokens, loss, metrics = model(**batch, **fwd_kwargs)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nFAIL at forward: {type(e).__name__}: {e}")
        return 1
    print(f"      draft_tokens {tuple(draft_tokens.shape)} | loss {loss.item():.4f} "
          f"| finite={torch.isfinite(loss).item()}")

    print("[3/3] backward ...", flush=True)
    try:
        loss.backward()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nFAIL at backward: {type(e).__name__}: {e}")
        return 1
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None and ".experts." not in n]
    ok = torch.isfinite(loss).item() and not no_grad
    print(f"      core params missing grad: {no_grad[:4] or 'none'}")
    print(f"\n{'PASS' if ok else 'FAIL'} — DSV4 DSpark training smoke "
          f"(build + forward + backward on the trainer contract)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
