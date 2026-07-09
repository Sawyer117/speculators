#!/usr/bin/env python3
"""Multi-step training loop + memory probe for DSV4DSparkDraftModel.

Two things:
  (1) LEARNS?  overfit a single fixed batch for N steps with AdamW — the loss
      should fall steadily (the model memorizes the batch). If it doesn't drop,
      the gradient path is broken even though a single step "runs".
  (2) MEMORY?  on NPU, report the peak device memory after fwd, after bwd, and
      the training-step peak (params + grads + Adam state + activations), plus
      the analytic params/grads/optimizer breakdown, so the full-model footprint
      can be extrapolated.

The backbone is CURRENTLY PURE TORCH EAGER (no NPU fused kernels yet — #5), so
the measured activation memory is an UPPER BOUND; the fused sink-attention /
grouped-GEMM MoE / Sinkhorn kernels will cut it.

Config is env-scalable so you can probe memory at real-ish dims on NPU::

  # small (CPU ok):
  python dsv4_dspark_train_loop.py
  # real-ish widths, few experts so it fits one card (extrapolate experts):
  DEVICE=npu HIDDEN=4096 HEADS=64 HEAD_DIM=512 QLR=1024 OLR=1024 OGROUPS=8 \
  INTER=2048 EXPERTS=8 TOPK=6 LAYERS=3 BLOCK=5 SEQ=4096 ANCHORS=512 MARKOV=256 \
  STEPS=20 python dsv4_dspark_train_loop.py
"""
from __future__ import annotations

import os
import sys

import torch


def _env(name, default, cast=int):
    return cast(os.environ.get(name, default))


def build(dev):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.dsv4_dspark.core import (
        DSV4DSparkConfig,
        DSV4DSparkDraftModel,
    )
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    H = _env("HIDDEN", 128)
    V = _env("VOCAB", 256)
    block = _env("BLOCK", 5)
    heads = _env("HEADS", 4)
    hd = _env("HEAD_DIM", 32)
    tl = Qwen3Config(
        hidden_size=H, vocab_size=V, num_hidden_layers=_env("LAYERS", 3),
        num_attention_heads=heads, num_key_value_heads=1, head_dim=hd,
        intermediate_size=_env("INTER", 128), rms_norm_eps=1e-6,
        max_position_embeddings=max(2048, _env("SEQ", 48) + 64),
    )
    cfg = DSV4DSparkConfig(
        transformer_layer_config=tl,
        draft_vocab_size=V,
        block_size=block,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=V - 1,
        speculators_config=SpeculatorsConfig(
            algorithm="dsv4_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
        markov_rank=_env("MARKOV", 32),
        num_heads=heads, head_dim=hd, rope_head_dim=_env("ROPE_DIM", 8),
        q_lora_rank=_env("QLR", 64), o_lora_rank=_env("OLR", 64),
        o_groups=_env("OGROUPS", 2), window_size=_env("WINDOW", 16),
        n_routed_experts=_env("EXPERTS", 8), n_shared_experts=1,
        n_activated_experts=_env("TOPK", 2), moe_inter_dim=_env("INTER", 128),
        hc_mult=_env("HC_MULT", 2), hc_sinkhorn_iters=_env("HC_ITERS", 2),
    )
    model = DSV4DSparkDraftModel(cfg).to(dev)
    model.load_vocab_mappings(None, None)
    with torch.no_grad():
        for p in [model.embed_tokens.weight, model.lm_head.weight,
                  model.verifier_lm_head.weight, model.verifier_norm.weight]:
            if torch.isnan(p).any():
                p.normal_(std=0.02) if p.dim() > 1 else p.fill_(1.0)
    return model, cfg


def main() -> int:
    torch.manual_seed(0)
    want = os.environ.get("DEVICE", "cpu")
    npu = False
    if want == "npu":
        try:
            import torch_npu  # noqa: F401
            dev = "npu:0"
            npu = True
        except Exception as e:  # noqa: BLE001
            print(f"!! torch_npu unavailable ({e}); falling back to cpu")
            dev = "cpu"
    else:
        dev = want
    dtype = torch.bfloat16 if npu else torch.float32

    model, cfg = build(dev)
    model = model.to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"torch {torch.__version__} | device {dev} | dtype {dtype}")
    print(f"params total {n_params:,} | trainable {n_train:,} | experts "
          f"{cfg.n_routed_experts} layers {cfg.transformer_layer_config.num_hidden_layers}\n")

    H = cfg.transformer_layer_config.hidden_size
    V = cfg.transformer_layer_config.vocab_size
    n_aux = len(cfg.aux_hidden_state_layer_ids)
    T = _env("SEQ", 48)
    anchors = _env("ANCHORS", 8)
    g = torch.Generator().manual_seed(1)
    batch = {
        "hidden_states": torch.randn(1, T, n_aux * H, generator=g).to(dev, dtype),
        "input_ids": torch.randint(0, V, (1, T), generator=g).to(dev),
        "loss_mask": torch.ones(1, T).to(dev),
        "verifier_last_hidden_states": torch.randn(1, T, H, generator=g).to(dev, dtype),
        "document_ids": torch.zeros(1, T, dtype=torch.long).to(dev),
        "position_ids": torch.arange(T).unsqueeze(0).to(dev),
    }
    from speculators.models.metrics import kl_div_loss

    fwd_kwargs = {
        "loss_config": {"kl_div": (kl_div_loss, 1.0)}, "gamma": 4.0,
        "max_anchors": anchors, "confidence_head_alpha": 1.0,
        "per_position_loss_weight": "fixed-exp-decay", "dpace_alpha": 0.5,
    }
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=_env("LR", 1e-3, float)
    )
    steps = _env("STEPS", 20)

    def mem_mb():
        return torch.npu.max_memory_allocated() / 1e6 if npu else 0.0

    losses = []
    fwd_peak = bwd_peak = step_peak = 0.0
    for step in range(steps):
        if npu:
            torch.npu.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(**batch, **fwd_kwargs)
        if npu:
            torch.npu.synchronize()
            fwd_peak = max(fwd_peak, mem_mb())
        loss.backward()
        if npu:
            torch.npu.synchronize()
            bwd_peak = max(bwd_peak, mem_mb())
        opt.step()
        if npu:
            torch.npu.synchronize()
            step_peak = max(step_peak, mem_mb())
        losses.append(loss.item())
        if step < 5 or step % 5 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    dropped = losses[-1] < losses[0] - 1e-3
    print(f"\nloss: {losses[0]:.4f} -> {losses[-1]:.4f}  "
          f"({'DROPPED ✅' if dropped else 'did NOT drop ❌'})")
    if npu:
        pbytes = 2  # bf16 params
        params_mb = n_train * pbytes / 1e6
        grads_mb = n_train * pbytes / 1e6
        adam_mb = n_train * 8 / 1e6  # fp32 m+v
        print(f"\nNPU memory (MB): fwd_peak {fwd_peak:.0f} | after_bwd {bwd_peak:.0f} "
              f"| step_peak {step_peak:.0f}")
        print(f"  analytic (trainable {n_train:,}): params(bf16) {params_mb:.0f} + "
              f"grads(bf16) {grads_mb:.0f} + AdamW(fp32 m,v) {adam_mb:.0f} = "
              f"{params_mb + grads_mb + adam_mb:.0f}")
        print(f"  activations ~ step_peak - (params+grads+adam) = "
              f"{step_peak - params_mb - grads_mb - adam_mb:.0f} MB")
        print("  NB: pure-torch EAGER (no NPU fused kernels) -> activation upper bound.")
    else:
        print("\n(run with DEVICE=npu on an NPU card for fwd/bwd/peak memory numbers)")
    return 0 if dropped else 1


if __name__ == "__main__":
    sys.exit(main())
