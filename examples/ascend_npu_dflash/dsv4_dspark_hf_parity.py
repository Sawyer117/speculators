#!/usr/bin/env python3
"""Component parity: our clean-room DSV4 backbone vs HF transformers `deepseek_v4`.

Stage-1 correctness — the eager-fp32 "vs official" check, using **HF as the
oracle** (transformers `deepseek_v4` is the clean-torch port of the release; on
this box its mHC == the official kernel, confirmed). Method (per component):

  1. build the HF module and ours from a matched small config,
  2. random-init ONE side (HF) and copy its state_dict into ours through a
     key-name map (our clean-room names differ: wq_a↔q_a_proj, attn_sink↔sinks,
     experts.i.w1↔experts.gate_up_proj[i][:inter], …),
  3. run identical fp32 input through both and compare outputs (expect ~0).

Covers the components whose forward HF exposes 1:1: RMSNorm, mHC
HyperConnection / HyperHead, the sink-softmax core, and the MoE block (routing +
experts). The draft-only parts (main_proj / block-gamma / Markov / confidence)
are not in HF — those are gated by the fwd/bwd smoke in dsv4_dspark_parity.py
and by reading the official forward_spec.

Needs transformers>=5.7 with deepseek_v4. Run::  python dsv4_dspark_hf_parity.py
"""
from __future__ import annotations

import sys

import torch

from speculators.models.dsv4_dspark.config import DSparkDraftConfig

FAILURES: list[str] = []
TOL = 2e-5  # fp32 math-equivalence tolerance


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load_mapped(ours: torch.nn.Module, mapped: dict[str, torch.Tensor]) -> None:
    res = ours.load_state_dict(mapped, strict=False)
    if res.missing_keys:
        check("  (no missing keys)", False, f"{res.missing_keys[:6]}")
    if res.unexpected_keys:
        check("  (no unexpected keys)", False, f"{res.unexpected_keys[:6]}")


def diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def hf_config(cfg: DSparkDraftConfig):
    from transformers.models.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=1,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.moe_inter_dim,
        sliding_window=cfg.window_size,
        rms_norm_eps=cfg.rms_norm_eps,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        n_routed_experts=cfg.n_routed_experts,
        num_experts_per_tok=cfg.n_activated_experts,
        scoring_func=cfg.score_func,
        routed_scaling_factor=cfg.route_scale,
        swiglu_limit=cfg.swiglu_limit,
        mlp_bias=False,
        q_lora_rank=cfg.q_lora_rank,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        index_n_heads=cfg.num_heads,
        index_head_dim=128,
        index_topk=8,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        partial_rotary_factor=cfg.rope_head_dim / cfg.head_dim,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": cfg.rope_theta,
                     "partial_rotary_factor": cfg.rope_head_dim / cfg.head_dim},
            "compress": {"rope_type": "default", "rope_theta": 100000.0,
                         "partial_rotary_factor": cfg.rope_head_dim / cfg.head_dim},
        },
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
    )


def test_rmsnorm(cfg) -> None:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm

    from speculators.models.dsv4_dspark.backbone.norm import RMSNorm

    hf = DeepseekV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).float()
    torch.nn.init.normal_(hf.weight, std=0.3)
    ours = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).float()
    load_mapped(ours, {"weight": hf.weight.detach()})
    x = torch.randn(2, 7, cfg.hidden_size)
    check("RMSNorm == HF", diff(ours(x), hf(x)) < TOL, f"max_abs={diff(ours(x), hf(x)):.2e}")


def test_hyper(cfg) -> None:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HyperConnection,
        DeepseekV4HyperHead,
    )

    from speculators.models.dsv4_dspark.backbone.hyper import HyperConnection, HyperHead

    hcfg = hf_config(cfg)
    streams = torch.randn(2, 7, cfg.hc_mult, cfg.hidden_size)

    hf = DeepseekV4HyperConnection(hcfg).float()
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.02)
    ours = HyperConnection(cfg).float()
    load_mapped(ours, {k: v.detach() for k, v in hf.state_dict().items()})  # fn/base/scale match
    hp, hc, hcol = hf(streams)
    op, oc, ocol = ours(streams)
    check("mHC post == HF", diff(op, hp) < TOL, f"max_abs={diff(op, hp):.2e}")
    check("mHC comb == HF", diff(oc, hc) < TOL, f"max_abs={diff(oc, hc):.2e}")
    check("mHC collapsed == HF", diff(ocol, hcol) < TOL, f"max_abs={diff(ocol, hcol):.2e}")

    hf_h = DeepseekV4HyperHead(hcfg).float()
    for p in hf_h.parameters():
        torch.nn.init.normal_(p, std=0.02)
    ours_h = HyperHead(cfg).float()
    load_mapped(ours_h, {k: v.detach() for k, v in hf_h.state_dict().items()})
    check("mHC head == HF", diff(ours_h(streams), hf_h(streams)) < TOL,
          f"max_abs={diff(ours_h(streams), hf_h(streams)):.2e}")


def test_moe(cfg) -> None:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

    from speculators.models.dsv4_dspark.backbone.moe import MoE

    hcfg = hf_config(cfg)
    hf = DeepseekV4SparseMoeBlock(hcfg, layer_idx=0).float()
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.05)
    ours = MoE(cfg).float()

    # Map HF's packed experts -> our per-expert Linears.
    sd = hf.state_dict()
    inter = cfg.moe_inter_dim
    mapped: dict[str, torch.Tensor] = {
        "router.weight": sd["gate.weight"],
        "router.bias": sd.get("gate.e_score_correction_bias", torch.zeros(cfg.n_routed_experts)),
        "shared_experts.w1.weight": sd["shared_experts.gate_proj.weight"],
        "shared_experts.w3.weight": sd["shared_experts.up_proj.weight"],
        "shared_experts.w2.weight": sd["shared_experts.down_proj.weight"],
    }
    gate_up = sd["experts.gate_up_proj"]  # [E, 2*inter, hidden]
    down = sd["experts.down_proj"]        # [E, hidden, inter]
    for e in range(cfg.n_routed_experts):
        mapped[f"experts.{e}.w1.weight"] = gate_up[e][:inter].contiguous()   # gate
        mapped[f"experts.{e}.w3.weight"] = gate_up[e][inter:].contiguous()   # up
        mapped[f"experts.{e}.w2.weight"] = down[e].contiguous()
    load_mapped(ours, mapped)

    x = torch.randn(4, 6, cfg.hidden_size)
    check("MoE == HF", diff(ours(x), hf(x)) < 1e-4, f"max_abs={diff(ours(x), hf(x)):.2e}")


def test_sink(cfg) -> None:
    """Our sink kernel vs the HF/gpt-oss eager sink formula (inlined, dense non-causal).

    HF's ``eager_attention_with_sinks``: append a per-head sink logit column,
    subtract the row max over [logits | sink], softmax, drop the sink column,
    matmul V. Transcribed here (not imported) so the check is robust to HF's
    internal function name/signature.
    """
    import torch.nn.functional as F

    from speculators.models.dsv4_dspark.backbone.attention import sink_block_attention

    h, d, sq, sk = cfg.num_heads, cfg.head_dim, 5, 20
    q = torch.randn(2, sq, h, d)
    k = torch.randn(2, sk, h, d)
    v = torch.randn(2, sk, h, d)
    sink = torch.randn(h)
    scale = d ** -0.5

    ours = sink_block_attention(q, k, v, sink, scale)  # [2, sq, h, d]

    qb, kb, vb = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B,H,S,D]
    logits = torch.matmul(qb, kb.transpose(2, 3)) * scale  # [B,H,Sq,Sk]
    sinks = sink.reshape(1, -1, 1, 1).expand(qb.shape[0], -1, sq, -1)
    combined = torch.cat([logits, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1)[..., :-1]
    hf = torch.matmul(probs, vb).transpose(1, 2)  # [B,Sq,H,D]
    check("sink attention == HF formula", diff(ours, hf) < 1e-4, f"max_abs={diff(ours, hf):.2e}")


def main() -> int:
    import transformers

    print(f"torch {torch.__version__} | transformers {transformers.__version__}\n")
    cfg = DSparkDraftConfig().small()
    for name, fn in [("RMSNorm", test_rmsnorm), ("mHC", test_hyper),
                     ("MoE", test_moe), ("sink", test_sink)]:
        print(f"{name}:")
        try:
            fn(cfg)
        except Exception as e:  # noqa: BLE001
            check(f"{name} ran", False, f"{type(e).__name__}: {e}")
        print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL PASS — backbone components match HF deepseek_v4 (eager fp32).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
