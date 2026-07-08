# DSpark → MindSpeed injection spec (track B, single-machine EP)

How to wire the DSpark method (`examples/ascend_npu_dflash/dspark_method.py`) onto MindSpeed's DSV4
MTP, **all gated behind `--enable-dspark`** so MindSpeed's stock MTP path (base-model training) is
untouched. The DSV4 backbone (MLA + 256-MoE + mHC + sink attention + EP) is REUSED from MindSpeed; we
add the DSpark-specific pieces. Reference for every change = the official DSpark
`inference/model.py` (`DSparkBlock` / `DSparkAttention` / `DSparkMarkovHead` / `DSparkConfidenceHead`)
and upstream speculators #677 loss semantics. Verified against the live gitcode MindSpeed-LLM
(HEAD adb9f4c).

## 0. New CLI args (in `pretrain_deepseek4.py` / arguments)

```
--enable-dspark                 # master gate; off => stock MTP, no behavior change
--dspark-block-size 5           # gamma
--dspark-markov-rank 256
--dspark-target-layer-ids 40 41 42
--dspark-noise-token-id 128799
# implied when --enable-dspark: --mtp-num-layers 3  (validate, don't overload the count as the switch)
```

## 1. Injection points (file : location : change : which dspark_method.py piece)

| # | File | Location | Change (only under `enable_dspark`) | Uses |
|---|---|---|---|---|
| 1 | `mindspeed_llm/core/transformer/multi_token_prediction.py` | `mtp_layer_init_wrapper` (~L55) | stage 0: add `main_proj = Linear(dim*len(target_layer_ids), dim)` + `main_norm = RMSNorm`. last stage: add `markov_head`, `confidence_head`, final `norm`, `hc_head_*`. Mirrors official `DSparkBlock.__init__` stage_id gating. | `DSparkMarkovHead`, `DSparkConfidenceHead` |
| 2 | same | `mtp_layer_forward` L120–127 | replace `enorm/hnorm/eh_proj` → `main_x = main_norm(main_proj(cat(target_hidden[target_layer_ids])))` (official `forward_embed`). Input ids = block-γ. | `build_dspark_block` |
| 3 | same | `get_mtp_layer_input` (~L225) | build the γ-block input ids `[anchor, noise*(γ-1)]` + the target-hidden gather (layers 40/41/42, `hidden_states[id+1]` offset). | `build_dspark_block` |
| 4 | same | `mtp_block_forward` L253–268 | after the layers: hc_head collapse → `lm_head` → **Markov AR loop** (per block pos: `logits[:,i] += markov_head(prev_id)`; sample next) → `confidence`. Replace `compute_language_model_loss` (CE) with `dspark_compound_loss`. Mirrors official `forward_head`. | `DSparkMarkovHead`, `DSparkConfidenceHead`, `dspark_compound_loss` |
| 5 | `mindspeed_llm/tasks/models/spec/deepseek4_spec.py` | `mtp_spec` (~L61) | swap `self_attention: DeepSeek4MTPSelfAttention` → `DSparkMTPSelfAttention` (block-non-causal window). Keep `attn_mhc/mlp_mhc` (mHC) + the 256-MoE mlp as-is (reused). | — |
| 6 | `mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py` | new `DSparkMTPSelfAttention` + `get_dspark_topk_idxs` | block-γ attention: topk_idxs = `[window ctx] + [full block]` (non-causal within block); `attn_sink` already present (reuse). Training backward = the einsum+sink path (validated in `dspark_attn_ref_bench.py`), NOT the causal g2 triton bwd. | `dspark_block_mask` (ref) |

## 2. Forward flow (must match official `DSparkBlock`)

```
forward_embed (stage 0):   main_x = main_norm(main_proj(cat(target_hidden[40,41,42])))
                           x = embed([anchor, noise×4]) -> mHC-expand (hc_mult=4)
3× DSparkBlock:            MLA(block mask + sink) conditioned on main_x  ->  256-MoE  ->  mHC
forward_head (last stage): hc_head -> lm_head(norm(x)) = block logits
                           for i in γ: logits[:,i] += markov_head(prev_id); next = sample(...)
                           confidence = confidence_head(cat[hidden, markov_embed])
loss:                      dspark_compound_loss(draft_logits, target_logits, target_tokens, conf)
                             = Σ_k exp(-(k-1)/γ) · [0.1·CE + 0.9·L1 + conf·BCE(conf, 1-d_TV)], target detached
```

`target_logits` for the TV/BCE term = `lm_head(verifier_last_hidden)` — from the cached target hidden
(no live target in the loss). embed + lm_head shared & FROZEN (train only the 3 draft layers + heads).

## 3. MEGATRON primitive swaps (in `dspark_method.py`, marked `# MEGATRON:`)

- `DSparkMarkovHead.markov_w1`: `nn.Embedding` → `ParallelEmbedding`; `markov_w2`: `nn.Linear` → column-parallel `ParallelHead`.
- `DSparkConfidenceHead.proj`: keep fp32; small, replicate (no TP needed).
- The 256-MoE: **unchanged** — MindSpeed's `expert_model_parallel_size` (EP) handles it (this is the whole point of track B; EP=8/16 fits single-machine).

## 4. Parity checkpoints (before trusting training)

1. **Backbone**: MindSpeed DSV4 layer forward ↔ `examples/ascend_npu_dflash/dsv4_mla_ref.py` (CPU, same bf16 weights) ↔ official `inference/model.py`. + gsm8k on the MindSpeed base ≈ 97%.
2. **DSpark heads/loss**: `dspark_method.py::_selftest` (fwd+bwd, shapes, decay, detach) on a torch box.
3. **Draft block**: assembled DSpark MTP forward ↔ official DSpark `inference/model.py` on one γ-block with the released (dequantized bf16) weights — logits + confidence.
4. **Data**: 2S1T — target-HS producer (2 A2, shared-storage shards: `target_hidden_states[40,41,42]` + `target_last_hidden_states` + tokens) → MindSpeed trainer (1 A2, EP) reads with prefetch.

## 5. Open (confirm on-box)

- `DeepSeek4MTPSelfAttention` may already be topk_idxs-driven — if so, `DSparkMTPSelfAttention` = feed
  the block topk_idxs (no new kernel); confirm the einsum/backward path for the block-non-causal mask.
- `kv_lora_rank=512` (MindSpeed arg; official config omits it — confirm the MLA KV path).
- Whether to init the 3 draft layers from the released DSpark ckpt (dequant bf16; needs an
  `mtp.N.* → mtp_layers.N.*` name map à la `convert/mtp`) or train from scratch.
