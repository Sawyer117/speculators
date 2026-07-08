# DSpark-on-DeepSeek-V4-Flash — landing plan (team fork)

Faithful reproduction of DeepSeek's **DSpark draft training for DeepSeek-V4-Flash**, to live in the
team's `speculators` fork. Supersedes the earlier `ascend-npu-dsv4-dspark-training-port.md` (which
predated the Alloy/hf-npu-binder discovery and the official-gold anchoring). Plan 甲 — project decision.

## 0. Decisions locked (with evidence)

- **Training regime = A (teacher-forced on CACHED target hidden states), one-way. NOT on-policy.**
  Triple-confirmed: DSpark paper §2 + upstream `speculators/train/data.py` (schema `hidden_states` +
  `verifier_last_hidden_states`, offline `generate_hidden_states`, training never runs the target) +
  canonical **DeepSpec** `deepspec/data/target_cache_dataset.py` (sharded bf16 target-HS cache). The
  "verify" in DSpark is **inference-time** (confidence-scheduled verifier), not training.
- **Online (hard constraint; 4×A2, NO A3, shared storage) = 2S1T.** 2 A2 = one bf16 target-HS
  **producer**; **1 A2 = draft trainer** (fits via expert-parallel — see MoE below); the 4th A2 = spare
  (eval serve / producer-assist). HS transported via **shared storage** (DeepSpec-style sharded target-HS
  cache, streamed as a ring buffer). Frozen target ⇒ streaming ≡ offline mathematically; one-way.
  Bottleneck = target forward (43 vs 3 layers) → freeing a trainer A2 is the right direction. Anti-stall =
  trainer prefetches N shards.
- **Draft size reality check:** the draft is **NOT tiny** — 3 layers × 256-expert MoE ≈ **~20B params**
  (~40 GB bf16; the 13 GB released file is fp4/fp8 quantized). Full fp32 Adam ~320 GB.
- **MoE training parallelism = EXPERT PARALLEL (EP), not FSDP2-all-gather.** No 8-bit Adam (fidelity).
  Math: full-fp32-Adam 20B on **1 A2 (8 cards)** = 40 GB/card sharded + the FSDP2 **all-gather of all 256
  experts** (~26 GB fwd+bwd) ⇒ ~66 GB > 64 GB → **OOM**. So the FSDP2-gather-all-experts path is dropped.
  **EP is the correct primary** (not a fallback): it is the canonical DSV4 training layout (MindSpeed uses
  `expert_model_parallel_size=32`), matches how the model is served, and is memory-correct — each card
  holds only 256/EP experts, so the all-gather peak disappears. EP=8 on 1 A2 fits full fp32 Adam (~50
  GB/card) → **unlocks 2S1T with no 8-bit**. Design = **EP for the experts (~19 B) + FSDP2/DDP for the
  small dense parts (MLA/mHC/norms ~0.6 B)**, one EP-MoE impl with an `EP_size` knob (1 = dense/smoke,
  8 = 1 A2, 16 = 2 A2). EP=1 degenerates to dense for GPU smoke tests. Cost: implement EP dispatch
  (`torch.dist.all_to_all`, device-agnostic NCCL/HCCL) + EP-aware autograd in the torch-native path;
  expert **compute** reuses `hf-npu-binder` grouped-GEMM, we add the **routing** all-to-all layer.
- **No megatron / MindSpeed in the draft training path** (torch-native FSDP2 + EP).
- **Gold source: MindSpeed/MindSpeed-LLM = PRIMARY reference** (Ascend-official, team-defensible; agrees
  with the official code on architecture + rope-factor). **Official code = the weights-matched
  cross-check** (`deepseek-ai/DeepSeek-V4-Flash` `inference/model.py`+`config.json` HEAD 60d8d70;
  `…-DSpark` HEAD 62af8ff). `alloy/references/dsv4` is a transformers `deepseek_v4` snapshot (2 steps from
  official) — a CANDIDATE to verify, never the anchor. Do NOT unilaterally pick a side on any
  MindSpeed↔official discrepancy — **list them (see §8) and ask the team/Ascend/DeepSeek.** (Correction:
  the earlier "rope factor 40 vs 16 drift" was a misread — MindSpeed sets `--rope-factor 16` = official;
  the `40` is a separate `--rope-scaling-factor` arg. No drift on the primary factor.)
- **Team fork ⇒ no private-repo dependency in the shareable core.** Vendor the DSV4 backbone slice into
  the fork (backend-agnostic torch); NPU kernels (`hf-npu-binder`) = **optional conditional-import**
  (flag off → torch fallback → GPU-clean); the HS producer is **internal NPU tooling** (may use
  `hf-npu-binder`, since teammates don't "receive" it).

## 1. Pinned gold / reference trees (my sandbox)

| Path | What | Role |
|---|---|---|
| `/workspace/dsv4-official` | base `inference/model.py`+config (60d8d70) | **parity anchor (backbone)** |
| `/workspace/dsv4-dspark-official` | DSpark `inference/model.py`+config (62af8ff) | **parity anchor (draft + method)** |
| `MindSpeed-LLM` / `MindSpeed` | Ascend megatron impl (gitcode live) | read-spec authority + gsm8k bridge |
| `DeepSpec` | canonical DSpark training (add63ba) | data/loss/schedule reference |
| `speculators/src/.../models/dspark`,`dflash` | dense-line DSpark + DFlash | method reuse (heads/loss/trainer/data) |
| `alloy` / `hf-npu-binder` | HF-native DSV4 + NPU kernels (private) | verified candidate / internal kernels |

## 2. Definitive config (official, both configs agree)

**Backbone (base):** 43 layers, hidden 4096, heads 64, kv_heads **1** (MLA), head_dim 512, qk_rope 64,
q_lora 1024, **kv_lora 512** (from MindSpeed — official config omits it), o_lora 1024, o_groups 8; MoE 256 routed + 1 shared, topk 6, moe_int 2048,
`sqrtsoftplus`/`noaux_tc`, route_scale 1.5, n_hash_layers 3; **mHC** hc_mult 4 / sinkhorn_iters 20 /
eps 1e-6; sliding_window 128; RoPE yarn **factor 16** / orig_max 65536 / theta 10000, compress_rope_theta
160000, compress_ratios `[0,0,4,128,…,4,0]`; vocab 129280, rms_eps 1e-6, swiglu_limit 10.0,
tie_word_embeddings false. (Official ships fp4 experts + fp8; we run **bf16** dequantized — parity in bf16.)

**Draft (DSpark):** `n_mtp_layers 3`, `dspark_block_size` (γ) **5**, `dspark_target_layer_ids [40,41,42]`,
`dspark_markov_rank 256`, `dspark_noise_token_id 128799`.

## 3. Draft architecture (from DSpark `inference/model.py` — definitive; matches the released weights)

`DSparkBlock(Block)` × 3, stored under `mtp.*`, each a full DSV4 block (`DSparkAttention` MLA+sink+SW,
256-MoE, mHC). Extra parts:
- **stage 0**: `main_proj = Linear(dim*len(target_layer_ids)=4096*3, 4096)` + `main_norm = RMSNorm`.
- **stage 2 (last)**: `norm` + `markov_head = DSparkMarkovHead(vocab, 256)`
  (`markov_w1=Embed(vocab,256)`, `markov_w2=Head(vocab,256)` → low-rank vocab→vocab bias) +
  `confidence_head = DSparkConfidenceHead(dim+256)` (`Linear(4096+256, 1, fp32)` on `cat[hidden, markov_embed]`)
  + `hc_head_fn/base/scale` (mHC head).
- embed + lm_head **shared** with target, **frozen**.

**Forward (semi-autoregressive block-γ):**
1. `forward_embed`: `main_x = main_norm(main_proj(cat(target_hidden[40,41,42])))`;
   `draft_input_ids = [real_token, noise×4]` (block 5); `x = embed(ids)` → mHC-expand (`repeat hc_mult=4`).
2. 3 DSV4 blocks (attention conditioned on `main_x`).
3. `forward_head`: `hc_head` collapse → `logits = lm_head(norm(x))` (block logits); **Markov AR loop**
   `for i in block: logits[:,i] += markov_head(prev_id); next = sample(logits[:,i])`; **confidence** =
   `confidence_head(cat[hidden, markov_embed])`.

## 4. Loss (teacher-forced; speculators dense-line + DeepSpec + paper)

Per block position k=1..γ, decay `w_k = exp(-(k-1)/γ)`:
`L = Σ_k w_k · [ CE(p_k, target_token_k) + λ_tv·TV(p_k, q_k) + α·BCE(conf_k, 1 − d_TV(p_k,q_k)) ]`
- `q_k` (target dist) = `softmax(frozen lm_head(cached target_last_hidden))` — no live target in the loss.
- TV = distribution-overlap acceptance term (`1 − Σ min(p,q)`); confidence BCE target is the **detached**
  soft accept rate. STS temperature calibration is a post-hoc pass on the confidence head.

## 5. Phased plan (each phase gated by a parity/behaviour check)

**Phase 1 — DSV4 backbone (backend-agnostic torch), anchored to official base `model.py`.**
Build/vendor the DSV4 decoder layer in the fork (`speculators/models/dspark/backbone/`): MLA (wq_a/b,
wkv, wo_a/b + o_groups, q/kv RMSNorm, g2 dual-yarn RoPE factor 16) + per-head **sink** (concat→softmax→
`[...,:-1]`) + 256-MoE (routed+shared, sqrtsoftplus/noaux_tc, grouped-GEMM) + **mHC** (hc_mult 4, sinkhorn
20). Logic guided by MindSpeed read-spec; **numerically verified vs `dsv4-official/inference/model.py`**
(single layer, random input, same bf16 weights). Run alloy's `dsv4_*` through the SAME parity as a
candidate — adopt whichever passes; never blind-copy. NPU kernels = optional conditional-import (off →
torch fallback → GPU-clean).
*Gate:* single-layer fp32-eager `max_abs` vs official < tol; assembled base model gsm8k ≈ 97% (bridges to
MindSpeed's measured 96.59).

**Phase 2 — DSpark draft (3 layers + method), anchored to DSpark release `model.py`.**
Assemble the draft: 3 Phase-1 blocks + `main_proj`/`main_norm` (input = target layers [40,41,42]) +
frozen shared embed/lm_head + `DSparkMarkovHead(256)` + `DSparkConfidenceHead(4096+256)` + block-γ=5
(`DSparkAttention` block mask, noise_token 128799). Reuse the dense-line DSpark heads/losses from
`speculators/models/dspark` (adapt to the sparse backbone). Weight loader: DSpark release ckpt → draft
(names already match: `mtp.N.*`, `markov_head.markov_w1/w2`, `confidence_head.proj`, `main_proj/main_norm`,
`hc_*`) for init/eval.
*Gate:* draft-forward parity vs `dsv4-dspark-official/inference/model.py` on one γ-block with the released
weights (logits + confidence).

**Phase 3 — Online 2S2T target-HS pipeline (shared storage).**
- **Producer (2 A2):** frozen DSV4 target forward over rollout sequences → write `{target_hidden_states
  (layers 40/41/42), target_last_hidden_states, token_ids}` as DeepSpec-style **sharded bf16 cache** to
  **shared storage** (ring buffer). Internal NPU tool — may use `hf-npu-binder` fused kernels.
- **Trainer (2 A2):** `speculators` FSDP2 trainer reads the shard cache with **prefetch depth N**
  (anti-stall) — schema aligns with `speculators/train/data.py` + DeepSpec `target_cache_dataset`.
- Rollout (the running A3/A2 serve) supplies the token sequences; producer consumes them.
*Gate:* producer↔trainer throughput matched (no starvation past the target-forward ceiling); a few shards
train end-to-end without stall.

**Phase 4 — End-to-end train + eval + STS.**
Train the draft (2 A2 FSDP2) on the streamed HS. Eval **accept length** via spec-decode (target serve +
draft) with the existing Evaluator; compare to the num_spec=7 baseline (gsm8k 6.189 / math500 6.095 /
humaneval 5.524 / mbpp 5.191 / mt-bench 3.747). Add STS post-hoc confidence calibration.
*Gate:* accept length ≥ baseline (target: DSpark's paper +16–31% vs DFlash/Eagle).

## 6. Team-fork hygiene (cross-cutting)

- Shareable core (`speculators` draft training + DSpark method + vendored DSV4 backbone) is
  **self-contained + GPU-runnable**; zero private-repo imports.
- NPU acceleration = optional conditional-import backend; absence → torch fallback.
- HS **producer** is internal NPU tooling (separate from the shareable core), may depend on
  `hf-npu-binder`.
- Every model piece is parity-anchored to the **official** gold (§1); MindSpeed = read-spec; alloy = a
  verified candidate, never a blind source.

## 7. Open items

- kv-latent naming: official config exposes `head_dim 512` (single KV head) with no separate
  `kv_lora_rank`; confirm the MLA KV path against `model.py` when coding Phase 1.
- Shared-storage shard format + prefetch depth + ring-buffer cap (sized to the 2-A2 producer rate).
- Whether to init the draft from the released DSpark ckpt (dequantized bf16) or train from scratch.

## 8. MindSpeed ↔ official discrepancies to ASK (don't unilaterally decide)

Mostly yarn long-context knobs; at our train seq (~4–8k ≪ original 65536) yarn is inactive, so these
likely don't affect training/parity — list for completeness, low priority.

| Field | MindSpeed recipe | Official config | Question |
|---|---|---|---|
| `rope-scaling-factor` | **40** (separate from `rope-factor 16`) | only `factor: 16` | Does the `40` take effect? What does it map to? (`rope-factor 16` already matches official.) |
| `rope-scaling-original-max-position-embeddings` | **4096** | `original_max_position: 65536` (MindSpeed also has `original-seq-len 65536`) | Which one drives the yarn scaling? |
| `max-position-embeddings` | 163840 | 1048576 | Training cap vs model max — confirm harmless. |
| `kv-lora-rank` | 512 | (omitted) | Using **512** (MindSpeed fills the official gap) — confirm. |

No discrepancy on the substantive architecture: layers 43, experts 256+1 topk 6, MLA ranks, mHC
(hc_mult 4 / sinkhorn 20), sink, sliding_window 128, `sqrtsoftplus`/`noaux_tc` — MindSpeed, official
base, and official DSpark all agree.
