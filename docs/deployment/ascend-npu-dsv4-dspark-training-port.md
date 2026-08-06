# DSpark-on-DeepSeek-V4-Flash — training-port spec for `speculators` (Plan 甲, faithful)

> **Role:** this is the faithful **architecture / port spec** — the "how the draft is built" companion to the
> canonical training doc [`ascend-npu-dsv4-dspark-ep-training.md`](./ascend-npu-dsv4-dspark-ep-training.md)
> (design decisions + results + validation matrix). For pipeline overview see
> [`ascend-npu-dsv4-dspark-pipeline.md`](./ascend-npu-dsv4-dspark-pipeline.md) stage 4. Architecture facts here
> are current; for STATUS/RESULTS always defer to ep-training + the eval-results ledger.

Goal: reproduce **DeepSeek's DSpark draft TRAINING for DeepSeek-V4-Flash** inside the `speculators`
framework, faithful to the released `deepseek-ai/DeepSeek-V4-Flash-DSpark` checkpoint (Plan 甲 —
project decision, non-negotiable). Sources: the checkpoint weight index (`/workspace/dspark_extract/`),
the paper **arXiv:2607.05147** ("DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation"), and vllm-ascend's inference impl (PR #11196 / `dsa_v1.py`).

## 0. What DSpark is (method)
Semi-autoregressive drafter: one draft forward emits a **block of γ tokens in parallel** (the
"parallel backbone"), then a **low-rank Markov head** injects intra-block causal dependency, and a
**confidence head** predicts per-position acceptance (used by a confidence-scheduled verifier).
Two "lines": **dense** (targets Qwen3) and **sparse** (targets DeepSeek-V4). **Our checkpoint = the
sparse line** (MoE `mtp.N` layers). Upstream `speculators` DSpark (#677) = the **dense** line.

## 1. Draft architecture (weights + paper, unified)
- **3 draft layers** (`mtp.0/1/2`), each a **full DSV4-style decoder layer** (paper: "3-MoE-layer
  backbone with mHC and sliding-window attention 128"; ablation: 2-layer DSpark > 5-layer DFlash).
- **Attention = MLA** (the target's own sparse-MLA / `flash_mla` family), per draft layer:
  `wq_a→wq_b` (low-rank query), `wkv` (joint KV latent), `wo_a→wo_b` (grouped output),
  `q_norm/kv_norm` (latent RMSNorms), **RoPE** (reused from target), **sliding window = 128**,
  **non-causal within the draft block**. NB the DRAFT does **not** carry the DSA `indexer/compressor`
  sparse machinery (those keys are target-only, `layers.N`) — so draft attn = MLA + sink + SW mask.
- **Attention SINK = per-head, per-layer LEARNABLE scalar** in the softmax denominator:
  `p_j = exp(s_j) / (Σ_j exp(s_j) + exp(sink))` — a synthetic key that absorbs excess mass and
  contributes nothing to the value sum. Init 0 (GPT-OSS/MiMo scheme; confirm). The draft stores its
  OWN `mtp.N.attn.attn_sink` (not merely the target's). This is the piece `speculators` lacks.
- **MoE FFN**: `experts.0..255` (256 routed, w1/w2/w3) + `shared_experts` + `gate(+bias)`. Target
  V4-Flash MoE = 256 routed + 1 shared, top-6, expert intermediate 2048, hidden 4096; draft most
  plausibly inherits these — **confirm from `inference/config.json`**.
- **Manifold-constrained hyper-connections (mHC)**: `hc_attn_*`, `hc_ffn_*` (base/fn/scale) — residual
  generalization countering PreNorm hidden-state magnitude growth/collapse. `hc_mult` = mHC rate.
- **`main_proj`/`main_norm`**: project the frozen target's deep hidden state into the draft residual
  stream (analog of DFlash's `self.fc`).
- **DSpark heads**: `markov_head.markov_w1/w2` (low-rank r≈256 additive logit bias → causal block
  distribution) + `confidence_head.proj` (Linear→1, accept-rate predictor).
- **γ (block size) = 5** in production (DSpark-5); Markov rank ≈ 256. (Our measured baseline used
  num_spec=7 — confirm the checkpoint's γ from config.)

## 2. Training recipe (paper §2)
- **Frozen vs trained:** FREEZE the target's `embed_tokens` + `lm_head` (shared). TRAIN only the
  backbone (3 MLA-MoE-mHC layers + sinks), `main_proj`, Markov head, confidence head.
- **Loss** = position-weighted sum of three terms:
  `L = Σ_k w_k · [ CE + λ_tv·TV(draft,target) + α·BCE_conf ]`, `w_k = exp(−(k−1)/γ)`, k=1…γ.
  - **CE** vs target's next token (`ce_loss`).
  - **TV** total-variation distribution match (`l1_loss`; NB **L1 = 2·TV** — reconcile scale). This is
    the acceptance-optimizing term.
  - **Confidence BCE** (`confidence_loss`) against the **soft analytic accept rate = 1 − TV(p,q)**,
    computed **conditioned on accepted predecessors** (prefix survival).
- **STS (Sequential Temperature Scaling):** post-hoc per-position 1-D temperature calibration of the
  confidence head (order-preserving, minimizes ECE) so scores map to real acceptance probs.
- **Data / regime:** teacher-force on target outputs + the target's **deep hidden states**;
  **anchor-block sampling** (single forward over a γ-block from an anchor); **anchor-bounded sequence
  packing** with token-level attention indices so blocks don't attend across anchors. TV term ⇒
  self-distillation from the target.

## 3. Port change-list — HAVE vs NEED
**Already in `speculators` (upstream/main dspark, the DENSE line — reuse):**
- Markov head (vanilla/gated/rnn), Confidence head, `compound_loss` + confidence BCE (accept-rate
  target = 1−d_TV, matches), position decay `w_k`, anchor-block sampling, `main_proj` (DFlash `fc`),
  sliding-window + non-causal MASK (`sliding_window_non_causal`), DFlash `_backbone_forward` reuse.

**MUST ADD for 甲 (the sparse-line backbone — none of this is upstream):**
| # | Add | Effort | Ref |
|---|---|---|---|
| 1 | **MLA attention** (wq_a/b, wkv, wo_a/b, q/kv_norm, RoPE) as a trainable draft attn | **large** | HF `inference/model.py`; transformers `deepseek_v4` |
| 2 | **Per-head learnable `attn_sink`** in the draft softmax (concat-sink → softmax → slice; eager path — SDPA/flex can't carry it) | small–med | transformers `deepseek_v4` sink block; my `dspark_attn_ref_bench.py` (math validated) |
| 3 | **256-expert MoE FFN** (routed + shared + gate) as a TRAINABLE draft layer | **large** (routing + expert-parallel + memory) | HF `inference/model.py` |
| 4 | **mHC hyper-connections** (`hc_attn/hc_ffn`, `hc_mult`) | med | HF `inference/model.py`; rtuli `hc_head_project` |
| 5 | **DSV4 config + weight-name loading** (`hc_mult`, head_dim=512, layer_types SW/full, nested `rope_parameters["main"]`, freeze embed/lm_head) | med | rtuli `dflash-dsv4-training-v2:scripts/train.py:117`, `dflash/core.py:219` |
| 6 | **STS** post-hoc confidence calibration | small | paper §2 |
| 7 | Base everything on **upstream/main** (has dense-line dspark + `_backbone_forward`), not the stale `docs/dsv4-dspark` | — | — |

**Verdict:** 甲 = implement DSV4's **MLA + 256-MoE + sink + mHC decoder layer** as a trainable draft.
The DSpark *method* (heads/loss/sampling) is free from upstream; the **sparse backbone (1,3,4) is the
bulk** and is a real model-porting + MoE-training effort, not a head-add.

## 4. Phased plan (de-risk the MoE)
1. **Scaffold + smoke (no MoE):** DSV4 MLA + sink + mHC draft layer with a **dense FFN** stand-in;
   random-HS fwd+bwd on 1 NPU (extend `dspark_confidence_test.py`). Proves MLA+sink+mHC+heads+loss.
2. **Config + weight load:** construct the real DSV4 draft config; load/freeze target embed+lm_head.
3. **Add the 256-expert MoE** (routed+shared+gate) + training-side expert parallelism; single-node
   smoke, then FSDP/EP scale.
4. **End-to-end:** rollout → target-HS extraction → train (anchor-block packing) → ckpt → eval accept
   length with the existing Evaluator; add STS calibration.

## 5. Open items to confirm (from `DeepSeek-V4-Flash-DSpark/inference/config.json`)
`γ` (block size), `n_routed_experts`/`num_experts_per_tok` (draft MoE), `q_lora_rank`/`kv_lora_rank`
(MLA), `λ_tv`/`confidence_head_alpha`/`loss_decay_gamma`, `attn_sink` init. (WebFetch was blocked here;
read these on the box or from HF when implementing.)

Reported gains (paper): +60–85% per-user gen on V4-Flash over MTP-1 (lossless); offline accepted length
+16–31% vs DFlash/Eagle3; a 2-layer DSpark beats a 5-layer DFlash.
