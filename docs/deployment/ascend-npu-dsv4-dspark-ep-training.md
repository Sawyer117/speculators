# DSV4-DSpark Draft Training on Ascend NPU — Design Decisions & Results

> **STATUS: FINISHED (2026-07-14).** Design decisions and measured effects of DSV4-DSpark
> speculative-decoding draft training on Ascend NPU (A2, 8×64 GB), covering the two hidden-state
> extraction schemes (and why we chose the PR-based dumper over the native connector), the
> expert-parallel MoE refactor (DTensor-native, torchtitan-aligned), the tuning decisions, and
> the environment/run scripts (linked). Reports live under `docs/` (not `examples/`).
>
> One honest gap: §6 accept-length numbers are the **targets/baseline**; the converged draft
> numbers are pending a full training-to-eval pass (the last run was still LR-warming at
> accept_len ≈ 1.19). Everything else is measured.

## 0. Scope & audience

What this documents:
- How the **DSV4-Flash DSpark draft** is trained on Ascend A2 NPUs (8×64 GB) using the
  `speculators` FSDP2 trainer, with **expert-parallel (EP)** MoE.
- The **design decisions** made along the way, **why** each was chosen over the alternative,
  and the **measured effect**.
- The **environment-build and run scripts** (linked, not duplicated).

Not a from-scratch tutorial — assumes the reader knows speculative decoding, DSpark, and the
Ascend/vLLM stack. For the step-by-step recipe, follow the linked scripts.

## 1. System overview

```
┌── SERVE nodes (2× A2, bf16 verifier) ────────┐        ┌── TRAIN node (1× A2, 8 cards) ──┐
│  DeepSeek-V4-Flash bf16, TP8/DP2, EP off      │  HS    │  DSV4-DSpark draft               │
│  produces hidden states (Plan B HS_DUMP)      │ ─────► │  FSDP2 + EP8 grouped-GEMM MoE    │
│  -> hs_<row>.safetensors on shared /share     │ files  │  reads HS, trains draft          │
└───────────────────────────────────────────────┘        └──────────────────────────────────┘
```

- **Verifier (target):** DeepSeek-V4-Flash bf16, served across 2 A2 nodes (TP8/DP2, EP OFF —
  cross-node EP deadlocks; see §4).
- **Draft:** 3 decoder layers × 256 routed experts (DSV4-native: MLA + per-head sink + MoE +
  multi-hyper-connection + Markov/confidence heads), trained via `speculators` (HF-native,
  FSDP2 — NOT megatron).
- **Coupling:** online hidden-state (HS) extraction — the serve dumps the verifier's aux target
  hidden states to shared storage; the trainer reads them per sample (rolling buffer, produce-
  one / consume-one).

## 1.5 Architecture — verified from the weights & paper (NOT analogy)

> Every claim here is from a **reproducible primary source** (released checkpoint keys, released
> `config.json` fields, the DSpark/V4 papers, or vllm-ascend code) — cited inline `[like this]`
> and consolidated in **§10**. Reasoning-by-analogy burned this project repeatedly (the
> attention-sink flip-flops); the rule is *check the weight keys / the paper, not a sibling model*.

**Target — DeepSeek-V4-Flash** (43 layers, hidden 4096, vocab 129280, 64 heads). Each layer:
- **MLA** (Multi-head Latent Attention): q low-rank (`q_lora_rank=1024`: `wq_a→q_norm→wq_b`),
  single shared KV latent (`wkv→kv_norm`), grouped low-rank output (`wo_a→wo_b`).
  `[released config.json; weight keys layers.N.attn.{wq_a,wq_b,q_norm,wkv,kv_norm,wo_a,wo_b}]`
- **Per-head learnable attention sink** (all 43 layers). `[weight key layers.N.attn.attn_sink ×43]`
- **Hybrid long-context attention = SWA + CSA + HCA**, assigned per-layer by `compress_ratios`
  (44 entries, tiers `{0:3, 4:21, 128:20}`): ratio **0** = dense/SWA-only; ratio **4** = **CSA**
  (Compressed Sparse Attention, vLLM "c4a": KV ÷4, top-512); ratio **128** = **HCA** (Heavily
  Compressed Attention, vLLM "c128a": KV ÷128). Every layer also runs **SWA** (`sliding_window=128`)
  on uncompressed tokens. `[config compress_ratios / sliding_window=128 / index_topk=512;
  dsa_v1.py:1108 "vLLM-Ascend only support SWA-layer for Deepseek-V4"; vLLM blog c4a/c128a;
  arXiv:2606.19348]`
- **MoE**: 256 routed experts, top-6 (`num_experts_per_tok=6`), 1 shared, `moe_intermediate=2048`.
  `[config; weight keys layers.N.ffn.experts.{0..255} + shared_experts + gate]`
- **mHC** (Manifold-Constrained Hyper-Connections, Sinkhorn 20 iters) — replaces the residual.
  `[config hc_sinkhorn_iters=20; weight keys layers.N.hc_{attn,ffn}_{base,fn,scale}]`

**Draft — DSV4-Flash DSpark** (3 mtp layers, block γ=5). Each mtp layer is a DSV4 decoder layer
**minus DSA**:
- **MLA + per-head sink + MoE(256, top-6, shared) + mHC** — **NO compressor, NO indexer → NO
  DSA/CSA/HCA.** The draft attention is **dense sliding-window (128), bidirectional within the
  block.** `[released draft weight keys: mtp.N.attn.{wq_a,wq_b,q_norm,wkv,kv_norm,wo_a,wo_b,
  attn_sink} + ffn.experts.{0..255} + hc_{attn,ffn}, and NO compressor/indexer keys; DSpark paper
  arXiv:2607.05147 "the parallel backbone comprises three MoE layers with mHC and a sliding window
  attention of 128" + "All positions within a block attend bidirectionally"]`
- **Block heads**: `main_proj`+`main_norm` on **mtp.0** (block entry, injects the target context);
  `markov_head`(w1/w2, rank 256) + `confidence_head`(proj) + `hc_head` + `norm` on **mtp.2** (exit).
  `[released draft weight keys — heads appear once, on mtp.0 / mtp.2 respectively]`
- Target context = verifier hidden at `[40,41,42]`, projected (`main_proj`) and concatenated into
  every draft layer's K/V. Draft input = anchor token + γ mask embeddings (noise `128799`).
  `[config dspark_target_layer_ids=[40,41,42] / dspark_noise_token_id=128799; paper Eq. 2–3]`

⚠️ **"draft = MLA + DSA + SWA" is WRONG** (a natural guess, but false): the draft has **no DSA** —
no compressor/indexer weight keys, and the paper never gives the draft DSA. Correct = **MLA + sink
+ SWA**. DSA/CSA/HCA is **target-only** (long-context KV compression); the draft is tiny (3 layers,
γ=5), a 128 sliding window is already cheap.

```
 Draft forward (one block, γ=5) — from the weight keys + paper:

 TARGET(frozen) ─ hidden @ [40,41,42] ─────────────┐
                                                    ▼
 anchor + mask×5(=128799) ─embed─►  main_proj+main_norm (mtp.0)  → inject into every layer K/V
                                          │
              ┌── mtp.0 ──┐  each layer = mHC[ MLA + sink + SWA-128, bidir-in-block ]
              │   mtp.1   │              + mHC[ MoE 256 (top-6) + shared ]
              └── mtp.2 ──┘
                    │
              hc_head+norm (mtp.2)
             ┌──────┼──────────┐
             ▼      ▼          ▼
        lm_head  markov_head  confidence_head
        (5 tok)  (block dep)  (5× accept prob)
```

### Draft attention — the op landscape + train/infer windows

The draft attention is **SWA(128) + non-causal(bidirectional in block) + per-head sink**. The window
is asymmetric: `win_left = window + block_size − 1`, `win_right = block_size − 1`
`[va] dspark_attention.py:32-36`. Because training's `block_size` includes the anchor (6) but
inference's is the drafted count (5), the two sides use different but **equivalent** windows (the
anchor is a visible key either way — it's block-slot-0 in training, last-context-token at inference):

| side | block | `win_left / win_right` | KV = window+block |
|---|---|---|---|
| **training** (speculators, our forward) | **6** (anchor + 5 drafts) | **133 / 5** | 134 |
| **inference** (vllm-ascend SAS op) | **5** (γ) | **132 / 4** | 133 |

(NB: `134/6` is the **block7** Qwen3 case — `128+7−1`; not DSV4.) `[va] _dspark_sas_window`,
`[repo] models/dsv4_dspark/config.py block_size=6`.

**Why we can't reuse a fused op — no single op has all four properties:**

| op | SWA | non-causal | sink | backward | usable for… |
|---|---|---|---|---|---|
| vllm-ascend **SAS** (inference) | ✅ | ✅ (mask mode 4) | ✅ (`sinks=sinks`) | **❌ forward-only** | inference / the gold **reference** |
| SDPA / `npu_fusion_attention` | ✅ | ❌ causal | ❌ | ✅ | nothing here (missing sink + non-causal) |
| our **einsum** (`backbone/attention.py::sink_block_attention`) | ✅ | ✅ | ✅ | ✅ **but slow** | training **now** (correct, unfused) |
| **Triton** kernel (in dev) | ✅ | ✅ | ✅ | ✅ (goal) | training **target** (fused) |

**Does the operator need updating?**
- **Inference SAS op — code is fine, no update.** It already does SWA+non-causal+sink; the fork's
  `win_right>0` patch is in `dspark-dsv4` (PR #11196, AL 3.94). *If* a node's compiled `.so` computes
  causal-127 (upstream, missing the patch — its AscendC tiling asserts `oriWinRight==0`), that's a
  **build** issue on that node → rebuild vllm-ascend from the patched commit; verify with
  `diag_sas_window.py` (run at `BS=5` for DSV4). Not a code change.
- **Training op — the update-in-progress is the Triton kernel** (SWA+non-causal+sink **+ backward**;
  no existing fused op has all four). Until it lands, training uses the correct-but-slow einsum. SWA
  handoff repo: `Sawyer117/non-causal-swa-triton-ascend` (kernel matches the gold ref at fp32
  5.96e-7). See §5, §7.

## 2. HS extraction: two schemes, and why we chose the PR-based dumper (`HS_DUMP`)

The draft trains on the verifier's hidden states at target layers `[40,41,42]` **plus** the
final post-norm hidden. There are two ways to get those out of vLLM. This is the decision the
skeleton flagged as the headline example ("we did two schemes, chose the one based on a PR
rather than the native interface — why").

### Scheme A — native `extract_hidden_states` connector (`HS_EXTRACT=1`)
vLLM's built-in `extract_hidden_states` speculative method + the `ExampleHiddenStatesConnector`
(`kv_role: kv_producer`) writes `hs_*.safetensors`. This is the "official / upstream" interface.
On DSV4 it fails for **three independent reasons**, none of them a patchable bug:

1. **KV memory pathology (the killer).** `extract_hidden_states` implements a fake
   `CacheOnlyAttentionLayer` whose "KV cache" *is* the hidden-state store, allocated as a
   block-based cache **co-sized with the real KV pool**:
   `s_hidden(per token) = L_aux · H · d = 3·4096·2 = 24 KB` — **model-independent and
   uncompressed by construction**. DSV4's *real* KV is hyper-compressed (MLA single shared
   latent (c+r)=576, no k/v ×2, DSA compress-ratio ÷, sliding windows), so this fixed 24 KB/token
   becomes **~1/3–1/2 of the whole KV budget → OOM / squeezed KV**. On a GQA/MHA model (Qwen3-8B,
   where colleagues *did* validate extract) the same `s_hidden` is only ~5–10% of a big KV
   budget — a non-issue. **DSV4 is the worst possible case for extract, and no knob fixes it**
   (`L_aux·H` is uncompressed by design). A partial planner surgery (`feat/dsv4-supports-eagle3`
   @ `f7a1e25`) fit DP1 but DP0/head still OOM'd.
2. **vLLM version lock.** On our pinned **vLLM 0.23.0** the extract CacheOnly path crashes
   `'list' object has no attribute 'device'` (`extract_hidden_states.py:72`) — the KV arrives as
   a `list`. It's only fixed on vLLM **0.24-dev**, which *also* flips the v1→v2 model runner.
   Our vllm-ascend is version-locked to 0.23.0 (dozens of `vllm_version_is("0.23.0")` branches),
   so "just use latest vLLM" is a **full serve re-validation project**, not a bump.
3. **PD-disaggregation clash.** `kv_role=kv_producer` puts vLLM in **PD-disaggregated**
   (prefill/decode split) mode, which is **incompatible with Ascend balance-scheduling**
   (pydantic: *"enable_balance_scheduling only supports PD-mixed mode"*). The serve script has to
   force `VLLM_ASCEND_BALANCE_SCHEDULING=0` to even boot it (see
   `serve_dsv4_bf16_dualnode.sh:159-162`).

### Scheme B — PR-based dumper (`HS_DUMP=1`) ← **CHOSEN**
Instead of the generic connector, ride the **DSV4-model inference PR that already captures the
target hidden states**, and add a tiny dumper to spill them to disk.

- **What it rides on.** The DSV4 target model forward (vllm-ascend **`dspark-dsv4` / #11571**,
  `deepseek_v4.py:1190,1236-1242`) already fills an NPU scratch buffer
  `_dspark_hidden_buffer [max_num_batched_tokens, L_aux·H]` (~200 MB, overwritten each forward)
  with the mean-over-mHC target hidden at `[40,41,42]`, exposed via
  `get_mtp_target_hidden_states()`. **This capture is gated ONLY on
  `config.dspark_target_layer_ids` — independent of the speculative method** — so a *plain*
  serve populates it (no draft ckpt), as long as the target config carries
  `dspark_target_layer_ids` (we inject it via `--hf-overrides`).
- **What we added.** A runner **post-forward hook** (vllm-ascend fork
  `Sawyer117/vllm-ascend @ feat/dsv4-hs-dumper`, `vllm_ascend/dspark_hs_dumper.py` +
  `model_runner_v1.py`): after each `execute_model`, TP-rank-0 async-copies
  `get_mtp_target_hidden_states()` (aux) **and** the model's post-norm forward output
  (verifier-last) NPU→CPU, accumulates per request, and on request-finish writes
  `hs_<rollout_row_index>.safetensors` in the **standard `ArrowDataset` / extract-connector
  format** `{token_ids:[seq], hidden_states:[seq, num_aux+1, H]}` — so the trainer reads it
  unchanged. **No `kv_transfer`, no PD-disaggregation, no `HiddenStateCacheSpec`.**
- **Why chosen — it sidesteps all three Scheme-A failures at once:** zero extra NPU KV (just the
  ~200 MB scratch already there → no OOM), runs on the validated **0.23.0** dspark serve (no
  0.24 re-validation), and no kv_producer (no PD-disagg / balance-scheduling clash). CPU cost is
  ~2 GB RAM for 32 concurrent reqs (box has ~1.4 TB). Validated end-to-end (P2 smoke on A2
  dual-node, 3436 tok/s).
- **How it's driven.** ONLINE-only, teacher-forced **prefill-only** (`max_tokens=1`): a driver
  sends each rollout row's `input_ids` as a token-id prompt with request-id = the row index, so
  `token_ids` match by construction and the file lands at `hs_<index>.safetensors`. On serve +
  train: `DSPARK_HS_DUMP=1`, trainer `--hidden-states-path == DSPARK_HS_DIR`,
  `--on-missing generate --on-generate delete` (rolling buffer → no disk explosion). `loss_mask`
  is **not** in the HS file — `ArrowDataset` pulls it from the paired rollout Arrow dataset, so
  the producer needs no prompt/response split and no tokenizer.
- **One cluster gotcha (solved).** This cluster maps the same login to *different uids* on serve
  vs trainer, so the dumper pins file perms itself: `os.chmod(tmp, 0o777)` before atomic
  `os.replace`, and `chmod 0o777` on the out-dir (so `--on-generate delete` can unlink across
  uid). Fixed in `feat/dsv4-hs-dumper @ 4677f0b`. (Upstream speculators has the same shared-file
  design and simply assumes serve+train run as one uid.)

**Deferred, not dead.** The durable upstream path (native `extract` via `SupportsEagle3`) is
paused on branch `feat/dsv4-supports-eagle3 @ f7a1e25`; it needs vLLM 0.24 (for the `list.device`
fix + v2 runner) *and* the memory pathology solved for DP0/head. Full rationale and phased plan:
[`docs/deployment/ascend-npu-dsv4-hs-dumper-planB.md`](./ascend-npu-dsv4-hs-dumper-planB.md).

## 3. Expert-parallel MoE: the DTensor-native refactor (the big one)

**Problem.** The faithful draft has **256 routed experts × 3 layers**. Two pain points:
1. Per-shape MoE recompiles: dynamic per-expert token counts → the NPU recompiled the MoE
   kernel every step → **~160 s forward spikes**.
2. The 256-expert memory-fit trick was **per-expert FSDP** (each expert sharded across 8
   cards), which is **incompatible with a fused grouped-GEMM** (grouped needs each expert's
   full weight local).

**Rejected approach — plain-tensor experts via FSDP `ignored_params`.** First cut: keep the
`nn.ModuleList([Expert])`, exclude experts from FSDP (`ignored_params`) so they stay whole &
local, all-to-all the tokens. This *works* but the experts are **plain tensors** while
everything else is an **FSDP DTensor** → every bulk-param op mixes the two and crashes:
- `clip_grad_norm_` stacks DTensor + plain norms → crash (patched: split clip)
- AdamW `_foreach_mul_` batches DTensor + plain → crash (patched: two optimizers)
- checkpoint save gathers/collides plain expert names (patched: skip save)

→ a **whack-a-mole of patches**, and checkpoints don't work → **not upstreamable**. (This is
what the earlier commits `2506ca9`/`b39b5b5`/`390a325` did; all reverted in the refactor.)

**Chosen approach — GroupedExperts + `Shard(0)` DTensors (torchtitan-aligned).** After reading
torchtitan's EP (`GroupedExperts` in `torchtitan/models/common/moe.py`; the all-to-all token
dispatch in `torchtitan/models/common/token_dispatcher.py`; expert weights as `Shard(0)`
DTensors composed with FSDP), we mirrored the pattern:
- Routed experts become **stacked weights** (`w1/w3 [E,inter,dim]`, `w2 [E,dim,inter]`) in a
  `GroupedExperts` module (not a `ModuleList`) — the layout EP + grouped-matmul both want. Init
  matches `nn.Linear` (`U(±1/√fan_in)`); under EP each rank seeds its own slice via
  `torch.random.fork_rng`.
- Each rank builds only its `[E/EP, ...]` slice, wrapped as a **`Shard(0)` DTensor** on the
  **same DeviceMesh** as the FSDP-sharded rest (`shard_experts_as_dtensor`,
  `train/distributed.py:237`) → **every parameter is a uniform DTensor** → the optimizer /
  `clip_grad_norm_` / DCP checkpoint need **no special-casing**; all Stage-1 patches removed
  (single AdamW, single clip, normal save).
- MoE forward: `.to_local()` the local expert slice + **autograd-aware all-to-all** the tokens
  to their owner rank (`moe_ep._AllToAll`; backward = reverse all-to-all with swapped splits),
  local grouped-GEMM (`torch_npu.npu_grouped_matmul`), all-to-all back. Same math the CPU parity
  tests pin down (§8).
- `state_dict()` stays **stacked** (consistent with `named_parameters` / DCP / broadcast); the
  per-expert on-disk format (for the released loader / serve) is a **file-boundary conversion**,
  not a `state_dict` hook (a hook would diverge from `named_parameters`). *(User's call:
  disk-side per-expert, memory-side stacked.)*

**Why not TP (tensor parallel) too?** The draft is small (3 layers) and fits comfortably; it's
not memory- or compute-bound. TP would add a per-layer all-reduce (comm the small draft can't
amortize) and make it 3D-parallel (TP×EP×FSDP) for zero gain. **EP8 (experts) + FSDP (rest)** is
the right parallelism at this scale. Revisit TP only if the draft grows huge or moves to A3 (16
logical devices → EP16 mirrors the serve).

**Why not MindSpeed-native?** `speculators` (its FSDP2 trainer + DSpark method + eval + ckpt
format) is a hard requirement. MindSpeed-native = training in megatron's loop → conflicts. So we
**extracted the winning piece** (the fused grouped-GEMM op `torch_npu.npu_grouped_matmul`) into
the HF-native MoE and kept EP as our own all-to-all; MindSpeed-native shelved.

**Effect:** the faithful 256-expert step went from **~160 s recompile spikes → steady
~250–400 ms forward** (EP8 also distributes the expert compute 8×), clean training, checkpoints
work, no patches. See §6.

## 4. Other design decisions (with rationale + effect)

| Decision | Chosen | Why / effect |
|---|---|---|
| Serve EP on 2 nodes | **OFF** | `--enable-expert-parallel` on 2-node A2 → cross-node EP16 `HcclAllGather` deadlock (`shm_broadcast: No available block` hang). Confirmed by the AtomGit A2 two-node V4-Flash report + our own plog. EP OFF → MoE TP-sharded intra-node, all collectives stay intra-node. (`serve_dsv4_bf16_dualnode.sh` `ENABLE_EP`.) |
| Draft mask/noise token | **128799** (`--mask-token-id 128799`) | DSpark `noise_token_id` (`config.py`). Draft's masked positions embed as `embed_tokens[128799]`; **must match serve**. Without it, `resolve_mask_token_id` silently falls back to `pad_token_id=1` (wrong: collides with pad + mismatches serve). Caught by user during tuning. |
| `total_seq_len` | **3072** (down from 8192 default) | The collator **packs** rollout docs to fill exactly `total_seq_len` (not one-doc-padded). 3072 (vs 8192) cuts draft-forward activation memory (room for more anchors) + shortens the serve prefill. Anchor utilization = `max_anchors / seq_len`. Align serve `max_model_len ≈ 3072` so sequences fall within training length (no dumped tails the trainer truncates). |
| `max_anchors` | **128–256** (paper: 512) | Each fetched HS sequence trains `max_anchors` positions. Raising it makes HS fetch stop being the bottleneck: **`fetch_frac` 0.96 → 0.02–0.03**. Memory-capped: 128@3072 ≈ 49.7 GB (77%), 256@3072 ≈ 59.8 GB (93%, tight); **512 needs gradient checkpointing** (→ §5). |
| Optimizer / LR | **AdamW, lr 2e-4** | 6e-4 (DeepSpec `dspark_qwen3_4b` ref) **DIVERGED to NaN at ~step 931** the moment warmup reached it (`lr=5.98e-4`) — the warm-start's high early `grad_norm` (~200) overflows bf16 at 6e-4. Reverted to **2e-4** (the stable value the from-scratch runs used; `grad_norm` ~1). Single AdamW over uniform DTensors (§3); Muon available. |
| Grouped-GEMM op | `torch_npu.npu_grouped_matmul` | Extracted from MindSpeed; one grouped matmul per projection replaces the 256-way expert loop + kills per-shape recompiles. Self-written autograd backward, CPU + NPU parity-tested (§8). |
| Draft attention | **SDPA** (`--draft-attn-impl sdpa`) | Ascend has no `simple_flex_attention`. The non-causal-sink SWA fused kernel is a separate in-progress optimization (see §5, §7 handoff repo). |
| Rollout sampling | **greedy, temp=0 end-to-end** | Gen / train / eval all temp=0 → self-consistent (user's call). DeepSpec uses 0.7/top-p0.8; DSV4 official rec 1.0. Tripwire if ever benchmarked sampled. |
| Per-epoch validation | **off-able** (`--no-validation` / `NO_VAL=1`) | The 10% held-out val split has **no pre-dumped HS**, so `on_missing=generate` re-generates it **serially via the serve every epoch** (the val loader is `num_workers=0` — forked val workers corrupt the child heap right after the epoch-boundary DCP checkpoint: `free(): invalid pointer`). That dominates the epoch and *looks* hung (only the tqdm bar creeps). `--no-validation` skips the val pass **and** trains on the **FULL** dataset (`split_ratio=1.0`); `train_data_ratio` (default 0.9) only applies when val is **on**, so nothing is silently wasted. |
| MoE warm-start from target | **opt-in** (`--init-moe-from-target` / `INIT_MOE=1`) | A draft layer **is** a DSV4 target layer, so init each draft layer's MoE (routed experts + router + shared) from verifier layer `target_layer_ids[n]` (`[40,41,42]→[0,1,2]`) instead of random — a strong basin for the draft's job (predict next-token from the layer-40/41/42 hidden states). **1:1 copy** (verifier uses the official DeepSeek names; only rename `ffn.gate`→`router`; `w1`=gate/`w3`=up/`w2`=down). **Trainable** (not frozen, unlike the shared embed/lm_head). EP-aware: runs at build time when `GroupedExperts` are still plain per-rank tensors (before the `Shard(0)` wrap), so each rank copies only its `[ep_expert_offset : +n_local]` slice; no-op on meta params. **Faithful (256×2048) only** — fast-fails at startup on any dim/key mismatch. Verified on box: 256 experts, `moe_intermediate_size 2048`, `layers.{40,41,42}.ffn.{experts.{e}.w{1,2,3},gate.{weight,bias},shared_experts.w{1,2,3}}`. A/B vs from-scratch (early accept_len should start higher). Open (measurable): whether the **official** DSpark recipe warm-starts — `dspark_method.py` shows only loss+heads. |

## 5. Known follow-ups / not-yet-optimized

- **`argsort` int64 → AiCPU fallback** in the MoE dispatch (the recurring `[ArgSort] running on
  AiCpu` warning). Fix: cast the sort key to float32 (values 0–255 exact; drop `stable`,
  correctness holds via the inverse permutation) → runs on AiCore. *(Offered, not yet applied.)*
- **Fused MoE permute** (`npu_moe_init_routing`) — fuses sort+permute around the all-to-all,
  supersedes the argsort fix.
- **Fixed-shape MoE padding** — pad per-expert token counts to fixed buckets → static
  grouped-GEMM shapes → eliminates the residual ~7–14 s new-shape recompile spikes
  (`DSPARK_MOE_BUCKET`). The **shape-generic** alternative — `torch.compile`'d experts
  (`DSPARK_COMPILE=1`) — is validated (~1.74×, bit-exact) but **banked** (needs the torch-2.12
  stack); default off. See [`ascend-npu-dsv4-dspark-compile-recompile.md`](ascend-npu-dsv4-dspark-compile-recompile.md).
- **Gradient checkpointing (activation recompute)** — **DONE** (`DSPARK_RECOMPUTE=1`; `core.py`
  recomputes each draft layer in backward). Frees activation to run **384 @ 3072** (was
  memory-capped ~256); the paper's 512 still OOMs even with recompute, so 384 is the settled point.
- **SWA non-causal sink fused Triton kernel** — in development (replaces SDPA draft attention);
  handoff repo in §7.
- **Serve HS throughput** — `HS_DUMP` serve tuned to rollout-optimal (`MAXSEQS=32`, graph mode
  `EAGER=0`, `--async-scheduling`); ceiling ~0.60 rows/s @ concurrency 64 (KV-overflow bound,
  §8).

## 6. Results (measured)

| Config | fwd_ms (steady) | mem reserved | fetch_frac | loss | accept_len |
|---|---|---|---|---|---|
| faithful EP8, 256@3072, lr 6e-4 | **~250–400 ms** (was ~160 s spikes) | 59.8 GB (93%) | ~0.02–0.03 | ~0.46 | ~1.19 (LR still warming) |
| faithful EP8, 128@3072 | ~250 ms | 49.7 GB (77%) | ~0.03 | — | — |

- **Recompile spikes eliminated** (EP + grouped-GEMM). Residual ~7–14 s spikes are occasional
  new-shape recompiles (→ §5 fixed-shape padding).
- **HS no longer the bottleneck** once `max_anchors` raised (each fetched HS sequence trains more
  positions): `fetch_frac` 0.96 → 0.02–0.03.
- **Checkpoint save validated** (EP-DCP gather of the `Shard(0)` expert DTensors +
  `save_pretrained`, ~3 min/ckpt) after fixing a relative `from .backbone import moe_ep` in
  `core.py` that broke transformers' `custom_object_save` at save time (`a855bfb` → absolute
  import).

**Acceptance-length target (baseline, NOT yet the converged draft).** The DSV4 DSpark draft
drafts **γ = 5** tokens per block → served `num_speculative_tokens = 5` (not 7 — that's the
Qwen3 line; see note below). ⚠️ **Block-size convention gotcha (cost us a run):** the trainer's
`--block-size` is the **block width including the anchor** (slot 0 is the given anchor,
loss-masked; drafts `block_size − 1`, like DFlash `block16 → 15`). So **train with
`--block-size 6`** to draft 5; the released config's `dspark_block_size = 5` is **γ** (= served
num_spec), i.e. `block_size − 1`. Passing `--block-size 5` silently drafts only 4 (logs show
`position_1..4`, accept_len ceiling 5) — a wrong, one-short draft. A first train→eval pass IS now done
(see **6.1** below): the epoch-1 draft evals low (1.36) but that is a **serve bug, not training** —
weights vindicated. (Also: the `n_predict = dspark_block_size` inference quirk noted here is a
**pre-rewrite artifact** — the #12005 serve rewrite removes it, so num_spec is no longer forced ÷5.)
> **Provenance (this bit the project — full chain in §10.C):**
> - **DSpark's own forward masks the anchor slot** → trains/drafts `block_size − 1`:
>   `src/speculators/models/dsv4_dspark/core.py:343` (`mask_token_ids[:, ::block_size] = anchor token`)
>   and `:397` (`aligned_loss_mask[:, ::block_size] = 0`), plus `src/speculators/models/dspark/metrics.py:85,88,152`
>   (comment *"slot 0 is the anchor"*; `[:, 1:]`; `for pos in range(1, block_size)`).
> - **Shipped proposal config** `speculative_tokens = block_size − 1` (inherited: `DSV4DSparkDraftModel`
>   ← `DSparkDraftModel` ← `DFlashDraftModel`, not overridden) — `src/speculators/models/dflash/core.py:186-188`,
>   `# First block position is the anchor, not emitted during gen.`
> - **Inference sets `n_predict = dspark_block_size`, NO −1**: vllm-ascend (dspark path)
>   `patch/platform/patch_speculative_config.py:21`; test `tests/ut/spec_decode/test_dspark_config.py:36`
>   asserts `n_predict == 5`; `spec_decode/dspark_proposer.py:421` `block_size = num_speculative_tokens`.
> - **Qwen3 cross-check**: upstream trains `BLOCK_SIZE=8` (`examples/train/dspark_qwen3_0_6b_sharegpt_online.sh:36`)
>   ⇔ released `deepseek-ai/dspark_qwen3_4b_block7/config.json` `block_size=7` (Δ=1=anchor).
> - **accept_len ceiling = num_spec + 1 = 6**: `examples/ascend_npu_dflash/Evaluator.py:599`
>   `accept_length = 1.0 + d_acc/d_drafts`.

The bar to match/beat is the **released DeepSeek DSV4-Flash DSpark draft**, measured on NPU in
**vllm-ascend PR #11196** (QwertyJack) at `num_spec=5`:

| metric | released DSV4-Flash DSpark (PR #11196) |
|---|---|
| acceptance rate (AR) | **58.79%** |
| **accept length (AL)** | **3.94** (GPU reference: 3.86) |
| per-position accept | `[0.81, 0.68, 0.58, 0.48, 0.39]` (AL = 1 + Σ = 3.94) |

**eval /metrics counter resets mid-run** on vllm-ascend spec_decode — use the reset-aware poller
(`Evaluator.py @ a3c41a6`), or non-first-dataset accept lengths read low.

### 6.1 First train→eval pass (2026-07-16) — training side healthy; eval blocked on a SERVE bug, not the draft

epoch-1 ckpt `/home/a00652497/dspark_austin/run/ckpt_faithful_ep_20260715_213847/0` (faithful EP8,
arrow_0715, `INIT_MOE=1`, lr 2e-4, block 6):

- **Training side — healthy / on-track (count this, not just the eval).** Loss decreasing; **soft
  accept_len ~2.9–3.1 median** by epoch 1 (soft = `Σ_v min(p_v,q_v) = 1−d_TV` = E[len] under *sampling*,
  what `metrics.py` logs); **position_1 hard-argmax ≈0.82**. The draft *is* learning the target — the
  training-side signals are decent.
- **Eval side — 1.36, but a SERVE artifact.** epoch-1 ckpt → `scripts/convert_dspark_to_vllm.py` (UT-1
  bit-exact) → served → gsm8k **hard greedy accept_len 1.36** (pos0 ≈0.32). **This is a serve bug, NOT the
  weights:** the **released known-good draft scores the SAME ~1.34 on our serve** (proven — see worklog
  2026-07-16), root-caused to our fork's **pre-rewrite DSpark** (proposer piggybacks DFlash). Every
  statically-checkable piece (aux, non-causal window, sink, config, structure) matches DeepSeek's own
  `inference/model.py`.
- **⟹ Training is on track; the eval gap is the serve.** Fix = rebuild on the #12005 rewrite
  ([`vllm-ascend-dspark-rebuild.md`](./vllm-ascend-dspark-rebuild.md)); **re-measure the trained draft after
  that** (very possibly the number jumps once the serve is correct). Weights **VINDICATED — no retrain.**
- Metric caveat: training soft accept_len (~2.9) and serve hard greedy accept_len (1.36) are *different
  metrics* — don't equate them; the apples-to-apples signal was position_1 hard-argmax 0.82 (train) vs ~0.32
  (serve), and the released-draft test settled that it's the serve.

Fill the trained draft's per-dataset `num_spec=5` numbers here **after the #12005 serve rebuild** (a fair eval).

> **⚠ Do NOT conflate models.** The `dspark_qwen3_4b_block7` accept lengths (gsm8k 6.189 /
> math500 6.095 / humaneval 5.524 / mbpp 5.191 / mt-bench 3.747) are **Qwen3-4B, block7,
> num_spec=7, full-attention** — a *different* speculator. DSV4 is block5 / num_spec=5 /
> sliding-window+sink, and its released-draft AL is **3.94**, not ~6. (The two live side by
> side in our notes; the block/attention/model all differ.)

## 7. Environment build & run (linked, not duplicated)

**Install — training (SSOT):**
[`examples/ascend_npu_dflash/install_npu_env_dspark.sh`](../../examples/ascend_npu_dflash/install_npu_env_dspark.sh)
— gold-standard one-click NPU env (corrected toolchain: system gcc + CANN lld/ccec/bisheng,
zero conda compilers, numpy 2.3.5 forced last). Training needs `speculators` (editable) +
`transformers >=4.56.1,<5.14.0`; it does **not** need vllm-ascend.

**Install — serve / HS producer:**
[`examples/ascend_npu_dflash/setup_dsv4_env.sh`](../../examples/ascend_npu_dflash/setup_dsv4_env.sh)
— identical-on-every-node env: conda `dspark-dsv4-base`, torch/torch_npu 2.10.0, vLLM 0.23.0+empty,
**vllm-ascend @ `feat/dsv4-hs-dumper`** (the Plan-B HS dumper branch, based off `dspark-dsv4` @
`6036507`). Both serve nodes MUST be on the same vllm-ascend commit.

**Serve (verifier + HS producer):**
[`examples/ascend_npu_dflash/serve_dsv4_bf16_dualnode.sh`](../../examples/ascend_npu_dflash/serve_dsv4_bf16_dualnode.sh)
— `HS_DUMP=1 nohup bash ... head` (rank0/API) + `HS_DUMP=1 nohup bash ... worker` (rank1). Always
`nohup` the head — a bare foreground head dies to Ctrl-C (foreground poll + serve share a process
group). A3 single-node variant: `serve_dsv4_a3_singlenode.sh` (DP2/TP8/EP16, intra-node EP works).

**Train:**
[`examples/ascend_npu_dflash/train_dsv4_dspark.sh`](../../examples/ascend_npu_dflash/train_dsv4_dspark.sh)
— baked defaults now `LR=6e-4 EPOCHS=5 SEQLEN=3072 BLOCK=6`. Typical faithful run:
`INIT_MOE=1 NO_VAL=1 MAX_ANCHORS=384 RECOMPUTE=1 DSPARK_EP=1 DATA=<arrow_dir> bash ... faithful`.
Modes: `reduced` (1L×32E smoke) / `faithful` (3L×256E, EP8). Env knobs:
`SEQLEN MAX_ANCHORS LR EPOCHS BLOCK MASK_TOKEN CKPT_FREQ DSPARK_EP RECOMPUTE COMPILE NO_VAL
INIT_MOE DSPARK_GROUPED_MOE NPROC RUN SAVE_PATH VERIFIER DATA HS_DIR ENDPOINT CANN_ENV`
(the banner echoes their resolved values). Backgrounds a torchrun run; prints the tail command.
`NO_VAL=1` → no per-epoch val + train on full data; `INIT_MOE=1` → warm-start MoE from the
verifier; `RECOMPUTE=1` → activation recompute (needed past ~256 anchors). See §4 for each.

**Rollout (data gen):** `rollout_a3_shard.sh` / `rollout_shard.sh` (greedy, temp=0; prep →
Arrow). **HS-dump smoke:** `hs_dump_smoke.py`. **Eval:** `run_eval.sh` (background serve +
curl-wait + eval in ONE terminal).

**Repos / external:**
- Draft trainer fork: `https://github.com/Sawyer117/speculators` (branch `feat/dsv4-dspark`,
  HEAD `3953bc1`: `--init-moe-from-target`, `--no-validation`, `EPOCHS`/`SEQLEN 3072`/`LR 6e-4`
  knobs, `prepare_data --chat-template` port, `analyze_train_run --baseline` compare).
- HS-dumper serve fork: `https://github.com/Sawyer117/vllm-ascend` (branch `feat/dsv4-hs-dumper`,
  `@ 4677f0b`; rides `dspark-dsv4`/#11571's `_dspark_hidden_buffer`).
- SWA fused kernel handoff: `https://github.com/Sawyer117/non-causal-swa-triton-ascend`.
- EP pattern reference: torchtitan `torchtitan/models/common/moe.py` (`GroupedExperts`) +
  `torchtitan/models/common/token_dispatcher.py` (all-to-all dispatch), commit `ea3562e70`.

## 8. Appendix — parity tests & throughput study

**MoE parity tests (CPU, run before any NPU run):**
- `examples/ascend_npu_dflash/test_moe_grouped_gemm.py` — grouped-GEMM vs eager reference over
  `GroupedExperts`. Measured: forward diff **0.00**, backward diff **2.98e-08**.
- `examples/ascend_npu_dflash/test_moe_ep.py` — EP all-to-all dispatch: degenerate (size==1) and
  2-proc gloo. Measured: forward / grad_x diff **~1.49e-08 / 2.98e-08**.
These pin the EP + grouped-GEMM math independent of the NPU, so an NPU regression is isolatable.

**2×A2 bf16 serve throughput / KV-safety study:**
- KV cache holds **29,795 tokens**; beyond it the serve emits **HTTP-200 garbage** when
  over-batched (errors=0 ≠ quality — a silent-wrong trap).
- **Safe** = client concurrency **≤ 64** (32 seqs/replica × DP2). Clean rollout throughput
  **~0.60 rows/s** at that concurrency. `HS_DUMP` is prefill-only (`max_tokens=1`) so KV pressure
  is low; drop `MAXSEQS` to 16 for generation/eval if you see garbage.
- Rollout-optimal serve config: `MAXSEQS=32`, graph mode (`EAGER=0`, cudagraph
  `FULL_DECODE_ONLY`), `--async-scheduling`.

## 9. Validation matrix & green-check checklist

> **The point of this section.** "It runs" ≠ "it's correct." Every module needs a **correctness
> oracle** (a trusted thing to compare against), not just a smoke test. Below: what each module is
> validated against today, where we only check "runs / shapes" (🟡) or have nothing (🔴), and the
> plan to close each gap. Status legend: **✅** has a correctness oracle · **🟡** only runs /
> shape-checked (not proven correct) · **🔴** missing. Tier = where it can run: **CPU** (CI-able,
> no NPU), **NPU** (one card), **SERVE** (needs the live verifier).

### 9.1 Current status (what has a real oracle vs what doesn't)

| Module | Oracle (what it's compared against) | Tier | Status | Script |
|---|---|---|---|---|
| RMSNorm / mHC / sink-softmax / MoE block | HF `deepseek_v4`, per-component weight-copy, fp32 ≈ 0 | CPU | ✅ | `dsv4_dspark_hf_parity.py` |
| MLA attention | clean-room ref from official `inference/model.py` | CPU | ✅ | `dsv4_mla_ref.py` |
| Draft SWA + sink attention | vllm-ascend gold `_dspark_attention_reference` | NPU | ✅ | `dspark_attn_ref_bench.py` |
| mHC Sinkhorn | official tilelang kernel transcription | CPU | ✅ | `dsv4_dspark_parity.py` (A) |
| grouped-GEMM fwd/bwd | eager reference (fwd 0.00 / bwd 2.98e-8) | CPU | ✅ | `test_moe_grouped_gemm.py` |
| all-to-all dispatch | degenerate + 2-proc gloo parity | CPU | ✅ | `test_moe_ep.py` |
| Markov / Confidence heads + loss terms | formula-aligned to DeepSpec (BCE vs 1−TV, CE, L1) | NPU | 🟡 runs + formula, no numeric parity | `dspark_confidence_test.py`, `dspark_npu_op_check.py` |
| main_proj / block-γ | none in HF → fwd/bwd smoke only | CPU | 🟡 | `dsv4_dspark_parity.py` (B) |
| Full-draft numeric parity (load released weights → fwd) | planned "part C" | CPU/NPU | 🔴 **missing** | — (`dsv4_dspark_parity.py` C TODO) |
| **HS extraction — value correctness** | only layout/shape/seq today | SERVE | 🔴 **"can dump" ≠ "dumps right"** | `hs_dump_smoke.py` (format only) |
| Train loop actually learns / ckpt round-trip / serve↔train config match | — | CPU/NPU | 🔴 missing | — |

**Reading it:** the *components* almost all have a gold-standard oracle → high confidence in the
building blocks. The gaps are (a) the **assembled** draft's end-to-end numeric parity, and (b)
a handful of **silently-wrong** failure modes a smoke test can't catch (§9.3).

### 9.2 The two hard ones — how to give them a real correctness oracle

**HS extraction (the "怎么知道抽得对" question) — three oracles, cheap→gold:**
1. **Self-consistency via `lm_head` (single-machine, strong).** Extraction is teacher-forced
   prefill over known (prompt+response). So the **verifier-last** hidden (last slot of
   `hidden_states`), pushed through the target's `lm_head`, must give logits whose `argmax` hits
   the **next** teacher-forced token on the response region — matching at the model's own greedy
   rate. Near-zero / off-by-one match ⇒ we captured the wrong layer or slot (e.g. residual mixed
   in, off-by-one). This proves verifier-last end-to-end with no second machine.
2. **Cross-impl (gold).** Run the same token string through a trusted HF/CPU DSV4 forward and
   compare `[40,41,42]` + final hidden per-token (bf16 tol). Expensive; do it on a small slice.
   This also pins the aux-capture convention (post-layer, residual-folded — see
   `dsv4-aux-capture-convention`).
3. **Alignment (free).** Promote the `token_ids == rollout input_ids` check (already asserted in
   `train/data.py`) to a standalone assertion.

**Draft build correctness — two layers, do the cheap one first:**
- **Structural weight-key parity (no NPU, no forward — cheapest, must-have).** Our
  `named_parameters()` (name + shape) must match the **released DeepSeek DSV4-Flash-DSpark draft**
  keys (index at `/workspace/dspark_extract/{draft,index}.json`): every `mtp.N.attn.{wq_a,wq_b,
  wkv,wo_a,wo_b,q_norm,kv_norm,attn_sink}`, `experts.N.{w1,w2,w3}`, `shared_experts`, `gate`,
  `hc_{attn,ffn,head_*}`, `markov_head.markov_w{1,2}`, `confidence_head.proj`, `main_proj/norm`.
  A missing/extra/wrong-shape param = wrong build. **Also validates the stacked↔per-expert
  conversion (§A.2 follow-up).**
- **Numeric parity (part C).** Load the released weights into our module and compare a forward
  against the vllm-ascend inference op / an assembled reference (built from the component oracles
  above — don't wait for a nonexistent HF DSV4-DSpark forward).

### 9.3 Gates to add (silently-wrong modes a smoke test misses)

| # | Gate | Why it matters (the silent bug) | Tier |
|---|---|---|---|
| 1 | **EP-invariance** — EP=1 vs EP=2 vs EP=8, same input+seed, outputs/loss **bitwise-close** | all-to-all is data movement only; math must be invariant. Single most decisive test of the whole EP path (dispatch+grouped+combine). | CPU |
| 2 | **gradcheck on hand-written backward** — `_NpuGroupedMatmul`, `_AllToAll` (double, tiny shapes) | a wrong backward still lets loss go down — but to the wrong place. Only fwd-parity today. | CPU |
| 3 | **Overfit-one-batch** — same batch repeated, loss must drop to ≈0 | cheapest "the whole loop actually learns" signal; catches detached grad / wrong target / wrong loss. | NPU |
| 4 | **Checkpoint round-trip** — save→reload→fwd reproduces pre-save fwd; stacked↔per-expert round-trips | `check_ckpt.py` only checks integrity, not equivalence. | CPU/NPU |
| 5 | **serve↔train config guard** — assert `mask_token_id=128799`, `target_layer_ids=[40,41,42]`, `vocab`, and the **block convention** (train `--block-size 6` ⇔ served `dspark_block_size=5=γ`; drafts `block_size−1`) match between serve `--hf-overrides` and train args | a mismatch silently corrupts training — **we hit this** (mask fell back to pad=1; `--block-size 5` drafted 4 not 5). | CPU |
| 6 | **Data pipeline** — collator packing to `total_seq_len`, anchor-sample positions, `loss_mask`↔response alignment | speculators `test_data.py` only partially covers the DSV4 anchor/packing path. | CPU |

### 9.4 Green-check mechanism (three tiers)

A single `run_all_checks.sh` + pytest markers, printing a ✅/❌/⏭ table, split by run-tier
(most gates need NPU/serve, so tier the runner, don't assume one box does all):
- **CPU tier (CI-able):** structural parity · grouped/EP parity · EP-invariance · gradcheck ·
  `hf_parity` · overfit-one-batch (tiny) · config guard · data-pipeline.
- **Single-NPU tier:** `dspark_npu_op_check` · `dspark_attn_ref_bench` · `dspark_confidence_test`
  · overfit-one-batch (real).
- **Needs-serve tier:** HS self-consistency + cross-impl · `hs_dump_smoke` (format).

Status is recorded from **box runs** (the sandbox has no torch/NPU) — tick the checklist below as
each is confirmed on the box.

### 9.5 Checklist (tick as confirmed on box)

Priority order = cheap + catches big bugs first.

- [ ] **Structural weight-key parity** vs released DSpark draft (CPU) — also covers per-expert conversion
- [ ] **EP-invariance** EP=1 vs EP=N bitwise-close (CPU)
- [ ] **Overfit-one-batch** loss→≈0 (NPU)
- [ ] **HS self-consistency** verifier-last `@lm_head` argmax → next token (SERVE)
- [ ] **serve↔train config guard** (CPU)
- [ ] **gradcheck** grouped-GEMM + all-to-all backward (CPU)
- [ ] **Checkpoint round-trip** equivalence + stacked↔per-expert (CPU/NPU)
- [ ] **HS cross-impl** vs trusted HF/CPU DSV4 forward on a slice (SERVE)
- [ ] **Full-draft numeric parity** (`dsv4_dspark_parity.py` part C) (CPU/NPU)
- [ ] Data-pipeline packing / anchor / loss_mask alignment (CPU)
- [x] grouped-GEMM fwd/bwd parity · all-to-all parity · HF component parity · MLA ref · draft-attn gold · mHC Sinkhorn *(already green — §9.1)*

## 10. Evidence & provenance (every load-bearing claim → reproducible source)

> So a reviewer can check us, not take our word. Source tags: **[repo]** = this repo
> (`Sawyer117/speculators @ feat/dsv4-dspark`); **[va]** = `Sawyer117/vllm-ascend` (branch noted);
> **[HF]** = a released HuggingFace checkpoint's `config.json` / `model.safetensors.index.json`;
> **[paper]** = arXiv; **[blog]** = the vLLM V4 blog. DSpark claims cite DSpark/DSV4 code directly
> (not the DFlash base) except where a behavior is genuinely inherited (noted as such).

**A. Draft architecture (3 mtp layers = MLA + sink + MoE + mHC; NO DSA).**
- Draft = MLA + sink + MoE(256) + mHC, no compressor/indexer: **[HF]** released
  `deepseek-ai/DeepSeek-V4-Flash-DSpark` `model.safetensors.index.json` draft-shard keys —
  present: `mtp.N.attn.{wq_a,wq_b,q_norm,wkv,kv_norm,wo_a,wo_b,attn_sink}`, `mtp.N.ffn.experts.{0..255}`,
  `shared_experts`, `gate`, `mtp.N.hc_{attn,ffn}_{base,fn,scale}`; **absent**: any `compressor`/`indexer`
  key (those exist only under target `layers.N.attn.*`). **[paper]** arXiv:2607.05147: *"the parallel
  backbone comprises three MoE layers with mHC and a sliding window attention of 128"*, *"All positions
  within a block attend bidirectionally to each other and to the injected target context"* — no DSA
  mentioned for the draft.
- Head placement — `main_proj`/`main_norm` on **mtp.0**, `markov_head`(w1/w2)/`confidence_head`/`hc_head`/`norm`
  on **mtp.2**: **[HF]** draft keys (each head appears exactly once, at that mtp index).
- Draft input = anchor + γ mask embeddings; target context injected via K/V: **[paper]** arXiv:2607.05147
  Eq. 2–3; **[HF]** `dspark_noise_token_id=128799`, `dspark_target_layer_ids=[40,41,42]`.

**B. Target architecture (DeepSeek-V4-Flash: MLA + sink + SWA + CSA/HCA + MoE + mHC).**
- 43 layers, hidden 4096, 256 experts top-6, sliding_window 128, index_topk 512, mHC Sinkhorn 20:
  **[HF]** `deepseek-ai/DeepSeek-V4-Flash/config.json` (`num_hidden_layers=43`, `hidden_size=4096`,
  `n_routed_experts=256`, `num_experts_per_tok=6`, `sliding_window=128`, `index_topk=512`,
  `hc_sinkhorn_iters=20`, `vocab_size=129280`).
- Per-layer attention tier via `compress_ratios` (`{0:3, 4:21, 128:20}`): **[HF]** config
  `compress_ratios`. Ratio 4 = CSA/"c4a", 128 = HCA/"c128a", 0 = dense/SWA: **[blog]**
  vllm.ai/blog/2026-04-24-deepseek-v4 (c4a = ÷4 top-512, c128a = ÷128); **[paper]** arXiv:2606.19348.
- "SWA-only layer for V4" + lightning indexer: **[va]** `dspark-dsv4 dsa_v1.py:1108`
  (`"vLLM-Ascend only support SWA-layer for Deepseek-V4 now."`), `:1440-1458` (indexer/compressor).

**C. block_size / num_spec / accept_len (the off-by-one that cost a run).**
- Training block includes the anchor; drafts `block_size − 1` (DSpark's own code): **[repo]**
  `models/dsv4_dspark/core.py:135` (*"position 0 is the anchor"*), `:343` (`mask_token_ids[:, ::block_size] = anchor`),
  `:397` (`aligned_loss_mask[:, ::block_size] = 0`); `models/dspark/metrics.py:85,88` (`[:, 1:]`), `:152`
  (`for pos in range(1, block_size)`).
- Shipped proposal config `speculative_tokens = block_size − 1` (inherited, `DSV4DSparkDraftModel ←
  DSparkDraftModel ← DFlashDraftModel`, not overridden): **[repo]** `models/dflash/core.py:186-188`.
- Inference `n_predict = dspark_block_size`, **no −1**: **[va]** `patch/platform/patch_speculative_config.py:21`;
  test `tests/ut/spec_decode/test_dspark_config.py:36` (`assert patched.n_predict == 5`);
  `spec_decode/dspark_proposer.py:421` (`block_size = self.num_speculative_tokens`).
- DSV4 draft counts: **[HF]** `deepseek-ai/DeepSeek-V4-Flash-DSpark/config.json` `dspark_block_size=5`.
- Qwen3 cross-check (train 8 ⇔ release 7): **[repo]** `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh:36`
  (`BLOCK_SIZE=8`); **[HF]** `deepseek-ai/dspark_qwen3_4b_block7/config.json` `block_size=7`.
- accept_len = `1 + accepted/drafts`, ceiling `num_spec + 1 = 6`: **[repo]**
  `examples/ascend_npu_dflash/Evaluator.py:599`, `scripts/evaluate/perf_utils.py:344`.

**D. HS extraction (why Plan B over native extract).**
- Native extract KV pathology (CacheOnly cache co-sized with real KV): **[repo]**
  `docs/deployment/ascend-npu-dsv4-hs-dumper-planB.md` §1; upstream vLLM `extract_hidden_states.py:132,363-365`.
- 0.23.0 crash `'list' has no attribute 'device'`: `extract_hidden_states.py:72` (planB doc §0).
- PD-disagg vs Ascend balance-scheduling: **[repo]** `serve_dsv4_bf16_dualnode.sh:159-162`.
- Plan-B buffer already captured by the DSV4 forward (gated on `dspark_target_layer_ids`): **[va]**
  `dspark-dsv4 models/deepseek_v4.py:1190,1236-1242` (`_dspark_hidden_buffer`), `get_mtp_target_hidden_states()`.
- Dumper hook: **[va]** `feat/dsv4-hs-dumper vllm_ascend/dspark_hs_dumper.py` + `worker/model_runner_v1.py`;
  perm fix `@ 4677f0b`.

**E. Serve EP OFF (2-node A2) + KV-safety.**
- Cross-node EP16 `HcclAllGather` deadlock → EP OFF: **[repo]** `serve_dsv4_bf16_dualnode.sh:11-15,73-79`
  (+ AtomGit A2 two-node V4-Flash report cited there).
- KV holds 29,795 tok, garbage beyond concurrency 64: our serve plog / benchmark doc
  `docs/deployment/ascend-npu-dsv4-bf16-dualnode-benchmark.md`.

**F. Accept-length baseline (the bar to beat).**
- Released DSV4 draft AR 58.79% / AL 3.94 / per-pos `[0.81,0.68,0.58,0.48,0.39]`, GPU ref 3.86:
  **[va]** vllm-ascend **PR #11196** (QwertyJack) measurements.
- ⚠ Qwen3-4B `6.189/…` are a **different** model (block7/num_spec7/full-attn): **[HF]**
  `dspark_qwen3_4b_block7` (do not conflate — memory `dsv4-dspark-accept-baseline`).

**G. Key config values (must match serve).**
- `dspark_target_layer_ids=[40,41,42]`, `dspark_noise_token_id=128799`, `dspark_block_size=5`,
  `dspark_markov_rank=256`, `vocab=129280`: **[HF]** `DeepSeek-V4-Flash-DSpark/config.json`; mirrored in
  **[repo]** `models/dsv4_dspark/config.py:37,43-46`.

**H. EP internals (measured / code).**
- Grouped-GEMM + all-to-all parity numbers: **[repo]** `examples/ascend_npu_dflash/test_moe_grouped_gemm.py`,
  `test_moe_ep.py` (§8). `Shard(0)` DTensor conversion: `src/speculators/train/distributed.py:237`.
- Memory scaling (`max_anchors × block`): measured, memory `dsv4-dspark-fsdp2-memory-estimate`
  (block5@256=1280tok=59.82 GB; block6@256 OOM; block6@196 runs).
