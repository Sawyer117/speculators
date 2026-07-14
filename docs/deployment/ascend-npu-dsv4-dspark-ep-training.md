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
| Optimizer / LR | **AdamW, lr 6e-4** | DeepSpec `dspark_qwen3_4b` reference (the script's old 2e-4 default was 3× low). Single AdamW over uniform DTensors (§3). Muon available but AdamW chosen for this run. |
| Grouped-GEMM op | `torch_npu.npu_grouped_matmul` | Extracted from MindSpeed; one grouped matmul per projection replaces the 256-way expert loop + kills per-shape recompiles. Self-written autograd backward, CPU + NPU parity-tested (§8). |
| Draft attention | **SDPA** (`--draft-attn-impl sdpa`) | Ascend has no `simple_flex_attention`. The non-causal-sink SWA fused kernel is a separate in-progress optimization (see §5, §7 handoff repo). |
| Rollout sampling | **greedy, temp=0 end-to-end** | Gen / train / eval all temp=0 → self-consistent (user's call). DeepSpec uses 0.7/top-p0.8; DSV4 official rec 1.0. Tripwire if ever benchmarked sampled. |

## 5. Known follow-ups / not-yet-optimized

- **`argsort` int64 → AiCPU fallback** in the MoE dispatch (the recurring `[ArgSort] running on
  AiCpu` warning). Fix: cast the sort key to float32 (values 0–255 exact; drop `stable`,
  correctness holds via the inverse permutation) → runs on AiCore. *(Offered, not yet applied.)*
- **Fused MoE permute** (`npu_moe_init_routing`) — fuses sort+permute around the all-to-all,
  supersedes the argsort fix.
- **Fixed-shape MoE padding** — pad per-expert token counts to fixed buckets → static
  grouped-GEMM shapes → eliminates the residual ~7–14 s new-shape recompile spikes.
- **Gradient checkpointing** — needed to reach the paper's 512 anchors (memory-bound at 256).
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

**Acceptance-length target (baseline, NOT yet the converged draft).** The DSV4 DSpark draft is
**`block_size = 5` → `num_speculative_tokens = 5`** (not 7 — that's the Qwen3 line; see note
below). The last training run was still LR-warming (accept_len ≈ 1.19, out of a block-5 ceiling
of ~6); a converged draft has not yet been trained-to-eval on this EP stack.

The bar to match/beat is the **released DeepSeek DSV4-Flash DSpark draft**, measured on NPU in
**vllm-ascend PR #11196** (QwertyJack) at `num_spec=5`:

| metric | released DSV4-Flash DSpark (PR #11196) |
|---|---|
| acceptance rate (AR) | **58.79%** |
| **accept length (AL)** | **3.94** (GPU reference: 3.86) |
| per-position accept | `[0.81, 0.68, 0.58, 0.48, 0.39]` (AL = 1 + Σ = 3.94) |

Fill our trained draft's per-dataset numbers here after a full train→eval pass at `num_spec=5`.
**eval /metrics counter resets mid-run** on vllm-ascend spec_decode — use the reset-aware poller
(`Evaluator.py @ a3c41a6`), or non-first-dataset accept lengths read low.

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
— `SEQLEN=3072 MAX_ANCHORS=256 LR=6e-4 DSPARK_EP=1 bash ... faithful`. Modes: `reduced`
(1L×32E smoke) / `faithful` (3L×256E, EP8). Env knobs: `SEQLEN MAX_ANCHORS LR MASK_TOKEN
CKPT_FREQ DSPARK_EP NPROC RUN SAVE_PATH VERIFIER DATA HS_DIR ENDPOINT CANN_ENV`. Backgrounds a
torchrun run; prints the tail command.

**Rollout (data gen):** `rollout_a3_shard.sh` / `rollout_shard.sh` (greedy, temp=0; prep →
Arrow). **HS-dump smoke:** `hs_dump_smoke.py`. **Eval:** `run_eval.sh` (background serve +
curl-wait + eval in ONE terminal).

**Repos / external:**
- Draft trainer fork: `https://github.com/Sawyer117/speculators` (branch `feat/dsv4-dspark`,
  HEAD `a855bfb`).
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
