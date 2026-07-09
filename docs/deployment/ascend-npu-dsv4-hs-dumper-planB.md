# Plan B — DSpark-buffer HS dumper for DSV4 (memory-light, no HiddenStateCacheSpec)

**Status: DECIDED & SCAFFOLDED (2026-07-09).** This is the chosen path for producing DSV4
DSpark HS training data NOW. It rides the **validated dspark serve** and adds a small
**dumper** that copies the already-captured target hidden states out to disk in the
speculators format — moving the "hold until request-end" cost from **NPU** (where it
fights the KV budget) to **CPU RAM** (abundant).

## 0. Decision log & branch map (READ FIRST — how to resume either track)

Two parallel tracks. **Plan B = data now; 0.24-extract = durable upstream path, deferred.**

| Track | Branch (vllm-ascend, `Sawyer117/vllm-ascend`) | Base | Status | vLLM |
|---|---|---|---|---|
| **Plan B (THIS doc)** | `feat/dsv4-hs-dumper` | `dspark-dsv4` @ `6036507` | scaffolded; P1 next | 0.23.0 (validated) |
| **Extract / SupportsEagle3** | `feat/dsv4-supports-eagle3` @ `f7a1e25` | `dspark-dsv4` | **PAUSED** (see below) | needs 0.24 |

**Why Plan B now (the decision, 2026-07-09):**
- **Nobody has run DSV4 HS extraction end-to-end.** Cross-checked Mohammad (MohammadMahdi1375
  / `vLLM_NPU`): his extract path is validated on **Qwen3-8B (GQA)**, not DSV4; his DSV4 work is
  a separate track still in DSA decode-correctness debug (`--enforce-eager`, `max-num-seqs 1`).
- **The `extract` crash (`'list' object has no attribute 'device'` at vLLM-core
  `extract_hidden_states.py:72`) is a vLLM-version issue, not a patchable bug.** Mohammad runs
  vLLM **main / 0.24-dev** (`ae7c8ec22`, 2026-06-25), where the CacheOnly path was refactored
  (`kv_cache` is a per-layer tensor via `forward_context.slot_mapping` dict). Our base is
  vLLM **0.23.0** (release, 2026-06-14) where it arrives as a `list` → crash. Our vllm-ascend
  is version-locked to 0.23.0 (dozens of `vllm_version_is("0.23.0")` branches; 0.24 also flips
  v1→**v2 model runner**). So "go to latest vLLM" = a serve **re-validation project**, not a bump.
- **The DSV4 memory pathology (`HiddenStateCacheSpec` replicated per rank) is real & unrefuted**
  (see §1); `f7a1e25`'s planner surgery only fit DP1, DP0/head still OOM'd.
- **Plan B sidesteps ALL of the above** and reuses a capture we now know is correct (next bullet).

**Aux-capture correctness — SETTLED by DeepSpec canonical (do not re-litigate):**
`deepseek-ai/DeepSpec` `scripts/data/prepare_target_cache.py` captures via
`register_forward_hook` on each decoder LAYER, taking the layer's **output** = **POST-layer,
residual-fully-included**, catted over `target_layer_ids`; `target_last_hidden_states =
last_hidden_state` (final post-norm). The `dspark-dsv4` #11571 buffer capture
(`deepseek_v4.py:1240-1242`, `hidden_states.mean(dim=1)` AFTER the layer, no `+residual`,
mean over the mHC `hc_mult`) is **byte-consistent** with this. **Do NOT add `+residual`**:
DSV4's `hc_post` folds the residual internally, so the returned `hidden_states` already IS
the residual stream and the returned `residual` is a vestigial post-attn snapshot. (Mohammad's
DSV4 capture uses PRE-layer `hidden_states + residual` — correct for standard-decoupled Qwen3,
a latent double-count on DSV4; relay back to him.) See memory `dsv4-aux-capture-convention`.

**To resume the PAUSED extract/0.24 refactor track:** check out `feat/dsv4-supports-eagle3`
(`f7a1e25` — SupportsEagle3 model iface + one-flow refactor + KV-cache-grouping passthrough
surgery). It needs: (1) vLLM 0.24 (for the CacheOnly `list.device` fix + v2 runner) and its
vllm-ascend port; (2) the memory pathology still solved for DP0/head. ⚠️ `f7a1e25` may not be
on `origin` yet — push it before relying on it.

## 1. Why not the standard extract path (the memory pathology)

`method=extract_hidden_states` implements a fake `CacheOnlyAttentionLayer` whose "KV cache"
IS the hidden-state store: `num_heads = num_hidden_states (=L_aux=3)`, `head_size = hidden_size
(=H=4096)`, no k/v ×2 (`extract_hidden_states.py:132,363-365`). So it allocates a **block-based
KV cache co-sized with the real KV pool**:

```
s_hidden(per token) = L_aux · H · d        (model-INDEPENDENT, uncompressed)  = 3·4096·2 = 24 KB
S_hidden            = N_blocks · block_size · L_aux · H · d                    = N · 3 MB
```

DSV4's real KV is hyper-compressed (MLA single shared latent (c+r)=576, no ×2, further ÷ DSA
compress_ratio 4/128, + sliding windows), so `s_hidden` (fixed) becomes ~1/3–1/2 of the KV
budget → OOM / squeezed KV. On GQA/MHA (Mohammad's Qwen3) the same `s_hidden` is ~5-10% → a
non-issue. **DSV4 is the worst case for extract; Mohammad's code is identical, only his model's
KV is big.** No knob fixes this (it's `L_aux·H` uncompressed by construction).

## 2. Output contract (speculators v1 — `train/data.py:96-114`)

Each `hs_<id>.safetensors` (v1 format consumed by `standardize_data_v1`) carries:

- `input_ids`  : `[seq]`  (the sequence's token ids)
- `loss_mask`  : `[seq]`  (1 on tokens to train, e.g. response; 0 on prompt)
- `hidden_states` : a **per-layer list of `[seq, H]`** — the FIRST N are the target layers
  ([40,41,42]); the **LAST is the verifier-last = the target's final post-norm hidden** (the
  one lm_head consumes). `standardize_data_v1` then does
  `hidden_states = cat(list[:-1], dim=-1)` → `[seq, (N-1)·H]` and `verifier_last = list[-1]`.

⇒ The dumper must capture **TWO things**: the target layers [40,41,42] AND the final hidden.

## 3. Carrier serve (validated, memory-light)

Reuse the **dspark path** (proven on DSV4 by colleagues; no `HiddenStateCacheSpec`, so no KV
grouping/OOM problem). The target captures [40,41,42] (mean over mHC hc_mult) into the small
NPU scratch buffer `_dspark_hidden_buffer` `[max_num_batched_tokens, L_aux·H]` (~200 MB, over-
written each forward), exposed via `ForCausalLM.get_mtp_target_hidden_states()`.

- **VERIFY (step 0):** does the buffer capture fire in a *plain* serve (if the DSV4 target
  config carries `dspark_target_layer_ids`), or do we need `method=dspark` (loads the DSpark
  draft, whose own KV is separated via `get_draft_kv_cache_layer_names`, also fine)? Prefer
  plain serve (no draft ckpt) if the config triggers it; else `method=dspark`.
- Base branch = the **buffer-carrying** dspark branch (author's `#11571` / original
  `dspark-dsv4`), NOT our `feat/dsv4-supports-eagle3` (which retired the buffer for standard aux).

## 4. What to capture, and where the final hidden comes from

- Target [40,41,42]  ← `get_mtp_target_hidden_states()` (the dspark buffer), `[num_tokens, L_aux·H]`.
- Verifier-last (final hidden) ← the model's post-norm forward output (`DeepseekV4Model.forward`
  returns `self.norm(...)`), i.e. the `hidden_states` the runner already holds pre-lm_head,
  `[num_tokens, H]`. (Do NOT use layer-43 residual as verifier-last — it's pre-hc_head, not the
  final normed hidden.)

## 5. Dumper architecture

Two options; **recommend B** (avoids kv_transfer / PD-disaggregation, which is part of what we're
escaping):

**B — runner post-forward hook (recommended).** A vllm-ascend patch/callback that, after each
`execute_model`:
1. Async-copies `get_mtp_target_hidden_states()[:num_scheduled_tokens]` and the final
   `hidden_states[:num_scheduled_tokens]` NPU→CPU.
2. Routes each token to its request via the batch metadata (`input_batch.req_ids`,
   positions, `num_scheduled_tokens` per req).
3. Appends to a **per-request CPU accumulator** (dict req_id → growing `[seq, L_aux·H]` +
   `[seq, H]` + input_ids + loss_mask).
4. On request finish (runner marks it done), writes `hs_<req_id>.safetensors` to the shared
   path in the §2 layout, then drops the accumulator entry.

- Pro: no kv_transfer/PD-disagg, no HiddenStateCacheSpec, direct. Memory in CPU RAM.
- Where to hook: the same place model_runner_v1 already reads `get_mtp_target_hidden_states`
  (~L1835) for the MTP path — we know it's valid there.

**A — custom kv_connector (alt, more "standard" but more work).** A `DsparkHiddenStatesConnector`
mirroring `ExampleHiddenStatesConnector`'s per-request accumulate + save + `kv_transfer_params`
lifecycle, but sourcing hiddens from the dspark buffer instead of the cache_only_layers cache.
Reuses the connector plumbing; but the connector is welded to the HiddenStateCacheSpec today, so
rewiring its data source is non-trivial, and kv_producer re-enables PD-disagg paths we'd rather avoid.

## 5b. TP/DP awareness — write from TP-rank-0 (standard), do NOT shard

Clarification (corrected): the WRITE is already single-rank in the standard connector —
`ExampleHiddenStatesConnector` gates all disk I/O to TP rank 0 (`:161` "Only TP rank 0 writes …;
other TP ranks no-op", `:254` `get_tensor_model_parallel_rank()==0`). So there is **no 8×
write redundancy** to fix; it's day-1. The extract path's only pathology is the **replicated
per-rank CACHE** (each rank allocates a full HiddenStateCacheSpec; `get_hidden_size()` with no
TP division) — a MEMORY issue, orthogonal to the write.

Plan B follows the same, and this resolves the shard-vs-single-writer tension:

- The residual-stream hidden is **TP-replicated** (identical on every TP rank; rank 0 already
  holds the FULL `[L_aux·H]`). So **rank 0 writes the full tensor** — 1 writer, complete data,
  optimal I/O. **Do NOT TP-shard the HS.**
- Sharding (each rank stores 1/TP of `H`) would only be to shrink a big *cache*, and it would
  FORCE all TP ranks to write their slice (or an all-gather before write) — the opposite of
  single-writer. Plan B has no big cache (just the ~200 MB scratch), so there is nothing to
  shrink → sharding is pure downside here.
- Per **DP replica**: its own TP-rank-0 writes its own requests to the shared path (data-parallel
  → disjoint requests). 2 DP replicas → 2 writers, disjoint `hs_<i>`. (This is the same gating the
  connector already uses — plan B just reuses it, not a new optimization.)

## 6. Memory analysis (the whole point)

- NPU: only the ~200 MB dspark scratch (already there). **Zero extra NPU KV** — no
  HiddenStateCacheSpec.
- CPU RAM: per-request accumulator = concurrent_reqs × seq_len × (L_aux·H + H) × d. E.g. 32 reqs
  × 2048 tok × (4·4096) × 2 B ≈ **2 GB** CPU (box has ~1.4 TB). Negligible.
- I/O: one safetensors write per request at completion (same as the standard connector).

## 7. Teacher-forced regime (how we actually drive it)

Rollout is pre-generated (prompt+response). HS extraction = a **prefill-only forward over known
sequences** (regime A). Send each (prompt+response) as a completion with `max_tokens=1`; the
prefill computes all tokens' [40,41,42] + final hidden; the dumper writes them keyed by request.
`loss_mask` = 0 on prompt tokens, 1 on response tokens (from the request's prompt/response split).

## 7b. Granularity & length (derived from `train/data.py`, NOT a choice)

- **One `hs_<id>.safetensors` = one rollout sequence** (natural length). `ArrowDataset` does
  `file_list = list_files(datapath)` (data.py:57) — one file = one sample — and `__getitem__`
  loads one file then `slice_and_pad_to_length(t, max_len)` (data.py:44,160,175). So the TRAINER
  owns length (slice/pad to `max_len`); the producer must NOT pre-chunk to training length.
- Streaming ("produce one, consume one") is per-FILE via the rolling buffer
  (`on_missing="generate"`, `on_generate="delete"`, data.py:276-277).
- Align the serve's `max_model_len` ≈ trainer `max_len` so sequences fall within the training
  length and we don't dump tails the trainer truncates away (I/O waste), — an alignment, not a chunk.
- **verifier-last is confirmed = the target's FINAL post-norm hidden** (`self.norm(...)` output),
  used by the draft's `verifier_norm`+head to reconstruct the target distribution for the loss
  (speculators `dflash/core.py:349`, `mtp/core.py:97` "MTP only uses the last hidden layer").
  NOT layer-43 residual, and NOT derivable from #11571 (that's inference; verifier-last is a
  TRAINING input defined on the speculators side).
- **File-naming convention — DERIVED from code, model-agnostic (no choice):** the trainer reads
  `hidden_states_path / f"hs_{file_idx}.safetensors"` where `file_idx = index + start_file_idx`
  and `index` = row in the rollout dataset (`data.py:293,324-325,396`). So each file is
  `hs_<rollout_row_index>.safetensors` in `hidden_states_path` (default `datapath/hidden_states`).
  The extraction **client tags each request with its rollout row index** (via the per-request
  `kv_transfer_params.hidden_states_path`, exactly like the standard offline example) → the dumper
  writes `hs_<i>.safetensors`. Naming aligns to the trainer automatically; nothing to ask.
  speculators' DATA layer is model-agnostic — DSV4 needs no data-side support; only the DRAFT
  model side (our dspark_method) is DSV4-specific.

## 8. Open decisions before coding

1. Plain serve vs `method=dspark` for the capture (step 0 verify).
2. Hook (B) vs connector (A) — recommend B.
3. Exact on-disk safetensors key layout — match `ExampleHiddenStatesConnector` /
   `_maybe_load_hs_file` so `standardize_data_v1` reads it unchanged (verify keys: does it store
   `hidden_states` as a stacked `[seq, N, H]` or per-key? + `input_ids`/`loss_mask`).
4. Where `loss_mask` + the prompt/response split come from (request metadata vs a sidecar map).

## 9. Phased plan

- **P0 — RESOLVED (from code).** Capture trigger: the `_dspark_hidden_buffer` is filled in
  `DeepseekV4Model.forward` gated ONLY on `config.dspark_target_layer_ids` (`deepseek_v4.py:1190,
  1236-1242`), **independent of the speculative method** → a **plain serve** populates it (no draft
  ckpt needed) as long as the target config carries `dspark_target_layer_ids`. On-disk format: the
  reader (`speculators/train/data.py:175-206`) loads the **standardized 4-key** layout DIRECTLY
  (`hidden_states` `[seq, L_aux·H]`, `verifier_last_hidden_states` `[seq, H]`, `input_ids`,
  `loss_mask`); `standardize_data_v1` is NOT applied on file load, so the dumper writes those 4 keys.
- **P1 — DONE** (branch `feat/dsv4-hs-dumper`, commit `0ca26de`):
  - `vllm_ascend/dspark_hs_dumper.py` — `DsparkHSDumper`: TP-rank-0 writer, per-request CPU
    accumulate (timing-independent flush when accumulated len ≥ `num_prompt_tokens`), atomic
    tmp+rename write of the standardized 4-key safetensors.
  - `vllm_ascend/worker/model_runner_v1.py` — lazy-init + `capture(...)` hook in `execute_model`
    post-process (after aux/pcp handling, last PP rank only), sourcing the aux from
    `get_mtp_target_hidden_states()` and the verifier-last from the post-norm forward output.
  - **Enable on the serve:** `DSPARK_HS_DUMP=1 DSPARK_HS_DIR=/share/.../hidden_states`, plus ensure
    the DSV4 target config has `dspark_target_layer_ids=[40,41,42]` (inject via `--hf-overrides`
    if absent). Client drives prefill-only (`max_tokens=1`) and sets each request id to its rollout
    row index → `hs_<index>.safetensors`.
- **P2 — smoke (on box, NEXT):** plain dspark serve + `DSPARK_HS_DUMP=1`; send 1–2 teacher-forced
  requests; confirm `hs_<id>.safetensors` written with shapes `[seq, 3·4096]` + `[seq, 4096]` and
  load back through `ArrowDataset.__getitem__`. Watch that `max_num_batched_tokens ≥ max_model_len`
  so each sequence prefills in one chunk (else the accumulator spans chunks — supported, but verify).
- **P3 — full rollout:** drive the whole rollout (prefill-only), refine `loss_mask` (prompt/response
  split, §8.4), watch CPU RAM + write throughput; wire the `hs_sidecar.py` HTTP fetch if the trainer
  box has no shared FS to the serve box.
