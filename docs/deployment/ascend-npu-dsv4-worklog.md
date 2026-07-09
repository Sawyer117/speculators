# DSV4 DSpark on Ascend — worklog (exhaustive)

Chronological, warts-and-all record of the DSV4 DSpark draft-training effort on Ascend NPU:
decisions, dead-ends, cross-checks, numbers, commits. This is the **raw material to distill a
report from** — it captures *what happened and why*, including paths that failed. The structured
how-to lives in [`ascend-npu-dsv4-rollout-data.md`](ascend-npu-dsv4-rollout-data.md) and
[`ascend-npu-dsv4-hs-dumper-planB.md`](ascend-npu-dsv4-hs-dumper-planB.md). Append new entries at the
bottom; don't rewrite history.

**Goal:** reproduce DeepSeek DSpark speculative-decoding draft **training** on DSV4-Flash (bf16) on
Ascend NPU — produce hidden-state training data and train the DSpark draft. Sampling greedy (temp=0)
end-to-end.

**Branches (Sawyer117):** vllm-ascend `feat/dsv4-hs-dumper` (Plan B dumper) | speculators
`feat/dspark-confidence-head` (serve/smoke/driver/train adapter/fixes) | paused extract track
vllm-ascend `feat/dsv4-supports-eagle3@f7a1e25`.

---

## 2026-07-09 — HS extraction: extract path attempt → cross-checks → Plan B decision

### Extract path (standard `extract_hidden_states`) — attempted, then PAUSED
- Added `SupportsEagle3` to vllm-ascend `AscendDeepseekV4ForCausalLM` (`feat/dsv4-supports-eagle3`):
  model returns `(hidden, aux_hidden_states)` at aux layers [40,41,42]. Passed the EAGLE3 gate.
- Then broke **5× in a row** on DSV4/Ascend: `Model does not support EAGLE3` (fixed) → KV-cache-grouping
  `AssertionError` (HiddenStateCacheSpec is a `MLAAttentionSpec` subclass, mis-grouped) → planner OOM
  (13.41 GiB) → passthrough tensor sizing → finally vLLM-core `extract_hidden_states.py:72
  'list' object has no attribute 'device'` in dummy_run. Commits 9cfdb75…`f7a1e25`.
- **Root cause of the OOM (formula-verified):** `extract` fakes a `CacheOnlyAttentionLayer` whose KV
  cache IS the hidden store — `num_heads=L_aux=3`, `head_size=H=4096`, **no TP division, replicated per
  rank**. Per token = `L_aux·H·2 = 24 KB`, block-based, context-sized. DSV4's real KV is hyper-compressed
  (MLA latent 576, /TP-sharded, ÷DSA) so this fixed uncompressed replicated cache dwarfs it → ~1/3–1/2 of
  per-card budget → OOM. On GQA (Qwen3) the same code is ~5-10% = non-issue. **No knob fixes it.**

### Cross-check: Mohammad (MohammadMahdi1375 / vLLM_NPU)
- **No one has run DSV4 HS extraction end-to-end.** His extract is validated only on **Qwen3-8B (GQA)**
  (e2e test `MODEL="Qwen/Qwen3-8B"`); his DSV4 work is a separate track still in DSA decode-correctness
  debug (`debug/replicate-numheads`, `--enforce-eager`, `max-num-seqs 1`).
- The `'list' has no attribute device` crash is a **vLLM-version** issue, not a patchable bug: he runs
  vLLM **main/0.24-dev** (`ae7c8ec22`, 2026-06-25 — CacheOnly refactored to a per-layer tensor via
  `forward_context.slot_mapping`); we run vLLM **0.23.0** (`kv_cache` arrives as a `list`). Our
  vllm-ascend is version-locked to 0.23.0 (many `vllm_version_is("0.23.0")` branches; 0.24 also flips
  v1→v2 model runner) → "go latest vLLM" = a serve **re-validation project**, not a bump.

### Cross-check: DeepSpec (canonical DSpark) — settles the aux-capture convention
- `deepseek-ai/DeepSpec` `prepare_target_cache.py`: `register_forward_hook` on each decoder LAYER,
  captures its **output** = **POST-layer, residual-fully-included**, catted over `target_layer_ids`;
  `target_last_hidden_states = last_hidden_state` (final post-norm). `loss_mask` is **passed through from
  the dataloader**, NOT computed in the forward.
- DSV4 specifics: the layer's `hc_post` op **folds the residual internally** (proof: the passed-in
  `residual` is overwritten on entry / never used; the model tail is `self.norm(hidden_states)` with no
  residual). So the returned `hidden_states` already IS the residual stream. ⇒ our capture
  `hidden_states.mean(dim=1)` post-layer (no `+residual`) is **correct**; the `dspark-dsv4` #11571
  buffer already does exactly this. Mohammad's DSV4 capture uses PRE-layer `hidden_states + residual` —
  right for Qwen3, a **latent double-count on DSV4** (never triggered; he never ran DSV4 extract).
- I initially (wrongly) claimed "we're missing +residual" from a surface diff; DeepSpec + the hc_post
  trace overturned it. Lesson: trace the residual op before trusting a cross-repo capture diff.

### DECISION (user): Plan B now for data + 0.24-extract deferred in parallel
- Plan B = ride the **validated dspark serve** (0.23.0, #11571 buffer intact) + a small **runner-hook
  dumper** that copies the already-captured target hidden + post-norm final to CPU and writes speculators
  files. Sidesteps the extract memory pathology, the KV-grouping surgery, and the vLLM-core crash.
- Extract track (`f7a1e25`) parked; resume needs vLLM 0.24 + v2-runner migration + the memory pathology.

---

## 2026-07-09/10 — Plan B build → P2 smoke PASS → online-only

- **Branch** `feat/dsv4-hs-dumper` off `dspark-dsv4@6036507` (validated serve; NOT the extract branch).
- **Dumper** (`vllm_ascend/dspark_hs_dumper.py`, `0ca26de`): TP-rank-0 writer, per-request CPU accumulate
  (flush when accumulated len ≥ `num_prompt_tokens`), atomic tmp+rename. Hook in `model_runner_v1.py`
  `execute_model` post-process (last PP rank), sourcing aux from `get_mtp_target_hidden_states()` (the
  #11571 buffer) + verifier-last from the post-norm forward output.
- **Format correction** (`cf06451`): first wrote the deprecated `SampleFileDataset` 4-key format; switched
  to the **standard `ArrowDataset` / extract-connector format** `{token_ids:[seq], hidden_states:[seq,
  num_aux+1, H]}` (aux layers + verifier-last as last). This is why **`loss_mask` needs no producer work**
  — ArrowDataset takes it from the paired rollout dataset. (`SampleFileDataset` = `--legacy-data`,
  deprecated; we don't maintain offline.)
- **`_stem` fix** (`381bd0e`): file named `hs_<idx>` via regex `hs_\d+` (vLLM wraps X-Request-Id as
  `cmpl-hs_<idx>`).
- **Serve opt-in** (`serve_dsv4_bf16_dualnode.sh` `HS_DUMP=1`, `585c5a8`): plain serve + `DSPARK_HS_DUMP=1`
  + `DSPARK_HS_DIR` + `--hf-overrides '{"dspark_target_layer_ids":[40,41,42]}'` (buffer fires on plain
  serve, no draft ckpt). Mutually exclusive with `HS_EXTRACT`.
- **P2 smoke PASS** (A2 dualnode 115/116, `hs_dump_smoke.py`): `hidden_states=(16, 4, 4096)` (aux+verifier),
  `token_ids [seq]`, bf16. hs_0/hs_1 named cleanly (regex works). Junk `hs_chatcmpl-…` from the serve's
  built-in 数数 smoke are harmless (trainer reads only `hs_<idx>`).
- **Throughput bench** (`hs_dump_driver.py --bench`, `423f7b6`): **3436 tok/s** (conc 32, 1024-len,
  prefill + NPU→CPU + disk write, end-to-end). Sizing: **32 KiB/token** (4×4096×2). A 117 MB/s HTTP link
  carries ~3600 tok/s → extraction is below the threshold = **HTTP-hidden** (moot: /share is shared).
  Per-shard offline全量 would be ~2.8–5.7 TB (88.8k × ~1–2k tok × 32 KiB) — why online-only.
- **Online-only** (user: "我们永远不走offline"): offline `hs_dump_driver.py` kept as a probe/bench only.
- **Online rolling adapter** (`data.py`, `8638d07`): `DSPARK_HS_DUMP=1` makes `ArrowDataset._maybe_generate_hs`
  drive our serve (`X-Request-Id=hs_<file_idx>` + poll for the file) instead of the connector; standard
  `on_generate="delete"` then gives the rolling buffer (peak disk ≈ dataloader-workers files, no explosion).
  Strictly env-gated: unset = connector path untouched. `--hidden-states-path` must == serve `DSPARK_HS_DIR`.

---

## 2026-07-09/10 — Rollout data (A3 shard rollout track)

- Rolling `open-perfectblend` shards through the DSV4-bf16 serve, greedy temp=0, conc 64, max_tokens 3072,
  gsm8k gate (96.66%). A3 ~1.15 rows/s.
- **`--resume` error-row bug + fix** (`675d835`): the client writes an error record (same `idx`) on failure,
  and `load_seen` used to count it as done → failed rows permanently skipped. Symptom: a serve died mid-run,
  the client raced through the remaining ~56.9k rows writing instant ConnectionError rows → output *looked*
  complete (88,807 lines, 0-to-do on resume) but was ~31.9k valid + ~56.9k errors. Fix: `load_seen` skips
  `metadata.error` rows → resume retries them. One-time rescue: drop error rows, re-run against a healthy serve.
- **Progress bar** (`9a165bd`): `tqdm(total=…)` was `args.limit` (None for rollout) → bare count. Now counts
  local shard rows + `initial=already-done` → `X/Y + ETA`, and prints `Shard rows / already done / to do`.
- **Garbage filtering** (`detect_garbage.py --clean`, `7697ee8`): keep short/numeric, drop only REPEAT loops +
  errors → 99.95–99.96% clean (shard 00: 42 flagged / 38 REPEAT; shard 01: 32 / 26). Decision driven by
  `finish_reason`: flagged rows are mostly `stop` (valid short answers); true garbage is the `length` REPEAT
  handful. Clean with `--min-len 0 --min-alpha 0`. See rollout-data doc §5.

---

## 2026-07-10 — rollout → Arrow prep (the DSV4 tokenizer/chat-template saga)

Cleaned rollout (`out_bf16/rollout_0{0,1}.clean.jsonl`, ~99.95% kept) → `prepare_data.py` → Arrow.
Several DSV4-is-bleeding-edge blockers, each fixed:
- `speculators` not importable → `pip install -e speculators --no-deps` (setup script only cloned it).
- `AutoProcessor.from_pretrained` fails (DSV4 has no processor class) → `load_processor` falls back to
  `AutoTokenizer` (`ba8fd38`); that ALSO fails (`deepseek_v4` unregistered → generic config →
  rope_scaling `max_position_embeddings` crash) → 3rd fallback `PreTrainedTokenizerFast` (`ea93c18`).
  Upgrading transformers 5.5→5.12 gives native `deepseek_v4` + clean load; vLLM 0.23.0 still imports
  (`import vllm` OK) so it's safe — kept 5.12.
- Then `Processor does not support chat templates`: **DSV4 ships NO Jinja template** (uses vLLM's custom
  `encode_messages`). Added `--chat-template` (`562120c`) + reconstructed a **byte-exact** jinja
  (`7682be6`) from `encode_messages` chat mode. Verified `ALL MATCH: True` in-sandbox against
  **vllm-project v0.23.0** `encode_messages` (cloned to /workspace, NOT Mohammad's main). Community mlx
  jinja rejected (double-`</think>` bug).
- Prep succeeds: `Using HF assistant token mask`, sample viz shows correct format + blue=response, and
  `loss_mask` non-zero on all rows (frac 0.62–0.95). `--chat-template` + `{% generation %}` → loss on
  the response only. See rollout-data doc §6.

## 2026-07-10 — pre-launch audit of the DSpark train.py command (3 fixes + FSDP2/attn confirmations)

Before launching on 108, audited the recorded train.py command line-by-line against `scripts/train.py`
argparse + the DFlash loss/verifier/trainer code. All arg *names* matched, but found three things:

1. **The DSpark distribution loss was unreachable from the CLI** (commit `bcde32b`). Its factory
   `combo_ce_l1_loss` (ce·0.1 + l1·0.9) was wired only under the loss-fn name `combo`, consumed ONLY when
   `loss_fn=="..."` in `DFlash.get_trainer_kwargs` (core.py:209). But `parse_args()` ended with an
   unconditional `resolve_loss_fn(args.loss_fn)` validation whose map holds only the atomic losses
   {kl_div, ce, tv, nla} — so selecting it raised `ValueError: Unknown loss function` before training.
   This path had NEVER been exercised (DFlash-4B used ce/nla). Fix: (a) **renamed the selectable loss
   `combo` → `dspark`** (clearer; it's our own DSpark addition — commit `56066c1` — not upstream, so
   nothing to align to; the internal factory keeps the descriptive name `combo_ce_l1_loss`); (b) guard the
   validation `if args.loss_fn != "dspark": resolve_loss_fn(...)`. So the command must pass
   **`--loss-fn dspark`** — WITHOUT it the ce/l1 alphas are inert and training silently uses default kl_div.
2. **Missing `--draft-attn-impl sdpa`** (would crash on NPU). The default is `simple_flex_attention`, which
   is unavailable on Ascend (README §attn + all three ascend train scripts — `train_qwen3_{4b,8b}*.sh` —
   pass `--draft-attn-impl sdpa`; the SDPA block-attention path is the validated flex replacement, see
   `dspark_npu_op_check.py`). Added to the command.
3. Command hygiene: filled `--vllm-endpoint http://80.5.5.115:7000/v1` (API_PORT 7000, HEAD_IP 80.5.5.115),
   `--nproc_per_node 8` (108 = A2, 8 cards), `--num-workers 4` (online HS concurrency = ranks×workers = 32,
   under the serve's clean ceiling 64).

Confirmed (no change needed):
- **FSDP2 is already the trainer's mechanism** — `train/utils.py:apply_fully_sharded` uses `fully_shard`
  (FSDP2, with `MixedPrecisionPolicy` bf16 params / fp32 reduce), per-layer then whole-model; optimizers.py
  notes params become DTensors. `torchrun --standalone --nproc_per_node 8 scripts/train.py` → FSDP2 directly.
- **Verifier loading is memory-safe on a single 108 node.** `build_draft_model` → `from_training_args` →
  `load_verifier_weights` → `load_model_layers` reads ONLY `embed_tokens.weight` / `lm_head.weight` /
  `model.norm.weight` from the verifier's `model.safetensors.index.json` via `safe_open`+`get_tensor`
  (≈2 GB bf16), NOT the 568 GB full DSV4 model. `parse_vocab_mappings` falls through to "full verifier
  vocab" (no `--draft-vocab-size`/token_freq) via a config-only `AutoConfig.from_pretrained`.
- **draft depth (5) vs HS taps (3) are orthogonal.** `--num-layers 5` = the DRAFT's own Qwen3 decoder
  depth. `--target-layer-ids 40 41 42` = which 3 VERIFIER layers we tap for aux hidden states, fused by
  `fc = Linear(3·hidden → hidden)` as the draft's INPUT conditioning. The HS file carries `[seq, 4, H]` =
  those 3 aux (input) + the verifier's final hidden (last), which through `verifier_lm_head` forms the
  training TARGET distribution. num_target_layers = len([40,41,42]) = 3 (no auto layer-add).

## Open items / next

- **Launch DSpark draft training on 108** with the corrected online command (must include
  `--loss-fn dspark` and `--draft-attn-impl sdpa`). Pre-flight: (a) 108 env deployed on
  `feat/dspark-confidence-head` **including commit `bcde32b`** (fresh clone or `git pull`); (b) HS_DUMP
  serve up on 115/116, `DSPARK_HS_DIR == /share/canada_group_folder/dataset/dsv4_hs_dump`; (c) 108 can
  reach `http://80.5.5.115:7000/v1`. Watch: online HS-request concurrency ≈ world_size × num_workers —
  reduce `--num-workers` if the serve is stressed; draft as-is (`--num-layers 5 --block-size 5`), shrink
  only on OOM.
- Deploy env on 109 for 3–4 machine scale-out (setup script retargeted to `dspark_2026` + Plan B branches).
- Deferred: extract track on vLLM 0.24 (upstream-standard; needs the serve re-validation + memory pathology).
