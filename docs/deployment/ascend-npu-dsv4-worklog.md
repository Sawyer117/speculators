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

## 2026-07-14/15 — DSpark draft training launched → epoch-1 ckpt

Corrected online command shipped (pre-launch audit above + `ascend-npu-dsv4-dspark-ep-training.md`). Faithful
Plan-甲 draft (3×[MLA+sink+256-MoE+mHC] + markov/confidence heads) trained via speculators FSDP2 on 2×A2. Best
run = arrow_0715 / **whole-layer warm-start (`--init-layer-from-target` / `INIT_LAYER=1`)** / lr 2e-4 / EPOCHS 5; **epoch-1 ckpt =
`/home/a00652497/dspark_austin/run/ckpt_faithful_ep_20260715_213847/0`** (CKPT_FREQ 1.0). Epoch-0→1 showed NO
accept jump (soft accept_len ~2.9–3.1 median, slow creep). 6e-4 NaN'd (1% vs DeepSpec 4% warmup) → 2e-4.
**Branch convergence:** all train/serve/convert work now on ONE branch **`feat/dsv4-dspark`** (the old
`feat/dspark-confidence-head` / `-inference` split retired). See `ascend-npu-dsv4-dspark-ep-training.md`.

## 2026-07-16 — convert draft → vllm-ascend `mtp.*`; UT-1 bit-exact

`scripts/convert_dspark_to_vllm.py` (pure rename + expert-unstack, NO numeric transform): our stacked
`layers.{n}.*` → released `mtp.*` (per-expert). **Converter bug caught on the first real `--inspect`**
(`fb2151f`): our backbone names the target-hidden projection `fc`/`hidden_norm` (not `main_proj`/`main_norm`),
so it was silently dropping the serve-required `mtp.0.main_proj`+`main_norm` → fixed. `confidence_head` is
serve-skipped (training-only). **UT-1** (`scripts/verify_dspark_conversion.py`, independent inverse-map) =
**2378/2378 bit-exact** → converter RULED OUT.

## 2026-07-16 — serve bring-up (A2 bf16 dual-node, 115=head / 116=worker)

All had to be true: (1) **`VLLM_ASCEND_DSPARK_USE_STANDARD_DSA=1`** — default 0 routes the draft attn through
the custom `dspark_attention` TND path → **NaN at KV=window128+block5=133**; =1 = standard DSA PA_ND. (2) pin
aux `[40,41,42]` via BOTH channels (`--hf-overrides dspark_target_layer_ids` on target +
`draft_model_config.eagle_aux_hidden_state_layer_ids`) — else target emits 4-layer default (16384) vs draft's
3-layer `main_proj` (12288) → dim mismatch on first draft. (3) **both nodes on the SAME serve commit** — a
stale worker (116) crashed DP1 with the 16384 mismatch while the fixed head (DP0) lived (head-ok/worker-crash
was the tell). (4) draft on **`/share`** (node-local /home invisible to peer) + world-readable (`chmod a+rX`,
cross-user). Eval = `run_dspark_eval.sh`. Serve scripts do (1)+(2) when `DRAFT` set.

## 2026-07-16 — ★ ACCEPT COLLAPSE, and the test that flipped the diagnosis

Our epoch-1 draft on gsm8k = **accept_len 1.364** (pos0 32% / pos1 4% / pos2+≈0) vs released bar **3.94**. I
first mis-framed it as a soft-vs-hard metric gap (training soft accept_len 2.9 = `Σ min(p,q)` = E[len] under
sampling; serve = hard greedy argmax) — **user pushback: that can't explain per-position collapse; position_1
is hard-argmax in BOTH and was ~0.82 in training vs ~0.32 at serve.** Right.

Decisive move: **build a standalone RELEASED fp8 draft and serve it on OUR exact harness** (swap only the
draft). `scripts/build_released_draft_dir.py` (`8a6a57c`): `mtp.*` fp8 verbatim from released shards 46/47/48 +
`embed`/`head` borrowed from our bf16 draft + released fp8 `quantization_config` (`ignored_layers=[embed,head]`).
Result: **released draft = accept_len 1.344, pos0 0.3265 / pos1 0.0167 — IDENTICAL to ours.**
`num_draft_tokens/num_drafts = 110805/22161 = 5.0` ⇒ num_spec=5 IS live (the `block_size=2` load-log value is
the KV **page** size — red herring). **⟹ a known-good draft collapses the same way ⇒ the SERVE is the bug,
NOT our weights.** Training/conversion/definition VINDICATED, no retrain. (Needed `chmod a+rX` on the /share
draft dir first — same cross-user gotcha, surfaced as `FileNotFoundError`.)

## 2026-07-16 — exhaustive verify vs DeepSeek's reference → root cause

Pulled the official reference `DeepSeek-V4-Flash-DSpark/inference/model.py` and diffed our serve (sandbox
mirror `/workspace/vllm-ascend` @ box `60365071`) point-by-point. **ALL match:** aux reduction
(`hidden_states.mean(dim=1)` over hc_mult @ `deepseek_v4.py:1242` == official `h.mean(dim=2)` @ `model.py:921`
— dim index differs only because vLLM flattens batch×seq); aux post-layer, hc_post folds residual (layer fwd
1084-1103), cat[40,41,42]; non-causal `win_right=4` (`causal=False` @ proposer 371/505); `attn_sink` +
inverse-RoPE; config (block5/noise128799/markov256/hc4). Ruled out: method routing (`mtp`+`dspark_block_size`
→ `AscendDSparkProposer`, `self.method="dflash"` = expected), STANDARD_DSA, `block_size=2`.

**Root cause:** our fork `60365071` carries the **PRE-REWRITE DSpark, where the proposer piggybacks on the
DFlash path** (`self.method="dflash"`, 760-line proposer). The `num_spec % 5 == 0` constraint (which rejected
the official `num_spec=7`) traces to a **fork-local patch** (`patch_speculative_config.py`:
`n_predict=dspark_block_size`), unlike every other MTP. Upstream `vllm-project/vllm-ascend` main has **NO
DSpark** — it's fork-only. And the official 3.94 recipe (README: `method:"dspark", num_spec:7,
draft_sample_method:"greedy"`, 4×GB300, fp8 kv, EP, deep_gemm) is **upstream vLLM-core on GB300** — a
*different codebase* from our ascend fork's reimplementation.

## 2026-07-16 — the rewrite (#12004+#12005) is the fix

Fetched the post-#11196 PRs. #11196 (@QwertyJack, monolithic prototype) is being **decomposed into a 4-PR
stack** (issue #11126): **#12003** (noncausal DSA attn) → **#12004** (draft model) → **#12005** (eager
spec-decode = "correctness baseline") → **#12006** (FULL ACLGraph, perf). `#11431` (@drslark) is a *separate,
competing* refactor. The rewrite **materially changes drafting**: proposer 760→196 lines; DSpark split OUT of
DFlash into its own path (new code comment: "DSpark query length = num_spec, **unlike DFlash where it's
num_spec+1**"); the `n_predict=dspark_block_size` ÷5 constraint **removed**;
`VLLM_ASCEND_DSPARK_USE_STANDARD_DSA` **removed** (TND-NaN/PA_ND dilemma gone); method accepts `"dspark"` or
`"mtp"`. **Status: #12004/#12005/#12006 are DRAFT/WIP, none merged, all `mergeable=False`, human review early
(mostly bots); `#11765` (generic proposer dep) is MERGED to main.** `pr-12005` head folds 12003+12004
(self-contained, verified). ⟹ our collapse is almost certainly the old DFlash-piggyback impl. Write-up: memory
`dsv4-dspark-inference-conversion`; kernel handoff (now moot) `HANDOFF_dspark_accept_collapse_2026-07-16.md`.

## 2026-07-16/17 — DECISION: rebuild the serve on #12005 (fresh env) → building

Compat confirmed: **both HEAD and pr-12005 pin `VLLM_TAG=v0.23.0`** → reuse the box's vllm; `torch==torch-npu==
2.10.0` identical. All 15 fork commits are superseded old-DSpark → **cherry-pick NONE** (`60365071` w8a8 o_proj
fix is w8a8-only, irrelevant for bf16). CANN custom ops are **package-isolated** (`build_aclnn.sh` →
`${ROOT_DIR}/vllm_ascend/_cann_ops_custom`; `utils.py:323` prepends *this package's* ops to
`ASCEND_CUSTOM_OPP_PATH` at import) → rebuilding does NOT contaminate the running 60365071 serve. Plan (runbook
`vllm-ascend-dspark-rebuild.md`): fresh conda env **`dspark-dsv4-serving`** (clone `-base`, inherits vllm
0.23.0) + fresh github clone **`vllm-ascend-serving @ dspark-dsv4-v2 = 431a64b18b`** (pr-12005 head, a DRAFT →
pin the commit) + `pip install -e .` (setup.py auto-chains `build_aclnn`) after sourcing CANN 9.0.0 nnal/atb in
a fresh shell. Old env + `vllm-ascend-v4` (60365071) untouched for rollback. **Status: building.** vllm-ascend
version string `0.19.1rc2.dev1028` is its own tag lineage (≠ vllm core 0.23.0), NOT a regression. Live status:
memory `dsv4-dspark-serving-rebuild-worklog`.

## 2026-07-17 — serve fixed on #12006; root-caused OUR draft to `sample_from_anchor`; retraining

**The #12006 rebuild WORKED.** Serve = `vllm-ascend-serving @ dspark-dsv4-v3` (#12005 + the python-only ACLGraph
commit — no csrc recompile; DP token padding handled via `_pad_request_rows` for eager+graph). Built a **bf16**
released control with `scripts/build_released_draft_dir.py --dequant-bf16` (dequant the released fp8 `mtp.*` →
bf16 with the proven DeepSeek block-fp8 math; needed because the rewrite binds draft precision to the bf16
target). **The known-good released draft scores gsm8k accept_len 4.658 on our serve — smooth pos0 0.925 → pos4
0.538, ABOVE the official 3.94** ⇒ the serve is fixed/excellent. Our epoch-1 draft still evals ~1.758 with a
sharp pos0 0.646 → pos1 0.06 cliff on the SAME serve ⇒ **the problem was OUR weights, not the serve** (the earlier
"serve bug, weights vindicated, no retrain" conclusion is REVERSED — it held only for the old broken serve).

**Root cause: the fork DELETED upstream's `sample_from_anchor` switch and hardcoded the FALSE branch.** DSpark
serving samples EVERY block slot (slot 0 predicts the next token from the anchor/seed) ⇒ it needs
`sample_from_anchor=True`; we trained FALSE (targets rolled +1 → every slot off-by-one; slot 0 masked). Ruled out
first, all MATCHING the serve / canonical DeepSpec: RoPE pairing + YaRN (within-block Δangle 0.002–0.009 rad,
negligible — numerically checked), QuaRot (our converted config has no rotation), noise token (both 128799),
Markov head (both vanilla first-order), aux VALUES (the target `deepseek_v4.py` layer forward is byte-identical
between the dumper and serve builds), and the whole block scheme (DeepSpec `common.py` mask/noise/positions/
target-alignment byte-identical to ours). So the bug was never the block code — only the deleted switch.

**`sample_from_anchor=True` needs gating in THREE places (found in order):** (1) the **target roll**
(`dsv4_dspark/core.py`) — skip under True; (2) the **slot-0 loss mask** — don't zero slot 0 under True; (3) the
**per-position loss DECAY** (`models/metrics.py::dflash_loss_decay`) — the one we first MISSED. It hardcoded
`w * (pos_idx != 0)` → slot 0 weight 0 → zero gradient, so the first (gated 1+2 only) retrain gave
`position_0_acc ~0.03` FLAT while pos1-4 learned (~0.37 decay), which kills `hard_accept_len` (the sequential
accept starts at slot 0). A one-shot `DSPARK_DIAGNOSE=1` probe proved the targets are correctly aligned
(`argmax(targets[slot k]) == true token[anchor+k+1]` for all slots incl 0) → NOT a data bug; the decay was it.
**Fix: under True → `exp(-pos/gamma)` (slot 0 = highest weight 1.0, decays out).**

**Cross-validated vs canonical DeepSpec:** its `_build_loss_weight_mask` (`modeling/dspark/loss.py`) =
`exp(-arange(block_size)/gamma)` → position 0 weight 1.0 — **byte-equivalent to our fix**, and DeepSpec trained
the working released draft (pos0 = 0.925). **Upstream speculators HAS the same bug** (its `compute_metrics` has
`sample_from_anchor` for `start_pos` but never threads it into the `decay_fn`) → a real upstream bug worth a PR.

**Also fixed:** `train_dsv4_dspark.sh` **BLOCK 6→5** (under True `speculative_tokens = block_size`, so BLOCK=5
= released `dspark_block_size=5`; the old BLOCK=6 was the False `block_size-1=5` workaround, which under True
would draft 6 slots) and the **shared-KV sink attention** (`backbone/attention.py`: drop the per-head `.expand`,
`nkd` einsums — bit-identical (parity UT), −~2.1 GB, ~20× fwd, matches the #12005 shared-KV op — frees anchors).

**Commits (`feat/dsv4-dspark`):** `88ad6a4` roll+mask gate + config + UT · `c80b942` BLOCK 6→5 · `595d058`
shared-KV attn + parity UT · `d430086` `DSPARK_DIAGNOSE` probe · `928ea32` loss-decay slot 0 + UT.

**Topology (3-machine online HS):** **109 = trainer** (`train_dsv4_dspark.sh`, drives `--vllm-endpoint` so the
serve dumps HS on demand); **115 (head) + 116 (worker) = bf16 target serve** with `HS_DUMP=1` (env
`dspark-dsv4-austin` = `vllm-ascend-v4 @ feat/dsv4-hs-dumper`, **NOT** #12006 — the dump hook is our feature),
rolling HS to shared `/share/.../dsv4_hs_dump`. Eval later uses #12006 (`dspark-dsv4-serving`) + the draft.

**Status: RETRAINING** (109: `MAX_ANCHORS=512 BLOCK=5 sample_from_anchor=True INIT_LAYER=1 CKPT_FREQ=0.5`). Watch
`train/position_0_acc` (must climb from 0 — the decay fix's direct signal) + `hard_accept_len` → released ~3.9+;
per-position should become smooth (pos0 highest). No re-dump needed — aux verified identical between builds.

## 2026-07-17 (later) — a SECOND train↔serve mismatch: Markov prev_token off-by-one (slots ≥1); both fixes serve-validated; run restarted

The `sample_from_anchor=True` retrain was climbing correctly (`position_0_acc` 0.03 → **0.73**, decay fix works)
but `accept_len` stalled ~1.9 — **too low for a healthy slot 0**. Found the cause: a **second** train↔serve
mismatch, ORTHOGONAL to the decay one, in the **Markov head's `prev_token_ids`**.

**Our DSV4 draft (`DSV4DSparkDraftModel`) overrides only `_backbone_forward`; it INHERITS `DSparkDraftModel.forward`**,
whose `prev_token_ids` used the DFlash **shifted concat UNCONDITIONALLY** (`torch.cat([block_tokens[:,:1],
block_tokens[:,:-1]])`). **Cross-validated against the AUTHORITATIVE serve proposer** (`vllm-ascend`
`deepseek_v4_dspark_proposer._sample_sequential`, the code that produces the released draft's 4.658): the loop is
`prev_ids = seed_buffer` (= anchor token) → per step `logits = base[:,k] + markov_bias(markov_embed(prev_ids))` →
`prev_ids = draft_buffer[:,k]`. So **slot k's Markov prev = the token at position p+k** (seed=anchor for k=0, then
the autoregressively-drafted token). Teacher-forced equivalent = **raw `block_tokens[:, k]`**; the shifted concat
fed `block_tokens[:, k-1]` = position p+k−1 → **off-by-one for every slot ≥1**.

**KEY — the two bugs are ORTHOGONAL, and the decay A/B was NOT confounded:** slot 0's prev = `block_tokens[:,0]`
= anchor in BOTH old and new, so the Markov bug touches **only slots ≥1**; slot 0's problem was **purely the
decay**. So `decay → slot 0`, `Markov → slots 1–4`, independent. This is why the run showed `position_0_acc 0.73`
(decay fixed slot 0) yet `accept_len ~1.9` (slots 1–4 Markov still misaligned, capping the block).

**Fix:** gate `prev_token_ids` on `sample_from_anchor` (True → raw `block_tokens`, False → shifted), in the
inherited `dspark/core.py` forward. **`feat/dsv4-dspark @ 373deae`.** Verified line-by-line against the serve
proposer. **Both fixes are now required and both are serve-validated** (decay = slot 0, Markov = slots 1–4).

**Action:** STOPPED the (Markov-buggy) run at step ~1532; pull + restart with both fixes (same recipe).

**Upstream decay PR staged (not opened):** branch `fix/dspark-slot0-loss-decay @ 8a564c3` on `Sawyer117/speculators`
(off upstream main) — `dflash_loss_decay` gains a `sample_from_anchor` branch (True → `exp(-pos/gamma)`, slot 0 =
1.0), threaded from dspark+dflash `compute_metrics`; decay UT extended. Commit cites **no other PRs** (we're already
in that area). HOLD opening until the two-fix retrain proves end-to-end (train metric + serve accept_len, double
evidence). The Markov fix needs no PR of ours — it's the serve convention.

## Open items / next

- **[RESTART] The two-fix retrain** (109, `MAX_ANCHORS=512 BLOCK=5 CKPT_FREQ=0.5`, sample_from_anchor=True) — now
  with BOTH the decay slot-0 fix AND the Markov `prev_token_ids` fix (`373deae`). Success = `position_0_acc` high
  (~0.7) AND `accept_len`/`hard_accept_len` climb past the old ~1.9 toward released ~3.9+; per-position smooth.
  Then convert (`convert_dspark_to_vllm.py` on `ckpt_.../0`) → eval on #12006 (`dspark-dsv4-serving`) + draft.
- **[DONE 07-17] #12006 serve rebuild + validation** (`dspark-dsv4-v3`): released bf16 draft = **4.658** on our
  serve ⇒ serve fixed; our epoch-1 draft ~1.758 ⇒ weights were the bug → `sample_from_anchor` root cause + fix.
- **[DONE 07-17] Two train↔serve conventions fixed + serve-validated:** decay slot 0 (`928ea32`) + Markov
  prev_token (`373deae`). Orthogonal (slot 0 vs slots 1–4). Decay PR branch staged (`8a564c3`), held pending e2e.
- **Deferred:** track #12006 to merge or keep the pinned snapshot; extract track on vLLM 0.24 stays deferred.
