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

## 2026-07-18 — a THIRD training bug (FSDP2 mixed-precision) + upstream sync; ALL-FIXES run launched

**A third training-correctness bug, found via the GLM-5.2 DSpark article (§4.1): FSDP2 mixed-precision master
weights.** The trainer blanket-cast `self.model.to(bf16)` BEFORE `fully_shard`, so FSDP2's
`MixedPrecisionPolicy(param_dtype=bf16)` had **no fp32 master** → sub-ULP updates (norm ~1e-7) + AdamW weight
decay were rounded away in bf16 → **RMSNorm silently frozen, decay dropped for all params** (the article's
controlled ablation: **+13.5–28.9% accept_len**). Confirmed in our trainer (cast at setup_model, shard after).

**Fix = option B (memory-safe), `4354a6d`:** keep the SMALL trainable params (norm/MLA/mHC/Markov/confidence heads)
fp32 as masters; frozen params AND the big EP experts stay bf16 (full-fp32 experts ≈ +5–6 GB → OOM at 512 anchors;
expert grads are rounding-robust). FSDP's policy casts the fp32 masters to bf16 for compute → no autocast (NPU
autocast avoided). Cost: `mem/alloc` 18.95 → 19.43 GB (+~0.5 GB, the small-param fp32 masters + their fp32 AdamW
moments); `mem/reserved` unchanged ~56 GB → **no anchor trim**. (Why no memory explosion: only the tiny trainable
params go fp32; experts — the bulk — stay bf16.)

**Upstream sync (one-time merge).** The fork IS a real fork of `upstream/main` (merge-base `21033a7` #736, 133
ahead / 33 behind — an empty `merge-base` earlier was a SHALLOW-CLONE artifact, not "unrelated history"). Merged
`upstream/main` on an isolated branch, resolving 11 conflicts (took upstream's canonical #798/#806/#760, kept our
#805-equiv slot-0 metrics + AMP + EP + `dsv4_dspark` model). **Gained beyond the 4 we'd hand-ported:** #788
(divergence losses in float32 under bf16 — shifts tv/kl magnitudes UP, so **don't compare loss across the merge**),
#759 (metric double-reduction `.clone()`), #711's checkpointer §4.3 dtype normalization, + sliding-window / MRoPE
/ JSD features (inert for DSV4).

**Smoke caught a real latent bug — the AMP fix had NEVER run** (the earlier step-79-105 run was pre-AMP,
`264af2d`). Its first selective cast skipped the complex64 RoPE `freqs_cis` buffer (`is_floating_point()` is False
for complex) → NPU `aclnnIndex` crashed (`DT_COMPLEX64` not in its dtype support list). **Re-fixed in `4354a6d`:**
blanket `.to(bf16)` FIRST (buffers incl. `freqs_cis` get the old working treatment) THEN upcast small trainable →
fp32. **Re-smoke PASS:** no crash, no NaN, `mem/reserved 56.15 GB` (fits), metrics moving. `feat/dsv4-dspark`
fast-forwarded to `4354a6d`; **the smoke run IS the real run** (same recipe, let it continue).

**ALL FIXES now in one branch (`feat/dsv4-dspark @ 4354a6d`):** pos0 decay slot-0 · Markov `prev_token_ids` ·
metrics slot-0 · AMP fp32 masters · upstream merge. **AMP definitive check (do at 1st checkpoint):** compare the
ckpt's norm weights vs the verifier norm @[40,41,42] (the warm-start source) — **changed ⇒ norms training ⇒ AMP
works** (the buggy code froze them at the warm-start values, so the article's `max|w−1|==0` canary doesn't apply
to our warm-start).

## Open items / next

- **[RUNNING] The ALL-FIXES run** (`MAX_ANCHORS=512 BLOCK=5 CKPT_FREQ=0.5 DSPARK_EP=1 NO_VAL=1 RECOMPUTE=1`,
  `feat/dsv4-dspark @ 4354a6d`). Watch: `position_0_acc` climb past lr-peak (~step 245); **`hard_accept_len` break
  the killed run's ~2.4** (Markov fix signal); no NaN; mem ~56 GB. At 1st ckpt: the norm-changed AMP check. Then
  convert (`convert_dspark_to_vllm.py` on `ckpt_.../0`) → eval on #12006 (`dspark-dsv4-serving`) + draft.
- **[PARALLEL] torch-2.12 compile env** (`dspark-dsv4-compile`, clone of `-austin`): kills the grouped-GEMM
  recompile (62% wall-clock). CANN 9.0.0 compatible (validated). Gated on: this run stable + a fresh bit-exact
  re-check post-merge + a `transfer_to_npu` guard. Recipe in `ascend-npu-dsv4-dspark-compile-recompile.md`.
- **[DONE 07-17] #12006 serve rebuild + validation** (`dspark-dsv4-v3`): released bf16 draft = **4.658** on our
  serve ⇒ serve fixed; our epoch-1 draft ~1.758 ⇒ weights were the bug → `sample_from_anchor` root cause + fix.
- **[DONE 07-17] Two train↔serve conventions fixed + serve-validated:** decay slot 0 (`928ea32`) + Markov
  prev_token (`373deae`). Orthogonal (slot 0 vs slots 1–4). Decay PR branch staged (`8a564c3`), held pending e2e.
- **Deferred:** track #12006 to merge or keep the pinned snapshot; extract track on vLLM 0.24 stays deferred.

---

## 2026-08-04 — ★ CANONICAL REPRODUCIBLE COMMANDS (keep updated: best + last) + RoPE-fix from-scratch launch

**Why this section exists:** the per-ckpt `train_command.txt` (trainer writes it into every ckpt dir) captures the `train.py` **argparse ONLY** — it does NOT record the `DSPARK_*` **env vars** (`DSPARK_EP / RECOMPUTE / COMPILE / MOE_BALANCE(+RATE) / TEACHER_DOUBLE_NORM`, plus the `BF16_EXPERTS` force) that the launcher sets via `env … torchrun`. Recover those from the run log's `patch_getenv` INFO lines (`get env DSPARK_X = Y`) + the `[MOE-BALANCE]` echo. **Complete recipe = train_command.txt + those env lines.** Both resolved below.

### LAST run — `ckpt_faithful_ep_20260729_092941` (ep4p5; a RESUME of the bal-1e3 / fresh-router / dedup line)
Git SHA `da064ea` · speculators 0.5.0.dev448 · transformers 5.13.1 · torch 2.12.0+cpu · world_size 8 (EP8).
- **env (from log `patch_getenv`):** `DSPARK_EP=1  DSPARK_RECOMPUTE=1  DSPARK_COMPILE=0  DSPARK_MOE_BALANCE=1  DSPARK_MOE_BALANCE_RATE=1e-3  DSPARK_TEACHER_DOUBLE_NORM=0  DSPARK_HS_DUMP=1  BF16_EXPERTS=1` (forced → `--bf16-experts` even under EP=1, because A2 64G can't fit EP option-A fp32 experts).
- **argparse (`train_command.txt`, verbatim):**
  ```
  scripts/train.py --speculator-type dsv4_dspark --served-model-name dsv4 \
    --num-layers 3 --n-routed-experts 256 --block-size 5 --target-layer-ids 40 41 42 \
    --max-anchors 512 --dflash-decay-gamma 4.0 --sliding-window 128 --sliding-window-non-causal \
    --total-seq-len 3072 --mask-token-id 128799 --noise-std 0.05 --kd-temperature 1.0 \
    --draft-attn-impl sdpa --loss-fn '{"ce":0.1,"tv":1.8}' \
    --scheduler-type cosine --scheduler-warmup-ratio 0.04 --optimizer adamw --lr 2e-4 --epochs 5 \
    --checkpoint-freq 0.5 --no-validation --bf16-experts --on-missing generate --on-generate delete \
    --num-workers 12 --prefetch-factor 4 \
    --hidden-states-path /share/canada_group_folder/dataset/dsv4_hs_dump \
    --vllm-endpoint http://80.5.5.115:7000/v1 \
    --verifier-name-or-path /share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16 \
    --data-path /share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0730_77w_dedup \
    --save-path …/ckpt_faithful_ep_20260729_092941 --log-dir …/run
  ```
  ⚠ this run was a **RESUME** (`--save-path`=existing dir, **no `--init-*` flags**). The line's from-scratch INIT = `--init-layer-from-target --init-moe-no-router` (box-wide `train_command.txt` count: 23× `--init-layer-from-target`, 5× `+--init-moe-no-router`).

### BEST draft — `ep0p5-ropefix-77w` (RoPE-fix, 0.5ep, mean **3.84**) ★ NEW BEST (2026-08-05)
The RoPE-fix retrain is the new best across everything (ledger `ep0p5-ropefix` row); recipe = the LAST-run
recipe above **+ real-RoPE `feb0066`/`8db8f75`** (bal1e3 / fresh-router / dedup / lr2e-4 / anchor512 / EP8).
- *(prev best, now #2)* `ep1mid-f1-77w` (single-norm "f1", 1.5ep, mean **3.63** = 82% of released 4.42) — its
  trainer ckpt was overwritten; recipe known from the ledger: single-norm, NO balance, non-causal, anchor576,
  noise0.05, LR3e-4, A3 `launch_a3` (DP16/EP16, `INIT_LAYER=1`).

### ★ RoPE-fix FROM-SCRATCH launch (= LAST line's recipe + real cos/sin RoPE `feb0066`/`8db8f75`)
On 109 (A2, env `dspark-dsv4-compile`), with 115/116 HS serve up:
```
cd /home/a00652497/dspark_austin/speculators && git pull
DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3 \
INIT_LAYER=1 INIT_MOE_NO_ROUTER=1 \
LR=2e-4 EPOCHS=5 MAX_ANCHORS=512 CKPT_FREQ=0.5 \
DATA=/share/canada_group_folder/dataset/open_perfectblend.dsv4_rollout/arrow_0730_77w_dedup \
  bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful
```
Only changes vs the LAST line: **proper RoPE** (was degenerate scale-only), **from-scratch** (re-adds the init flags the resume lacked), fresh save-path. A/B target = beat the degenerate line's plateau (0.5ep 3.56 → 4.5ep 3.45); **tail pos2-4 is the tell**.

---

## 2026-08-05 — ★★★ RoPE-fix RESULT: eval-validated, NEW BEST (the A/B lands)

The from-scratch RoPE-fixed run (`ckpt_faithful_ep_20260804_165215`, the recipe above) converted at **0.5ep**
(`dsv4_dspark_ep0p5_ropefix_vllm-77w`, 2378/2378 bit-exact) and served on 176 (A3-single, ns5, conc48):

**mean accept_len 3.84 = NEW BEST across everything** — beats the prior best `ep1mid-f1` (3.63 @ 1.5ep) at 1/3
the epochs; = 86.9% of released 4.42. **ALL 5 datasets up vs the same-recipe degenerate `ep0p5-bal1e3`** (the
clean single-variable A/B — ONLY change is RoPE): gsm8k 4.050→**4.309**, math500 3.784→**4.068**, humaneval
3.890→**4.298**, mbpp 3.591→**3.908**, mt-bench 2.466→**2.627** (mean 3.56→3.84, +0.28). The gain is the
**TAIL** (gsm8k cumul pos2/3/4 59.8/46.3/35.0→65.6/53.6/42.8; conditional accept to pos4 stays ~80%) and the
diagnosed **`train↑/eval↓` divergence is RESOLVED — eval now tracks train.** ⟹ RoPE was THE remaining
train↔serve mismatch; the "gap = data/recipe/tail" and "serve bug / no retrain" reads across the older docs are
superseded (they've been annotated). Full row + per-position in the eval-results ledger. Trajectory (1.0ep `/0`,
1.5ep `/1`, …) being converted+evaled to see if it keeps climbing toward 4.42 or plateaus at a much higher level.

---

## 2026-08-06 — RoPE-fix line: full 0.5→2.0ep trajectory, gsm8k passes released; tooling; upstream PR #942

### Eval trajectory (the RoPE-fixed from-scratch run `ckpt_faithful_ep_20260804_165215`)

Every point is the SAME line (recipe = the LAST-run recipe in the canonical block above **+ real cos/sin
RoPE** `feb0066`/`8db8f75`; fresh router + `DSPARK_MOE_BALANCE=1 @1e-3`, LR 2e-4, anchor512, EP8,
dedup 77W). Converted with `convert_dspark_to_vllm.py` (each 2378/2378 bit-exact), served on 176 A3-single
`num_spec=5`, `DATASET=all` conc48.

| ckpt dir | epoch | gsm8k | math500 | humaneval | mbpp | mt-bench | **mean** | % of released |
|---|---|---|---|---|---|---|---|---|
| `/0` (epoch0_step12388) | 0.5 | 4.309 | 4.068 | 4.298 | 3.908 | 2.627 | **3.84** | 87.0% |
| `/0` (epoch0_end)       | 1.0 | 4.493 | 4.241 | 4.585 | 4.167 | 2.796 | **4.06** | 91.8% |
| `/1` (epoch1_step12448) | 1.5 | 4.628 | 4.408 | 4.706 | 4.300 | 2.865 | **4.18** | 94.6% |
| `/1` (epoch1_end)       | 2.0 | **4.701** | 4.431 | 4.745 | 4.412 | 2.980 | **4.25** | **96.3%** |

- **Monotonic** (Δ +0.22 / +0.12 / +0.07 — decelerating). The degenerate lines had already turned DOWN by
  2.0ep (3.56→3.45); this one has not.
- **gsm8k 4.701 > released 4.658 (100.9%)** — first dataset past the bar.
- **Conditional accept now matches/beats released**: gsm8k c0–c4 = 0.926/0.904/0.884/0.873/0.856 vs released
  0.928/0.892/0.885/0.868/0.842 ⟹ pos1–4 at or above. Residual gap = pos0 (−0.2pt) and **mt-bench (90.5%)**,
  which is a DATA-distribution issue (our rollout is 99.96% single-turn), not a bug.
- **Next lever for the last ~4% = anneal LR→0** (the DeepSpec full-anneal we have never run), not more
  epochs at constant LR. ⚠ **RETRACTED 2026-08-07 — see the 3.5ep entry**: this run was already
  annealing (`--scheduler-type cosine` over 5 epochs → exactly 0 at the last step; predicted
  `lr(gs=87138)=4.444e-05` matches the logged `4.45e-05`). The LR was never constant.

### MoE routing — the official regime, reproduced

- **Official draft, static:** `mtp.0.ffn.gate.bias` non-zero (≈10 ± 0.09) but the ≈10 is a routing-IRRELEVANT
  uniform shift (top-k is shift-invariant) — the real signal is the ~0.74 spread; `mtp.1`/`mtp.2` are
  **exactly 0**. **Dynamic probe (200 fwd):** N_eff **L0 ~70 / L1 ~55 / L2 ~47**, entropy 0.70–0.77,
  **rotating** (union still growing) ⟹ `gate.bias = 0` does **not** mean collapsed; the spread comes from the
  trained router weights. Official target regime = **moderate ~47–70 + rotating**.
- **This run reproduces it:** N_eff **L0 62 / L1 73 / L2 81**, entropy 0.60→0.74–0.79, union **252–256/256
  SATURATED** ⟹ analyzer verdict "✓ balanced — load balancing is NOT the bottleneck".
- **Definitions (so the numbers are reproducible):** `[MOE-LOAD]` entropy = normalized Shannon entropy
  `-(p·log p).sum()/log(E)`, E=256 (`core.py`); the analyzer's `eff.experts` = `E**entropy` = `exp(H)` = the
  load distribution's **perplexity** = mass-weighted effective expert count (`analyze_train_run.py`). N_eff
  measures **per-step** sparsity only — "fixed collapse vs rotating" needs the **union** column.
- **Balance A/B (⚠ all from the DEGENERATE-RoPE era — compare only within the table, never against the 4.25
  above):** no-balance ~18 fixed → **3.63**; warm+5e-3 ~20 → 3.47; fresh+1e-2 ~120 → **2.66** (tail collapse:
  gsm8k pos1 77%→50%, pos2 57%→28%). ⟹ **forced un-collapse is harmful; the goal is moderate sparsity +
  rotation, not uniformity.**

### Tooling added (all git-tracked under `examples/ascend_npu_dflash/`)

- `stitch_train_logs.py` — merge the logs of a kill+resume run into ONE continuous `global_step` curve
  (de-dupes the resume overlap, reports gaps). Parses exactly like `analyze_train_run.py`.
- `consolidate_run.py` — `--auto-chain` walks `global_step` back from the deepest log to reconstruct a whole
  resume LINE, then writes one recipe-named dir: `train_full.log` + `recipe.txt` (argparse **and** the
  resolved `DSPARK_*` env) + `MANIFEST.txt` + **symlinks to every real trainer checkpoint**.
- `verify_safetensors_dir.py` — verify a downloaded model dir is complete/intact (shard count from the
  `-of-N` naming, per-shard header-vs-size, JSON parse, zero-byte files; skips `.cache/` downloader
  internals; `--manifest` diffs against the HF file tree incl. LFS sha256).
- `plot_best_vs_baseline.py` — gained the RoPE-fix line as **"current best"** + `--group best` (default) and
  **`--latest`** (released vs the newest ckpt only = the report front-page figure).

### Upstream PR

- **[vllm-project/speculators#942](https://github.com/vllm-project/speculators/pull/942)** —
  `fix(loss): normalize training loss by global (cross-rank) token count`. `loss_function` normalized by the
  **per-rank** supervised-token count; under DDP/FSDP grad mean-averaging that makes the objective a
  rank-local mean of ratios instead of the token-weighted one. Packing does NOT prevent it: the multipack
  sampler balances **total** tokens against a budget and never inspects `loss_mask`. Fix = all-reduce the
  (detached) denominator to the global count and scale by `world_size`; **gradient-identical when ranks are
  token-balanced**. +11 lines in `metrics.py` + a 2-rank gloo test. DeepSpec's `_build_loss` does the
  identical math. ⚠ **SpecForge is NOT a precedent** (its `reduce_metrics` reduces over the **SP group**,
  differentiably, on num AND den; the base adapter doesn't reduce at all) — that citation was removed before
  submitting. Status: DCO green (needs `git commit -s`), CodeRabbit/docs green, blocked only on the
  approved-reviewer list. Related upstream PR #871 (Ulysses SP) touches the same lines for a DIFFERENT
  problem (SP-group reassembly) — noted in a PR comment, will rebase if #871 lands first.

### Doc hygiene done this day

Index (`ascend-npu-dsv4-dspark-pipeline.md`) gained a **"Cross-cutting docs & tooling"** section — the
worklog, the acceptance-troubleshooting tree, run-comparison, the serve-rebuild and Mooncake docs were all
reachable only by knowing they existed. Stage 5 was renamed **Convert → Serve → Eval** and now names
`convert_dspark_to_vllm.py` / `verify_dspark_conversion.py`: a trainer checkpoint **cannot** be served
directly, and that step was missing from the index — the single most likely place for someone reproducing
this to get stuck.

---

## 2026-08-07 — 3.0ep checkpoint converted; ★ the confidence head is trained but NEVER served

### 3.0ep (`epoch2_end`) landed and converted

`ckpt_faithful_ep_20260804_165215/2` → `dsv4_dspark_ep3p0_ropefix_vllm-77w`, **2378/2378 bit-exact**.
Save grid confirmed exactly: `step_interval = 12448` = 0.5 epoch, so one epoch = 24,896 steps and
**74,688 = 3 × 24,896** — the trainer printed `Training epoch 3/5 completed` at `global_step=74,692`.
This save **overwrote** the 2.5ep mid-epoch weights in the same `/2` dir (2.5ep had already been
converted + evaluated, so nothing was lost — but this is the standing hazard, convert promptly).

**Measured end-to-end rate: 3.17–3.18 s/step**, twice independently (71,504→73,167 over 88 min;
73,167→74,144 over 51m34s). The log's `profile/step_ms ≈ 1.95–1.98e3` is ~60% optimistic — the
difference is time outside the step timer (`--on-missing generate` waiting on the HS serve,
dataloader stalls, checkpoint writes). **Use 3.17 s/step for ETAs, not `step_ms`.**

### ★ Finding: `1 skipped: confidence_head.proj.bias` — chased down, three facts

`convert_dspark_to_vllm.py` reports `1 skipped (target-only/buffers): ['confidence_head.proj.bias']`
on every checkpoint (input 84 tensors → output 2378, identical for all six converted ckpts). Chased:

1. **The serve drops the WHOLE head, not just the bias.**
   `vllm_ascend/models/deepseek_v4_draft.py::_remap_dspark_name`:
   `if rest.startswith("confidence_head."): return None`. Neither `proj.weight` nor `proj.bias` is
   ever loaded. **The confidence head is dead code at inference today.**
2. **Training detaches its inputs**, `src/speculators/models/dspark/core.py:181-187`:
   `conf_features = cat([hidden_blocks.detach(), prev_emb.detach()...])`. The BCE gradient updates
   **only** `confidence_head.proj.{weight,bias}` and can never reach the backbone.
   ⟹ **Zero effect on every accept_len number in the ledger** — structurally, not "small". It does
   enter the reported `train/loss` (`0.1×ce + 1.8×tv + conf` = `0.1156+0.405+0.312 = 0.832`, matches
   the log) and the global grad-norm; at `grad_norm ≈ 0.61` no clipping fires, so no second-order
   coupling either.
3. **Architecture drift:** our `ConfidenceHead` is `nn.Linear(input_dim, 1)`
   (`models/dspark/model_definitions.py:88`) = `bias=True`, but the released `mtp.*` layout has no
   bias — `dsv4_dspark/weights.py::expected_draft_keys` lists only `confidence_head.proj.weight`, so
   that key set is itself inconsistent with the model it claims to describe.

**When it bites:** the day adaptive / dynamic draft length lands (stage 1 of the adaptive-speculation
proposal). Then confidence actually gates drafting, and both the missing serve wiring and the bias
mismatch become real work. **Cost to fix is low** — because the features are detached, the head can be
re-fit on a FROZEN backbone in minutes; no retrain. Fix = `nn.Linear(input_dim, 1, bias=False)`, either
outright (load old ckpts `strict=False`) or behind a `confidence_head_bias` flag defaulting False for
DSV4, since the module is shared with the Qwen3 DSpark model.

**Still open:** read the released draft's safetensors header
(`/share/canada_group_folder/ckpt/released_draft_bf16_standalone`) and list keys containing
`confidence` — decides whether released is `proj.weight`-only (⟹ just flip `bias=False`) or has no
confidence head at all (⟹ the head is entirely ours and was never part of the served contract).

### 3.0ep eval — mean **4.35 = 98.4% of released**; non-chat **99.4%**; the plateau call is retracted

`dsv4_dspark_ep3p0_ropefix_vllm-77w`, 176 A3-single, conc48, ns5, `DATASET=all`, log
`~/eval_ep3p0_ropefix_all.txt`.

| dataset | 2.5ep | **3.0ep** | Δ | released | % |
|---|---:|---:|---:|---:|---:|
| gsm8k | 4.753 | **4.796** | +0.043 | 4.658 | **103.0%** |
| math500 | 4.485 | **4.519** | +0.034 | 4.661 | 97.0% |
| humaneval | 4.818 | **4.855** | +0.037 | 4.942 | 98.2% |
| mbpp | 4.428 | **4.512** | +0.084 | 4.535 | **99.5%** |
| mt-bench | 2.988 | **3.046** | +0.058 | 3.294 | 92.5% |
| **mean** | 4.29 | **4.35** | +0.05 | 4.42 | **98.4%** |
| **non-chat (4)** | 4.621 | **4.671** | +0.050 | 4.699 | **99.4%** |

Three things this run changes:

1. **"Approaching plateau" is RETRACTED.** Deltas per half-epoch: +0.22 / +0.12 / +0.07 / +0.04 / **+0.06**.
   The 2.0→2.5ep step (+0.04) was read as the onset of a plateau and produced the "next lever = anneal
   LR→0, not more epochs" call in the `ep2p0-ropefix` ledger row. **The 2.5→3.0ep step is LARGER.** At this
   scale that could still be run-to-run noise, but there is no flattening in the data, so that call does not
   stand on the evidence. Keep running to 5.0ep before deciding on annealing.
2. **Per-position is fully at/above released.** gsm8k conditional c0–c4 = **0.933 / 0.911 / 0.894 / 0.882 /
   0.870** vs released 0.928 / 0.892 / 0.885 / 0.868 / 0.842 — **c0 crosses for the first time** (it was
   0.927, −0.1pt, at 2.5ep). ⟹ nothing about the per-slot mechanism is behind any more.
3. **The residual gap is one dataset.** Non-chat 99.4%; mt-bench 92.5%. That is the rollout distribution
   (99.96% single-turn), not a model defect — the lever is multi-turn rollout data, not the recipe.

⚠ **`~/eval_ep2p5_ropefix_all.txt` was destroyed.** The 3.0ep eval was first launched with the *2.5ep* `tee`
target and aborted at the smoke test; `tee` truncates on open, so that file now holds only the aborted
header. The 2.5ep numbers survive only as the ledger transcription. **Rule: the `tee` target is part of the
command — change it before re-running, not after.**

### `ConfidenceHead` → `bias=False`

Acting on the finding above, `models/dspark/model_definitions.py:88` is now
`nn.Linear(input_dim, 1, bias=False)`, so our module matches the released `mtp.*` layout (and its own
`expected_draft_keys` contract). Existing checkpoints are unaffected: `train/checkpointer.py:311` already
loads with `strict=False`, so a legacy `confidence_head.proj.bias` is reported unexpected and ignored —
verified. The exemption in `verify_dspark_conversion.py` is kept and annotated as legacy-only.

**A/B on the full unit suite** (to prove the change is inert): pristine HEAD = **18 failed / 476 passed**,
with the change = **16 failed / 479 passed**. **New failures introduced: zero.** The two that flipped green
are `test_dspark_metrics.py::{test_perfect_draft_low_loss_high_accept, test_confidence_cumprod_bias_sign}`,
which were **already red** — they encoded the OLD `sample_from_anchor=False` convention and were never
updated when the default flipped. Fixed to the production convention (all-ones mask → accept_len 3.0 at
block_size=2; full mismatch → cumprod bias 1.0) and a new `test_perfect_draft_accept_len_anchor_convention`
pins the legacy convention explicitly at 2.0, so the two can no longer silently swap.

The remaining **16 pre-existing failures** are unrelated to this line and are recorded here so nobody
re-diagnoses them: `train/test_draft_config_init.py` ×7 (fake args lack `init_on_meta`),
`models/test_mtp_model.py` ×5, `models/test_mtp_attention.py`, `models/test_mtp_frozen_weights.py`,
`convert/test_eagle3_converter.py`, `train/test_data.py` ×1 each.

### 3.5ep eval — mean **4.36 = 98.7%**; the climb IS flattening; ★ the LR was already annealing all along

`dsv4_dspark_ep3p5_ropefix_vllm-77w` (ckpt `/3`, `global_step` 87,136), 176 A3-single, conc48, ns5,
log `~/eval_ep3p5_ropefix_all.txt`.

| dataset | n | 3.0ep | **3.5ep** | Δ | released | % |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 1309 | 4.796 | **4.822** | +0.026 | 4.658 | **103.5%** |
| math500 | 490 | 4.519 | **4.548** | +0.029 | 4.661 | 97.6% |
| humaneval | 154 | 4.855 | **4.832** | **−0.023** | 4.942 | 97.8% |
| mbpp | 247 | 4.512 | **4.504** | **−0.008** | 4.535 | 99.3% |
| mt-bench | 70 | 3.046 | **3.107** | +0.061 | 3.294 | 94.3% |
| **mean** | | 4.346 | **4.363** | **+0.017** | 4.418 | **98.7%** |
| **non-chat (4)** | | 4.671 | **4.677** | +0.006 | 4.699 | **99.5%** |

1. **The plateau call is back on, and this time the data supports it.** Deltas
   +0.22/+0.12/+0.07/+0.04/+0.06/**+0.02**. The +0.06 at 3.0ep — which made me retract "approaching
   plateau" — now reads as jitter around a decelerating trend, not a re-acceleration. **Non-chat is
   static (+0.006).** The only real gain is **mt-bench +0.061** (90.7% → 92.5% → 94.3% over the last
   three checkpoints): the laggard is closing while everything else has converged.
2. **Two datasets moved DOWN** — humaneval −0.023, mbpp −0.008. They are the two SMALLEST sample sets
   (154 / 247). Greedy temp0 makes a *given* draft reproducible, so these are real draft-to-draft
   differences rather than serve noise; but at n=154 a −0.02 is not worth acting on.
3. gsm8k conditional c0–c4 = 0.934/0.912/0.897/0.888/0.871, still above released at every position.

### ★ Correction: "anneal LR→0" is not a future lever — this run has been annealing since step 4,979

The `ep2p0-ropefix` ledger row concluded "**next lever for the last ~4%: anneal LR→0 (the DeepSpec
full-anneal we have never run)**". That is **wrong**. The run's own launcher passes
`--scheduler-type cosine --scheduler-warmup-ratio 0.04` over `EPOCHS=5`, i.e. a cosine decay to
**exactly zero** at the final step. Verified against the log rather than assumed:

```
TOT = 5 x 24,896 = 124,480    warmup = round(0.04 x TOT) = 4,979
lr(gs) = 2e-4 * 0.5 * (1 + cos(pi * (gs-4979)/(124480-4979)))
lr(87,138) = 4.444e-05        <-- log at that step prints lr=4.45e-05   MATCH
```

so the remaining schedule is **4.44e-5 (3.5ep) → 2.07e-5 (4.0ep) → 5.31e-6 (4.5ep) → 0 (5.0ep)**.
⟹ **the last 1.5 epochs ARE the full anneal.** There is no anneal-or-not decision to make; the
correct action is to let the run finish. Whatever the anneal is worth will show up in the 4.0 / 4.5 /
5.0ep checkpoints, and *that* is the number to compare against the 3.5ep plateau.

Remaining: 4.0ep @ gs 99,584 (~11 h), 4.5ep @ 112,032, 5.0ep @ 124,480 (~33 h at the measured
3.17 s/step). Three checkpoints left, not one — the run writes two saves per integer epoch dir.

### 4.0ep eval — ★ the non-chat average PASSES the released draft (100.6%); the 3.5ep plateau was noise

`dsv4_dspark_ep4p0_ropefix_vllm-77w` (ckpt `/3` = epoch3_end, `global_step` 99,584), 176 A3-single,
conc48, ns5, log `~/eval_ep4p0_ropefix_all.txt`. 2378/2378 bit-exact.

| dataset | n | 3.5ep | **4.0ep** | Δ | released | % |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 1309 | 4.822 | **4.831** | +0.009 | 4.658 | **103.7%** |
| math500 | 490 | 4.548 | **4.573** | +0.025 | 4.661 | 98.1% |
| humaneval | 154 | 4.832 | **4.924** | +0.092 | 4.942 | 99.6% |
| mbpp | 247 | 4.504 | **4.574** | +0.070 | 4.535 | **100.9%** |
| mt-bench | 70 | 3.107 | **3.062** | −0.045 | 3.294 | 93.0% |
| **mean** | | 4.363 | **4.393** | +0.030 | 4.418 | **99.4%** |
| **non-chat (4)** | | 4.677 | **4.726** | +0.049 | 4.699 | ★ **100.6%** |

**Two datasets now exceed the released draft** (gsm8k 103.7%, mbpp 100.9%) and humaneval is at 99.6%.
**The four non-chat datasets average ABOVE released for the first time.** The whole remaining headline
gap is mt-bench, i.e. multi-turn chat = a rollout data-distribution problem, not a model one.

### ★ Retraction: the 3.5ep "the climb IS flattening" call was over-reading one point

Written at 3.5ep: *"The plateau call is back on, and this time the data supports it."* It did not.
Lining up 3.0 → 3.5 → 4.0 for the three SMALL sets shows them alternating and cancelling:

```
humaneval (n= 154)   4.855 → 4.832 (−0.023) → 4.924 (+0.092)
mbpp      (n= 247)   4.512 → 4.504 (−0.008) → 4.574 (+0.070)
mt-bench  (n=  70)   3.046 → 3.107 (+0.061) → 3.062 (−0.045)
```

At 3.5ep humaneval and mbpp happened to dip together while mt-bench rose, producing the +0.02 that
read as a plateau; at 4.0ep all three reversed. Meanwhile the two LARGE sets — gsm8k (n=1309) and
math500 (n=490) — have risen at **every** checkpoint with no inflection at all.

**Method note, for every future checkpoint:** the per-half-epoch deltas (+0.22/+0.12/+0.07/+0.04/
+0.06/+0.01/+0.03) are now the *same order of magnitude* as the per-checkpoint bounce of the small
sets. ⟹ **convergence cannot be called from a single point at ±0.03 resolution.** Read the trend on
gsm8k + math500 (which carry 1799 of the 2270 samples), quote the 5-set mean as the headline, and
treat any single-checkpoint move in humaneval / mbpp / mt-bench as provisional until the next point
confirms the direction. Both this and the earlier "anneal LR is a future lever" error came from the
same habit — turning one observation into a conclusion.

gsm8k conditional c0–c4 = **0.936/0.912/0.896/0.888/0.880** vs released 0.928/0.892/0.885/0.868/0.842:
above at every position, and the **c4 margin has widened to +3.8pt** (was +1.4pt at 2.5ep). LR here is
2.07e-05, i.e. two thirds through the cosine anneal.

⚠ Both 4.0ep and 4.5ep were saved before either was converted (the 4.0ep notification was missed).
4.0ep sits in `/3`, which nothing writes to again — safe. 4.5ep sits in `/4` and **is overwritten by the
5.0ep epoch-end save**; both were converted in time. Checkpoint→dir mapping for this run:
`/0`←0.5,1.0ep · `/1`←1.5,2.0 · `/2`←2.5,3.0 · `/3`←3.5,4.0 · `/4`←4.5,5.0.

### 4.5ep eval — mean **4.41 = 99.7%**, THREE datasets above released; convergence finally has large-sample support

`dsv4_dspark_ep4p5_ropefix_vllm-77w` (ckpt `/4` = epoch4_step12448, `global_step` 112,032), 176
A3-single, conc48, ns5, log `~/eval_ep4p5_ropefix_all.txt`. 2378/2378 bit-exact.

| dataset | n | 4.0ep | **4.5ep** | Δ | released | % |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 1309 | 4.831 | **4.840** | +0.009 | 4.658 | **103.9%** |
| math500 | 490 | 4.573 | **4.564** | −0.009 | 4.661 | 97.9% |
| humaneval | 154 | 4.924 | **4.954** | +0.030 | 4.942 | **100.2%** |
| mbpp | 247 | 4.574 | **4.553** | −0.021 | 4.535 | **100.4%** |
| mt-bench | 70 | 3.062 | **3.122** | +0.060 | 3.294 | 94.8% |
| **mean** | | 4.393 | **4.407** | +0.014 | 4.418 | **99.7%** |
| **non-chat (4)** | | 4.726 | **4.728** | +0.002 | 4.699 | **100.6%** |

**humaneval crosses this step**, so gsm8k / humaneval / mbpp are all now above the released draft and
the overall mean is 0.3% short of it.

**Applying the method note from 4.0ep — read the LARGE sets, not the mean:**

```
gsm8k  (n=1309)  +0.184 +0.135 +0.073 +0.052 +0.043 +0.026 +0.009 +0.009
math500(n= 490)  +0.173 +0.167 +0.023 +0.054 +0.034 +0.029 +0.025 −0.009   <- first decline
```

gsm8k's per-checkpoint gain has decayed monotonically by 20× and is **flat at +0.009 for two
consecutive checkpoints**; math500 declines for the first time after eight straight rises. **This is
the convergence signal — on 1799 of the 2270 samples, not on a single-point mean move.** Unlike the
3.5ep call it does not rest on the noisy small sets. Still: it is one step from the end, 5.0ep is the
confirmation, and **no decision rides on it** — the run terminates at 5.0ep regardless and the LR is
already down to 5.31e-06.

gsm8k conditional c0–c4 = **0.936/0.914/0.900/0.886/0.878** vs released 0.928/0.892/0.885/0.868/0.842
— above at every position, c4 margin +3.6pt.

### Next: AR (no-spec) baseline at conc48 — closes the open `no-spec base` TODO

Decided with the user: measure the autoregressive baseline so every ledger row gets a speedup
denominator. It is **draft-independent** (target + serve only), so it is measured once and reused.

- **conc48 first** — same concurrency as every existing row, so the comparison is apples-to-apples
  with the whole ledger. ⚠ Expect a modest number: at conc48 the serve is **throughput-bound**, and
  spec decode spends compute on drafting that would otherwise serve more requests. This is the same
  effect already seen when accept_len rose while tok/s did not track.
- **conc1 later** — that is where spec decode actually pays, and it is the number that matches what
  people mean by "speedup". Needs BOTH arms re-run at conc1 (2 runs), so it is a separate exercise.
- **Method:** identical command, serve started **without `DRAFT=`**. `run_dspark_eval.sh` tolerates
  this — its step [3/4] counter check is `grep ... || echo`, so zero spec counters only print a
  notice and the benchmark proceeds. Compare **output tok/s** and **mean ITL**; **TTFT must be
  ~unchanged** between the arms (it is prefill, draft-independent) — that is the free validity check
  that the two runs saw the same machine state.

### AR (no-spec) baseline @ conc48 — **mean speedup 1.42× (1.21–1.77×)**; the long-open denominator is closed

Serve started **without `DRAFT=`**, otherwise the identical command (`DATASET=all CONCURRENCY=48`).
`run_dspark_eval.sh` tolerates the missing spec counters as predicted — step [3/4] printed its notice
and the benchmark ran. Log `~/eval_ar_base_conc48.txt`. Denominator is draft-independent, so this is
measured once and reused for every row in the ledger.

| dataset | AR tok/s | spec tok/s (4.5ep) | ×(thru) | ×(wall) | ×(E2E) | ms/token AR→spec |
|---|---:|---:|---:|---:|---:|---|
| gsm8k | 460.7 | 596.4 | **1.29×** | 1.29× | 1.28× | 102.6 → 79.7 |
| math500 | 612.3 | 819.6 | **1.34×** | 1.34× | 1.28× | 66.8 → 52.4 |
| humaneval | 320.9 | 568.6 | **1.77×** | 1.81× | 1.43× | 103.2 → 73.6 |
| mbpp | 669.0 | 988.9 | **1.48×** | 1.47× | 1.39× | 61.5 → 44.1 |
| mt-bench | 446.8 | 540.9 | **1.21×** | 1.18× | 1.12× | 114.3 → 99.9 |
| **mean** | | | **1.42×** | | | |

Three independent estimators (throughput ratio, wall-clock ratio, E2E-latency ratio) agree to ~1%.

### ★ Correction: "TTFT must be unchanged" is NOT the validity check for spec-vs-AR

Stated in the 4.5ep entry when planning this measurement. Measured TTFT is **1.50–2.43× HIGHER** with
spec (gsm8k 467→1134, math500 422→904, humaneval 558→1092, mbpp 485→803, mt-bench 492→740 ms).

That is **expected, not a broken comparison**. At conc48 TTFT is dominated by **queueing**, and with
drafting on, every engine step costs a draft forward plus a 6-token verify, so requests are admitted
more slowly. The original rule was derived from comparing **two drafts** (both paying the same
per-step cost, so TTFT genuinely isolates machine state) — it does not transfer to spec-vs-AR.
Likewise **`Mean/Median ITL` is not comparable across arms**: spec emits an accepted block at once, so
its ITL measures the gap between *blocks*, not tokens (median 416 ms for ~4.8 tokens ≈ 87 ms/token,
against AR's 48.8 ms — which would wrongly suggest spec is slower).

**The correct check is total output tokens** — greedy over identical prompts ⟹ identical work:
+0.6 / −0.4 / −1.9 / +0.5 / +2.2 % across the five sets, all within ±2.2%. Use this from now on.

### Observation worth following up: our spec decode is not bit-lossless vs AR

Greedy + a *lossless* speculative decoder should emit **token-for-token identical** output, so the
totals above should match exactly rather than to ±2%. Most likely cause is **batch-dependent numerics
in the target forward** (spec runs different batch shapes ⟹ different reduction order ⟹ an occasional
different argmax at a near-tie), which is not by itself a defect. But it means **"bit-identical to the
AR path" is something we have never verified**. A direct check (same prompts, greedy, diff the output
strings) is cheap and has not been run.

### Why speedup does not track accept_len

gsm8k has the highest non-chat accept_len (4.840) but the **lowest** non-chat speedup (1.29×), while
humaneval (4.954) gets **1.77×**. At fixed concurrency the win depends on how saturated the engine is:
humaneval's 154 samples finish in 31–56 s and never fill the batch, so the latency win shows; gsm8k /
math500 / mbpp run long and saturate, so the gain is capped by throughput. ⟹ **conc48 systematically
understates the draft. 1.42× is a conservative lower bound** — quote it as such. The conc1 pair (both
arms re-run) is the figure that isolates the draft's contribution, and is the open TODO.

## 2026-08-10 — 🏁 RUN COMPLETE: `ckpt_faithful_ep_20260804_165215`, 10/10 checkpoints, LR annealed to 0

Final checkpoint `/4` (epoch4_end, `global_step` 124,480 = 5 × 24,896) → `dsv4_dspark_ep5p0_ropefix_vllm-77w`,
2378/2378 bit-exact. Eval log `~/eval_ep5p0_ropefix_all.txt`.

### The full curve (5-dataset mean accept_len)

| ep | gsm8k | math500 | humaneval | mbpp | mt-bench | mean | %rel | non-chat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 4.309 | 4.068 | 4.298 | 3.908 | 2.627 | 3.842 | 87.0% | 4.146 |
| 1.0 | 4.493 | 4.241 | 4.585 | 4.167 | 2.796 | 4.056 | 91.8% | 4.371 |
| 1.5 | 4.628 | 4.408 | 4.706 | 4.300 | 2.865 | 4.181 | 94.6% | 4.511 |
| 2.0 | 4.701 | 4.431 | 4.745 | 4.412 | 2.980 | 4.254 | 96.3% | 4.572 |
| 2.5 | 4.753 | 4.485 | 4.818 | 4.428 | 2.988 | 4.294 | 97.2% | 4.621 |
| 3.0 | 4.796 | 4.519 | 4.855 | 4.512 | 3.046 | 4.346 | 98.4% | 4.671 |
| 3.5 | 4.822 | 4.548 | 4.832 | 4.504 | 3.107 | 4.363 | 98.7% | 4.677 |
| 4.0 | 4.831 | 4.573 | 4.924 | 4.574 | 3.062 | 4.393 | 99.4% | 4.726 |
| 4.5 | 4.840 | 4.564 | 4.954 | 4.553 | 3.122 | 4.407 | 99.7% | 4.728 |
| **5.0** | **4.849** | **4.565** | **4.933** | **4.555** | **3.079** | **4.396** | **99.5%** | **4.726** |
| *released* | 4.658 | 4.661 | 4.942 | 4.535 | 3.294 | 4.418 | 100% | 4.699 |

**Headline:** mean **99.5%** of the released DeepSeek draft; the four **non-chat** datasets average
**100.6% — above it**; gsm8k **104.1%** and mbpp **100.4%** individually exceed it. **End-to-end speedup
over autoregressive decoding = 1.39× mean (1.18–1.64×) at conc48**, a conservative lower bound.

### Convergence — decided on the large-sample criterion, as agreed at 4.0ep

```
gsm8k (n=1309) per-checkpoint delta:
  +0.184 +0.135 +0.073 +0.052 +0.043 +0.026 +0.009 +0.009 +0.009
last three means: 4.3928 / 4.4066 / 4.3962   -> spread 0.0138
```

A 20× monotone decay settling on the **same +0.009 three checkpoints running**, and a three-point mean
spread of 0.0138 — smaller than the single-step bounce of the small sets (mt-bench alone swings ±0.06).
⟹ **4.0 / 4.5 / 5.0ep are statistically one point.** Over the last three steps every dataset oscillates
about zero except gsm8k's steady +0.009. The LR reached exactly 0, so the anneal is complete too.

### Deliverable = 5.0ep, deliberately NOT the highest-mean checkpoint

4.5ep has the highest mean (4.4066 vs 4.3962, **+0.011**) but that is *inside* the bounce. 5.0ep is the
end of the schedule (LR=0) and is where the **largest-sample** dataset peaks — gsm8k 4.849, its maximum
over all ten checkpoints. Picking 4.5ep on +0.011 would repeat exactly the single-point over-reading
that had to be retracted at 3.5ep and again at 4.0ep. **Recording the reasoning, not just the choice.**

### What this run cost / measured

- 124,480 steps at a measured **3.17–3.20 s/step** end-to-end (the log's `profile/step_ms ≈ 1.95–2.03e3`
  is ~60% optimistic — the rest is HS fetch waits, dataloader stalls and checkpoint writes).
- Checkpoint→dir map: `/0`←0.5,1.0ep · `/1`←1.5,2.0 · `/2`←2.5,3.0 · `/3`←3.5,4.0 · `/4`←4.5,5.0.
  All ten were converted and evaluated; none was lost.

## 2026-08-11 — PR #942 A/B: global loss normalization is free, and the imbalance it fixes is systematic

Reviewer `eldarkurtic` asked on the PR for runtime implications and short training runs with accuracy
evals. Run `ckpt_faithful_ep_20260810_234322` is that experiment: the canonical recipe with
`DSPARK_GLOBAL_LOSS_REDUCE=1` added and nothing else changed, against the completed
`ckpt_faithful_ep_20260804_165215` as the OFF arm.

### Accuracy at 0.5ep — indistinguishable

| dataset | OFF | ON | Δ |
|---|---:|---:|---:|
| gsm8k | 4.309 | 4.298 | −0.011 |
| math500 | 4.068 | 4.048 | −0.020 |
| humaneval | 4.298 | 4.326 | +0.028 |
| mbpp | 3.908 | 3.911 | +0.003 |
| mt-bench | 2.627 | 2.626 | −0.001 |
| **mean** | **3.8420** | **3.8418** | **−0.0002** |
| **non-chat (4)** | **4.1457** | **4.1458** | **+0.0000** |

The non-chat four sum to the same number (16.583) in both arms. Largest per-dataset move is 0.028 on
humaneval, the smallest sample set, inside the ±0.03 resolution established at 4.0ep.

**Read it as "no regression", not as "no effect worth having".** The argument for the change is that
the per-rank objective depends on `world_size` and on how the sampler happened to shard the data;
this run says adopting the correct objective costs nothing measurable.

### The imbalance is real, and it does not average out

From `profile/sup_tokens_*` in the same run (@2095 logged steps):

```
cumulative tokens/rank: r0=5731147 r1=5466186 r2=5256060 r3=5142604
                        r4=5027599 r5=4887614 r6=4845435 r7=4801747
heaviest/lightest = 1.194   r0 +11.4%  …  r7 −6.7%
first half r0 +11.5%   second half r0 +11.3%   deviation correlation +0.997
24/2095 steps had a rank with ZERO supervised tokens
```

Three independent signals that this is structural, not sampling noise:

1. **It does not decay.** Cumulative skew read 1.185 @889 steps, 1.190 @1745, 1.194 @2095. Zero-mean
   noise would shrink like 1/√n — doubling the steps should have taken +11.3% to about +8%.
2. **Cumulative totals are monotone in rank index** (r0 > r1 > … > r7). Random assignment does not do
   that; the sampler distributes in index order, so rank 0 systematically receives the densest packs.
   *(The sampler mechanism is inference; the monotonicity and stability are measured.)*
3. **Half-vs-half correlation +0.997** — the same ranks are heavy and light in both halves.

⟹ Under per-rank normalization a token on r7 is weighted ~19% more than an identical token on r0,
permanently. Because the packing order (not the sample identity) determines it, the bias attaches to
supervision-DENSITY: dense packs are systematically down-weighted.

### Runtime — no measurable cost

step_ms steady across three measurements: **+20 / +50 / +40 ms** (~1–2% of 2080 ms). It is not stable,
whereas one scalar all-reduce would add a constant; the spread is the same order as the ~2.6% machine
drift already on record from the 1.5→2.0ep throughput dip. A dedicated micro-benchmark would be needed
to resolve the true cost, which is expected to be microseconds.

⚠ **`~/eval_ep0p5_ropefix_all.txt` was destroyed** — this eval was tee'd over it, the same
name-reuse mistake that took out the 2.5ep log. The OFF-arm numbers survive only as the ledger
transcription. **The `tee` target is part of the command; change it before pressing enter.**

## 2026-08-13 — PR #942 A/B run status @ gs 59,180; ⚠ the 2.5ep window will close before tomorrow

Run `ckpt_faithful_ep_20260810_234322` (ON arm, `DSPARK_GLOBAL_LOSS_REDUCE=1`) is at
**global_step 59,180 / 124,480 = 47.5%**. At the measured 3.17 s/step the remaining 65,300 steps
are **~2.4 days**.

### Save grid — what is safe and what is not

Interval 12,448 (= 0.5 epoch); dirs cycle `/0 /1 /2 /3 /4` and **each dir is written exactly twice,
then never again**, so a dir becomes permanent once its second (integer-epoch) save lands.

| step | epoch | dir | state as of gs 59,180 |
|---:|---|---|---|
| 24,896 | 1.0 | `/0` | ✅ permanent (already converted + eval'd) |
| 37,344 | 1.5 | `/1` | ❌ overwritten by 2.0ep |
| **49,792** | **2.0** | **`/1`** | ✅ **permanent — convertible any time** |
| **62,240** | **2.5** | **`/2`** | ⏳ lands ~2 h 40 m from gs 59,180 |
| 74,688 | 3.0 | `/2` | overwrites 2.5ep — **11.0 h window** |

**Checkpoint handling — settled.** The user converts **every** checkpoint as it lands, so the
11 h window on each mid-epoch save is covered by prompt conversion; the evals themselves are batched
later (serve boxes busy) against the already-converted weights. No copy-out workaround needed. The
window numbers above remain the reason conversion must happen promptly rather than in bulk.

**Recording rule for this run.** Each eval result is appended to
[`ascend-npu-dsv4-dspark-eval-results.md`](ascend-npu-dsv4-dspark-eval-results.md) **as it arrives**,
one row per checkpoint, with the OFF-arm value alongside — this run is an A/B, so a row without its
paired OFF-arm number is not usable. Record per row: 5 per-dataset accept_len, the 5-dataset macro
mean, the non-chat mean, per-position cumulative accept rates, throughput, the trainer ckpt dir, the
converted draft name, and the eval log path. Do not defer these to the end of the run.

### OFF-arm reference (same recipe, per-rank loss normalization) — for whichever point gets eval'd

5-dataset mean accept_len: 1.0ep **4.056** · 2.0ep **4.254** · 2.5ep **4.294** · 3.0ep **4.346**.
Convert cmd pattern and the expected `2378/2378 bit-exact` / 83 input tensors / 0 skipped are in the
`ascend-npu-dsv4-dspark-pipeline.md` §4 block; substitute the run TS and the `epXpY_lossreduce` name.

### Still outstanding (all one-liners on the box, no rush)

- `~/eval_ar_base_conc48.txt` per-token latency — needed to replace the DERIVED ~57 ms AR value in the
  eval-results ledger's mt-bench divisor-bug note with a measured one.
- Rollout truncation rate (`finish_reason=length` share of the cleaned jsonl) — the 公众号 article
  currently says "触及 3072 上限的样本占比较低" with no number.
- Dedup before/after row counts (`wc -l` of `out_bf16/rollout_*.clean.jsonl` vs
  `out_bf16_clean/rollout_all.clean.jsonl` = 775,965) — the article says "按提示去重" with no percentage.
- **`faithful_ep_20260810_234322.log`** — the ON arm is the only run with `profile/sup_tokens_ranks`;
  it is the data source for the per-rank supervised-token figure (§4.3), the one finding in the
  article that has hard numbers but no plot.

## 2026-08-13 — eval no longer spends the warmup samples (`KEEP_WARMUP=1` is the new default)

`Evaluator.py` shuffled each dataset with a fixed `random.seed(42)`, sent the first 10 as warmup, then
**dropped them** (`samples = samples[actual_warmup:]`). Every ledger row up to today is therefore on
1309/490/247/154/**70** instead of 1319/500/257/164/**80** — 12.5% of mt-bench and 6.1% of humaneval
were paid for and thrown away.

**The old numbers are not wrong.** The seed is fixed, so the *same* 10 were dropped in every run ever
made, on every draft — released bar, ON/OFF arms, every checkpoint. All comparisons were on byte-identical
sample sets. What the drop costs is a fixed offset between our absolute value and the full-set mean,
of order `√10·σ/N`: ~0.03 on mt-bench, ~0.019σ on humaneval, ~0.002 on gsm8k. A constant, not scatter.

It could have been much worse: mt-bench's 80 questions ship ordered by category (8 × 10). The
`random.shuffle` sits at `Evaluator.py:127`, *before* the warmup slice — had the slice come first,
`samples[:10]` would have deleted an entire category.

**Change (user's call: "宁愿测出来速度有点误差").** New `--keep-warmup-samples` flag: warm up on the
first 10, then **flush the prefix cache a second time** and measure all N including those 10. The
second flush is why the speed cost is smaller than feared — without it those 10 would enter the timed
phase with their prefill cached.

- `Evaluator.py` — flag **defaults OFF**, so the shared team client still reproduces every earlier
  number byte-for-byte (its docstring mandates cross-team identity).
- `run_eval.sh`, `run_dspark_eval.sh`, `eval_trainsample.py` — new `KEEP_WARMUP` env var, **default 1**.
  `KEEP_WARMUP=0` reproduces any pre-cutover row.

⚠ **Do not mix the two modes inside one comparison.** The A/B in flight
(`ckpt_faithful_ep_20260810_234322` vs `ckpt_faithful_ep_20260804_165215`) has its OFF arm already
measured under the old mode, so **every remaining A/B eval must run `KEEP_WARMUP=0`** — the OFF-arm
reference values (1.0ep 4.056 · 2.0ep 4.254 · 2.5ep 4.294 · 3.0ep 4.346) are on the 70/154/247 sets.
Switch to the new default only for runs whose whole comparison set is post-cutover.

## 2026-08-17 — training line closed: the run finished, then one batch re-measured everything

### The run

`ckpt_faithful_ep_20260810_234322` (the `DSPARK_GLOBAL_LOSS_REDUCE=1` arm) reached **global_step
124,480 = 5.0ep**, LR annealed to exactly 0. The save grid held: `/0`…`/4` each written twice, so the
five integer-epoch checkpoints are permanent. The 4.5ep save was converted inside its ~1 h window
before 5.0ep overwrote `/4` — worth it, because 4.5ep is where the OFF arm peaked.

Disk was cleared first: 38 stale trainer checkpoint dirs deleted (31 empty shells + 5×113G + 2×226G)
= **1.0 TB**, `/` from 93% to 80%. Both keepers verified intact at 565G each.

### The batch driver — `examples/ascend_npu_dflash/eval_all_drafts.sh`

Evaluating 17 drafts by hand is 17 chances to pair a number with the wrong weights. The driver does
stop-serve → start-serve-with-DRAFT → wait-ready → full 5-dataset eval → stop-serve, appending a
banner carrying the **absolute weight path** directly above each result, and printing consolidated
accept-length and tok/s tables at the end. **18/18 OK, 0 failed, 7h37m unattended.**

Three things it had to learn on the way, all worth keeping:

1. **`pkill -f vllm` is not enough.** vLLM renames its subprocesses `VLLM::EngineCore` / `VLLM::Worker`
   — UPPERCASE — while the API server is `vllm serve`. `pkill -f` is case-sensitive, so the naive
   pattern killed the server and left the engine cores holding every byte of HBM. All kills use `-i`
   now. (The user's own habit, `ps -ef | grep VLLM`, was right and mine was wrong.)
2. **Idle HBM is ~3.1 GB/chip on this box**, so gating teardown on an absolute MB threshold either
   never fires or fires instantly depending on the box. The right question is *"is any process holding
   a device"*, which `npu-smi info`'s process table answers directly (`No running processes found`).
3. **24 zombie `[VLLM::Worker] <defunct>` processes, 17 days old**, all parented to a root-owned
   `sleep infinity` — the container's PID 1, which never `wait()`s. A zombie holds no NPU and no signal
   can touch it, but `pgrep` still reports it, so every teardown burned its full 180 s kill timeout:
   ~1.8 h across the batch. `procs_alive()` now skips `Z` state. **Do not kill that parent** — it is
   the container init.

### What the batch actually changed

Full numbers: [`ascend-npu-dsv4-dspark-eval-results.md`](ascend-npu-dsv4-dspark-eval-results.md),
new top section. Three results that change how earlier rows should be read:

★ **A ±0.02 difference on the 5-dataset mean is noise, demonstrated rather than argued.** Three #942
arm pairs were measured twice with identical weights — old sample set vs full — and **two of the three
flipped sign** (0.5ep −0.0002 → +0.0226; 4.5ep −0.0170 → +0.0050) purely from restoring 10 prompts per
dataset, 0.4% more data. This retires the open question in the `ep4p5-lossreduce` row and, more
usefully, invalidates the "CURRENT BEST" promotions in the ⭐ matrix rows, which step by 0.001–0.02.

★ **#942 is settled on the accuracy side.** Six pairs spanning the whole run: +0.0226 / −0.0130 /
+0.0238 / +0.0110 / +0.0050 / +0.0024, mean **+0.0086**, 5 of 6 positive, max |Δ| 0.024 — every one
inside the noise band. No effect measurable, no regression. Single seed per arm, so this bounds the
effect rather than proving zero. The PR was rebased onto main after #951 moved the loss layer into
`speculators/losses/` (relocation only, logic unchanged, 4/4 tests pass including two 2-rank gloo
gradient-equivalence tests) and force-pushed; it is now `mergeable`, waiting only on reviewer approval.

★ **The training line is done.** From 4.0ep onward every checkpoint is the same model —
**99.7–99.9% of the released draft on the 5-dataset mean, 100.5–100.9% on the non-chat four** — and the
last three convergence steps (+0.024 / −0.001 / +0.006) are inside noise with LR already at 0. gsm8k
**4.846 = 103.9%**, mbpp 100.9%, humaneval 100.6% all exceed the released draft. **mt-bench 95.5% is the
only remaining gap**, and restoring the dropped prompts narrowed it without closing it.

The AR denominator was re-measured on the same full set (460.11 / 624.98 / 337.50 / 684.63 / 464.13
tok/s), removing the caveat that hung on every speedup number. ⚠ But at conc48 the throughput ratio
does **not** discriminate: ours 1.36–1.38× and the released draft 1.380× are the same number despite
accept_len differing by 0.17 on gsm8k. Per-token latency is the honest statement — **22–34% off the AR
cost** — and a real "speculative speedup" figure still needs conc1 for both arms, which nobody has run.

## 2026-08-17 — conc1, under a deadline: **2.27×**, and why it isn't 4.8×

The box was being reclaimed, so the conc1 plan (both arms, five datasets, ~12 h) was cut to what
actually answers the question: **gsm8k, spec arm full 1319 prompts, AR arm 200**. Numbers and caveats
in the [eval ledger](ascend-npu-dsv4-dspark-eval-results.md), new top section. Three things to carry
forward:

**① The conc48 speedup was an artifact, confirmed.** Same checkpoint, same dataset: 1.302× at conc48,
**2.27× at conc1** — 74% higher. Every speedup figure in this repo predating today was measured where
the engine is throughput-bound, and the conc48 table also shows the ratio failing to separate our
draft from the released one (1.36–1.38× vs 1.380×) despite a 0.17 accept-length gap. **conc48
throughput is not evidence about draft quality and should never be quoted as such.**

**② accept_len is concurrency-independent** — 4.831 at conc1 vs 4.845 at conc48 on identical weights.
That is a free revalidation of the entire conc48 matrix: batching moves throughput, not acceptance.

**③ ★ The MoE verify tax — the finding worth chasing.** I predicted 3.4–3.9× and was wrong; the
mechanism is the interesting part. Decomposed:

    one AR step   = 38.51 ms -> 1 token
    one spec step = 71.39 ms -> 4.831 tokens      (1.85x the cost of an AR step)
    ideal if steps cost the same = 4.83x ;  realized 2.61x ;  efficiency 54%

The dense-model intuition — at batch 1 decode is memory-bandwidth-bound on weight loading, so
verifying 6 tokens is nearly free — **fails on a 256-expert MoE**. One token routes to 6 experts per
layer; six tokens route to up to 36 distinct experts, and the step pays for pulling all of them. The
draft's own 3-layer forward cannot explain the gap: 3 layers against 43 is ~7%, while the observed
overhead is ~85% of an AR step. **Not profiled — this is a hypothesis with an arithmetic bound
around it**, and it is the first thing to profile if anyone wants a bigger number.

⚠ **And the user immediately found the competing explanation, which points the opposite way.** These
runs are **graph mode** (`EAGER=0`, ACLGraph FULL_DECODE_ONLY): the decode graph is captured at
*padded* shapes. At conc1 with `num_spec=5` a step verifies 6 tokens, which very likely pads to a
captured size of 8 — in which case verify is **flat** in `num_spec` up to that boundary and the MoE
routing story measures nothing. One data point at `num_spec=5` fits both models; both are calibrated
to it.

They disagree about the prize, not the direction:

    num_spec   accept_len   padding model   linear-cost model
       5         4.831         2.61x            2.61x
       7         5.799       * 3.13x *          2.68x
       8         6.185         3.34x            2.66x

★ **Both say go UP.** My first write-up suggested a *smaller* `num_spec` and that was wrong — it
looked only at the step getting more expensive and skipped the marginal arithmetic. The marginal
speculated token costs ~11.6 ms per token gained even under the pessimistic model, against a current
average of 14.78 ms, so the average keeps improving until roughly `num_spec=7-8`.

**The decisive test is cheap and needs no retrain:** sweep `num_spec = 1..5` at conc1 on ~100 gsm8k
prompts (all within what the current draft emits) and watch the step time. Flat ⟹ padding dominates
⟹ retraining at `--block-size 8` is clearly worth it (+20%). Rising with k ⟹ the MoE tax is real and
a retrain buys ~3%. Five runs, under an hour.

⚠ `num_spec > 5` is a **training-side** change: our draft emits exactly 5 (`block_size=6` = anchor +
5 mask slots) and so does the released one. No serve flag reaches past that.

**Still missing at conc1:** math500 (abandoned at 61%), humaneval, mbpp, mt-bench. mt-bench will be
materially worse (accept_len 3.15 vs gsm8k 4.85). The AR arm's 200-prompt sampling is fine for a
per-token rate but is not the strict same-set pairing the conc48 rows have. And DP2×TP8 idles half
the deployment at conc1 — the *ratio* is robust, the absolute ms/token is not; a real single-request
deployment would be TP16.

## 2026-08-17 — `num_spec=7`: the block width, not the model, is what is costing us

The user's point: DSpark can be trained short and served long, so test it. Three drafts at
`num_spec=7`, conc48, full sets — released, `ep5p0-ropefix`, `ep5p0-lossreduce`. Validity gate passed
on all three (`num_draft_tokens / num_drafts` = **7.000** exactly). Numbers in the
[ledger](ascend-npu-dsv4-dspark-eval-results.md), new top section.

**It works, and it replicates.** Our two independently-trained arms gain **+0.4678** and **+0.4706**
on the 5-dataset mean — agreeing to 0.003, far inside the ±0.025 noise band established yesterday.

**But the released draft gains more (+0.5188), so our standing falls at the official setting.** And
`num_spec=7` *is* the official setting — DeepSeek's README recipe is `num_spec: 7`, which means every
ns5 row in this ledger, the released bar included, is a non-official operating point:

    5-set mean vs released     ns5  99.84% / 99.90%     ns7  98.83% / 98.93%
    non-chat four              ns5 100.84% / 100.69%    ns7  99.92% / 99.99%

★ **Where it goes is unambiguous.** gsm8k conditional acceptance: we are higher than released at
**every position the draft was trained on** (c0–c5: 0.926/0.905/0.893/0.879/0.863/0.792 vs
0.921/0.889/0.870/0.862/0.818/0.768) and lower only at **c6** (0.657 vs 0.693). Per dataset, every
point we lose is at pos5/pos6. That is over-fitting to the block width, not a weaker draft. The extra
slots also perturb the trained positions (pos0–4 drop 1–3 pt, −0.105 token), which a matched block
width would not.

★★ **Action: the next training run uses `--block-size 8`.** With pos5/pos6 in-distribution and
following our own decay, gsm8k projects to **≈5.78** against released's 5.221, and the pos0–4
perturbation disappears. We have been training at a block width that does not match the serving
configuration the model is meant to run at.

**It also overturns our own earlier finding.** `ep1mid-f1-blk7` ran this exact experiment months ago
and concluded "pos6 nearly dead (2–6%) → γ=6 likely the sweet spot". That checkpoint was trained
under the **degenerate RoPE** (complex freqs cast to bf16 = scale-only, no rotation) and only 1.5
epochs; today the same experiment gives pos5 **44.98%** and pos6 **29.56%**, a 4.7× larger mean gain.
A model with a broken positional encoding cannot extrapolate to unseen positions — so that was a
false negative, and **the RoPE fix restored positional extrapolation, not just in-distribution
acceptance.** That consequence of `feb0066`/`8db8f75` had gone unnoticed. The row is marked
SUPERSEDED with the reasoning attached.

⚠ **Unresolved, and the box is gone: is ns7 faster at batch 1?** conc48 throughput moved only
+3.5%/+5.4%, which means nothing there. The graph-padding and linear-cost models predict **2.95× vs
2.52×** — opposite sides of ns5's measured 2.61×. One 5-minute conc1 run settles it. Until then ns7
is an acceptance win and a latency unknown; **do not quote a speedup for it.**

## 2026-08-18 — ⚠ PRECONDITION for merging the upstream sync: hs_connectors

Not a problem on this branch — recorded here because it **will** be, the moment the
2026-08-13 upstream sync lands, and because the symptom is actively misleading.

Upstream **#735** (`1afb3b2`) turned `hs_connectors` into a **uv workspace member**:
`pyproject.toml` lists `hs-connectors` under `dependencies` and resolves it via
`[tool.uv.sources] hs-connectors = { workspace = true }`. **`pip install -e .` cannot read that
table.** It looks for `hs-connectors` on PyPI, finds nothing, and leaves the dependency
unsatisfied — while `scripts/train.py:94` imports it unconditionally. Every rank dies before a
single module is built.

That matters specifically because **our deployment contract is "install this commit"**: the
install SSOT (`install_npu_env_dspark.sh`) builds every training env with plain pip. The day the
sync merges, that documented path stops producing a working env.

**The symptom hides the cause.** torch_npu's own excepthook recurses ~996 frames while formatting
the failure, so the log is a wall of `RecursionError: maximum recursion depth exceeded`. The real
error is one line at the very bottom, under `Original exception was:`. **When reading any NPU
training failure, go to that marker first** — everything above it can be noise from the handler
rather than from the program.

Reproduced away from the box, which also explains it: with the repo root on `sys.path`,
`import hs_connectors` binds the OUTER `hs_connectors/` directory as a namespace package and
shadows the real one —

    resolved -> None _NamespacePath(['.../hs_connectors'])
    ImportError: cannot import name 'HiddenStatesBackend' from 'hs_connectors' (unknown location)

— verbatim the box's message. Prepending `hs_connectors/src` resolves to
`.../hs_connectors/src/hs_connectors/__init__.py` and the import succeeds.

**Both fixes already exist on `feat/dspark-next-port` (`f48572b`) and MUST come across with the
sync:**

1. `train_dsv4_dspark.sh` prepends `$REPO_ROOT/hs_connectors/src` to `PYTHONPATH` — repairs
   every env that already exists, without reinstalling, and is a no-op where the package is
   installed properly (same files, not a competing copy).
2. `install_npu_env_dspark.sh` installs the workspace member directly after speculators, asserts
   the import, and exits 2 if it still fails. Guarded on the directory existing so it stays
   correct on a pre-#735 checkout.

⚠ Do not "fix" this by reverting anything on this branch. `cc3d2ef` predates #735, has no such
import, and trains fine — there is nothing broken here to revert.
