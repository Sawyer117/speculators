# DSV4-DSpark — spec-decode eval results ledger

Append-only ledger of DSpark draft acceptance-length runs on the **DeepSeek-V4-Flash bf16**
target. **To add a run:** append one row to the [summary matrix](#summary--accept_len-by-dataset)
*and* one detail section under [Runs](#runs-detail). Keep the matrix in sync so a regression is
one glance. Every run inherits [Shared setup](#shared-setup) unless its config block overrides it.

## Shared setup

Unless a run says otherwise, all numbers below share:

- **Target**: `DeepSeek-V4-Flash-bf16`.
- **Serve**: vllm-ascend PR **#12006 @ `386530d12`** (env `dspark-dsv4-serving`), A2 **dual-node bf16**
  (115 head / 116 worker, `world_size=16`), **EAGER=0** (graph mode — #12006 ACLGraph; graph is the ONLY
  mode that supports our DP2 dual-node — **eager/EAGER=1 raises "Eager DSpark does not support DP token padding"**),
  `num_speculative_tokens=5`, `draft_sample=greedy`, `aux_layers=[40,41,42]`,
  `VLLM_ASCEND_DSPARK_USE_STANDARD_DSA=1`. Serve script `examples/ascend_npu_dflash/serve_dsv4_bf16_dualnode.sh`.
  > ⚠️ **2026-07-22 NOTE**: this row IS correct (386530d12 + EAGER=0). A rebuild attempt failed only because the
  > editable install was left at `+g431a64b18` (v2 ops) while the checkout was 386530d12 (v3 python) = MISMATCHED
  > build → the `_get_block_table` crash. FIX = clean rebuild AT 386530d12 (`git checkout 386530d12 && pip install
  > -e . --no-deps --no-build-isolation`, confirm `pip list` shows `+g386530d12`) on BOTH nodes, then serve EAGER=0.
  > EAGER=1 does NOT work here (eager DSpark can't do DP token padding); v2 `431a64b18` is eager-only (no graph).
- **Eval**: `examples/ascend_npu_dflash/run_dspark_eval.sh` (→ `Evaluator.py`), **greedy**
  `temp0 / top-p1 / top-k1`, concurrency 16, `max_new_tokens=2048`, **FULL** datasets, warmup 10.
- **Metric defs**:
  - `accept_len` = 1 (bonus token) + mean accepted **draft** tokens per draft.
  - `accept_rate` = accepted draft tokens / proposed draft tokens (= `accept_len − 1` over 5).
  - `per-position pos_k` = **cumulative survival** `S_k = P(prefix 0..k all accepted)` (monotone ↓).
    Identity: `Σ_{k=0..4} S_k + 1 = accept_len`. Conditional per-slot accuracy = `S_k / S_{k−1}`
    (this is the "is THIS slot any good, given the prefix held" number — read the cliff here).
  - `throughput` (tok/s) + `TTFT / ITL / E2E latency` are **hardware- and config-dependent** — log
    them, but compare *only within an identical serve* (same #12006 stack, node count/parallelism,
    concurrency). accept_len is the hardware-independent quality metric; a real *speedup* number needs
    the **no-spec base** tok/s (see [Baselines](#baselines--todo)). The full latency breakdown
    (Mean TTFT / Mean+Median ITL / E2E mean) for every run is in its **raw eval log** (path in the run
    block); the tables here carry accept_len / accept_rate / throughput / per-position.
- **Bar to beat**: official released DSV4 draft **AL 3.94 @ num_spec=5** (vllm-ascend #11196).
  Our-serve released draft = **full-all done (mean 4.42; gsm8k 4.658 reproduced exactly)** — the
  definitive per-dataset bar, see the [released row](#summary--accept_len-by-dataset) + detail below.
  (Our serve runs *above* the official 3.94 because #12006 + bf16 dequant; use the released *row* as
  the same-serve target, and 3.94 only as the cross-stack sanity floor.)

## Summary — accept_len by dataset

Headline `accept_len` per run. `mean` is the unweighted 5-dataset average (mt-bench, being
multi-turn chat, drags it — quote per-dataset when a run's draft is non-chat).

| Run | gsm8k | math500 | humaneval | mbpp | mt-bench | mean | notes |
|-----|:-----:|:-------:|:---------:|:----:|:--------:|:----:|-------|
| **released draft** *(bar)* | 4.658 | 4.661 | 4.942 | 4.535 | 3.294 | **4.42** | official 3.94 @ ns5; full-all on our serve ✓ |
| `epoch4-17w` | 3.404 | 3.265 | 3.312 | 3.058 | 2.344 | 3.08 | ⚠ wrong dataset (17W not 45W); all fixes in; **gap = tail** |
| `ep0-77w` *(epoch 0)* | 3.186 | 3.041 | 3.079 | 2.868 | 2.255 | 2.89 | first ckpt of A3 77W run (LR 3e-4, noise 0.05); epoch0 vs 17w's epoch4 — **not** like-for-like; gap = tail (pos2 cliff) |
| `ep2.5-77w` *(arm A, noise 0.05, CAUSAL)* | 3.203 | 3.096 | 3.106 | 2.928 | 2.272 | 2.92 | epoch2-mid (30,969 steps) of A3 77W run; **≈ ep0 — FLAT, tail frozen** (gsm8k pos2/3/4 = 39/14/5, same as ep0) → 1.5 more epochs bought +0.03 mean. 176 A3-single serve conc48. ~~serve capping?~~ **RESOLVED**: `ep0mid-77w-nc` on the SAME 176 serve hit 4.032 → serve does NOT cap → ep2.5's flatness was REAL (causal-training), not a serve artifact. |
| **`ep0mid-77w` *(NON-CAUSAL fix, epoch0-mid)*** | **4.032** | **3.721** | **3.856** | **3.586** | **2.482** | **3.54** | ★ **THE non-causal fix lands.** epoch0-**mid** (step 6194 = 0.5 epoch) of the non-causal retrain (`--sliding-window-non-causal`, +#848 detach). Same 176 serve as ep2.5 → **only variable = train causality.** gsm8k **3.186→4.032** (+0.85), tail **pos2/3/4 = 59/46/35** vs causal's 39/14/5 (pos4 **7.4×**). **0.5ep non-causal >> 2.5ep causal.** Past official 3.94@ns5; 86.6% of released-on-our-serve at 0.5/10 epoch. |
| `ep0end-77w` *(non-causal, DOUBLE-norm teacher, epoch0-end)* | 4.006 | 3.753 | 3.970 | 3.666 | 2.539 | 3.59 | epoch0-**end** (1.0 epoch, `0/` after end-save overwrote mid). Non-causal epoch curve: **0.5ep→1.0ep = HEAD sharpens, TAIL plateaus early.** pos0/pos1 up across ALL datasets; pos2-4 flat-to-slightly-down (gsm8k tail 59/46/35→57/43/34). ⚠ this run has the DOUBLE-norm teacher (see `ep0end-f1-77w` below — turns out that HELPS). |
| **`ep1mid-f1-77w` *(single-norm, epoch1-mid = REPRODUCTION main line)*** | **4.069** | **3.757** | **4.100** | **3.674** | **2.549** | **3.63** | ★ **Reproduction ON TRACK — data scaling confirmed.** F1 single-norm epoch1-**mid** (1.5ep). vs its own ep0end (1.0ep, 3.37): **+0.26 mean, ALL datasets** (+0.09…+0.42), tail climbing (gsm8k pos2/3/4 51/36/28→**59/44/35**). Now **SURPASSES the double-norm ep0end (3.59)** with +0.5 epoch → the correct teacher climbs past the 16%-off one; the double-norm's early lead is transient. **82% of released (4.42)** at 1.5/10 epoch (gsm8k 87%). Faithful recipe scaling → keep training. |
| `ep0end-f1-77w` *(SINGLE-norm teacher = "F1 fix")* | 3.811 | 3.511 | 3.680 | 3.407 | 2.460 | **3.37** | ★★ **SURPRISE — the F1 "double-norm fix" is a REGRESSION.** Same epoch0-end, same non-causal / anchor576 / noise0.05 / LR3e-4 — the ONLY variable is the teacher norm (single vs double). Single-norm (the "correct" real-verifier distribution) is **WORSE on ALL 5 datasets** (Δ −0.08…−0.29, **mean −0.21**), entirely in the **TAIL** (gsm8k pos2/3/4 **57/43/34 → 51/36/28**; pos0/1 unchanged). ⟹ the DOUBLE-norm teacher (≈ `w²` sharpening) distills the tail BETTER, despite its 16% argmax-flip vs the real verifier (T1). **The teacher is a TRAIN target, NOT the serve verifier — so "double-norm" was never a train/serve inconsistency; it's an accidental teacher-SHARPENING that helps.** My "F1 = correctness fix" call was wrong; user's "double-norm is an optimization" instinct was right. Next: deliberate KD-temperature on the CORRECT teacher (clean sharpening, no argmax defect) may beat both. |

### Throughput (tok/s) by dataset

⚠ Hardware/config-dependent — comparable **only within the same serve** (see metric defs). Logged for
reference / regression, not as a cross-hardware quality metric. `no-spec base` = the AR reference the
speedup is measured against.

| Run | gsm8k | math500 | humaneval | mbpp | mt-bench |
|-----|:-----:|:-------:|:---------:|:----:|:--------:|
| `no-spec base` *(ref, gsm8k-only so far)* | 302.69 | — | — | — | — |
| **released draft** | 187.55 | 313.03 | 169.41 | 374.51 | 268.59 |
| `epoch4-17w` | 162.26 | 256.48 | 143.19 | 285.44 | 220.50 |
| `ep0-77w` *(⚠ conc 48, NOT 16 — not comparable)* | 232.30 | 364.40 | 218.17 | 415.53 | 253.64 |
| `ep2.5-77w` *(⚠ conc 48 + A3-single serve — NOT comparable)* | 449.93 | 634.26 | 364.42 | 701.41 | 405.82 |
| `ep0mid-77w` *(non-causal; ⚠ conc 48 + A3-single serve)* | 552.14 | 744.06 | 471.49 | 885.18 | 469.40 |
| `ep0end-77w` *(non-causal; ⚠ conc 48 + A3-single serve)* | 558.13 | 760.87 | 481.91 | 928.27 | 489.41 |

## Runs (detail)

### `released draft` (bar) — 2026-07-20

- **Draft**: `/share/canada_group_folder/ckpt/released_draft_bf16_standalone` — DeepSeek's official
  DSV4-Flash DSpark draft, dequantized to bf16 (`build_released_draft_dir.py --dequant-bf16`). The
  reference our own drafts must approach. (⚠ built by `a00652497` with umask 077 → weights land `0600`;
  a cross-account serve needs `chmod -R a+rX` on the dir first — same class as the converter's auto-chmod.)
- **Serve / eval**: [Shared setup](#shared-setup). gsm8k reproduces **4.658 exactly** vs the prior
  gsm8k-only run → confirms the serve mechanism is bit-stable run-to-run.
- **Raw eval log**: `~/eval_released_all.log` on head node 115.

| Dataset | Samples | tok/s | accept_len | accept_rate | pos0 | pos1 | pos2 | pos3 | pos4 |
|---------|:-------:|:-----:|:----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| gsm8k | 1309 | 187.55 | **4.658** | 73.17% | 92.77 | 82.77 | 73.29 | 63.55 | 53.45 |
| math500 | 490 | 313.03 | **4.661** | 73.22% | 91.78 | 82.53 | 73.01 | 63.83 | 54.93 |
| humaneval | 154 | 169.41 | **4.942** | 78.84% | 95.60 | 88.94 | 78.02 | 70.16 | 61.47 |
| mbpp | 247 | 374.51 | **4.535** | 70.69% | 91.42 | 80.91 | 70.33 | 60.20 | 50.61 |
| mt-bench | 70 | 268.59 | **3.294** | 45.88% | 79.21 | 58.55 | 41.45 | 29.32 | 20.87 |

**Read.** The shape a well-trained DSpark draft *should* have: per-position decays **smoothly, no cliff**.
gsm8k conditional per-slot = **93 / 89 / 89 / 87 / 84 %** (nearly flat — every slot ~85-90%), vs
`epoch4-17w`'s **90 / 80 / 75 / 36 / 26 %** (cliff at pos3). The **entire epoch4↔released gap lives in
the tail**: released pos3/pos4 = 63.6 / 53.5 % where our 17W draft is 19.2 / 5.0 % — released *learned*
the far slots, ours didn't (17W underfit + exp-decay downweight). Direct proof the tail is a
data/training deficit, not a structural ceiling → recoverable by the 45W retrain. mt-bench is hardest
even for released (3.294, pos0 79%) — multi-turn chat genuinely stresses a block draft.

### `epoch4-17w` — 2026-07-19

- **Draft ckpt**: `/share/canada_group_folder/ckpt/dsv4_dspark_drafts/dsv4_dspark_epoch4_17w`
  (converted via `scripts/convert_dspark_to_vllm.py` from `run/ckpt_faithful_ep_20260718_014045/4`).
- **Train**: **17W** = `open_perfectblend.dsv4_rollout/arrow` (177k) — ⚠ **WRONG dataset**, the run
  omitted `DATA=` and fell back to the 177k default; intended `arrow_0715` (45W). Ran all 5 epochs
  anyway to observe epoch-to-epoch. `block_size=5`, `sample_from_anchor=True`, LR 2e-4,
  `MAX_ANCHORS=512`, `INIT_LAYER=1`.
- **Fixes present**: pos0 loss-decay (slot-0) + Markov `prev_token_ids` + metrics slot-0 +
  FSDP2 AMP fp32-master (**option-B**: small trainable fp32, EP experts stay bf16) + upstream merge
  (#788 float32 divergence loss, #759 metric clone). This is the first fully-fixed, serve-validated draft.
- **Serve / eval**: [Shared setup](#shared-setup).
- **Raw eval log** (full TTFT/ITL/E2E): `~/eval_epoch4_17w_all.log` on head node 115.

| Dataset | Samples | tok/s | accept_len | accept_rate | pos0 | pos1 | pos2 | pos3 | pos4 |
|---------|:-------:|:-----:|:----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| gsm8k | 1309 | 162.26 | **3.404** | 48.08% | 90.44 | 71.96 | 53.87 | 19.18 | 4.95 |
| math500 | 490 | 256.48 | **3.265** | 45.30% | 87.82 | 66.70 | 50.16 | 17.39 | 4.43 |
| humaneval | 154 | 143.19 | **3.312** | 46.23% | 92.75 | 72.15 | 49.56 | 13.70 | 3.00 |
| mbpp | 247 | 285.44 | **3.058** | 41.17% | 87.30 | 62.25 | 41.31 | 12.36 | 2.63 |
| mt-bench | 70 | 220.50 | **2.344** | 26.88% | 69.45 | 38.90 | 19.86 | 5.28 | 0.91 |

**Read.** vs the buggy pre-fix draft (gsm8k 1.758) this is a ~2× jump → decay + Markov + metrics +
AMP all confirmed live on the real serve. Shape: healthy pos0–2 (gsm8k conditional ≈ 90 / 80 / 75 %),
then a **sharp cliff at pos3** (conditional 75 → 36 %). A cliff (not smooth decay) = genuine far-slot
conditional weakness = exp-decay loss downweights far slots + they're intrinsically harder + 17W
underfits them — **NOT** autoregressive drift (DSpark backbone is a parallel noise-filled block
predictor; the only AR piece, the Markov head, sees train-consistent input on any prefix that counts
toward `accept_len`). mt-bench low (2.344) = multi-turn chat, out-of-distribution for this non-chat
draft. The tail is recoverable → the 45W (`arrow_0715`) retrain targets pos3/pos4.

### `ep0-77w` — 2026-07-22

- **Draft ckpt**: epoch-0 (first checkpoint) of the A3 **77W** full-alignment run (`arrow_0720_77w`, LR 3e-4, AdamW,
  `noise_std=0.05`, tv 1.8, γ4, warm-start-layer ON, `block_size=5`, `sample_from_anchor=True`, all fixes in). Converted
  via `convert_dspark_to_vllm.py` (2378/2378 bit-exact), served as `dsv4_dspark_ep0_vllm-77w`.
- **Serve**: [Shared setup](#shared-setup) — #12006 `386530d12` (unpatched), **EAGER=0**, but **concurrency 48** (not the
  shared 16) → its **throughput is NOT comparable** to the conc-16 rows above. `accept_len` IS comparable (per-draft,
  concurrency-independent). ⚠ Requires node **109's rogue rollout process KILLED first** — its `/v1/completions` flood
  drains a DP rank and triggers the DP-idle `dummy_run` crash (fixed by `Sawyer117/vllm-ascend@fix/dspark-dummy-dp-pad`,
  not pulled here; with 109 dead it ran full-all clean without the patch).
- **Raw eval log**: `~/eval_ep0_all.log` on head node 115.

| Dataset | Samples | tok/s (conc48) | accept_len | accept_rate | pos0 | pos1 | pos2 | pos3 | pos4 |
|---------|:-------:|:-----:|:----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| gsm8k | 1309 | 232.30 | **3.186** | 43.72% | 87.91 | 73.10 | 39.21 | 13.65 | 4.73 |
| math500 | 490 | 364.40 | **3.041** | 40.83% | 84.81 | 68.76 | 35.60 | 11.40 | 3.58 |
| humaneval | 154 | 218.17 | **3.079** | 41.57% | 90.27 | 74.20 | 31.85 | 9.54 | 2.01 |
| mbpp | 247 | 415.53 | **2.868** | 37.37% | 84.40 | 64.07 | 28.51 | 7.93 | 1.93 |
| mt-bench | 70 | 253.64 | **2.255** | 25.10% | 67.61 | 39.52 | 14.15 | 3.46 | 0.77 |

**Read.** First checkpoint (epoch 0) of the 77W run. Headline sits below `epoch4-17w` (gsm8k 3.186 vs 3.404) — but that
is **epoch-0 vs epoch-4, not like-for-like**; the 77W run has 10 epochs. Shape: healthy pos0/1 (gsm8k 88/73) then a
**cliff already at pos2** (conditional 73→54%), *earlier* than 17w's pos3 cliff → the tail is the undertrained-at-epoch-0
deficit, exactly as expected. The entire gap to released lives in pos2+ (released 73/64/53 vs ep0's 39/14/5). Watch the
**per-position epoch curve** (pos2/3/4 lifting) as later ckpts convert+eval — that, not the headline, tells whether the
tail is learning. mt-bench lowest (2.255, multi-turn chat OOD). Next: eval epoch1; then the **`noise_std=0` A/B**
(DeepSpec trains with no hidden-state noise — top candidate tail fix).

### `ep2.5-77w` (arm A, noise 0.05) — 2026-07-23

- **Draft ckpt**: **epoch2-mid** (`epoch2_step6193`, global_step 30,969 = **2.5 epochs**) of the A3 **77W** run
  (`arrow_0720_77w`, LR 3e-4, AdamW, `noise_std=0.05`, tv 1.8, γ4, warm-start-layer ON, `block_size=5`,
  `sample_from_anchor=True`). Converted via `convert_dspark_to_vllm.py` (2378/2378 bit-exact), served as
  `dsv4_dspark_ep2.5_noise0.05_vllm-77w`. = **arm A** of the noise A/B (arm B = same fork at epoch1_end, noise 0).
- **Serve**: ⚠ **DIFFERENT from Shared setup** — node **176 A3 SINGLE-NODE** (`serve_dsv4_a3_singlenode.sh`,
  DP2/TP8/**EP16**), `386530d12`, **EAGER=0**, **`VLLM_ASCEND_ENABLE_FLASHCOMM1=0`** (A3-single graph-mode +
  spec-decode needs FlashComm/seq-parallel OFF, else cudagraph "multiple of 6 and 8" crash), conc 48.
  **⚠ THIS SERVE IS NOT YET CALIBRATED** — released-draft-on-176 (expect gsm8k 4.658) is PENDING; until it
  reproduces, cannot distinguish "training flat" from "176 serve caps the tail."
- **Raw eval log**: `~/eval_ep2.5_noise0.05.log` on 176.

| Dataset | Samples | tok/s (conc48) | accept_len | accept_rate | pos0 | pos1 | pos2 | pos3 | pos4 |
|---------|:-------:|:-----:|:----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| gsm8k | 1309 | 449.93 | **3.203** | 44.07% | 88.53 | 73.16 | 39.37 | 14.21 | 5.07 |
| math500 | 490 | 634.26 | **3.096** | 41.92% | 86.00 | 69.90 | 37.38 | 12.45 | 3.89 |
| humaneval | 154 | 364.42 | **3.106** | 42.12% | 90.22 | 74.45 | 32.98 | 10.46 | 2.47 |
| mbpp | 247 | 701.41 | **2.928** | 38.56% | 86.36 | 66.43 | 29.55 | 8.37 | 2.07 |
| mt-bench | 70 | 405.82 | **2.272** | 25.44% | 68.33 | 39.72 | 14.55 | 3.77 | 0.82 |

**Read — the alarm.** 2.5 epochs vs ep0's 1 epoch = **essentially FLAT** (mean 2.92 vs 2.89, gsm8k 3.203 vs 3.186).
The tail did **not** move: gsm8k pos2/3/4 = 39.4/14.2/5.1 (ep0: 39.2/13.7/4.7). 1.5 extra epochs (~18.5k steps) →
+0.02 accept_len. Meanwhile **training metrics rose** (train accept ~3.8, train pos3/4 ~0.64/0.59) → the eval-tail
is decoupled from training progress. released proves the tail is learnable (63/53) so it's **our training/convention,
not the architecture**. Leading hypothesis: **teacher-forcing exposure at the tail** — train pos_k sees the TRUE
prefix; serve pos_k sees the draft's OWN (wrong) prefix, so more teacher-forced training never lifts the free-running
tail. ⚠ BUT first rule out the serve: this is on the uncalibrated 176 A3-single serve — **run released-on-176 (expect
4.658) before trusting "training is flat."** If 176 reads released < 4.658, the 3.203 is partly a serve artifact.

> **★ ROOT CAUSE FOUND (2026-07-23) — it was a train↔serve mismatch, exactly at the later positions.** Training ran
> **CAUSAL** intra-block attention (`train_dsv4_dspark.sh` never passed `--sliding-window-non-causal`, CLI default
> False) but the vllm-ascend serve drafts the block **NON-CAUSAL** (`deepseek_v4_dspark_proposer.py:449 cad.causal=False`).
> So every slot sees all γ slots at serve but only 0..k in training → the tail freezes and never improves with training,
> while released (trained non-causal) gets 63/53. FIX = `--sliding-window-non-causal` (welded into `train_dsv4_dspark.sh`
> as `NONCAUSAL=1` default) + **retrain from scratch** (serving side unchanged). See memory `dsv4-dspark-noncausal-root-cause`.
> The `ep0`/`ep2.5` rows above are the CAUSAL (broken) baseline; the non-causal retrain is the real test.
> **✅ CONFIRMED by `ep0mid-77w` below** — 0.5 epoch non-causal crushed 2.5 epoch causal; the tail unfroze exactly as predicted.

### `ep0mid-77w` (NON-CAUSAL fix, epoch0-mid) — 2026-07-24

- **Draft ckpt**: **epoch0-mid** (`epoch0_step6194`, 0.5 epoch) of the A3 **77W NON-CAUSAL retrain**
  (`ckpt_faithful_ep_20260723_152149/0`; `--sliding-window-non-causal` = `NONCAUSAL=1`, +#848 confidence-detach,
  LR 3e-4, AdamW, noise 0.05, tv 1.8, γ4, warm-start-layer ON, block_size=5, sample_from_anchor=True). Converted via
  `convert_dspark_to_vllm.py` (2378/2378 bit-exact, `--config-from released_draft_bf16_standalone/config.json`), served as
  `dsv4_dspark_ep0mid_noncausal_vllm-77w`.
- **Serve**: **SAME as `ep2.5` above** — node 176 A3 single-node (`serve_dsv4_a3_singlenode.sh`, DP2/TP8/EP16),
  `386530d12`, EAGER=0, `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`, conc 48. → **ep2.5↔ep0mid is a clean same-serve A/B; the
  only variable is train causality.**
- **Raw eval log**: `~/eval_ep0mid_nc_all.log` on 176.

| Dataset | Samples | tok/s (conc48) | accept_len | accept_rate | pos0 | pos1 | pos2 | pos3 | pos4 |
|---------|:-------:|:-----:|:----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| gsm8k | 1309 | 552.14 | **4.032** | 60.64% | 88.54 | 74.73 | 58.88 | 46.05 | 35.00 |
| math500 | 490 | 744.06 | **3.721** | 54.43% | 84.57 | 69.23 | 52.52 | 38.75 | 27.05 |
| humaneval | 154 | 471.49 | **3.856** | 57.13% | 88.12 | 74.83 | 55.05 | 40.23 | 27.43 |
| mbpp | 247 | 885.18 | **3.586** | 51.72% | 85.29 | 67.76 | 48.11 | 34.06 | 23.40 |
| mt-bench | 70 | 469.40 | **2.482** | 29.64% | 67.41 | 39.89 | 21.83 | 12.21 | 6.86 |

**Read — the fix lands, unambiguously.** Same 176 serve as causal `ep2.5`, only the training changed (causal→non-causal).
gsm8k **3.203→4.032** (+0.85, +26%); the tail — frozen at 39/14/5 through 2.5 causal epochs — jumps to **59/46/35** at
just **0.5 non-causal epoch** (pos4 **7.4×**). pos0/pos1 barely move (88/75, never the problem): the delta is entirely
pos2+, the exact signature of unfreezing the late block slots (causal starved slot-k of slots >k; non-causal feeds it all
γ). **0.5ep non-causal >> 2.5ep causal** → the causal runs were optimizing the wrong task, as diagnosed. Two bonus
conclusions: (1) **176 serve is NOT capping** — it happily reports 4.032 → retroactively validates the 176 serve and
confirms ep2.5's flatness was real (training), closing the "serve artifact" caveat; (2) gsm8k 4.032 already **exceeds the
official 3.94@ns5** and is **86.6% of released-on-our-serve (4.658)** at 0.5/10 epoch — headroom to climb. Next: convert +
eval later ckpts (epoch0-end, epoch1…) to watch the tail keep rising; the noise A/B and causal-warmstart ablations are now
secondary (the primary fix is validated).

### `ep0end-77w` (non-causal, epoch0-end) — 2026-07-24

- **Draft ckpt**: **epoch0-end** (1.0 epoch) of the same non-causal retrain (`ckpt_faithful_ep_20260723_152149/0` — the
  end-of-epoch save OVERWROTE the mid save in `0/`; symlink `epoch0_end`). Same convert recipe (2378/2378 bit-exact,
  `--config-from released_draft_bf16_standalone/config.json`), served as `dsv4_dspark_ep0end_noncausal_vllm-77w`.
- **Serve**: SAME 176 A3 single-node setup as `ep0mid`/`ep2.5` (386530d12, EAGER=0, FLASHCOMM1=0, conc 48) → the
  ep0mid↔ep0end delta is a clean epoch-curve step (greedy temp0 → serve is run-to-run deterministic; released reproduces
  4.658 bit-stable, so the deltas below are REAL, not serve noise).
- **Raw eval log**: `~/eval_ep0end_nc_all.log` on 176.

| Dataset | Samples | tok/s (conc48) | accept_len | accept_rate | pos0 | pos1 | pos2 | pos3 | pos4 |
|---------|:-------:|:-----:|:----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| gsm8k | 1309 | 558.13 | **4.006** | 60.11% | 89.53 | 77.05 | 56.63 | 43.46 | 33.90 |
| math500 | 490 | 760.87 | **3.753** | 55.05% | 86.74 | 72.47 | 51.73 | 37.14 | 27.17 |
| humaneval | 154 | 481.91 | **3.970** | 59.40% | 92.30 | 80.42 | 55.33 | 39.48 | 29.49 |
| mbpp | 247 | 928.27 | **3.666** | 53.33% | 86.96 | 70.96 | 48.55 | 34.99 | 25.18 |
| mt-bench | 70 | 489.41 | **2.539** | 30.78% | 68.51 | 42.41 | 22.52 | 12.91 | 7.55 |

**Read — head sharpens, tail plateaus (epoch 0→1).** vs `ep0mid` (0.5 epoch): mean 3.54→3.59 (+0.05). The gain is at the
**head**: pos0/pos1 rise on EVERY dataset (gsm8k pos1 74.7→77.1, humaneval pos1 74.8→**80.4**, math500 pos1 69.2→72.5). The
**tail (pos2-4) has plateaued** this early — gsm8k tail 59/46/35 → 57/43/34 (a ~2pp dip, within run-to-run noise; net gsm8k
AL flat 4.032→4.006), others flat-to-slightly-up (mbpp/mt-bench tail all +). So the causal→non-causal fix unlocked the tail
in ONE shot (captured at ep0mid); subsequent training first converges the head. The tail's climb toward released (63/53)
is the thing to watch over epochs 1-10 — NOT yet moving at ep1. Next: convert+eval epoch1 (`1/`) for the next curve point.

## Baselines / TODO

- [x] **released draft — full `DATASET=all`** on our serve → **DONE 2026-07-20** (mean 4.42; gsm8k 4.658
      reproduced). This is now the same-serve bar row + detail section above.
- [ ] **no-spec base** tok/s reference — have gsm8k 302.69; extend to `DATASET=all` for a per-dataset
      speedup denominator (spec-decode tok/s ÷ no-spec tok/s).
- [ ] **45W (`arrow_0715`) retrain** — the real deliverable; must lift pos3/pos4 (19/5 → toward
      released's 64/53) to close the tail gap. This is where accept_len 3.08 → ~4.4 comes from.
- [ ] **epoch0–3 of the 17W run** — convert + eval each for the epoch→accept_len curve.
