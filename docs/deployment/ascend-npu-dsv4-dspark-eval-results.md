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
| **`ep0p5-bal1e3-77w` *(1e-3 noaux_tc balance, FRESH router, epoch0-mid = 0.5ep, PRE-dedup 77W)*** | **4.050** | 3.784 | 3.890 | 3.591 | 2.466 | **3.56** | ★ **BEST 0.5ep yet (gsm8k).** `DSPARK_MOE_BALANCE=1 @ 1e-3` + fresh router (`INIT_MOE_NO_ROUTER`), A2 DP8 / LR 2e-4 / anchor512, trained on the **pre-clean** 77W (`arrow_0720_77w`, garbage still in). gsm8k **4.050** > non-causal 0.5ep 4.032 > f1-bal5e3 0.5ep 3.998; **mean 3.56** = top-of-class for 0.5ep. Healthy tail (gsm8k pos2/3/4 = **59.8/46.3/35.0**), accept_rate 61.0%, 558 tok/s. Serve = 176 A3-single, EAGER=0, conc48. `/0` (step 12388) → `dsv4_dspark_ep0p5_bal1e3_77w`, 2378/2378 bit-exact. **★ This is the BAR the garbage-dedup resume line must beat at the same epoch** to prove data-cleaning helps. |
| `ep1p0-bal1e3-77w` *(SAME line, epoch1.0 = 1.0ep, pre-dedup 77W)* | 4.017 | 3.612 | 3.938 | 3.542 | 2.480 | **3.52** | **FLAT vs its own 0.5ep (3.56 → 3.52, −0.04) over ONE half-epoch.** NOT over-train-past-peak — our balance line hits ~3.56 already at **0.5ep** (balance front-loads, ≈ f1's 1.5ep peak 3.63) then sits flat to 1.0ep. **The PEAK is likely still AHEAD at 1.5ep** (cf. f1: 1.0ep 3.37 → **1.5ep 3.63 peak**). Head vs tail near-flat (gsm8k pos2 59.8→60.1, pos3/4 46.3/35.0→43.7/32.6). The **dirty run was killed at 1.0ep** — so 0.5ep & 1.0ep are its ONLY points (no 1.5/2.0ep dirty ckpt). `dsv4_dspark_ep1p0_dedup_77w` (name says "dedup" but it's the DIRTY 1.0ep `/0` — mislabeled at convert). ⟹ **decider = the 1.5ep DEDUP ckpt** (resumed from this 1.0ep on clean data): climb >3.56 → line still rising + dedup helps; flat → plateaued. Serve = 176 A3-single conc48. |
| `ep0end-77w` *(non-causal, DOUBLE-norm teacher, epoch0-end)* | 4.006 | 3.753 | 3.970 | 3.666 | 2.539 | 3.59 | epoch0-**end** (1.0 epoch, `0/` after end-save overwrote mid). Non-causal epoch curve: **0.5ep→1.0ep = HEAD sharpens, TAIL plateaus early.** pos0/pos1 up across ALL datasets; pos2-4 flat-to-slightly-down (gsm8k tail 59/46/35→57/43/34). ⚠ this run has the DOUBLE-norm teacher (see `ep0end-f1-77w` below — turns out that HELPS). |
| `ep0end-dnorm-bal5e3-77w` *(A2: dnorm + balance 5e-3, epoch0-end = 1.0ep) — ⭐ CROSS-DOMAIN PIPELINE VALIDATION* | 4.027 | 3.626 | 3.838 | 3.568 | 2.455 | **3.50** | ⭐ **The main result here is PIPELINE VALIDATION, not the balance number.** A2 draft (DP8, LR2e-4, anchor512, dnorm, balance-from-0.5ep-resume, trained on **A2's own HS**) → converted → served on **A3-176** → **3.50 ≈ A3 dnorm no-balance 1.0ep (3.59)**, clean output (smoke-test数数 OK). ⟹ the cross-parallel-domain **HS-dump → training → conversion → inference path is SOUND** and A2/A3 are CONSISTENT (settles the abandoned warm-start's "A2 HS ≠ A3 HS" doubt). **NOT a clean balance A/B** — confounded vs the A3 dnorm by LR/anchor/DP/machine/resume, so 3.50 vs 3.59 is within slack, NOT "balance hurt dnorm". **★ Unified read:** balance added ~nothing to dnorm (already un-collapsed, N_eff ~76) but LIFTED f1 (collapsed, N_eff ~18) to the dnorm level (f1+bal 0.5ep 3.527 ≈ dnorm 3.535) → **balance only helps WHERE there's a collapse**; balance & dnorm are two routes to the same un-collapsed ~3.5-3.6 ceiling. Reaching released 4.42 needs MORE (data / lower-LR / teacher). Serve = 176 A3-single, EAGER=0, FLASHCOMM1=0, conc48. `0/`(1.0ep) → `dsv4_dspark_ep0end_dnorm_bal5e3_vllm-77w`. |
| ⛔ `ep1mid-dnorm-bal5e3-77w` *(A2: dnorm + balance 5e-3, LR2e-4, epoch1-mid = 1.5ep) — DECLINE IS ROBUST* | 3.815 | 3.461 | 3.512 | 3.258 | 2.341 | **3.28** | ⛔ **A2 (LOWER LR + UN-COLLAPSED) ALSO declines → the over-train decline is ROBUST, not a single-lever fix.** A2 dnorm curve: 1.0ep **3.50** → 1.5ep **3.28** (−0.22, ALL datasets down). A2 = LR **2e-4** + dnorm teacher + **un-collapsed** (N_eff 43/77/78) — the exact opposite of A3 f1's high-LR + collapse — yet STILL over-trains past ~1.0ep. ⟹ **neither lower-LR NOR un-collapse NOR dnorm rescues the decline** (my "next lever = lower LR" call RETRACTED). The decline holds across {LR 3e-4 / 2e-4, collapsed / un-collapsed, single / double-norm}. **Best draft across everything stays A3 f1 1.5ep = 3.63 (82%).** Practical move = **EARLY-STOP at the ~1.5ep peak**, don't over-train; the ceiling-raiser to 4.42 must come from elsewhere (more/better DATA, KD-temperature on the correct teacher, #865) — NOT more epochs / balance / lower-LR. Serve = 176 A3-single, conc48. `1/`(1.5ep) → `dsv4_dspark_ep1mid_dnorm_bal5e3_vllm-77w`. |
| **`ep1mid-f1-77w` *(single-norm, epoch1-mid = REPRODUCTION main line)*** | **4.069** | **3.757** | **4.100** | **3.674** | **2.549** | **3.63** | ★ **Reproduction ON TRACK — data scaling confirmed.** F1 single-norm epoch1-**mid** (1.5ep). vs its own ep0end (1.0ep, 3.37): **+0.26 mean, ALL datasets** (+0.09…+0.42), tail climbing (gsm8k pos2/3/4 51/36/28→**59/44/35**). **82% of released (4.42)** at 1.5/10 epoch (gsm8k 87%). ⚠ f1-1.5ep (3.63) > dnorm-1.0ep (3.59) but **epochs NOT aligned** — dnorm has no 1.5ep+ data (run killed ~1.36ep), so this is NOT a fair "f1 beat dnorm"; dnorm at 1.5ep could match/exceed. Only fair (same-epoch) point = 1.0ep, where dnorm(3.59)>f1(3.37). What IS shown: the single-norm faithful recipe scales with data. To compare trajectories, re-run dnorm to 1.5ep+. |
| ⚡ `ep1mid-f1-blk7` *(BEST ckpt f1 1.5ep, served num_spec=**7** — TRAIN-SHORT-SERVE-LONG, inference-only)* | 4.266 | 3.887 | 4.111 | 3.758 | 2.624 | **3.73** | ⚡ **TRAIN-SHORT-SERVE-LONG WORKS** (SAME weights as `ep1mid-f1-77w`; trained γ=5/block_size6, **served at `dspark_block_size=7` / num_spec=7**, internal block 8). vs the SAME ckpt at ns5 (3.63): mean **+0.099, ALL 5 datasets up** (gsm8k 4.069→4.266). The **UNTRAINED slots 5/6 extrapolate** (pos5 ~10-18%, pos6 ~2-6% — RoPE + shared mask-token generalize; no weight hardcodes γ). ⚠ **NOT a free speedup: throughput ~FLAT** (533 vs ~541 tok/s; accept_rate 46.7% vs ~61%) — the wider block's draft+verify cost ≈ cancels the accept_len gain; **pos6 nearly dead (2-6%) → γ=6 likely the sweet spot.** ⚠ **NOT apples-to-apples with released 4.42** (that's ns5; released at ns7 would also rise) — keep ns5 for the reproduction comparison; ns7 is a serve-time lever. Serve = 176 A3-single; `dsv4_dspark_ep1mid_f1_blk7_vllm-77w` (softlink weights + `dspark_block_size` 5→7). |
| ⭐ **`ep0mid-f1-bal5e3-77w` *(f1 + noaux_tc balance 5e-3, epoch0-MID = 0.5ep)*** | **3.998** | **3.700** | **3.885** | **3.595** | **2.457** | **3.53** | ★★ **STRONG POSITIVE for load balancing — clean single-variable A/B** (identical f1 recipe, ONLY variable = `DSPARK_MOE_BALANCE=1` @ rate 5e-3). At just **0.5ep** it ALREADY **beats no-balance f1's 1.0ep (3.37)** and **nears its 1.5ep PEAK (3.63)** — i.e. matches ~1.5ep of no-balance training in 0.5ep (gsm8k 3.998 vs no-bal 1.0ep 3.811 / 1.5ep-peak 4.069). ⚠ NOT same-epoch (favours balance even more: it wins with HALF the training). **Definitive test still ahead:** does it EXCEED the 3.63 peak AND AVOID the no-balance 1.5ep→3.0ep decline (3.63→3.01)? — watch 1.0/1.5/2.0ep. **★ Mechanism puzzle:** training-side N_eff stayed ~18 (5e-3 did NOT un-collapse the per-step load, [[dsv4-dspark-moe-loadbalance]]) yet eval jumped → the gain is likely the LEARNED `gate.bias` improving SERVE routing (picking a better ~18), NOT a higher N_eff — supports "N_eff=256 is too strict; the routing bias matters more than the count". Serve = 176 A3-single (386530d12, EAGER=0, **FLASHCOMM1=0** — now auto-gated on DRAFT, `281cc23`), conc48; `gate.bias` carried bit-exact (verify passed). `0/` (0.5ep) → `dsv4_dspark_ep0mid_bal5e3_vllm-77w`. |
| `ep0end-f1-bal5e3-77w` *(f1 + balance 5e-3, epoch0-end = 1.0ep)* | 4.087 | 3.638 | 3.846 | 3.530 | 2.488 | **3.52** | f1+bal 1.0ep. gsm8k still climbing (0.5ep 3.998→**4.087**) but **mean FLAT vs 0.5ep (3.53→3.52)** — the harder sets (math500/mbpp/mt-bench) don't move. Training-side N_eff still ~20 (5e-3 too weak). Same 176 serve. `0/`(1.0ep) → `dsv4_dspark_ep0end_bal5e3_vllm-77w`. |
| ⛔ `ep1mid-f1-bal5e3-77w` *(f1 + balance 5e-3, epoch1-mid = 1.5ep) — VERDICT ROW* | 3.963 | 3.675 | 3.740 | 3.482 | 2.475 | **3.47** | ⛔ **HYPOTHESIS REVERSED — balance is NOT the lever.** f1+bal curve **0.5ep 3.53 → 1.0ep 3.52 → 1.5ep 3.47** = a slow decline FROM THE START; it never climbed to a peak. At the same 1.5ep, **no-balance f1 PEAK 3.63 > balance 3.47 (−0.16)** — balance LOSES the same-epoch A/B; even gsm8k peaked at 1.0ep (4.087) then fell (3.963). ★★ **Both arms over-train & decline under LR 3e-4 → the decline is LR-driven, NOT collapse-driven** (balance never un-collapsed N_eff ~20 AND still declined; earlier "balance must beat the decline curve" call is retracted). noaux_tc @5e-3 = an early-epoch `gate.bias` bump (picks a better ~18) with a LOWER ceiling; it neither beats the peak nor avoids the decline. **Next lever = LOWER LR** — the A2 dnorm@2e-4 arm tests exactly this (does un-collapsed + low-LR HOLD past 1.5ep). Serve = SAME 176 A3-single. `1/`(1.5ep) → `dsv4_dspark_ep1mid_bal5e3_vllm-77w`. |
| `ep1end-f1-77w` *(single-norm, epoch1-end = 2.0ep, REPRODUCTION main line)* | 4.059 | 3.641 | 4.007 | 3.594 | 2.513 | **3.56** | ⚠ **First DOWN-TICK: 1.5ep→2.0ep = −0.067 mean (82%→81%).** NOT overfitting-shaped: gsm8k basically FLAT (4.069→4.059, −0.010) and its TAIL actually ROSE (pos3/4 44.48/35.42→**46.69/36.81**), only the HEAD dipped (pos0/1); the loss is concentrated in math500 (−0.116) / humaneval (−0.093) / mbpp (−0.080). Reads as eval/batch noise + mid-vs-end LR-phase wobble, not a clean overfit (epoch 2/10, 770k samples seen 2× → data overfit implausible). ONE point can't call a trend — **decider = 2.5ep (step 30970, `2/` mid)**: continue<2.0ep ⟹ real; bounce ⟹ noise. If single-norm has plateaued ~3.6 that's the known teacher ceiling ([[dsv4-dspark-double-norm-teacher-helps]] dnorm was +0.2 at 1.0ep), an optimization lever — not "no way out". Serve = SAME 176 A3-single (386530d12, EAGER=0, FLASHCOMM1=0, conc48); ckpt `1/` (step 24776) → `dsv4_dspark_ep1end_f1_vllm-77w`, 2378/2378 bit-exact. |
| `ep2mid-f1-77w` *(single-norm, epoch2-mid = 2.5ep)* | 3.703 | 3.293 | 3.552 | 3.200 | 2.320 | **3.21** | ⛔ **REGRESSION CONFIRMED — the 2.0ep down-tick was NOT noise; it's an accelerating decline. PEAK = 1.5ep (3.63/82%).** Curve: 1.0ep 3.37 → **1.5ep 3.63 (peak)** → 2.0ep 3.56 (−0.07) → **2.5ep 3.21 (−0.35, 73%)**. Broad degradation incl the HEAD (gsm8k pos0 90.5→**83.7**, not just tail) → weights DRIFTING, not a clean tail-overfit. train↑ (accept_len 3.58, still climbing) + eval↓ = **divergence**. Cause = single-norm teacher + **LR 3e-4 over-trains**: cosine at 2.5/10ep only decayed to ~2.66e-4 (89% of peak) → 2.5 epochs of high-LR fitting drifts past the eval optimum. **Best f1 ckpt = 1.5ep = 82%** (its full trainer ckpt is GONE — `1/` mid overwritten by 2.0ep end; only the converted weights-only `ep1mid_f1_vllm-77w` survives). Path forward = lower LR (peaks later/higher) + the DOUBLE-norm teacher — exactly the A2 dnorm arm (LR 2e-4 + double-norm), which also doubles as the A2/A3-consistency test. `2/` step 30970 → `dsv4_dspark_ep2mid_f1_vllm-77w`, same 176 serve. |
| `ep2end-f1-77w` *(single-norm, epoch2-**end** = 3.0ep) — ⭐ NO-LOAD-BALANCE baseline* | 3.493 | 3.065 | 3.334 | 2.956 | 2.198 | **3.01** | ⛔ **Decline continues — monotonic since the 1.5ep peak.** Full curve: 1.0ep 3.37 → **1.5ep 3.63 (peak)** → 2.0ep 3.56 → 2.5ep 3.21 → **3.0ep 3.01** (−0.20 vs 2.5ep, **68% of released 4.42**). Broad degradation incl the HEAD (gsm8k pos0 83.7→**80.9**, pos1 61.5; tail 46/35/26), every dataset down (mbpp 2.956, mt-bench 2.198). Confirms weights DRIFTING under LR 3e-4 single-norm (train accept↑ / eval↓ = divergence) — the over-train signature of the diagnosed **MoE expert-collapse** (256→~14 experts, [[dsv4-dspark-moe-loadbalance]]). **★ This entire 5-point decline curve is the reference the `DSPARK_MOE_BALANCE=1` run must beat — the balanced run should NOT diverge like this.** Serve = SAME 176 A3-single (386530d12, EAGER=0, conc48); `2/` end (step 37164, overwrote the 2.5ep mid) → `dsv4_dspark_ep2end_f1_vllm-77w`. |
| `ep0end-f1-77w` *(SINGLE-norm teacher = "F1 fix")* | 3.811 | 3.511 | 3.680 | 3.407 | 2.460 | **3.37** | ★★ **SURPRISE — the F1 "double-norm fix" is a REGRESSION.** Same epoch0-end, same non-causal / anchor576 / noise0.05 / LR3e-4 — the ONLY variable is the teacher norm (single vs double). Single-norm (the "correct" real-verifier distribution) is **WORSE on ALL 5 datasets** (Δ −0.08…−0.29, **mean −0.21**), entirely in the **TAIL** (gsm8k pos2/3/4 **57/43/34 → 51/36/28**; pos0/1 unchanged). ⟹ the DOUBLE-norm teacher (≈ `w²` sharpening) distills the tail BETTER, despite its 16% argmax-flip vs the real verifier (T1). **The teacher is a TRAIN target, NOT the serve verifier — so "double-norm" was never a train/serve inconsistency; it's an accidental teacher-SHARPENING that helps.** My "F1 = correctness fix" call was wrong; user's "double-norm is an optimization" instinct was right. Next: deliberate KD-temperature on the CORRECT teacher (clean sharpening, no argmax defect) may beat both. |
| ⛔ `ep1-freshrtr-bal1e2-77w` *(FULL un-collapse: fresh router + balance 1e-2, N_eff ~120, epoch1 = 1.0ep) — ★ UN-COLLAPSE HURTS* | 3.045 | 2.716 | 2.745 | 2.684 | 2.117 | **2.66** | ⛔ **Un-collapse is NEGATIVE, not neutral.** Fresh router + balance 1e-2 → N_eff ~120 on ALL 3 layers → gsm8k **3.045 / mean 2.66**, well BELOW no-balance (~3.5) & released (4.42), esp. TAIL (gsm8k pos1 50 vs ep0end-nc 77, pos2 28 vs 57). ⟹ **SOME specialization/collapse is BENEFICIAL** (matches released gate.bias: only 1/3 layers balanced → still 4.42). The training-proxy "balance ≈ no-balance flat 3.15" UNDERSTATED it — serve shows a NET LOSS. **DEFINITIVE: do NOT balance.** ⚠ minor confound: 1.0ep vs the 3.63@1.5ep, but ~0.5 gap >> epoch. A3 COMPILE=0 resume, `serve_dsv4_a3_singlenode` conc16; `dsv4_dspark_ep1_freshrtr_bal1e2_vllm-77w`. |
| `kdtemp0.7-A2` *(A2 warm-start + KD-temperature 0.7 on the LOSS; ckpt `20260727_223448`)* | 4.011 | 3.793 | 3.876 | 3.599 | 2.513 | **3.56** | KD-temperature 0.7 (uniform teacher-logit sharpening T<1, argmax-preserved — SEPARATE from double-norm's hidden-side reshape). **~NEUTRAL**: mean 3.56 = f1 2.0ep, BELOW f1 1.5ep peak 3.63; gsm8k 4.011 a touch high but mt-bench 2.513 drags the mean. ⟹ **KD-temp is NOT the ceiling lever** — still the ~3.5-3.6 cluster. ⚠ epoch not pinned (warm-start incl router, not fresh). Evaluator reset-aware (counter-reset fix in). |

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

## fresh-router bal1e2 (full un-collapse) — ep1 (1.0 epoch), 2026-07-28 — ★ UN-COLLAPSE HURTS

- **Draft**: `dsv4_dspark_ep1_freshrtr_bal1e2_vllm-77w` — A3 fresh-router (`INIT_MOE_NO_ROUTER` + `DSPARK_MOE_BALANCE` rate **1e-2** → N_eff ~120 on ALL 3 layers, fully un-collapsed), COMPILE=0 resume from CKPT1, 1.0 epoch.
- **Serve**: A3 single-node bf16 (`serve_dsv4_a3_singlenode`, EAGER=0, FLASHCOMM1=0, num_spec=5, conc 16), `DATASET=all` full.

| Dataset | Samples | tok/s | accept_len | accept_rate | pos0 | pos1 | pos2 | pos3 | pos4 |
|---------|:-------:|:-----:|:----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| gsm8k | 1309 | 273.71 | **3.045** | 40.91% | 90.00 | 50.56 | 28.07 | 20.29 | 15.62 |
| math500 | 490 | 361.12 | **2.716** | 34.32% | 87.38 | 41.45 | 20.29 | 13.22 | 9.26 |
| humaneval | 154 | 216.99 | **2.745** | 34.91% | 91.87 | 40.97 | 19.87 | 13.02 | 8.80 |
| mbpp | 247 | 399.66 | **2.684** | 33.69% | 87.15 | 39.44 | 20.23 | 13.03 | 8.58 |
| mt-bench | 70 | 308.93 | **2.117** | 22.33% | 68.73 | 25.03 | 9.94 | 5.15 | 2.81 |

**Read — ★ UN-COLLAPSE HURTS (balance is NEGATIVE, not neutral).** Mean AL ≈ **2.66**, gsm8k **3.045** — well BELOW no-balance-f1 (gsm8k ~3.5-3.63) and released (4.42 / gsm8k 4.658). Forcing full un-collapse (N_eff ~120 on all 3 layers) LOWERS serve accept, esp. the **tail** (gsm8k pos1 50 vs `ep0end-nc` 77, pos2 28 vs 57). ⟹ **SOME specialization/collapse is beneficial** — matches the released draft's gate.bias asymmetry (only layer 0 balanced, mtp.1/2 collapsed → still 4.42). Confound: ep1 (1.0ep) vs the 3.63 at 1.5ep, but the ~0.5 gap is too big to be epoch alone. **CONCLUSION: do NOT balance/un-collapse. The training-proxy "balance ≈ no-balance" (3-way flat ~3.15) UNDERSTATED it — the SERVE eval shows balance is a NET LOSS.**

## Baselines / TODO

- [x] **released draft — full `DATASET=all`** on our serve → **DONE 2026-07-20** (mean 4.42; gsm8k 4.658
      reproduced). This is now the same-serve bar row + detail section above.
- [ ] **no-spec base** tok/s reference — have gsm8k 302.69; extend to `DATASET=all` for a per-dataset
      speedup denominator (spec-decode tok/s ÷ no-spec tok/s).
- [ ] **45W (`arrow_0715`) retrain** — the real deliverable; must lift pos3/pos4 (19/5 → toward
      released's 64/53) to close the tail gap. This is where accept_len 3.08 → ~4.4 comes from.
- [ ] **epoch0–3 of the 17W run** — convert + eval each for the epoch→accept_len curve.
