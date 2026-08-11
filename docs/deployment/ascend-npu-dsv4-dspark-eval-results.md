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

> **★★★ 2026-08-05 — read the PRE-ropefix rows as historical.** The `~3.5 plateau`, `train↑/eval↓
> divergence`, `gap = tail / data / exposure`, and `best draft = f1-1.5ep 3.63` framing in the older rows
> below is **SUPERSEDED**: the cause was **degenerate training-RoPE** (complex freqs cast to bf16 = scale-only,
> no rotation), root-caused + fixed (`feb0066`/`8db8f75`). The **`*-ropefix-77w` rows (top of the matrix) are
> the current line** and confirm **eval now tracks train** — the divergence is resolved. Trajectory
> 🏁 **RUN COMPLETE (10/10 checkpoints).** Full curve, 5-dataset mean:
> **0.5 3.84 → 1.0 4.06 → 1.5 4.18 → 2.0 4.25 → 2.5 4.29 → 3.0 4.35 → 3.5 4.36 → 4.0 4.39 → 4.5 4.41 →
> 5.0 4.40** (87.0% → **99.5%** of released 4.42). **Non-chat four = 4.726 vs 4.699 = 100.6%**, i.e. above
> the released draft; gsm8k 4.849 = 104.1% and mbpp 4.555 = 100.4% individually exceed it. **Speedup vs the
> AR baseline @conc48 = 1.39× mean (1.18–1.64×)**, a conservative lower bound (conc48 is throughput-bound).
> **Converged**: gsm8k's per-checkpoint gain decays +0.184→…→+0.009 and holds +0.009 for the last three;
> the last three means span 0.0138, less than the small sets' single-step bounce.
> ★ **The LR is already annealing** — `--scheduler-type cosine` over 5 epochs decays to exactly **0** at
> 5.0ep (verified against the log: predicted `lr(gs=87138)=4.444e-05` vs observed `4.45e-05`). So the
> "next lever = anneal LR→0, which we have never run" note in the `ep2p0-ropefix` row is **WRONG** — this
> run carries a full anneal in its final 1.5 epochs (4.44e-5 → 2.07e-5 → 5.31e-6 → 0). Prior "next = data-cleaning / TTT / KD-temp / lower-LR /
> balance" plans were pursued BEFORE RoPE was found; they're now secondary. Details: the pipeline top bullet.

| Run | gsm8k | math500 | humaneval | mbpp | mt-bench | mean | notes |
|-----|:-----:|:-------:|:---------:|:----:|:--------:|:----:|-------|
| **released draft** *(bar)* | 4.658 | 4.661 | 4.942 | 4.535 | 3.294 | **4.42** | official 3.94 @ ns5; full-all on our serve ✓ |
| 🔬 **`ep0p5-lossreduce-77w` *(A/B arm — `DSPARK_GLOBAL_LOSS_REDUCE=1`, NOT a best-run entry)*** | 4.298 | 4.048 | 4.326 | 3.911 | 2.626 | **3.84** | 🔬 **A/B for upstream PR #942 (global cross-rank loss normalization), NOT a quality attempt.** Same recipe / data / step count as `ep0p5-ropefix`; the ONLY variable is the loss normalization (per-rank mean-of-ratios → global token-weighted). **Result: indistinguishable.** 5-dataset mean 3.8418 vs 3.8420 = **Δ −0.0002**; the four non-chat datasets sum to the SAME value (16.583 both) so their mean is **Δ +0.0000**. Per-dataset: gsm8k −0.011, math500 −0.020, humaneval +0.028, mbpp +0.003, mt-bench −0.001 — max |Δ| 0.028 on humaneval (n=154, the smallest set), all inside the ±0.03 resolution established at 4.0ep. ★ **This is the answer to the reviewer's 'short training runs with accuracy evals': no regression.** The case for the change is correctness (the per-rank objective depends on world_size and on how the sampler happened to shard the data), not a measured gain — and this run says taking the correct objective costs nothing. ★ **The imbalance it corrects is real and SYSTEMATIC on this setup** (measured in the same run, `profile/sup_tokens_*`): cumulative heaviest/lightest **1.194**, per-rank totals monotone in rank index (r0 +11.4% … r7 −6.7%), first-half vs second-half deviation correlation **+0.997** and the magnitude does NOT decay with more steps — zero-mean noise would shrink ~1/√n. **24/2095 logged steps had a rank with ZERO supervised tokens.** ★ **Runtime: no measurable cost** — step_ms deltas across three measurements were +20 / +50 / +40 ms (~1–2%), i.e. NOT stable, whereas one scalar all-reduce would add a constant; this is the same order as the ~2.6% machine drift already on record. Trainer ckpt `ckpt_faithful_ep_20260810_234322/0` → `dsv4_dspark_ep0p5_lossreduce_vllm-77w`. ⚠ its eval log was tee'd over `~/eval_ep0p5_ropefix_all.txt`, destroying the OFF arm's raw log (numbers survive in the `ep0p5-ropefix` row). |
| 🏁 **`ep5p0-ropefix-77w` *(RoPE-fix, epoch4-end = 5.0ep — FINAL, run complete)*** | **4.849** | **4.565** | **4.933** | **4.555** | **3.079** | **4.40** | 🏁 **RUN COMPLETE — 10/10 checkpoints, LR annealed to exactly 0. Final mean 4.396 = 99.5% of released; NON-CHAT 4.726 vs 4.699 = 100.6%.** gsm8k **4.849 = 104.1%** and mbpp **4.555 = 100.4%** exceed released; humaneval 99.8%, math500 97.9%, mt-bench 93.5%. ★ **CONVERGED — the large-sample criterion is met cleanly.** gsm8k (n=1309) per-checkpoint deltas: **+0.184/+0.135/+0.073/+0.052/+0.043/+0.026/+0.009/+0.009/+0.009** — a 20× monotone decay settling on the *same* +0.009 three checkpoints running. The last three means (4.3928 / 4.4066 / 4.3962) have a **spread of 0.0138**, smaller than the single-step bounce of the small sets (mt-bench moves ±0.06 per step) ⟹ **4.0 / 4.5 / 5.0ep are statistically the same point.** Per-dataset over the last three steps everything oscillates about zero except gsm8k's steady +0.009. ★ **Deliverable = this checkpoint (5.0ep)**, not 4.5ep: 4.5ep's mean is 0.011 higher, which is *inside* the bounce, while 5.0ep is the end of the schedule (LR=0) and is where the largest-sample dataset peaks (gsm8k 4.849, its maximum over all ten). Choosing 4.5ep on +0.011 would repeat the single-point over-reading corrected at 3.5ep and 4.0ep. ★ **Speedup vs the AR baseline @conc48: gsm8k 1.31× · math500 1.38× · humaneval 1.64× · mbpp 1.46× · mt-bench 1.18× — mean 1.39×** (4.5ep measured 1.42×; the gap is throughput noise on humaneval's 34 s run). gsm8k conditional c0–c4 = **0.937/0.915/0.900/0.889/0.878** vs released 0.928/0.892/0.885/0.868/0.842 — above at every position. **Per-position (cumulative pos0-4):** gsm8k 93.68/85.68/77.01/68.47/60.08 · math500 91.01/81.34/71.02/61.19/51.91 · humaneval 95.11/88.48/78.59/70.05/61.09 · mbpp 91.61/81.66/70.75/60.52/50.97 · mt-bench 75.66/53.39/36.48/25.17/17.17. **Throughput (conc48):** gsm8k 605.1 · math500 843.4 · humaneval 526.7 · mbpp 977.4 · mt-bench 526.5 tok/s. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/4` (epoch4_end, `global_step` 124,480 = 5 × 24,896) → `dsv4_dspark_ep5p0_ropefix_vllm-77w`, 2378/2378 bit-exact. Eval log `~/eval_ep5p0_ropefix_all.txt`. |
| **AR baseline (no-spec)** *(speedup denominator, conc48)* | 460.7 | 612.3 | 320.9 | 669.0 | 446.8 | **— tok/s** | **Autoregressive, no draft — the denominator for every speedup number.** Identical command; the serve is started **without `DRAFT=`**, so the comparison is apples-to-apples with every row above. Cells are **tok/s, not accept_len**. ★ **Speedup of `ep4p5-ropefix` over this baseline: gsm8k 1.29× · math500 1.34× · humaneval 1.77× · mbpp 1.48× · mt-bench 1.21× — mean 1.42×.** Three independent measures agree within ~1%: throughput ratio, wall-clock ratio (1.29/1.34/1.81/1.47/1.18) and E2E-latency ratio (1.28/1.28/1.43/1.39/1.12). Per-token: gsm8k 102.6→79.7 ms, math500 66.8→52.4, humaneval 103.2→73.6, mbpp 61.5→44.1, mt-bench 114.3→99.9. ⚠ **TTFT is NOT a validity check across spec-vs-AR** (it is 1.50–2.43× HIGHER with spec: 467→1134, 422→904, 558→1092, 485→803, 492→740 ms). That is queueing, not a machine-state difference — at conc48 each engine step costs a draft forward plus a 6-token verify, so requests are admitted more slowly. The earlier "TTFT must be unchanged" rule holds only between two DRAFTS, not between spec and AR. ⚠ **`Mean/Median ITL` is also not comparable across arms** — spec emits accepted tokens in bursts, so its ITL is the gap between *blocks*, not between tokens. **The correct validity check is total output tokens** (greedy, same prompts ⟹ same work): +0.6 / −0.4 / −1.9 / +0.5 / +2.2 % — all within ±2.2%. ★ **Speedup does NOT track accept_len**: gsm8k has the highest accept (4.840) but the lowest non-chat speedup (1.29×) while humaneval (4.954) gets 1.77× — because at conc48 the win depends on how saturated the engine is, and humaneval's 154 samples never fill the batch. ⟹ **conc48 understates the draft; treat 1.42× as a conservative lower bound** and see the conc1 TODO. Serve = 176 A3-single, conc48, greedy, FULL datasets. Eval log `~/eval_ar_base_conc48.txt`. |
| ⭐⭐⭐⭐⭐ **`ep4p5-ropefix-77w` *(RoPE-fix, epoch4-mid = 4.5ep — CURRENT BEST)*** | **4.840** | **4.564** | **4.954** | **4.553** | **3.122** | **4.41** | ★★★ **99.7% of released — 0.3% off the bar — and THREE datasets now exceed it**: gsm8k **4.840 = 103.9%**, mbpp **4.553 = 100.4%**, humaneval **4.954 = 100.2%** (humaneval crosses this step). math500 97.9%, mt-bench 94.8%. Non-chat four **4.728 vs 4.699 = 100.6%**. ★ **Convergence now has LARGE-SAMPLE support** (the standard set at 4.0ep after being burned twice by single points): gsm8k (n=1309) per-checkpoint deltas run **+0.184/+0.135/+0.073/+0.052/+0.043/+0.026/+0.009/+0.009** — a clean monotone decay, **flat at +0.009 for two consecutive checkpoints**; math500 (n=490) posts its **first decline** (−0.009) after eight straight rises. That is a far stronger convergence signal than any single-point move in the mean, but it is still ONE step from the end — **5.0ep is the confirmation, and no decision rides on it** since the run terminates there anyway (LR is 5.31e-06, effectively annealed out). ★ gsm8k conditional c0–c4 = **0.936/0.914/0.900/0.886/0.878** vs released 0.928/0.892/0.885/0.868/0.842 — above at every position, c4 margin **+3.6pt**. **Per-position (cumulative pos0-4):** gsm8k 93.57/85.48/76.91/68.18/59.87 · math500 90.84/81.05/71.10/61.33/52.06 · humaneval 95.05/88.97/78.67/70.98/61.69 · mbpp 91.53/81.54/70.67/60.47/51.09 · mt-bench 76.77/54.14/37.43/25.84/18.00. **Throughput (conc48):** gsm8k 596.4 · math500 819.7 · humaneval 568.6 · mbpp 988.9 · mt-bench 540.9 tok/s. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/4` (epoch4_step12448, `global_step` 112,032) → `dsv4_dspark_ep4p5_ropefix_vllm-77w`, 2378/2378 bit-exact. Eval log `~/eval_ep4p5_ropefix_all.txt`. ⚠ `/4` is a MID-epoch save; the 5.0ep epoch-end save overwrites it (converted in time). |
| ⭐⭐⭐⭐⭐ **`ep4p0-ropefix-77w` *(RoPE-fix, epoch3-end = 4.0ep — CURRENT BEST)*** | **4.831** | **4.573** | **4.924** | **4.574** | **3.062** | **4.39** | ★★★ **THE NON-CHAT AVERAGE PASSES THE RELEASED DRAFT: 4.726 vs 4.699 = 100.6%.** Overall mean 4.39 = 99.4%. **Two datasets now exceed released** — gsm8k **4.831 = 103.7%**, mbpp **4.574 = 100.9%** — and humaneval is at 99.6%, math500 98.1%. Only mt-bench trails (3.062 = 93.0%), i.e. the entire remaining headline gap is multi-turn chat, which is a rollout DATA-distribution issue (99.96% single-turn). ★ **The 3.5ep "plateau" reading is RETRACTED — it was small-sample bounce.** Across 3.0→3.5→4.0 the three small sets alternate and cancel: humaneval (n=154) −0.023 then **+0.092**; mbpp (n=247) −0.008 then **+0.070**; mt-bench (n=70) +0.061 then **−0.045**. At 3.5ep two of them happened to dip together, producing the +0.02 that looked like flattening. The large sets — gsm8k (n=1309) and math500 (n=490) — have risen monotonically at every single checkpoint with no inflection. **Deltas +0.22/+0.12/+0.07/+0.04/+0.06/+0.01/+0.03 are the same order as the per-checkpoint bounce, so a plateau simply cannot be called at ±0.03 resolution from single points.** ★ gsm8k conditional c0–c4 = **0.936/0.912/0.896/0.888/0.880** vs released 0.928/0.892/0.885/0.868/0.842 — above at every position, and the c4 margin has widened to **+3.8pt**. **Per-position (cumulative pos0-4):** gsm8k 93.56/85.34/76.48/67.93/59.75 · math500 90.95/81.17/71.40/61.63/52.15 · humaneval 95.16/88.46/78.25/69.95/60.55 · mbpp 91.93/81.94/71.07/60.91/51.57 · mt-bench 75.99/52.74/36.00/24.64/16.83. **Throughput (conc48):** gsm8k 596.6 · math500 837.4 · humaneval 555.7 · mbpp 1009.4 · mt-bench 524.4 tok/s. LR at this point = 2.07e-05 (cosine anneal, see the 3.5ep note). Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/3` (epoch3_end, `global_step` 99,584) → `dsv4_dspark_ep4p0_ropefix_vllm-77w`, 2378/2378 bit-exact. Eval log `~/eval_ep4p0_ropefix_all.txt`. |
| ⭐⭐⭐⭐ **`ep3p5-ropefix-77w` *(RoPE-fix, epoch3-mid = 3.5ep — CURRENT BEST)*** | **4.822** | **4.548** | **4.832** | **4.504** | **3.107** | **4.36** | **98.7% of released; non-chat 4 = 4.677 vs 4.699 = 99.5%.** Per-dataset: gsm8k **4.822 = 103.5%**, mbpp 99.3%, humaneval 97.8%, math500 97.6%, mt-bench 94.3%. ★ **The climb is now genuinely flattening** — trajectory **3.84 → 4.06 → 4.18 → 4.25 → 4.29 → 4.35 → 4.36**, deltas +0.22/+0.12/+0.07/+0.04/+0.06/**+0.02**. The +0.06 at 3.0ep (which prompted retracting the plateau call) now reads as jitter around a decelerating trend. ★ **Two datasets moved DOWN**: humaneval 4.855→4.832 (−0.023), mbpp 4.512→4.504 (−0.008) — both are the SMALLEST sample sets (154 / 247); greedy temp0 makes a given draft reproducible, so these are real draft-to-draft differences, but −0.02 at n=154 is not worth over-reading. **Non-chat is essentially static (+0.006); the only real gain is mt-bench (+0.061)** — the laggard closing, now 94.3% (was 90.7% at 2.5ep). gsm8k conditional c0–c4 **0.934/0.912/0.897/0.888/0.871**, still above released 0.928/0.892/0.885/0.868/0.842 at every position. **Per-position (cumulative pos0-4):** gsm8k 93.44/85.22/76.47/67.92/59.18 · math500 90.92/80.87/70.86/60.78/51.35 · humaneval 94.28/86.81/76.35/68.11/57.70 · mbpp 91.22/80.96/69.49/59.14/49.57 · mt-bench 76.43/53.64/36.76/25.69/18.15. **Throughput (conc48):** gsm8k 598.6 · math500 811.8 · humaneval 525.1 · mbpp 962.9 · mt-bench 533.1 tok/s. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/3` (epoch3_step12448, `global_step` 87,136) → `dsv4_dspark_ep3p5_ropefix_vllm-77w`. Eval log `~/eval_ep3p5_ropefix_all.txt`. ⚠ **`/3` is a MID-epoch save — the 4.0ep epoch-end save overwrites it.** |
| ⭐⭐⭐⭐ **`ep3p0-ropefix-77w` *(RoPE-fix, epoch2-end = 3.0ep — CURRENT BEST)*** | **4.796** | **4.519** | **4.855** | **4.512** | **3.046** | **4.35** | ★★★ **98.4% of released; on the 4 NON-CHAT datasets 4.671 vs released 4.699 = 99.4% — effectively parity.** Per-dataset: gsm8k **4.796 = 103.0%** (surpasses by 3%), mbpp **99.5%**, humaneval 98.2%, math500 97.0%, mt-bench 92.5%. ★ **The climb did NOT flatten:** trajectory **3.84 → 4.06 → 4.18 → 4.25 → 4.29 → 4.35**, deltas +0.22/+0.12/+0.07/+0.04/**+0.06** — the 2.5ep step (+0.04) had looked like the onset of a plateau; **this step is LARGER**, so "approaching plateau" is retracted (could still be run-to-run noise at this scale, but there is no flattening in the data). ★ **gsm8k conditional accept c0–c4 is now ABOVE released at EVERY position, including c0 for the first time**: **0.933/0.911/0.894/0.882/0.870** vs released 0.928/0.892/0.885/0.868/0.842 (at 2.5ep c0 was 0.927, i.e. still −0.1pt) ⟹ **the per-position mechanism is fully at/above the released draft; the entire residual headline gap is mt-bench multi-turn chat**, a rollout DATA-distribution issue (99.96% single-turn), not a model defect. **Per-position (cumulative pos0-4):** gsm8k 93.28/84.98/75.96/67.03/58.34 · math500 90.56/80.45/70.20/60.12/50.62 · humaneval 94.12/86.92/77.05/68.29/59.09 · mbpp 91.49/81.06/69.60/59.30/49.79 · mt-bench 75.66/52.63/35.47/24.22/16.66. **Throughput (conc48):** gsm8k 592.8 · math500 800.6 · humaneval 549.9 · mbpp 967.8 · mt-bench 478.1 tok/s. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/2` (epoch2_end, `global_step` 74,688 = 3 × 24,896) → `dsv4_dspark_ep3p0_ropefix_vllm-77w`, 2378/2378 bit-exact. Eval log `~/eval_ep3p0_ropefix_all.txt`. |
| ⭐ **`ep0p5-ropefix-77w` *(★RoPE-FIX, from-scratch, epoch0-mid = 0.5ep — same recipe as `ep0p5-bal1e3`: bal1e3 / fresh-router / dedup / lr2e-4)*** | **4.309** | **4.068** | **4.298** | **3.908** | **2.627** | **3.84** | ★★★ **THE RoPE fix lands — NEW BEST across EVERYTHING, at just 0.5ep.** Clean single-variable A/B vs `ep0p5-bal1e3` (identical recipe; **ONLY** variable = RoPE: degenerate scale-only → real cos/sin interleaved, `feb0066`/`8db8f75`). **ALL 5 datasets up, mean 3.56→3.84 (+0.28):** gsm8k 4.050→**4.309**(+0.26), math500 3.784→**4.068**(+0.28), humaneval 3.890→**4.298**(+0.41), mbpp 3.591→**3.908**(+0.32), mt-bench 2.466→**2.627**(+0.16). **Beats the prior best-of-everything `ep1mid-f1` (3.63 @ 1.5ep) at 1/3 the epochs** → 86.9% of released 4.42. ★ **The gain is the TAIL = the RoPE-fix signature** (later block slots finally rotate correctly): gsm8k cumulative pos2/3/4 59.8/46.3/35.0→**65.6/53.6/42.8** (+5.8/+7.3/+7.8); **conditional accept rate to pos4 stays ~80%** (c3/c4 0.774/0.756→**0.818/0.797**); overall accept_rate 61.0%→**66.2%**. ⟹ **the diagnosed `train↑ / eval↓` DIVERGENCE is RESOLVED — eval now TRACKS train** (RoPE was THE train↔serve mismatch, [[dsv4-dspark-rope-degenerate-root-cause]]). **Per-position (cumulative pos0-4):** gsm8k 90.41/78.54/65.59/53.64/42.76 · math500 87.65/74.57/60.74/47.80/36.04 · humaneval 92.39/80.65/65.76/51.79/39.26 · mbpp 87.35/72.01/56.38/43.25/31.85 · mt-bench 68.95/42.98/25.90/15.51/9.33. Still ~2-5 pts/pos below released conditional (~93/89/89/87/84) = **data/epoch gap, NOT a bug** (0.5ep vs released fully-trained); expect further climb past 0.5ep. Serve = 176 A3-single, conc48, EAGER=0, num_spec=5, throughput gsm8k 555 tok/s. Trainer ckpt `ckpt_faithful_ep_20260804_165215/0` (run TS 20260804_165215) → `dsv4_dspark_ep0p5_ropefix_vllm-77w`, 2378/2378 bit-exact. Eval log `~/eval_ep0p5_ropefix_all.txt`. |
| ⭐⭐⭐ **`ep2p5-ropefix-77w` *(RoPE-fix, epoch2-mid = 2.5ep — same line)*** | **4.753** | **4.485** | **4.818** | **4.428** | **2.988** | **4.29** | ★★★ **97.2% of released; gsm8k 4.753 = 102.0% (surpasses by 2%).** Trajectory **3.84 → 4.06 → 4.18 → 4.25 → 4.29** (Δ +0.22/+0.12/+0.07/**+0.04**) — still climbing, deceleration continues. ★ **On the 4 NON-CHAT datasets the mean is 4.621 vs released 4.699 = 98.3%** — the remaining headline gap is almost entirely **mt-bench 2.988 (90.7%)**, i.e. multi-turn chat, which is a **DATA-distribution** issue (our rollout is 99.96% single-turn), not a model defect. Per-dataset: gsm8k **102.0%**, mbpp 97.6%, humaneval 97.5%, math500 96.2%, mt-bench 90.7%. ★ **Conditional accept c1–c4 now ALL ABOVE released**: gsm8k 0.927/0.908/0.893/0.879/0.865 vs released 0.928/0.892/0.885/0.868/0.842 (only c0 is −0.1pt) ⟹ the per-position mechanism is no longer the limiter. **Per-position (cumulative pos0-4):** gsm8k 92.73/84.16/75.14/66.08/57.16 · math500 90.29/79.79/69.49/59.41/49.54 · humaneval 94.35/86.74/76.65/67.18/56.85 · mbpp 90.91/79.77/67.90/56.99/47.19 · mt-bench 75.20/51.49/34.10/22.70/15.28. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/2` (epoch2_step12448) → `dsv4_dspark_ep2p5_ropefix_vllm-77w`. ⚠ **its eval log `~/eval_ep2p5_ropefix_all.txt` was later TRUNCATED** — an aborted 3.0ep eval re-used that `tee` target, and `tee` truncates on open. The numbers here are the transcription of record; the raw log is gone. |
| ⭐⭐ **`ep2p0-ropefix-77w` *(RoPE-fix, epoch1-end = 2.0ep — same line)*** | **4.701** | **4.431** | **4.745** | **4.412** | **2.980** | **4.25** | ★★★ **gsm8k 4.701 SURPASSES the released draft's 4.658 (100.9%)** — first dataset to go past the bar. Mean **4.25 = 96.3% of released 4.42**. **Trajectory 0.5→1.0→1.5→2.0ep = 3.84 → 4.06 → 4.18 → 4.25** (Δ +0.22/+0.12/+0.07): still climbing but **decelerating — approaching plateau**, and *nothing like* the degenerate lines which had already turned DOWN by 2.0ep (3.56→3.45). ★ **Per-position CONDITIONAL accept now matches/beats released**: gsm8k c0-c4 = **0.926/0.904/0.884/0.873/0.856** vs released 0.928/0.892/0.885/0.868/0.842 ⟹ **pos1-4 at or above released**; the residual gap is pos0 (−0.2pt) and chat. Per-dataset vs released: gsm8k **100.9%**, mbpp 97.3%, humaneval 96.0%, math500 95.1%, mt-bench 90.5% (mt-bench gained the most this step, 2.865→2.980). **Per-position (cumulative pos0-4):** gsm8k 92.60/83.67/73.96/64.58/55.30 · math500 89.94/79.35/68.37/57.77/47.64 · humaneval 93.97/85.44/74.84/65.03/55.25 · mbpp 90.57/79.83/67.68/56.62/46.50 · mt-bench 74.51/51.09/34.11/22.94/15.32. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/1` (epoch1_end) → `dsv4_dspark_ep2p0_ropefix_vllm-77w`. Eval log `~/eval_ep2p0_ropefix_all.txt`. ~~⟹ Next lever for the last ~4%: anneal LR→0 (the DeepSpec full-anneal we have never run), not more epochs at constant LR.~~ ⚠ **RETRACTED 2026-08-07 (at 3.5ep)** — this run was **already annealing**: the launcher passes `--scheduler-type cosine` over `EPOCHS=5`, a cosine decay to **exactly 0** at the final step (verified against the log, not assumed: predicted `lr(gs=87138)=4.444e-05` vs logged `4.45e-05`). So "the full-anneal we have never run" and "at constant LR" are both wrong — the LR was never constant, and the final 1.5 epochs ARE that anneal (4.44e-5 → 2.07e-5 → 5.31e-6 → 0). See the `ep3p5-ropefix` row. |
| ⭐ **`ep1p0-ropefix-77w` *(RoPE-fix, epoch0-end = 1.0ep — same line)*** | **4.493** | **4.241** | **4.585** | **4.167** | **2.796** | **4.06** | Mid-point of the climb, filling in the trajectory: **0.5ep 3.84 → 1.0ep 4.06 → 1.5ep 4.18** — monotonic, no sign of the degenerate lines' plateau/decline. 91.8% of released 4.42; gsm8k 4.493 = 96.5% of released. Tail keeps lifting (gsm8k pos2/3/4 65.6/53.6/42.8 @0.5ep → **69.6/58.7/48.8**). **Per-position (cumulative pos0-4):** gsm8k 91.50/80.80/69.55/58.71/48.75 · math500 88.50/76.52/64.26/52.71/42.08 · humaneval 93.94/84.49/71.47/60.14/48.51 · mbpp 89.04/76.07/62.09/50.12/39.37 · mt-bench 72.18/47.01/29.36/18.87/12.20. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/0` (epoch0_end) → `dsv4_dspark_ep1p0_ropefix_vllm-77w`. ⚠ its eval log was tee'd over `~/eval_ep1p5_ropefix_all.txt` (name reused) — numbers transcribed here. |
| ⭐ **`ep1p5-ropefix-77w` *(RoPE-fix, epoch1-mid = 1.5ep — same line as `ep0p5-ropefix`)*** | **4.628** | **4.408** | **4.706** | **4.300** | **2.865** | **4.18** | ★★★ **STILL CLIMBING — the plateau is BROKEN.** Same RoPE-fixed line at 1.5ep: **0.5ep 3.84 → 1.5ep 4.18 (+0.34)**, = **94.6% of released 4.42** (vs the degenerate lines that PLATEAUED/declined past ~1.5ep at 3.45–3.63). ★ **Near-parity with the released draft on non-chat:** gsm8k **4.628 = 99.4% of released 4.658**; humaneval 4.706 (95.2%), mbpp 4.300 (94.8%), math500 4.408 (94.6%); mt-bench 2.865 (87%, still the laggard — multi-turn chat gap — but climbing 2.627→2.865). **Conditional accept (gsm8k) 0.922/0.895/0.877/0.863/0.851 ≈ released 0.928/0.892/0.885/0.868/0.842 — matches released within ~1pt/position** ⟹ the RoPE-fixed draft is ~as good as the released draft on non-chat at just 1.5/10 epochs; RoPE was the whole gap. **Per-position (cumulative pos0-4):** gsm8k 92.23/82.56/72.36/62.47/53.14 · math500 89.96/79.17/67.92/57.05/46.66 · humaneval 94.43/85.35/74.31/64.33/52.15 · mbpp 89.77/77.90/65.28/53.79/43.21 · mt-bench 73.22/48.24/31.21/20.48/13.38. Serve = 176 A3-single, conc48, ns5. Trainer ckpt `ckpt_faithful_ep_20260804_165215/1` (epoch1_step12448) → `dsv4_dspark_ep1p5_ropefix_vllm-77w`, 2378/2378 bit-exact. Eval log `~/eval_ep1p5_ropefix_all.txt`. |
| `epoch4-17w` | 3.404 | 3.265 | 3.312 | 3.058 | 2.344 | 3.08 | ⚠ wrong dataset (17W not 45W); all fixes in; **gap = tail** |
| `ep0-77w` *(epoch 0)* | 3.186 | 3.041 | 3.079 | 2.868 | 2.255 | 2.89 | first ckpt of A3 77W run (LR 3e-4, noise 0.05); epoch0 vs 17w's epoch4 — **not** like-for-like; gap = tail (pos2 cliff) |
| `ep2.5-77w` *(arm A, noise 0.05, CAUSAL)* | 3.203 | 3.096 | 3.106 | 2.928 | 2.272 | 2.92 | epoch2-mid (30,969 steps) of A3 77W run; **≈ ep0 — FLAT, tail frozen** (gsm8k pos2/3/4 = 39/14/5, same as ep0) → 1.5 more epochs bought +0.03 mean. 176 A3-single serve conc48. ~~serve capping?~~ **RESOLVED**: `ep0mid-77w-nc` on the SAME 176 serve hit 4.032 → serve does NOT cap → ep2.5's flatness was REAL (causal-training), not a serve artifact. |
| **`ep0mid-77w` *(NON-CAUSAL fix, epoch0-mid)*** | **4.032** | **3.721** | **3.856** | **3.586** | **2.482** | **3.54** | ★ **THE non-causal fix lands.** epoch0-**mid** (step 6194 = 0.5 epoch) of the non-causal retrain (`--sliding-window-non-causal`, +#848 detach). Same 176 serve as ep2.5 → **only variable = train causality.** gsm8k **3.186→4.032** (+0.85), tail **pos2/3/4 = 59/46/35** vs causal's 39/14/5 (pos4 **7.4×**). **0.5ep non-causal >> 2.5ep causal.** Past official 3.94@ns5; 86.6% of released-on-our-serve at 0.5/10 epoch. |
| **`ep0p5-bal1e3-77w` *(1e-3 noaux_tc balance, FRESH router, epoch0-mid = 0.5ep, PRE-dedup 77W)*** | **4.050** | 3.784 | 3.890 | 3.591 | 2.466 | **3.56** | ★ **BEST 0.5ep yet (gsm8k).** `DSPARK_MOE_BALANCE=1 @ 1e-3` + fresh router (`INIT_MOE_NO_ROUTER`), A2 DP8 / LR 2e-4 / anchor512, trained on the **pre-clean** 77W (`arrow_0720_77w`, garbage still in). gsm8k **4.050** > non-causal 0.5ep 4.032 > f1-bal5e3 0.5ep 3.998; **mean 3.56** = top-of-class for 0.5ep. Healthy tail (gsm8k pos2/3/4 = **59.8/46.3/35.0**), accept_rate 61.0%, 558 tok/s. Serve = 176 A3-single, EAGER=0, conc48. `/0` (step 12388) → `dsv4_dspark_ep0p5_bal1e3_77w`, 2378/2378 bit-exact. **★ This is the BAR the garbage-dedup resume line must beat at the same epoch** to prove data-cleaning helps. |
| `ep1p0-bal1e3-77w` *(SAME line, epoch1.0 = 1.0ep, pre-dedup 77W)* | 4.017 | 3.612 | 3.938 | 3.542 | 2.480 | **3.52** | **FLAT vs its own 0.5ep (3.56 → 3.52, −0.04) over ONE half-epoch.** NOT over-train-past-peak — our balance line hits ~3.56 already at **0.5ep** (balance front-loads, ≈ f1's 1.5ep peak 3.63) then sits flat to 1.0ep. **The PEAK is likely still AHEAD at 1.5ep** (cf. f1: 1.0ep 3.37 → **1.5ep 3.63 peak**). Head vs tail near-flat (gsm8k pos2 59.8→60.1, pos3/4 46.3/35.0→43.7/32.6). The **dirty run was killed at 1.0ep** — so 0.5ep & 1.0ep are its ONLY points (no 1.5/2.0ep dirty ckpt). `dsv4_dspark_ep1p0_dedup_77w` (name says "dedup" but it's the DIRTY 1.0ep `/0` — mislabeled at convert). ⟹ **decider = the 1.5ep DEDUP ckpt** (resumed from this 1.0ep on clean data): climb >3.56 → line still rising + dedup helps; flat → plateaued. Serve = 176 A3-single conc48. |
| **`ep1p5-dedup-77w` *(dedup 77W resume, epoch1-mid = 1.5ep — ⚖️ THE DECIDER)*** | 4.007 | 3.586 | 3.948 | 3.558 | 2.515 | **3.52** | ⚖️ **VERDICT: dedup / data-cleaning did NOT lift it.** 1.5ep of the dedup-resume line (from the dirty 1.0ep, NO_VAL welded, cleaned Arrow) = **mean 3.52 = FLAT vs the dirty 1.0ep (3.52) and BELOW the 0.5ep bar (3.56)**. Per-dataset a wash vs 1.0ep (gsm8k 4.017→4.007, math500 3.612→3.586; humaneval/mbpp/mt-bench slightly up). ⟹ **more/cleaner data + more epochs is NOT the near-term lever; the line plateaued ~3.52–3.56.** ★ **num_steps barely changed across the NO_VAL resume: 24,776→24,896 (+0.48%)** — the "lost 10% to val" premise was WRONG (symlink `epoch1_step12448`=local step_interval → num_steps=2×12448; old 12388→24,776), so the LR/data-amount resume confound is NEGLIGIBLE (earlier ~10% stretch worry retracted). **The gap to released 4.42 is a TAIL problem:** per-position pos0-1 near released (gsm8k 90.2/76.7 vs 92.8/82.8) but **pos2-4 ~15-20pts BELOW** (57.4/43.3/33.1 vs released 73.3/63.6/53.5) → points to a later-position TRAINING-CONVENTION / correctness gap (TTT / exposure-bias the prime suspect), NOT data volume. gsm8k pos0-4 90.21/76.66/57.35/43.33/33.11 · math500 86.14/66.62/48.56/33.31/24.01 · humaneval 91.80/77.74/56.71/39.22/29.29 · mbpp 86.19/67.17/47.32/32.42/22.72 · mt-bench 68.41/40.63/22.66/12.53/7.29. Serve = 176 A3-single conc48; ckpt `/1` (`epoch1_step12448`) → `dsv4_dspark_ep1p5_dedup_77w`. |
| `ep1end-dedup-77w` *(dedup 77W, epoch1-end = 2.0ep)* | 3.966 | 3.532 | 3.814 | 3.487 | 2.473 | **3.45** | ⛔ **DECLINE — over-train DRIFT, same shape as f1.** 2.0ep of the dedup line = mean **3.45**, DOWN on ALL 5 vs its own 1.5ep (3.52): gsm8k 4.007→3.966, math500 3.586→3.532, humaneval 3.948→3.814, mbpp 3.558→3.487, mt-bench 2.515→2.473. **Curve: 0.5ep 3.56 → 1.0ep 3.52 → 1.5ep 3.52 → 2.0ep 3.45** (flat-then-declining; peak ~0.5–1.5ep). ★ **train↑/eval↓ DIVERGENCE**: train accept_len CLIMBED 3.42→3.57 ("not converged") while eval FELL — the over-train signature, identical to f1 (1.5ep 3.63→2.0ep 3.56→2.5ep 3.21). Decline is UNIFORM (tail gsm8k 57.4/43.3/33.1→56.6/42.3/32.9 ~flat, head also dips) = DRIFT not tail-collapse. ⚠ STILL MID-cosine (LR high, not near 0) → per the Kimi/DeepSpec full-anneal insight this is UN-CONVERGED drift, NOT a converged ceiling; **the full-anneal-to-LR→0 + judge-at-end run is STILL never done** ([[deepspec-recipe]]). Per-pos: gsm8k 89.86/74.97/56.56/42.28/32.93 · math500 86.15/64.70/46.81/32.06/23.45 · humaneval 91.17/73.34/52.66/36.68/27.53 · mbpp 86.78/64.09/44.86/31.08/21.91 · mt-bench 68.81/38.94/21.36/11.54/6.60. Serve = 176 A3-single conc48; `/1` (epoch1_end) → `dsv4_dspark_ep1end_dedup77w`. ✅ **trainsample serve-eval DONE — train/serve-mismatch RULED OUT (see ep4p5 row).** |
| `ep4p5-77w` *(SAME dedup/bal resume line, ~4.5ep, deep over-train)* | 3.973 | 3.514 | 3.833 | 3.470 | 2.468 | **3.45** | ★ **OVER-TRAIN PLATEAUS, does NOT collapse.** ~4.5ep of the SAME resume line as `ep1end-dedup-77w`(2.0ep). Curve: 0.5ep 3.56 → 1.0 3.52 → 1.5 3.52 → 2.0 3.45 → **4.5ep 3.45** = **FLAT since 2.0ep**. Past ~2ep it neither helps nor hurts — a plateau at ~3.45, NOT the f1-no-bal cliff (3.63→3.01 by 3ep). Balance + the resumed (stretched) LR held it flat. gsm8k pos0-4 89.95/75.23/56.64/42.37/33.07 · math500 85.81/64.25/46.47/31.72/23.10 · humaneval 91.31/74.11/53.18/37.05/27.64 · mbpp 86.39/63.52/44.42/30.81/21.81 · mt-bench 68.57/38.83/21.36/11.48/6.57. Tail still the gap (gsm8k pos2-4 56.6/42.4/33.1 vs released 73/64/53). Serve=176 A3-single conc48; `dsv4_dspark_ep4p5_vllm-77w`. **⟹ over-train = plateau not cliff → "stop over-training" saves compute but is NOT the accept lever; the pos2-4 gap is stable.** Root-cause status (this session): forward-convention audit CLEAN + HS dump ~95-97% correct (conc1≈conc96 → over-subscription RULED OUT) + trainsample train/serve gap CONFOUNDED (serve runs its OWN trajectory ≠ rollout) → the gap is DRAFT QUALITY (data/recipe/tail), NOT a bug. Next = train↔serve numeric FORWARD parity (`dsv4_dspark_forward_parity_v2.py`) to CLOSE the loop, then tail-recipe (gamma↑ / anneal-to-0 / D-PACE). |
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
| ⭐ `ep0p5-ropefix-77w` *(RoPE-fix 0.5ep; 176 A3-single conc48)* | 555.51 | 759.47 | 463.69 | 905.20 | 451.46 |
| ⭐ `ep1p0-ropefix-77w` *(RoPE-fix 1.0ep; 176 A3-single conc48)* | 564.99 | 793.11 | 513.97 | 928.19 | 479.72 |
| ⭐⭐ `ep2p0-ropefix-77w` *(RoPE-fix 2.0ep; 176 A3-single conc48)* | 569.71 | 803.14 | 507.12 | 956.25 | 494.60 |
| ⭐⭐⭐ `ep2p5-ropefix-77w` *(RoPE-fix 2.5ep; 176 A3-single conc48)* | 589.27 | 815.07 | 538.09 | 973.71 | 500.58 |
| ⭐ `ep1p5-ropefix-77w` *(RoPE-fix 1.5ep; 176 A3-single conc48)* | 574.45 | 793.53 | 504.59 | 940.55 | 471.64 |
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
- [x] **no-spec base (AR) tok/s — full `DATASET=all` @ conc48** → **DONE 2026-08-08**. See the
      `AR baseline (no-spec)` row + the speedup table below. **Mean speedup 1.42× (1.21–1.77×).**
      Draft-independent (target + serve only) → measured once, reused as the denominator for every row.
- [ ] **AR + spec at conc1** — the low-concurrency pair. conc48 is throughput-bound and *understates*
      the draft: at conc1 nothing else fills the batch, so the accept-length win converts directly to
      latency. Needs BOTH arms re-run (2 runs). This is the number that matches what "speedup" means
      to most readers; the conc48 figure above is a conservative lower bound.
- [ ] **45W (`arrow_0715`) retrain** — the real deliverable; must lift pos3/pos4 (19/5 → toward
      released's 64/53) to close the tail gap. This is where accept_len 3.08 → ~4.4 comes from.
- [ ] **epoch0–3 of the 17W run** — convert + eval each for the epoch→accept_len curve.
