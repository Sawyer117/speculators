# DSV4-DSpark — spec-decode eval results ledger

Append-only ledger of DSpark draft acceptance-length runs on the **DeepSeek-V4-Flash bf16**
target. **To add a run:** append one row to the [summary matrix](#summary--accept_len-by-dataset)
*and* one detail section under [Runs](#runs-detail). Keep the matrix in sync so a regression is
one glance. Every run inherits [Shared setup](#shared-setup) unless its config block overrides it.

## Shared setup

Unless a run says otherwise, all numbers below share:

- **Target**: `DeepSeek-V4-Flash-bf16`.
- **Serve**: vllm-ascend PR **#12006 @ `386530d12`** (env `dspark-dsv4-serving`), A2 **dual-node bf16**
  (115 head / 116 worker, `world_size=16`), **EAGER=0** (graph mode — #12006 fixed graph),
  `num_speculative_tokens=5`, `draft_sample=greedy`, `aux_layers=[40,41,42]`,
  `VLLM_ASCEND_DSPARK_USE_STANDARD_DSA=1`. Serve script `examples/ascend_npu_dflash/serve_dsv4_bf16_dualnode.sh`.
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

### Throughput (tok/s) by dataset

⚠ Hardware/config-dependent — comparable **only within the same serve** (see metric defs). Logged for
reference / regression, not as a cross-hardware quality metric. `no-spec base` = the AR reference the
speedup is measured against.

| Run | gsm8k | math500 | humaneval | mbpp | mt-bench |
|-----|:-----:|:-------:|:---------:|:----:|:--------:|
| `no-spec base` *(ref, gsm8k-only so far)* | 302.69 | — | — | — | — |
| **released draft** | 187.55 | 313.03 | 169.41 | 374.51 | 268.59 |
| `epoch4-17w` | 162.26 | 256.48 | 143.19 | 285.44 | 220.50 |

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

## Baselines / TODO

- [x] **released draft — full `DATASET=all`** on our serve → **DONE 2026-07-20** (mean 4.42; gsm8k 4.658
      reproduced). This is now the same-serve bar row + detail section above.
- [ ] **no-spec base** tok/s reference — have gsm8k 302.69; extend to `DATASET=all` for a per-dataset
      speedup denominator (spec-decode tok/s ÷ no-spec tok/s).
- [ ] **45W (`arrow_0715`) retrain** — the real deliverable; must lift pos3/pos4 (19/5 → toward
      released's 64/53) to close the tail gap. This is where accept_len 3.08 → ~4.4 comes from.
- [ ] **epoch0–3 of the 17W run** — convert + eval each for the epoch→accept_len curve.
