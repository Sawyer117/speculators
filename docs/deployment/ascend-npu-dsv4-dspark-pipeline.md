# DSV4-DSpark on Ascend NPU — End-to-End Pipeline (Index)

> **Top-level entry point.** The whole DSV4-Flash DSpark speculative-decoding pipeline on Ascend
> NPU, stage by stage: **environment build → data generation (rollout) → serving (verifier + HS
> producer) → draft training → evaluation.** Each stage links its archived write-up and its
> runnable script. For the deep design rationale (HS-extraction scheme choice, EP MoE refactor,
> decisions + results + validation matrix) see the EP-training doc linked in stage 4.

```
[1] ENV BUILD ──► [2] DATA GEN ──► [3] SERVE ─────────► [4] TRAIN ──► [5] EVAL
   install/setup    rollout→Arrow    verifier + HS_DUMP    FSDP2+EP8      accept-len
   (2 conda envs)   (greedy temp0)   hs_<row>.safetensors  DSpark draft   vs released draft
                          │                  │                  ▲
                          └──── row index ───┴── HS files ──────┘  (trainer reads Arrow + HS)
```

The coupling that makes it one chain: **rollout row index = HS file name = trainer sample key.**
The rollout produces an Arrow dataset (row `i` = one prompt+response with `loss_mask`); the serve,
driven prefill-only over each row's `input_ids`, dumps `hs_<i>.safetensors`; the trainer reads
Arrow row `i` **and** `hs_<i>` together (`loss_mask` from Arrow, hidden states from the HS file).

## The chain

| # | Stage | What it does | Archived doc | Script(s) |
|---|---|---|---|---|
| 1 | **Env build** | Two conda envs: **train** (speculators editable + transformers) and **serve** (vLLM 0.23.0 + vllm-ascend `feat/dsv4-hs-dumper`). Corrected toolchain, numpy 2.3.5 pinned last. | [`ascend-npu-dsv4-dspark-w8a8-inference.md`](./ascend-npu-dsv4-dspark-w8a8-inference.md) §3–4 (toolchain/env notes; **w8a8 track, archived**) | `install_npu_env_dspark.sh` (train, SSOT) · `setup_dsv4_env.sh` (serve, per-node) |
| 2 | **Data generation** | Rollout the target over a prompt set (greedy, temp=0, max-tokens 3072) → prompt+response+`loss_mask` → **Arrow dataset**. Sharded across nodes; tuning keeps short/numeric answers, drops only garbage. | [`ascend-npu-dsv4-rollout-data.md`](./ascend-npu-dsv4-rollout-data.md) (pipeline) · [`ascend-npu-dsv4-a3-rollout-handoff.md`](./ascend-npu-dsv4-a3-rollout-handoff.md) (A3 runbook) · [`ascend-npu-dsv4-rollout-benchmark.md`](./ascend-npu-dsv4-rollout-benchmark.md) (throughput/time) | `rollout_shard.sh` · `rollout_a3_shard.sh` · `rollout_stats.py` · `gen_tiny_dsv4_dataset.py` |
| 3 | **Serving (verifier + HS producer)** | DeepSeek-V4-Flash bf16 served (A2 two-node TP8/DP2 **EP off**, or A3 single-node). With `HS_DUMP=1` a plain serve becomes a hidden-state producer (Plan B dumper) — no PD-disagg, no `HiddenStateCacheSpec`. | [`ascend-npu-dsv4-bf16-dualnode-benchmark.md`](./ascend-npu-dsv4-bf16-dualnode-benchmark.md) (A2) · [`ascend-npu-dsv4-a3-singlenode-benchmark.md`](./ascend-npu-dsv4-a3-singlenode-benchmark.md) (A3) · [`ascend-npu-dsv4-hs-dumper-planB.md`](./ascend-npu-dsv4-hs-dumper-planB.md) (**HS extraction: two schemes, why Plan B**) · [`ascend-npu-dsv4-supports-eagle3-issue.md`](./ascend-npu-dsv4-supports-eagle3-issue.md) (native-extract path, paused) | `serve_dsv4_bf16_dualnode.sh` · `serve_dsv4_a3_singlenode.sh` · `hs_dump_driver.py` · `hs_dump_smoke.py` |
| 4 | **Draft training** | Faithful DSV4-native DSpark draft (3×[MLA+sink+256-MoE+mHC]+Markov/confidence heads), `speculators` FSDP2 **+ EP8 grouped-GEMM MoE**. Reads Arrow + rolling HS buffer. | [`ascend-npu-dsv4-dspark-ep-training.md`](./ascend-npu-dsv4-dspark-ep-training.md) (**design decisions + results + §9 validation matrix**) · [`ascend-npu-dsv4-dspark-training-port.md`](./ascend-npu-dsv4-dspark-training-port.md) (Plan 甲, faithful port spec) · [`ascend-npu-dsv4-dspark-compile-recompile.md`](./ascend-npu-dsv4-dspark-compile-recompile.md) (**WIP: throughput — anchor/recompute + recompile→compile**) | `train_dsv4_dspark.sh` · `scripts/train.py` · `test_compile_grouped_mm.py` |
| 5 | **Evaluation** | Serve verifier + trained draft; measure accept length / throughput vs the released-draft baseline (**num_spec=5, AL 3.94**, PR #11196). Reset-aware metrics poller. | (in the EP-training doc §6 + serve benchmark docs; standalone eval write-up TODO) | `run_eval.sh` · `run_eval_full.sh` · `Evaluator.py` · `gsm8k_eval.py` |

## Status at a glance

- **Pipeline: run end-to-end** (env → rollout → HS-producing serve → EP training → checkpoint
  save all validated on the box). **First train→eval pass done (2026-07-16):** training side healthy
  (soft accept_len ~2.9–3.1, position_1 ~0.82) but the epoch-1 draft **evals low (gsm8k hard 1.36)** —
  proven a **SERVE bug, not the weights** (the released known-good draft scores the same ~1.34 on our
  serve; root-caused to the fork's pre-rewrite DSpark). **Fix in progress: rebuild the serve on the
  #12005 rewrite** ([`vllm-ascend-dspark-rebuild.md`](./vllm-ascend-dspark-rebuild.md)); re-measure the
  trained draft after. Weights vindicated — no retrain. Details: EP-training §6.1 + the worklog.
- **Training levers** (`feat/dsv4-dspark @ 3953bc1`): MoE warm-start from the target
  (**`INIT_LAYER=1` whole-layer — the chosen best; `INIT_MOE=1` MoE-only was worse/unstable**),
  skip-validation + train-on-full-data (`NO_VAL=1`), activation recompute
  (`RECOMPUTE=1`, 384@3072), defaults `EPOCHS=5 / SEQLEN=3072 / LR=6e-4`; A/B two runs with
  `analyze_train_run.py --baseline`. Rationale in EP-training §4, run knobs in §7.
- **Correctness validation:** see the **validation matrix + green-check checklist** in
  [`ascend-npu-dsv4-dspark-ep-training.md`](./ascend-npu-dsv4-dspark-ep-training.md) §9 — component
  oracles mostly green; assembled-draft numeric parity + a few silently-wrong-mode gates still to
  write.

## Historical / superseded (not on this branch — on `feat/dspark-confidence-head`)

Kept for provenance; **not** part of the current chain, so not carried onto `feat/dsv4-dspark`:
- `ascend-npu-dsv4-dspark-landing-plan.md` — early landing plan; its "training framework = **MindSpeed-native** (DECIDED)" is **reversed**: MindSpeed-native was shelved (`speculators` is a hard req; GPU-safe matters). Superseded by the Track-A / Plan-甲 training-port + EP-training docs.
- `ascend-npu-dsv4-dspark-mindspeed-injection.md` — track-B MindSpeed injection spec; shelved with the above.
- `ascend-npu-dspark-install.md` / `ascend-npu-dspark-report.md` — **Qwen3-4B** DSpark *inference* baseline (the predecessor line), not DSV4. Useful as origin context; different model.

## Misc / archived (on this branch, kept for provenance)

- [`ascend-npu-dsv4-dspark-w8a8-inference.md`](./ascend-npu-dsv4-dspark-w8a8-inference.md) — the **early w8a8 inference track** (INT8 w8a8-mtp target). **Blocked on a vLLM-0.23/main w8a8 regression** (proven not CANN) — that block is exactly why we **pivoted to a bf16 target + the DSV4-native DSpark draft** (the current chain above). Retains reusable **toolchain/env notes** (§3–4: system-gcc-not-conda-clang, exit-127 `patch` gotcha, guaranteed v0.22.1rc1 baseline) and the drafter-extraction (13 GB) recipe. Its §8 status/next are archived, NOT current.
