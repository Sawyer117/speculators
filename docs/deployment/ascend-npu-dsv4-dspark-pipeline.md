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
| 1 | **Env build** | **3 roles / 2 build-types**: (a) **train-compute** (speculators editable + transformers + torch_npu, **no vLLM**); (b) **serve** = ONE vllm-ascend build, run `HS_DUMP=1` → HS producer (stage 3) **or** `DRAFT=` → eval spec-decode (stage 5). numpy 2.3.5 pinned last. *(Physically one conda env per machine; the two serve envs `austin`/`serving` can later collapse to a single HS_DUMP-gated build — **deferred, don't touch the working stack**.)* | env-build scripts = the Script column. Toolchain gotchas only: [`w8a8-inference.md`](./ascend-npu-dsv4-dspark-w8a8-inference.md) §3–4 (system-gcc-not-conda-clang, exit-127 `patch`) — *archived track, notes only.* | **train:** `install_npu_env_dspark.sh` (SSOT) · `setup_dsv4_train_compile.sh` (A3 torch-2.12) · **serve:** `setup_dsv4_serve_a3.sh` (A3 #12006) · `setup_dsv4_env.sh` (per-node) |
| 2 | **Data generation** | Rollout the target over a prompt set (greedy, temp=0, max-tokens 3072) → prompt+response+`loss_mask` → **Arrow dataset**. Sharded across nodes; tuning keeps short/numeric answers, drops only garbage. | [`ascend-npu-dsv4-rollout-data.md`](./ascend-npu-dsv4-rollout-data.md) (pipeline) · [`ascend-npu-dsv4-a3-rollout-handoff.md`](./ascend-npu-dsv4-a3-rollout-handoff.md) (A3 runbook) · [`ascend-npu-dsv4-rollout-benchmark.md`](./ascend-npu-dsv4-rollout-benchmark.md) (throughput/time) | `rollout_shard.sh` · `rollout_a3_shard.sh` · `rollout_stats.py` · `gen_tiny_dsv4_dataset.py` |
| 3 | **Serving (verifier + HS producer)** | DeepSeek-V4-Flash bf16 served. **A3 single-node (current): expert-parallel ON** (`ENABLE_EP=1` — intra-node EP works). *A2 two-node (deprecated): EP OFF — cross-node EP all-gather DEADLOCKS on Ascend A2 (hard-won lesson).* ⚠️ this is the **verifier-serve's** EP, distinct from **training's EP8**. With `HS_DUMP=1` a plain serve becomes a hidden-state producer (Plan B dumper) — no PD-disagg, no `HiddenStateCacheSpec`. | [`ascend-npu-dsv4-bf16-dualnode-benchmark.md`](./ascend-npu-dsv4-bf16-dualnode-benchmark.md) (A2) · [`ascend-npu-dsv4-a3-singlenode-benchmark.md`](./ascend-npu-dsv4-a3-singlenode-benchmark.md) (A3) · [`ascend-npu-dsv4-hs-dumper-planB.md`](./ascend-npu-dsv4-hs-dumper-planB.md) (**HS extraction: two schemes, why Plan B**) · [`ascend-npu-dsv4-supports-eagle3-issue.md`](./ascend-npu-dsv4-supports-eagle3-issue.md) (native-extract path, paused) | `serve_dsv4_bf16_dualnode.sh` · `serve_dsv4_a3_singlenode.sh` · `hs_dump_driver.py` · `hs_dump_smoke.py` |
| 4 | **Draft training** | Faithful DSV4-native DSpark draft (3×[MLA+sink+256-MoE+mHC]+Markov/confidence heads), `speculators` FSDP2 **+ EP8 grouped-GEMM MoE**. Reads Arrow + rolling HS buffer. | **[`ascend-npu-dsv4-dspark-ep-training.md`](./ascend-npu-dsv4-dspark-ep-training.md) — THE canonical training doc** (design decisions + results + §9 validation matrix). Detail companions: [`…-training-port.md`](./ascend-npu-dsv4-dspark-training-port.md) (Plan 甲 faithful **arch spec**) · [`…-compile-recompile.md`](./ascend-npu-dsv4-dspark-compile-recompile.md) (**throughput axis** — anchor/recompute + recompile→compile, WIP). | `train_dsv4_dspark.sh` · `scripts/train.py` · `test_compile_grouped_mm.py` |
| 5 | **Evaluation** | Serve verifier + trained draft; measure accept length / throughput vs the released draft. **Primary bar = released-on-OUR-serve `DATASET=all` mean 4.42 (gsm8k 4.658 reproduced)**; official 3.94 @ num_spec=5 (PR #11196) = the cross-stack sanity floor only. Reset-aware metrics poller. | [**`ascend-npu-dsv4-dspark-eval-results.md`**](./ascend-npu-dsv4-dspark-eval-results.md) (**append-only results ledger** — accept_len matrix + per-dataset/per-position detail per run/baseline) · EP-training doc §6 (design rationale) | `run_dspark_eval.sh` · `Evaluator.py` |

## Status at a glance

- **★★★ RoPE root-caused, fixed, and EVAL-VALIDATED (2026-08-05).** The long-standing accept-length
  gap was **degenerate training-RoPE**: the draft's `freqs_cis` was complex64, NPU aclnn can't handle
  complex so the trainer cast the model to bf16 which dropped the imaginary part → `apply_rotary_emb`'s
  complex×real went from a real ROTATION to a **scale-only** op, while the serve rotated properly = the
  train↔serve divergence. Fixed to real cos/sin interleaved (`feb0066`+`8db8f75`, matches
  vLLM-Ascend/MindSpeed/torchtitan-npu). **A from-scratch RoPE-fixed run — same recipe as the degenerate
  `ep0p5-bal1e3`, ONLY variable = RoPE — evals at just 0.5ep to mean 3.84 = NEW BEST across everything**
  (beats the prior best `f1-1.5ep` 3.63 at 1/3 the epochs; 87% of the released bar 4.42; ALL 5 datasets up
  +0.16…+0.41). The gain is the **tail** (later block slots finally rotate) and the diagnosed
  **`train↑/eval↓` divergence is RESOLVED — eval now tracks train.** Full row + per-position in the stage-5
  ledger. ⟹ the earlier "gap = data/recipe/tail / serve bug / no retrain" conclusions below are SUPERSEDED.
- **A3 two-box move + eval baselines locked (2026-07-20).** New topology: **182 = A3 inference + training-HS
  producer**, **176 = A3 training on the torch-2.12 COMPILE stack** (data/weights migrating A3
  `/home/canada_group_folder` → A2 `/share`). One-shot env builds: `setup_dsv4_serve_a3.sh` (#12006 serve, now
  pins transformers==5.13.0 + VLLM_ENGINE_READY_TIMEOUT_S=1800 for the 543 GB load) and
  `setup_dsv4_train_compile.sh` (torch 2.12.0+cpu / torch_npu 2.12.0rc1 / inductor_npu_ext / triton-ascend, no
  vLLM). **No-shared-FS HS transport SOLVED:** `hs_sidecar.py` serves the dumped `hs_<id>.safetensors` over
  HTTP(S) to the remote trainer (`HS_FETCH_BASE`), and the validated **Plan-B dumper is ported onto the #12006
  A3 stack** — `Sawyer117/vllm-ascend@dspark-dsv4-v3-hsdump` (zero-risk: #12006 already exposes
  `get_mtp_target_hidden_states()`; pure-python, no rebuild), enabled by `serve_dsv4_a3_singlenode.sh HS_DUMP=1`.
  **Eval baselines** now in the append-only ledger (stage 5): released draft full-`DATASET=all` **mean 4.42**
  (gsm8k 4.658 reproduced); our best-at-the-time `epoch4-17w` **mean 3.08 = 70%** (gap then blamed on the
  pos3/pos4 tail — ⟹ **SUPERSEDED: the tail gap was degenerate RoPE; new best = `ep0p5-ropefix` 3.84**, see the top bullet). Static
  scoreboard `plot_best_vs_baseline.py`; `analyze_train_run.py` overlays the 3 released refs. **77W** dataset
  (775,965 deduped, the newest/most-complete — supersedes 17W/45W) registered (§3.1) + being prepped to
  `arrow_0720_77w` for the next retrain. See the **2026-07-20 worklog section** for the live bring-up detail.
- **ALL FIXES in one branch + upstream-synced; the real training run is going (2026-07-18).** `feat/dsv4-dspark
  @ 4354a6d` now carries FIVE fixes: (1) pos0 **decay** slot-0, (2) **Markov** `prev_token_ids` alignment (slots
  ≥1), (3) **metrics** slot-0 (soft accept_len/confidence), (4) **AMP fp32 master weights** (norm-frozen /
  weight-decay-dropped bug — option B: small trainable params fp32, EP experts+frozen bf16; +~0.5 GB, no OOM), and
  (5) a one-time **`git merge upstream/main`** (fork is a real fork, merge-base #736; gained #788 float32
  divergence loss, #759 metric double-reduction, #711 checkpointer dtype). Fixes 1–2 are serve-validated against
  the DSV4 proposer (slot k prev = pos p+k = raw block_tokens); the two are ORTHOGONAL (decay=slot 0,
  Markov=slots 1–4). A smoke run caught + fixed a latent complex64 `freqs_cis` crash from the AMP selective cast;
  re-smoke clean (no NaN, mem ~56 GB, fits). **⟹ that same complex64→bf16 cast is exactly the
  degenerate-RoPE mechanism later root-caused (2026-08-05, `feb0066`): casting the complex rotary to
  bf16 silently drops the imaginary part → scale-only, no rotation. Fixed there; see the top bullet.** Watch: `position_0_acc` past lr-peak + **`hard_accept_len` breaking
  the killed run's ~2.4**; AMP proof = norm-changed-vs-verifier at the 1st ckpt. Detail: the **2026-07-18 worklog
  section**. (torch-2.12 compile env for the recompile bottleneck = a parallel later track.)
- **Serve FIXED, then root-caused OUR draft (2026-07-17, REVERSES the earlier "serve bug, no retrain").**
  Rebuilt the serve to **#12006** (`dspark-dsv4-v3`); the known-good released draft (dequant'd bf16 via
  `build_released_draft_dir.py --dequant-bf16`) scores **gsm8k accept_len 4.658** on our serve (smooth
  pos0 0.925→pos4 0.538, above official 3.94) ⇒ the serve is fixed. Our epoch-1 draft still evals ~1.758
  (sharp pos0→pos1 cliff) on the SAME serve ⇒ **the weights were the problem.** Root cause: the fork
  **DELETED upstream's `sample_from_anchor` switch and hardcoded FALSE**; DSpark serving needs **True**
  (every slot sampled). It gates in THREE places — target **roll**, slot-0 **loss-mask**, and the
  per-position **loss decay** (`dflash_loss_decay` zeroed pos 0 → slot-0 got no gradient, `position_0_acc
  ~0.03`; the piece we first missed). **Fixed + cross-validated byte-for-byte vs canonical DeepSpec**
  (`exp(-pos/gamma)`, pos0 highest); upstream independently merged the identical fix as **#798** (we've since
  synced the fork to `upstream/main` — see the 07-18 bullet). Plus BLOCK 6→5 and a bit-identical shared-KV
  attention (−2.1 GB). **Superseded by the 2026-07-18 all-fixes run.** Detail: the **2026-07-17 worklog
  section** + EP-training §6.1.
- **Training levers** (`feat/dsv4-dspark @ 928ea32`): MoE warm-start from the target
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
