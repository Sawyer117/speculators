# DFlash Qwen3-4B — NPU speculative-decoding evaluation (acceptance rate)

How to evaluate a trained DFlash draft on Ascend NPU with vLLM + GuideLLM, plus the
first validated result. **Milestone: vllm-ascend (vLLM 0.20.2) can serve a DFlash
speculator and run speculative decoding** — the "qwen3 draft not supported in vLLM"
training warning was NOT a blocker.

## First result (2026-06-23)

- **Checkpoint:** `outputs/qwen3-4b-dflash-npu/checkpoints/0/` (2.5 GB) — Qwen3-4B DFlash,
  **1 epoch**, half50 data, `kl_div` loss (early/pre-alignment run).
- **Verifier:** Qwen3-4B   **Dataset:** `RedHatAI/speculator_benchmarks`, subset `math_reasoning`
- **Load:** 80 requests, **0 errors**, ~1560 gen tok/s, ITL ~13.5 ms.

| Metric | Value |
|---|---|
| **Acceptance length** | **4.75** |
| Num drafts | 32,199 |
| Num draft tokens | 482,985 |
| Num accepted tokens | 120,878 |

Per-position acceptance rate (block size 16):

| pos | rate | pos | rate | pos | rate |
|----|------|----|------|----|------|
| 0 | 0.848 | 5 | 0.256 | 10 | 0.057 |
| 1 | 0.678 | 6 | 0.195 | 11 | 0.040 |
| 2 | 0.537 | 7 | 0.146 | 12 | 0.028 |
| 3 | 0.421 | 8 | 0.109 | 13 | 0.018 |
| 4 | 0.329 | 9 | 0.079 | 14 | 0.012 |

**Reading it:** acceptance length 4.75 = ~4.75 tokens produced per verifier step (1 verified
+ ~3.75 accepted draft tokens) → up to ~4.75× fewer decode steps (real wall-clock speedup is
less, due to draft + multi-token-verify overhead). Solid for a 1-epoch / kl_div / lower-
effective-LR draft; an aligned `ce` and/or longer-trained checkpoint should beat it.

> Use the **evaluate.py aggregate** acceptance_length (4.75). vLLM's tail `SpecDecoding
> metrics` log line is an instantaneous snapshot and reads lower (~3.5).

## How to reproduce

### 0. Deps — install GuideLLM ONLY
```bash
pip install 'guidellm==0.6.0'
```
⚠️ Do **NOT** `pip install -r scripts/evaluate/requirements.txt` — it pins `vllm>=0.12.0`
and would overwrite vllm-ascend. guidellm is the only missing piece.

### 1. Serve the trained draft as a speculator (one free NPU card)
```bash
CKPT=outputs/qwen3-4b-dflash-npu/checkpoints/0      # = SAVE_DIR/<epoch>
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve "$CKPT" \
  --port 8108 \
  --max-model-len 5120 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  2>&1 | tee /tmp/spec_serve.log &
until curl -sf --noproxy '*' http://localhost:8108/health >/dev/null 2>&1; do sleep 3; done
echo ready
```
Why these flags:
- `--max-num-batched-tokens 8192 --max-num-seqs 32` — without them, spec-decode scheduling
  computes a **negative** `max_num_scheduled_tokens` and vLLM refuses to start.
- `--max-model-len 5120` — evaluate.py (throughput mode) hard-codes `max_tokens=4096`
  (`evaluate.py:95`), so the server must allow prompt + 4096 output. 5120 ≈ 4096 + headroom.

Startup log should show `Resolved architecture: DFlashDraftModel` (draft loaded) +
`Qwen3ForCausalLM` (verifier).

### 2. Run the acceptance eval (math_reasoning ≈ GSM8K-style)
```bash
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"   # else localhost 504s via corp proxy
python scripts/evaluate/evaluate.py \
  --target http://localhost:8108/v1 \
  --dataset RedHatAI/speculator_benchmarks \
  throughput --subsets math_reasoning --max-requests 80
```
Valid subset names (there is **no** `GSM8K`):
`HumanEval, math_reasoning, qa, question, rag, summarization, tool_call, translation, writing`.
Results land in `outputs_<model>_<timestamp>/` (`acceptance.csv` + `artifacts/run_<subset>.json`).

### 3. Stop the server when done
```bash
pkill -f 'launch_vllm|EngineCore|APIServer'   # vLLM forks/retitles; kill the family
```

## Gotchas recap (all hit during the first run)

| Symptom | Cause | Fix |
|---|---|---|
| `max_num_scheduled_tokens = -1536`, serve won't start | spec draft slots > default batched-tokens | `--max-num-batched-tokens 8192 --max-num-seqs 32` |
| `504 Gateway Time-out` on /metrics & /v1/models | corp proxy intercepts localhost | `export no_proxy=localhost,127.0.0.1,::1` |
| `Couldn't find GSM8K.jsonl` | `GSM8K` is not a subset | use `math_reasoning` |
| `max_tokens=4096 > max_model_len=2048` (400) | eval hard-codes 4096 output | `--max-model-len ≥ prompt+4096` (e.g. 5120) |
| pip would break vllm-ascend | requirements.txt pins `vllm>=0.12` | install `guidellm==0.6.0` only |

## Notes

- This checkpoint is the early **kl_div / half50 / 1-epoch** run, not the aligned `ce` /
  open_perfectblend baseline. Re-run the same eval on the aligned checkpoint to compare
  (expect a higher acceptance length).
- Acceptance length / per-position rates are the right cross-run comparison metric — **not**
  the training loss magnitude (which is reduction-dependent; see `ascend-npu-dflash-loss.md`).

---

## Team-internal alignment harness (FORK-ONLY — do NOT upstream)

The GuideLLM path above is the upstream-official one. For day-to-day **team baseline
sync** we standardise on a lighter, dependency-free client + the standard public
spec-decode datasets, so every member reports comparable numbers. **These scripts and
this section are fork-only — they are never part of an upstream PR (we only upstream
`src/` core changes).**

Scripts (in `examples/ascend_npu_dflash/`):

| file | role |
|---|---|
| `run_server.sh`  | serve the Qwen3-4B verifier + the trained DFlash draft via `--speculative-config` |
| `Evaluator.py`   | the shared benchmark client (== colleague's `bench_dflash_vllm_all.py`, **keep identical across the team**) — pure `requests`, no GuideLLM |
| `run_eval.sh`    | one-command wrapper that runs `Evaluator.py` with the team-standard args |

### Pinned alignment params (everyone identical, or numbers are not comparable)

| knob | value | why |
|---|---|---|
| datasets | `gsm8k, math500, humaneval, mbpp, mt-bench` | standard spec-decode benchmarks |
| `num_speculative_tokens` | **15** | = training `block_size` (16) − 1 |
| temperature / top_p / top_k | **0 / 1 / 1** | greedy = deterministic headline; top_p/top_k inert at temp 0 |
| sample shuffle seed | **42** | fixed in `Evaluator.py` (`random.seed(42)`) |
| concurrency / warmup / max-new-tokens | 8 / 10 / 2048 | team defaults |

Defaults in `run_server.sh` derive from `config_qwen3_4b.sh` (`TARGET_MODEL`) and the
training output (`SAVE_DIR/checkpoint_best`), so after a training run it just works.

### Run it

```bash
# 1) serve (one free NPU card; defaults already aligned)
bash examples/ascend_npu_dflash/run_server.sh
curl -s --noproxy '*' http://localhost:30000/v1/models | head   # model listed = ready

# 2) benchmark all datasets (acceptance length + per-position + throughput)
bash examples/ascend_npu_dflash/run_eval.sh
```

The client reads the **same** vLLM Prometheus spec-decode counters as `evaluate.py`
(`vllm:spec_decode_num_drafts_total / num_accepted_tokens_total / ...per_pos_total`)
and computes acceptance length the same way (`1 + accepted/drafts`), so its numbers are
directly comparable to the GuideLLM result above (4.75) — only the dataset differs.

### Two NPU "if"s (match the colleague's config first; only change if you hit these)

1. **vLLM refuses to start with a negative `max_num_scheduled_tokens`** — spec-decode
   reserves `max_num_seqs*(1+num_speculative_tokens)` token slots/step. Set
   `EXTRA_FLAGS="--max-num-batched-tokens 8192" bash run_server.sh` **and tell the team**
   so everyone adds it (stay aligned).
2. **Long-prompt 400s** (prompt + `max_new_tokens` > `--max-model-len`). Default is now
   **8192**. 4096 looked fine on the first machine but a cross-machine repro hit a boundary
   prompt at 4097 > 4096 → 400 → crash: cross-machine noise (NPU-numeric generation-length
   drift or a tokenizer-version diff of ±1 token) tips borderline prompts over. 8192 removes
   the sensitivity. `max_model_len` is server context CAPACITY, not the generation length
   (`max_new_tokens` stays 2048) — raising it does NOT change the measured acceptance, only
   whether requests run, so 4096-run and 8192-run numbers are directly comparable. **Use the
   same value (8192) on every machine.**

`run_eval.sh` exports `no_proxy=localhost,...` so the corp proxy (`netentsec`) does not
504 the localhost `/metrics` + `/v1` calls.

### Baseline result (2026-06-28) — rollout draft, the team reference numbers

Checkpoint: rollout-trained DFlash Qwen3-4B (`ce` loss, on-policy
`open_perfectblend.qwen3-4b-rollout` data, `checkpoint_best`). Served `num_speculative_tokens=15`,
`max_model_len 8192`, `max_num_seqs 64`; benchmarked temp 0 / top_p 1 / top_k 1, seed 42,
concurrency 8, `max_new_tokens 2048`, FULL datasets.

**Accept length = baseline ± err** (err = spread of TWO independent reproductions — two NPU
machines, each retrained from scratch on the same data/config; centre = mean):

| Dataset | **Accept length (baseline ± err)** | Accept rate | Samples |
|---|---|---|---|
| **gsm8k** | **5.876 ± 0.004** | ~32.5% | 1309 |
| math500 | **5.446 ± 0.075** | ~29.6% | 490 |
| humaneval | **4.108 ± 0.024** | ~20.7% | 154 |
| mbpp | **4.006 ± 0.068** | ~20.0% | 247 |
| mt-bench | **2.670 ± 0.067** | ~11.1% | 70 (140 turns) |

These are the **team alignment targets** + their **noise floor**. Use them two ways:
- **Reproduction**: a fresh run reproduces the baseline if its accept length is within `± err`.
- **Signal vs noise**: a change (e.g. a different loss) is a *real* improvement only if it
  **exceeds `+ err`**; anything inside the band is noise. **gsm8k is the tightest (±0.004) → the
  best discriminator.** Easy rule: **±0.08 everywhere, ±0.01 for gsm8k.**

Caveats: (a) `± err` is an **n=2 point estimate** (a 3rd repro firms it up, esp. the ~0.07 on
math500/mbpp/mt-bench). (b) This is the **cross-machine** err (separately retrained drafts) — the
conservative, *portable* threshold. A *same-machine* comparison (e.g. CE/KL/LK on one box) has a
smaller same-machine err; don't apply the cross-machine band there. (c) bit-level reproduction is
impossible on NPU (kernel nondeterminism) — accept length is the robust cross-run metric, not loss.

Ordering math > code > chat is expected; per-position decays monotonically. Beats the earlier
kl_div/half50/1-epoch checkpoint (4.75 on math_reasoning above), confirming the rollout + `ce` recipe.

gsm8k per-position accept rate (block size 16, the headline curve):

| pos | rate | pos | rate | pos | rate |
|----|------|----|------|----|------|
| 0 | 88.4% | 5 | 36.4% | 10 | 12.3% |
| 1 | 75.3% | 6 | 30.0% | 11 | 9.3% |
| 2 | 63.4% | 7 | 24.4% | 12 | 7.0% |
| 3 | 53.0% | 8 | 19.8% | 13 | 5.1% |
| 4 | 44.0% | 9 | 15.9% | 14 | 3.5% |
