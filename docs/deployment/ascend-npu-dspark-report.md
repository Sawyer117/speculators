# DSpark / DFlash Speculative Decoding on Ascend NPU — Reproduction & Benchmark Report

> **Fork-only, team-internal.** Companion to `ascend-npu-dspark-install.md` (that doc = the
> from-scratch environment build; this doc = **the fixes, the benchmark methodology, the results,
> and the transferable learnings**). Read both to fully reproduce or to port speculative decoding
> into another project.

Status: DSpark **inference** reproduced on Ascend 910b (Qwen3-4B), fixes merged into the upstream
feature PR, DSpark-vs-DFlash benchmarked head-to-head on the same machine. Dates: 2026-06-30 →
2026-07-02.

---

## 1. What was reproduced

- **DFlash** = a lightweight draft model: a 1-layer cross-attention backbone that block-drafts
  `num_spec` tokens in parallel (query = mask tokens attending to the target's hidden states).
- **DSpark** = **DFlash backbone + a Markov head** (a low-rank bigram logit bias applied
  per-position in a short sequential loop) `[+ a confidence head, dormant at inference / used for
  dynamic block length in "v2"]`. So DSpark's delta over DFlash = the Markov refinement.
- Both run as the **draft model** in vLLM V1 speculative decoding; the **target** is stock Qwen3-4B.
- Qwen3 needs **zero custom Ascend kernels** — SparseMLA/TiLang/fused-deepseek kernels are
  DeepSeek-V4-only. Qwen3 (GQA) = DFlash backbone (already on vllm-ascend) + the per-step Markov bias.

## 2. Stack (see install doc for the full from-scratch build)

| component | version | note |
|---|---|---|
| CANN | 9.0.0 / 910b | unchanged from the DFlash box |
| torch / torch_npu | 2.10.0 | pinned by vllm-ascend main |
| numpy | 1.26.4 | **re-pin** after every build (2.x breaks triton-ascend) |
| vLLM | v0.23.0 built `VLLM_TARGET_DEVICE=empty` | "0.23.0+empty" |
| vllm-ascend | `Sawyer117/vllm-ascend` @ `dspark-npu-fixes` (`ee549165`) | PR #11153 + our fixes |
| triton-ascend | 3.2.1 | REQUIRED at runtime (block_table slot-mapping kernel) |
| clang-15 + **gxx_linux-aarch64** | conda-forge | triton JIT needs clang **and** libstdc++ headers |

## 3. The fixes (branch `dspark-npu-fixes`, merged into PR #11153)

Two clean commits on top of chenaoxuan's DSpark feature PR (vllm-ascend #11153):

**Commit 1 — load on vLLM 0.23.0 + the released checkpoint** (`patch/platform/patch_dspark_proposer.py`)
1. `DFlashQwen3Model._init` → `__init__` (v0.23.0 renamed it; import crash otherwise).
2. `dspark_markov_rank` → `markov_rank` (the actual released-ckpt field).
3. `_linear_output(...)` (undefined) → unpack `ReplicatedLinear` `(output, bias)`.
4. `confidence_head.proj` `bias=True` (released ckpt has the bias weight).

**Commit 2 — proposer runtime + cudagraph-safe drafting** (`spec_decode/llm_base_proposer.py`, `dflash_proposer.py`)
5. `hf_confif` → `hf_config` typo (crashed the proposer for **every** request).
6. **Cudagraph-safe Markov drafting**: the loop used a per-call `torch.empty` and seeded Markov
   position 0 from a per-step-reassigned `self._next_token_ids`, so a captured graph copied the
   **stale capture-time seed** → accept length collapsed (~5.7 eager → ~3.4 graph). Fix = the
   standard fixed-buffer pattern: persistent seed/draft buffers + in-place `copy_` of the seed, so
   graph replay always reads fresh data. Restores correct accept **in graph mode** (~5× throughput).

**Two required runtime settings** (not code — stack/config specific):
- `"method": "dflash"` in `--speculative-config` → routes DSpark onto `AscendDflashProposer` (the
  HS pipeline + `_next_token_ids`). Without it → generic `draft_model` proposer → crash.
- The released DSpark ckpt declares `architectures: ["Qwen3DSparkModel"]`, which vLLM 0.23.0
  doesn't register → **edit `config.json` `architectures` → `["DFlashDraftModel"]`** (loads via the
  DFlash path; the patch attaches the heads when it sees `markov_head_type`). Upstream vLLM PR
  #46995 would register `Qwen3DSparkModel` and remove this edit.

## 4. Serve + eval (one card)

```bash
export TARGET=/path/to/Qwen3-4B
export DRAFT=/path/to/dspark_qwen3_4b_block7   # architecture-edited
vllm serve "$TARGET" --trust-remote-code --tensor-parallel-size 1 \
  --max-num-seqs 64 --max-model-len 8192 \
  --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"method\":\"dflash\",\"draft_tensor_parallel_size\":1}" \
  --host 0.0.0.0 --port 30000
# eval client (accept length + throughput per benchmark):
cd examples/ascend_npu_dflash && bash run_eval.sh
```
Drop `--speculative-config` entirely for the **no-spec baseline**. Drop `--enforce-eager` (default)
for **graph mode** (needs triton-ascend + the cudagraph-safe fix).

## 5. Benchmark methodology — READ THIS BEFORE TRUSTING ANY NUMBER

These are the traps we hit; they generalize to any spec-decode benchmarking:

1. **Accept length ≠ speedup.** Accept length = mean tokens accepted per verify step
   (`1 + accepted/drafts`); it is the **theoretical ceiling** and is **hardware-independent**.
   Wall-clock **speedup = spec_tok/s ÷ no-spec_tok/s** is always lower (draft + verify overhead).
2. **Same-baseline discipline.** Speedup depends entirely on the denominator. A **graph** no-spec
   baseline is ~1.8× faster than an **eager** one, so "spec-graph ÷ no-spec-eager" **inflates** the
   number. Always compare **same machine, same card, same graph/eager config**. Our clean gsm8k
   decomposition: graph-opt **1.79×** × pure-spec **3.42×** = 6.11× (the inflated "vs eager" number).
3. **`num_speculative_tokens` = the checkpoint's block size**, not block−1. DSpark `block7` → **7**
   (paper config); using 6 dropped the 7th draft position and ~0.5 accept length. DFlash `block16`
   → 15. Wrong value silently lowers accept.
4. **Metrics can be unreliable.** vllm-ascend's Prometheus `vllm:spec_decode_*_total` counters
   **reset mid-run** in this build, so a naive before/after delta went `nan`/negative for
   non-first datasets. Fix = **reset-aware polling** (Evaluator.py `a3c41a6`: poll `/metrics`
   during each dataset, sum positive increments, treat drops as resets). Throughput (tokens/wall)
   is unaffected; only the /metrics-derived accept length was.
5. **Efficiency = speedup / accept length ≈ 0.53** here — this is **normal** for spec decode
   (typical 0.5–0.7). The ~47% gap to the ceiling is per-step overhead, **not** a bug.

## 6. Results (Qwen3-4B, Ascend 910b, same card, graph, greedy temp 0, concurrency 8)

**No-spec baselines (tok/s):** eager (old) gsm8k 302.69; **graph (correct denominator)** gsm8k
**541.41**, math500 573.52, humaneval 549.14, mbpp 580.43, mt-bench 483.65.

**DFlash (num_spec=15) vs DSpark (num_spec=7), speedup ÷ graph no-spec:**

| dataset | DFlash accept / tok-s / **speedup** / eff | DSpark accept / tok-s / **speedup** / eff | DSpark lead |
|---|---|---|---|
| gsm8k | 5.891 / 1709.34 / **3.16×** / 0.54 | 6.198 / 1848.76 / **3.42×** / 0.55 | +8% |
| math500 | 5.431 / 1602.27 / **2.79×** / 0.51 | 6.071 / 1874.56 / **3.27×** / 0.54 | +17% |
| humaneval | 4.090 / 1219.83 / **2.22×** / 0.54 | 5.486 / 1617.30 / **2.95×** / 0.54 | +33% |
| mbpp | 3.998 / 1191.57 / **2.05×** / 0.51 | 5.191 / 1534.17 / **2.64×** / 0.51 | +29% |
| mt-bench | 2.738 / 746.66 / **1.54×** / 0.56 | 3.779 / 989.42 / **2.05×** / 0.54 | +33% |

**Verdict:** efficiency is ~equal (~0.53) for both → the shared per-step overhead dominates, so
**wall-clock speedup tracks accept length**. DSpark's Markov head **holds accept far better on hard
distributions** (humaneval 5.49 vs 4.09, mt-bench 3.78 vs 2.74) → **DSpark wins every dataset, and
the lead widens (+8% → +33%) as tasks get harder.** DSpark gets higher accept at a *smaller* block
(7 vs 15), which is also cheaper per step. The Markov head pays off.

**Where the overhead goes (num_spec sweep, gsm8k):** solving `τ(n)=accept/tput=T_fixed+n·T_markov`
from num_spec=1 (accept 1.935, 686.03 tok/s) and num_spec=7 (6.198, 1848.76): the **Markov loop is
only ~18%** of per-step time; ~59% of the *draft* overhead is **fixed** (DFlash backbone forward +
`precompute_and_store_context_kv` HS→KV setup + per-step framework overhead). So to raise
efficiency, target the fixed draft overhead, **not** the Markov loop.

## 7. Transferable learnings (for adding spec decode to another project)

1. **Draft model plugs in via vLLM's `--speculative-config`**; the proposer method routes on the
   draft architecture / an explicit `method`. Draft ckpt must declare a **registered** architecture.
2. **Block/chain drafting**: a block drafter (DFlash) does one parallel forward → all `num_spec`
   positions; DSpark adds a **short sequential refinement loop** (Markov). Sequential loops are
   latency-bound — keep them cheap and **cudagraph-safe (persistent buffers, no per-call alloc, no
   reads of per-step-reassigned tensors)** or they silently break under graph capture.
3. **Benchmark honestly**: same-machine same-config baseline; report accept length (ceiling) AND
   measured speedup (reality); watch for unreliable metrics counters.
4. **Higher accept beats bigger block**: DSpark shows a smarter, higher-accept drafter at a smaller
   block wins over a plain drafter at a bigger block — both on quality and per-step cost.
5. **Efficiency ~0.5–0.7 is the reality**; don't expect accept-length× speedup. The loss is the
   backbone forward + verify + framework per-step overhead, shared by any drafter.

## 8. References

- vllm-ascend PR **#11153** (DSpark for Qwen3, author chenaoxuan) — our fixes merged into its
  `dspark` branch.
- Branch `Sawyer117/vllm-ascend@dspark-npu-fixes` (`ee549165`) — the 2 clean commits.
- vLLM PR **#46995** — would register `Qwen3DSparkModel` (removes the config.json edit).
- Env build + every gotcha: **`ascend-npu-dspark-install.md`** (same directory).
- Eval harness: `examples/ascend_npu_dflash/{run_eval.sh, Evaluator.py}` (Evaluator has the
  reset-aware /metrics fix).
