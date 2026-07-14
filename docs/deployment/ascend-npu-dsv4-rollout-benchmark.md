# DeepSeek-V4-Flash w8a8 rollout benchmark & time estimate (Ascend A2)

Measured 2026-07-04. Goal: estimate wall-clock to roll out the full **open-perfectblend**
(on-policy training data) through **DeepSeek-V4-Flash-w8a8-mtp**, and pick the serving config.

## Setup

- Hardware: **1 node = 8× Atlas 800 A2 (64 GB)**, single-node **DP1 / TP8**, expert-parallel.
- Stack: vLLM **0.23.0** + vllm-ascend `dspark-dsv4 @ 6036507` (#11196 + o_proj fix), **CANN 9.0.0**,
  torch_npu 2.10.0. Model = INT8 w8a8 (`--quantization ascend`).
- Client: `scripts/response_regeneration/script.py`, **temperature 0**, `max_tokens 3072`,
  non-thinking (DSV4 default). Frozen samples of the seed set drive the measurement.

## Dataset

`open_perfectblend_full.jsonl` = **1,420,905 prompts**. Reference (already-done Qwen3-4B rollout):
mean **581** completion tokens, 98.3% finish naturally / 1.7% hit the 3072 cap, **825 M** tokens total.

## What we measured (100-prompt sample unless noted)

DSV4 output length ≈ Qwen3: **mean 505** completion tokens (median 256, p99 2481, **0% hit 3072** in
sample) → full set ≈ **~717 M** tokens.

| Config | batch (max-num-seqs) | 100-prompt wall | throughput | note |
|---|---|---|---|---|
| AR | 16 / conc 32 | 442 s | 0.226 rows/s, ~114 tok/s | initial (latency-tuned) |
| **AR** | **64 / conc 64 + multistream** | **176.6 s** | **0.57 rows/s, ~286 tok/s** | **2.5× — chosen** |
| AR | 128 / conc 128 (400-prompt steady) | — | **0.57 rows/s** | = seqs 64 → **batch saturated** |
| MTP | 64 / conc 64 | ~12 min, stalled | ~4× slower | KV over-subscribed, thrashes |
| MTP | 32 / conc 32 (official) | ~470 s | 2.7× slower than AR64 | spec is a net loss at batch |

### Key findings

1. **Throughput saturates at batch ~64.** Going 64→128 gave nothing: the 284B-total / 13B-active MoE
   is compute-bound at batch ~64. **Ceiling on 8×A2 ≈ 0.57–0.60 rows/s ≈ ~286 tok/s aggregate.**
2. **Speculative decoding (MTP) is a net *throughput* loss for batched rollout** — expected, not a bug.
   vLLM docs: spec decode targets "medium-to-low QPS, memory-bound workloads"; vLLM blog: at
   compute-bound / high batch "the overhead … can outweigh its benefits". Corroborating issues:
   vllm #42505 (MoE DFlash slower than baseline at concurrency > 8), vllm-ascend #8967 / #9247 (Ascend
   MTP throughput regression / DSV4-Flash MTP prefix-cache penalty). So the full run uses **plain AR**.

## How the time estimate is derived

Rollout is embarrassingly parallel: N machines each run an independent 8-card serve on 1/N of the
prompts (round-robin shards), zero cross-machine comms → **linear** scaling.

```
per-machine throughput  T = 0.57 rows/s   (measured, steady-state, batch 64)
full-run wall (N machines) = total_rows / (T × N)
                           = 1,420,905 / (0.57 × N)  seconds
                           ≈ 28.9 / N  days
```

| Machines N | Full 1.42 M rollout |
|---|---|
| 1 | **~29 days** |
| 4 | ~7.2 days |
| 8 | ~3.6 days |
| **16** | **~1.8 days** |
| N | **≈ 28.9 / N days** |

To hit a deadline of X days → need ≈ **28.9 / X** machines (e.g. 5 days → 6 machines; 2 days → 15).
A **subset** scales linearly too (e.g. 500 k prompts = 0.35× → ~10 days single / ~1 day on 16).

`max_tokens 3072 → 4096` adds only **~1–3%** (only the ~1–2% of prompts that hit the 3072 cap generate
more; the 98%+ that finish naturally are unaffected; paged KV means no batch penalty).

## Execution plan (16-way)

1. Pre-split once on shared storage (round-robin = balanced, no shuffle):
   `split -n r/16 -d -a 2 --additional-suffix=.jsonl open_perfectblend_full.jsonl $SHARDDIR/shard_`
   → `shard_00.jsonl … shard_15.jsonl`, ~88,800 prompts each.
2. Each machine i runs `examples/ascend_npu_dflash/rollout_shard.sh <i>` under nohup/tmux — it starts
   that machine's AR serve, waits ready, and rolls out `shard_<i>` with `--resume` (crash-safe) to a
   shared per-shard output file.
3. Collate: `cat $OUTDIR/rollout_*.jsonl > open_perfectblend.dsv4-w8a8-rollout.full.jsonl`.

## Caveats

- Throughput from 100/400-prompt steady-state samples; first-N deterministic (comparable across
  configs). Real full-run may vary a few % with the length tail.
- Linear machine scaling assumes each node ≈ same 8×A2 throughput and independent serves (holds here —
  no shared bottleneck; shards read from shared NFS, generation dominates).
- AR-vs-MTP **content is byte-identical at temperature 0** (spec is lossless) — MTP was dropped for
  speed, not correctness.
