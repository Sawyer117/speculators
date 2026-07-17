# DeepSeek-V4-Flash **bf16** two-node serve & rollout benchmark (Ascend A2)

Measured 2026-07-05. Full-precision (bf16) DeepSeek-V4-Flash served across **2× Atlas 800 A2**
(DP2 / TP8, expert-parallel OFF), and the throughput / full-run time estimate for rolling out
**open-perfectblend** (1,420,905 prompts).

## Setup

- **Hardware:** 2 nodes × 8× Atlas 800 A2 (64 GB) = **16 cards**. One serve spans both nodes.
- **Stack:** vLLM **0.23.0** + vllm-ascend (`dspark-dsv4`), **CANN 9.0.0**, torch_npu 2.10.0.
- **Model:** DeepSeek-V4-Flash **bf16** (dequantized from the released FP4/FP8 mixed weights), ~550 GB.
- **Client:** `scripts/response_regeneration/script.py`, temperature 0, max_tokens 3072,
  **concurrency 64** — the validated-clean ceiling (see "Concurrency ceiling" below). Estimates use a
  **conservative 0.54 rows/s** (same-basis vs the c128 point); the live rollout actually ran faster
  (0.74, open-perfectblend's shorter prompts) — see Throughput. We estimate long, not short.

## Parallelism — DP2 / TP8, **NO expert-parallel**

- **TP = 8** (intra-node). DeepSeek-V4-Flash has `o_groups = 8`; `wo_a` (ColumnParallelLinear) needs
  ≥1 group per TP rank → **TP is hard-capped at 8**. A single 8-card node cannot hold bf16 (experts
  alone ≈ 64.5 GiB/card > 60.96 GiB HBM), so a **second node (DP2)** is required.
- **DP = 2** across the two nodes.
- **`--enable-expert-parallel` MUST be OFF.** Ascend does not support cross-node EP dispatch
  (`aclnnMoeDistributeDispatchV4`, error **561000**); with it on, the first cross-node
  `HcclAllGather` deadlocks at startup (`No available shared memory broadcast block` forever).
  With it **off**, experts are **TP-sharded across all 16 cards** (`moe_tp_size = world_size = 16`),
  so per-card ≈ **38 GB** → fits, and MoE routing uses the AllGather comm path (works cross-node).
  Memory is identical whether experts are EP-sharded or TP-sharded at the same degree (total/16);
  the flag only selects the routing op.

Serve flags (per node; worker adds `--headless --data-parallel-start-rank 1`):
```
--data-parallel-size 2 --data-parallel-size-local 1
--data-parallel-address <head_ip> --data-parallel-rpc-port 13389
--tensor-parallel-size 8            # NO --enable-expert-parallel
--max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 64
--block-size 128 --no-enable-prefix-caching --async-scheduling
--additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'   # peak; --enforce-eager = baseline
--gpu-memory-utilization 0.90       # worker uses 0.85 (avoids cudagraph-warmup OOM)
```
Key env: `HCCL_INTRA_PCIE_ENABLE=1`, `HCCL_BUFFSIZE=1024`, `HCCL_HOST/NPU_SOCKET_PORT_RANGE`,
`VLLM_ASCEND_ENABLE_FLASHCOMM1=0` (Flash Comm v1 asserts EP=True — incompatible with EP-off),
`VLLM_ASCEND_BALANCE_SCHEDULING=1`, `USE_MULTI_BLOCK_POOL=1`, `TRITON_ALL_BLOCKS_PARALLEL=1`.
Script: `examples/ascend_npu_dflash/serve_dsv4_bf16_dualnode.sh` (`EAGER=0 MAXSEQS=64` = peak).

## Throughput (one serve = 2 nodes)

| Config | max-num-seqs | client concurrency | seqs/DP-replica | rows/s | quality |
|---|---|---|---|---|---|
| eager (baseline) | 16 | 32 | 16 | 0.10 | clean |
| **graph FULL_DECODE_ONLY** | 64 | **64** | **32** | **~0.54** † | **clean (gsm8k 96.1 %)** |
| graph, over-batched | 64 | 128 | 64 | 0.63 | **GARBAGE (gsm8k 13.3 %) — do NOT use** |

- **Graph mode ≈ 6× over eager** (eager at small batch is launch-overhead bound; cudagraph removes it).
- **† Two numbers, different purposes.** Same-basis as the c128 point (batch 32/replica ≈ 85 % of
  batch 64) the clean config is **~0.54 rows/s** — use this for estimates (conservative). The live
  open-perfectblend rollout actually sustained **0.74 rows/s** (cumulative over 40 k rows / 15 h),
  higher only because that dataset's prompts are shorter, so real completion is likely *faster* than
  the 0.54 estimate. Do NOT compare 0.74 to the 300-prompt 0.63 (different data → rows/s not comparable).
- **The batch-64/replica "peak" (0.63) is only ~5 % over 0.54 but emits GARBAGE** — see next section.
  The clean ceiling is **client concurrency 64 (= 32 seqs / DP-replica)**.

## Concurrency ceiling — the KV-overflow garbage bug (hardest-won lesson #2)

The serve **silently emits garbage** (incoherent tokens — `No`, `Hateful`, repeated fragments) when
too many long sequences are in flight, **even though every request returns HTTP 200 (`errors=0`)**.
Full gsm8k (1319) is the litmus:

| client concurrency | seqs / DP-replica | gsm8k (1319) | verdict |
|---|---|---|---|
| 1 | 1 | 100 % | clean |
| 64 | 32 | **96.1 %** | clean |
| 128 | 64 | **13.3 %** | garbage |

**Root cause = KV-cache overflow.** From the engine log: bf16 weights take **36.84 GiB/card**, leaving
only **15.17 GiB → GPU KV cache = 29,795 tokens** per DP-replica. At 64 seqs/replica that is
`29,795 / 64 ≈ 465 tokens/seq`, but CoT answers need ~600–1000 (tail to 3072) → the KV pool overflows,
and vllm-ascend's overflow handling (preempt/recompute) **corrupts the output instead of degrading
gracefully** — a genuine vllm-ascend/DSV4 bug. At 32 seqs/replica → `~930 tokens/seq` → fits → clean.
Raising `max-num-seqs` makes it **worse** (more seqs → less KV each → more overflow); the only real
lever is more KV, which bf16 can barely spare (log suggests 15.2 → 19.8 GiB, still ~64-seq territory).

**Operating rules (baked into the rollout):**
- **Client concurrency ≤ 64** (≤ 32 seqs/DP-replica). Never over-send past the server's per-replica cap.
- **Validate every new serve config with the full gsm8k before rollout** — `errors=0` does NOT mean
  correct; only scoring the output catches the garbage.
- **Scale throughput by adding serve-pairs, not batch.** bf16 weights starve the KV cache, so the safe
  per-serve batch is hard-capped ~32/replica; more nodes ≠ bigger safe batch.

## Full-run estimate — open-perfectblend (1,420,905 prompts)

bf16 uses **2 machines per serve**. Serves are independent (embarrassingly parallel; round-robin
`--resume`-safe shards), so time scales linearly with serve count.

```
per-serve throughput = 0.54 rows/s (conservative — estimate long, not short)
full-run (S serves)  = 1,420,905 / (0.54 × S) seconds ≈ 30 / S days
```

| Machines | Serves (2 machines each) | Full 1.42 M rollout |
|---|---|---|
| 2 | 1 | **~30 days** |
| 4 | 2 | ~15 days |
| 8 | 4 | ~7.6 days |
| **16** | **8** | **~3.8 days** |
| N | N/2 | **≈ 61 / N days** |

The live rollout is running at **0.74 rows/s** (open-perfectblend's shorter prompts), so real time is
likely closer to **~44/N days** (16 machines ≈ 2.8 days) — treat that as upside; **plan for 61/N.**

## bf16 vs w8a8

| | throughput | per-machine | full 1.42 M, 16 machines |
|---|---|---|---|
| **w8a8** (single node) | 0.57 rows/s **per machine** | 0.57 rows/s | 16 serves → **~1.8 days** |
| **bf16** (2 nodes/serve) | 0.54 rows/s **per serve** | 0.27 rows/s | 8 serves → **~3.8 days** |

bf16's per-machine efficiency ≈ **47 %** of w8a8 (it needs 2× the weight bandwidth and 2 machines
per serve, and — because the bf16 weights starve the KV cache — can only run half the safe batch).
For the same machine count bf16 is **~2.1× slower** — the cost of full precision. (Conservative 0.54;
the live rollout's 0.74 narrows the gap somewhat.)

## Caveats

- Estimates use a conservative **0.54 rows/s** (same-basis vs the c128 garbage point). The live
  rollout's cumulative rate was **0.74** (open-perfectblend, shorter prompts; instantaneous swings
  0.53–0.84 with the length mix) — real time is likely faster, but plan for 0.54.
- **`errors=0` is NOT a quality signal** — garbage still returns HTTP 200. Always spot-check the actual
  generated text (and gsm8k-score any new serve config) before trusting a rollout.
- `--enable-expert-parallel` remains the single hardest-won lesson: it is REQUIRED off for two-node
  A2 (561000). See `docs/deployment/ascend-npu-dsv4-dspark-w8a8-inference.md` and the memory notes.
