# DeepSeek-V4-Flash **bf16** two-node serve & rollout benchmark (Ascend A2)

Measured 2026-07-05. Full-precision (bf16) DeepSeek-V4-Flash served across **2× Atlas 800 A2**
(DP2 / TP8, expert-parallel OFF), and the throughput / full-run time estimate for rolling out
**open-perfectblend** (1,420,905 prompts).

## Setup

- **Hardware:** 2 nodes × 8× Atlas 800 A2 (64 GB) = **16 cards**. One serve spans both nodes.
- **Stack:** vLLM **0.23.0** + vllm-ascend (`dspark-dsv4`), **CANN 9.0.0**, torch_npu 2.10.0.
- **Model:** DeepSeek-V4-Flash **bf16** (dequantized from the released FP4/FP8 mixed weights), ~550 GB.
- **Client:** `scripts/response_regeneration/script.py`, temperature 0, max_tokens 3072, on a frozen
  300-prompt open-perfectblend sample (shard_00).

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

## Throughput (300-prompt sample, one serve = 2 nodes)

| Config | max-num-seqs | concurrency | wall (300) | rows/s | tok/s (agg) | per-req tok/s |
|---|---|---|---|---|---|---|
| eager (baseline) | 16 | 32 | 3126 s | 0.10 | 51 | 1.6 |
| **graph FULL_DECODE_ONLY (peak)** | 64 | 128 | **473 s** | **0.63** | **389** | 3.2 |

- **Graph mode + batch 64 = 6.6× wall-clock, 7.6× tok/s** over eager. Eager at small batch is
  launch-overhead bound; cudagraph removes it.
- Sample completion length: mean ~530–614 tokens (median ~160–220; 2–11 % hit the 3072 cap).
  Because greedy decoding differs slightly between eager and graph kernels, **tok/s (389) is the
  fairest cross-config metric**; rows/s is sensitive to the sample's length tail.

## Full-run estimate — open-perfectblend (1,420,905 prompts)

bf16 uses **2 machines per serve**. Serves are independent (embarrassingly parallel; round-robin
`--resume`-safe shards), so time scales linearly with serve count.

```
per-serve throughput = 0.63 rows/s (measured, peak)
full-run (S serves)  = 1,420,905 / (0.63 × S) seconds ≈ 26 / S days
```

| Machines | Serves (2 machines each) | Full 1.42 M rollout |
|---|---|---|
| 2 | 1 | **~26 days** |
| 4 | 2 | ~13 days |
| 8 | 4 | ~6.5 days |
| **16** | **8** | **~3.2 days** |
| N | N/2 | **≈ 52 / N days** |

Token-normalized to the dataset's ~505-token average (this sample ran longer at 614) → ~21 days/serve,
i.e. ~**42 / N days**. Use ~52/N as the conservative figure.

## bf16 vs w8a8

| | throughput | per-machine | full 1.42 M, 16 machines |
|---|---|---|---|
| **w8a8** (single node) | 0.57 rows/s, 286 tok/s **per machine** | 0.57 rows/s | 16 serves → **~1.8 days** |
| **bf16** (2 nodes/serve) | 0.63 rows/s, 389 tok/s **per serve** | 0.315 rows/s | 8 serves → **~3.2 days** |

bf16's per-machine efficiency ≈ **55–68 %** of w8a8 (it needs 2× the weight bandwidth and 2 machines
per serve). For the same machine count bf16 is **~1.5–1.8× slower** — the cost of full precision.

## Caveats

- Numbers from a 300-prompt steady-state sample; the full run varies a few % with the length tail.
- The two configs' outputs are not byte-identical (eager vs graph greedy kernels diverge slightly at
  temp 0) — expected; pick one config (graph) for the actual rollout and it is self-consistent.
- `--enable-expert-parallel` remains the single hardest-won lesson: it is REQUIRED off for two-node
  A2 (561000). See `docs/deployment/ascend-npu-dsv4-dspark.md` and the memory notes.
