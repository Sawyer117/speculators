# DeepSeek-V4-Flash **bf16** single-node serve & rollout benchmark (Ascend **A3**)

Measured 2026-07-08. Full-precision (bf16) DeepSeek-V4-Flash served on **one Atlas 800 A3**
(16 logical devices in a single box, **DP2 / TP8 / EP16**, expert-parallel **ON**), benchmarked with
the same client and settings as the two-node A2 run so the numbers are directly comparable. Peer doc:
[`ascend-npu-dsv4-bf16-dualnode-benchmark.md`](ascend-npu-dsv4-bf16-dualnode-benchmark.md) (2×A2).

## TL;DR — A3 (single node, EP16) beats 2×A2 (TP-sharded, cross-node)

| Serve | parallelism | rows/s (graph, c64, clean) | per **physical box** | per **device** (÷16) |
|---|---|---|---|---|
| **A3 single node** | DP2 / TP8 / **EP16** | **1.15** † | **1.15** | 0.072 |
| 2×A2 (peer doc) | DP2 / TP8 / **EP off** | 0.54–0.74 | 0.27 | 0.034 |

**A3 is ~1.55× faster on the same rows/s basis** (1.15 vs the live 0.74), and ~2.1× vs the conservative
0.54. Per **device** it is ~2.1× (pure architecture win: EP16 all-to-all over intra-node HCCS vs the
cross-node TP-sharded-expert fallback A2 is forced into). Per **physical box** it is ~4.3× (A3 also
packs 16 devices in one node, so it needs 1 box where bf16-A2 needs 2). † A3's 1.15 is a small
300-prompt run (ramp/drain drags it); steady-state on a large run is higher — treat 1.15 as a floor.

## Setup

- **Hardware:** 1× Atlas 800 **A3** = 8 cards × 2 dies × 64 GB = **16 logical devices**, one HCCS fabric.
- **Stack:** vLLM **0.23.0** + vllm-ascend (`dspark-dsv4`), **CANN 9.0.0**, torch_npu 2.10.0.
- **Model:** DeepSeek-V4-Flash **bf16**, served from `…/ckpt/DeepSeek-V4-Flash-bf16`.
- **Client:** `scripts/response_regeneration/script.py`, temperature 0, max_tokens 3072,
  **concurrency 64** (= 32 seqs / DP-replica — the validated-clean ceiling, same as A2).
- **Data:** open-perfectblend shard (`shard_02`), 300 prompts, `--resume`.
- Serve script: `examples/ascend_npu_dflash/serve_dsv4_a3_singlenode.sh` (`EAGER=0` = graph = peak).

## Parallelism — DP2 / TP8 / **EP16** (the thing A2 can't do)

- All 16 devices are in **one node on the HCCS fabric**, so `--enable-expert-parallel` runs with the
  **EP dispatch/combine all-to-all staying intra-node** — no cross-node hang. This is exactly what
  A2 could NOT use: cross-node EP16 dispatch (`aclnnMoeDistributeDispatchV4`) errors **561000** and the
  first cross-node `HcclAllGather` deadlocks, forcing A2 into TP-sharded experts (dense AllGather every
  MoE layer, smaller per-rank GEMMs). Each rank owns 256/16 = **16 experts whole** → dense grouped-GEMM
  (fused MC2) + `multistream_overlap_shared_expert` hides the all-to-all latency.
- A3-specific env: `ASCEND_A3_ENABLE=1`, `VLLM_ASCEND_ENABLE_FUSED_MC2=1`, `HCCL_BUFFSIZE=1024`.

## Throughput (one A3 serve = 16 devices)

| Config | client concurrency | seqs/DP-replica | rows/s | gen tok/s | quality |
|---|---|---|---|---|---|
| eager (baseline) ‡ | 64 | 32 | ~0.33 | ~100–200 | clean |
| **graph FULL_DECODE_ONLY** | **64** | **32** | **1.15** | **657** | **clean (0 errors / 300)** |

- **graph run (measured):** `300 ok, 0 errors | 171,946 gen tokens in 261.7 s | 657.1 gen tok/s |
  1.15 samples/s` (avg ~573 gen tokens/row on this shard).
- **‡ eager is approximate** — measured on a partial, under-saturated run (~25 of 64 seqs in flight),
  so the eager rows/s is a rough floor and its tok/s (server `/metrics` two-sample ≈ 101) understates a
  saturated eager. The eager→graph lift on A3 is ≈ **3.5×** (0.33 → 1.15 rows/s); smaller multiple than
  A2's 6× only because A2 eager (0.10) was more launch-overhead-bound.
- **Metric note:** compare **rows/s** to the A2 doc (both are `response_regeneration/script.py`, same
  params) — it is the like-for-like number. `gen tok/s` is length-invariant and also A3-favorable, but
  A2 never reported it (derive A2 ≈ 0.74 × ~573 ≈ 420 tok/s for a cross-check → A3 657 ≈ 1.55×).

## Concurrency ceiling — same KV-overflow garbage rule as A2

The bf16-weights-starve-KV garbage bug carries over unchanged: **client concurrency ≤ 64
(≤ 32 seqs / DP-replica)**. Over-sending returns HTTP 200 with `errors=0` but corrupts output
(KV-overflow → preempt/recompute corruption). The 300-prompt run above was clean (0 errors); still
**gsm8k-score any new serve config before trusting a rollout** — `errors=0` is not a quality signal.
(EP16 does not change the KV budget: bf16 weights still take ~37 GB/card regardless of EP-vs-TP expert
sharding; the flag only picks the routing op.)

## Full-run estimate — open-perfectblend (1,420,905 prompts)

One A3 = one serve (16 devices), embarrassingly parallel across `--resume`-safe shards.

```
per-serve throughput = 1.15 rows/s (measured; conservative — small-run floor, steady-state higher)
full-run (S A3 nodes) = 1,420,905 / (1.15 × S) seconds ≈ 14.3 / S days
```

| A3 nodes | Full 1.42 M rollout | vs 2×A2 (same device count) |
|---|---|---|
| 1 (16 dev) | **~14.3 days** | 2×A2 (16 dev): ~30 days → **A3 ~2.1× faster** |
| 4 | ~3.6 days | |
| N | **≈ 14.3 / N days** | |

Per **physical machine** the gap is larger still: bf16 on A2 needs **2 boxes per serve** at 0.27 rows/s
per box, vs **1 A3 box** at 1.15 rows/s — ~4.3× the rollout throughput per machine.

## Caveats

- 1.15 rows/s is a **300-prompt** measurement (ramp-up + drain overhead); a large rollout's steady-state
  is higher — treat 1.15 as a floor, mirroring how the A2 live rollout (0.74) beat its 0.54 estimate.
- eager numbers are **approximate** (partial, under-saturated) — graph is the operating config, so eager
  wasn't fully benchmarked.
- Same as A2: `errors=0` is **not** a quality signal; spot-check text + gsm8k-score new serve configs.
- Rows/s across different shards is length-sensitive; both A3 and A2 used open-perfectblend so the
  distributions match, and the tok/s cross-check agrees — but for a strict comparison use tok/s.
