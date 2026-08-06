# Technical memos

Standalone write-ups that are not part of the DSV4-DSpark pipeline documentation. This branch
(`docs/adaptive-spec-graph-memo`) exists so the decks can be downloaded without touching the working
branch.

## `adaptive-spec-graph-mode-ascend.pptx`

**Adaptive speculative decoding under graph mode — what it costs on Ascend NPU, and where the real
blockers are.** 9 slides. Regenerate with `python3 make_adaptive_spec_memo.py --out <file>.pptx`
(requires `python-pptx`).

Adaptive speculation gives each request its own verification length `K_i`, so the target model only
verifies tokens worth verifying. Device graphs want the opposite: fixed shapes, fixed addresses, a
stable execution path. The memo asks whether the graph updates, workspace queries, tiling, host
metadata and padding needed to reconcile the two cost more than the compute they save.

Three findings, each grounded in source rather than inference:

1. **vLLM PR #48692 is narrower than it is usually described.** It is closed and unmerged, it
   explicitly does *not* allocate the verification budget online (a user-provided
   `num_speculative_tokens_per_batch_size` is used instead), it supports `FLASH_ATTENTION` only, and
   its author states that no significant speedup is measurable — the demonstrated benefit is
   acceptance rate under a fixed budget. It is a reference for how variable length coexists with a
   full graph, not evidence of a speedup.

2. **Device-side query lengths already exist on Ascend — on one of the two operator families.**
   FIA v2 (`npu_fused_infer_attention_score_v2`) takes a host-side int array, and every call site in
   vllm-ascend builds it with `.tolist()`. The DSA / lightning-indexer / SFA operators
   (`torch.ops._C_ascend.npu_vllm_*`) take `torch.Tensor`, and their call sites pass
   `query_start_loc[1:].clone()` with no host round-trip. DeepSeek-V4-Flash runs on the DSA path, so
   for that model the prerequisite everyone worries about is already met.

3. **The FIA workspace cache is keyed by exactly what adaptive `K` changes.** The key is the total
   token count for the round, so a static `K` hits the cache every step while adaptive `K_i` misses
   every step, once per layer. The fix is structural — key by a `(batch bucket, total-verify-token
   bucket)` pair and pre-allocate at capture time — not a micro-optimisation.

The memo closes with the measurements that should precede any implementation work: size the saving
first, then compare static-K full graph against adaptive-K in eager, piecewise and full graph modes,
and audit the `K_i` path for host round-trips.
