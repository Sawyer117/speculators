## Purpose

Follow-up to #952, split into two designs as @shanjiaz asked. This is the model
definition; the companion design (expert-parallel training) comes separately, and the two
are independent of each other.

Add a `dsv4_dspark` speculator: the DSpark algorithm (block drafting, a Markov transition
head, a confidence head) on a draft whose backbone mirrors DeepSeek-V4-Flash's decoder
layer. Three layers, ~21B total parameters, ~1.5B active.

One new directory, **2039 lines across 11 files**, plus 354 lines of tests. Zero deletions.

```
models/dsv4_dspark/
  __init__.py   19   config.py   130   core.py   789   checkpoint_mapping.py  242
  backbone/
    __init__.py 16   attention.py 171  moe.py    322   block.py   70
    hyper.py   119   rotary.py    112  norm.py    49
```

Three shared files, 34 lines between them, no behaviour change for any other model: three
lines of registration in `models/__init__.py`; two flags in `train/config/schema.py` that
only this model reads (`DraftArgs` already carries several of those); and in
`train/checkpointer.py` an optional hook so a model can translate its own on-disk layout
when a run resumes — a model that does not define it gets back the object it was passed,
which is a test rather than a claim.

**Why it needs its own model definition.** The draft is close to the target but not the
same, and the differences are in the attention: a per-head sink in the softmax denominator,
which no fused path takes, and a block drafter's access pattern — gamma-wide non-causal
queries over a sliding window of target context, no KV cache. Neither is reachable by
configuring an existing module.

**Not in this proposal.** Our tree carries four more files under `backbone/` — the
expert-parallel dispatch, a grouped GEMM, a kernel registry and a `torch.compile` wrapper.
None are here, and neither are the dispatch points: every heavy op calls its reference
implementation directly, so there is no plugin seam in this diff and no vendor call in it.
The dispatch belongs to the companion design; the rest is an accelerator story we would
rather bring separately.

**Three environment switches**, two of which we are happy to move: `DSPARK_RECOMPUTE`
(activation checkpointing per draft layer), `DSPARK_MOE_BALANCE_RATE` (the `noaux_tc` bias
update — the rate is also the switch), and `DSPARK_LOG_EXPERT_LOAD` (expert-utilisation
counters). The counters belong to the model, since routing collapse does not show in the
loss curve. The other two are recipe features that happen to live here — their own PR, or
config fields instead of environment reads, whichever you prefer. Either way, the numbers
below come from runs with both enabled.

## `--freeze-experts`

@zihanlin-ai's suggestion, and it removes the dependency on the expert-parallel design:
routed experts get `requires_grad=False` and drop out of the optimizer, so nothing is left
to shard and the two proposals can be reviewed in either order.

Two caveats. The memory does not go away, it stops being sharded — ~42 GB of frozen bf16
experts resident per rank, an 80 GB-class recipe. And we have not trained this way: every
number below is full-expert EP training, and we are not attaching an acceptance-length
claim to a regime we have not measured.

## `--init-from-target`

`--init-from-target attn moe hc norm` (or `all`) warm-starts those parts of each draft
layer from the matching verifier layer. Off by default: @zihanlin-ai found MTP lineage in
the released weights only in Flash-family layer 0, and we found the released draft's router
`gate.bias` balanced only in layer 0 and untouched elsewhere, yet scoring 4.42.

The router is never warm-started, only its experts. The verifier's `gate.weight` was fitted
to a much wider hidden distribution and concentrates the draft's routing on a few experts;
its balance bias solves for the verifier's own load. That is an argument, not a measurement
— we have not A/B'd inheriting the router — so it is stated here rather than left as a flag
whose default we could not defend.

@zihanlin-ai also asked for per-layer **and per-slot** expert-utilisation counters. This has
per-layer: used/dead counts, normalized entropy, effective expert count, and the hot-expert
set across steps, so a fixed collapsed subset can be told from a rotating one. Per-slot is
not there — the counters aggregate over a forward. Say the word and we will add it. Their
remaining points bear on expert-parallel training and are answered in the companion design.

## Performance

The expert compute is the reference implementation: correct, portable, not fast. We are not
quoting a speedup, because we have not benchmarked a fused grouped GEMM against it.

## Checkpoint layout — your call, not ours

Two conventions exist for a DSpark draft and they disagree:

| loader | reads |
|---|---|
| `vllm/models/deepseek_v4/nvidia/dspark.py` | `mtp.{0,1,2}.*` from the target checkpoint |
| `vllm_ascend/models/deepseek_v4/dspark.py` | `mtp.{i}.*` |
| `vllm/model_executor/models/qwen3_dspark.py` | speculators-native (`layers.*`, `d2t`/`t2d`) |

The split is by provenance, not hardware: drafts DeepSeek releases live in the target
checkpoint's `mtp.*` namespace, drafts speculators trains use this library's layout. We are
the first speculators-trained DSV4 draft, so the two have not collided before.

| | today | cost |
|---|---|---|
| **A** emit `mtp.*` | loads on GPU and Ascend unchanged | a speculators checkpoint that does not look like one |
| **B** emit speculators-native | nothing serves it | a loader branch in both vLLM and vllm-ascend |
| **C** emit native, ship an exporter | one documented step to serve | the step exists until B lands |

**We lean to A**: speculators and vLLM are one project, so a checkpoint this library trains
should be one the sibling engine can load. B does not exist, and C leaves the checkpoint
just as unloadable with an export step in front of it.

A is implemented as registered `WeightRenaming` / `WeightConverter` rules, the way
`transformers` handles Mixtral — no save override. It is also what keeps serving out of
scope: DeepSeek ships the draft *inside* the target checkpoint, 4,705 `mtp.*` tensors
alongside 67,606 target ones under a `config.json` that already carries the `dspark_*`
fields. A draft trained here writes exactly those 4,705, so nothing has to describe it.

Against A, and worth saying before someone finds it: the prefix is misleading and upstream
is already dealing with it. vLLM PR #52165 (open, closing issue #52111) puts it as
*"DeepSeek-V4-Flash-0731 and DeepSeek-V4-Pro-0813 do not have MTP heads — their `mtp.*`
tensors are DSpark drafters. vLLM routes them to `DeepSeekV4MTPModel` anyway and dies deep
in the weight loader."* If you would rather we emit the native layout and write the loader
changes as follow-up, we will.

One thing worth passing on either way: a round trip is not a sufficient check. A rule and
its own inverse agree whatever they emit, so a mapping can round-trip perfectly and still
write a layout no loader recognises. Ours did, until the key set was compared against the
release.

And the line we would draw: this proposal guarantees the checkpoint, not the engine. A run
that changes the recipe saves and reloads losslessly, and whatever the released layout has
no slot for keeps its module name rather than being forced into `mtp.*` under an invented
one. Whether an engine understands a recipe it has never seen is that engine's concern.
(Relatedly: `algos.py::update_dspark` falls back to `Qwen3DSparkModel`, so a
speculators-format DSV4 checkpoint reaches the wrong architecture. Choosing A is what keeps
this proposal independent of that.)

## Evidence

775,965 rows self-distilled from the target at temperature 0; 8 Ascend NPUs (EP8 + FSDP2),
hidden states supplied online by a live serving instance of the target. Scored on the same
stack, greedy, `num_speculative_tokens=5`:

| | gsm8k | math500 | humaneval | mbpp | mt-bench | 5-set mean | non-chat 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| released DeepSeek draft | 4.665 | 4.639 | 4.939 | 4.526 | 3.347 | **4.4232** | 4.6922 |
| this draft, 5 epochs | 4.845 | 4.565 | 4.971 | 4.546 | 3.154 | **4.4162** | **4.7317** |
| | | | | | | 99.84% | 100.84% |

Ahead on the four non-chat datasets; the remaining gap is multi-turn chat, which we read as
training-data coverage rather than architecture. The run is fully annealed and the last
three checkpoints span 0.014 on the five-set mean.

## Open questions

1. **What would you want to see before taking ~2000 lines in-tree?** We commit to
   maintaining it, keeping it working as the training stack moves, and testing it on the
   hardware we have. If there is a bar for a model this size — a test, a doc, a
   second-platform run — tell us and we will meet it.
2. **Is a reference expert implementation acceptable in-tree**, portable and correct but
   not fast? We think yes for a first landing, with performance as a separate conversation.
3. **Should `--freeze-experts` be the documented default** for anyone reproducing this
   without expert-parallel hardware, given we have not measured its quality?

## Tests

```
$ pytest tests/unit -q
797 passed, 5 skipped in 3:23
```

`make quality` passes as well, against the `ruff` and `mypy` versions `pyproject.toml`
pins.

Eleven of those tests are new and run on a CPU-sized model in about a second:

| test | what it pins |
|---|---|
| `test_saved_keys_match_the_released_draft` | the emitted key set equals the released draft's, exactly |
| `test_round_trip_is_bit_identical` | save then load returns the same tensors |
| `test_experts_are_stacked_in_the_module_and_per_expert_on_disk` | one stacked parameter in memory, one tensor per expert on disk |
| `test_resume_reads_back_the_layout_it_wrote` | the single-device resume path |
| `test_the_distributed_resume_path_loads_the_same_weights` | the same, through `set_model_state_dict` |
| `test_a_released_checkpoint_that_covers_nothing_is_refused` | `strict=False` cannot hide a total mismatch |
| `test_module_layout_checkpoints_still_load` | checkpoints written before the mapping existed |
| `test_the_checkpointer_hook_is_a_no_op_for_other_models` | the shared hook, on a Qwen3 DSpark draft |
| `test_freeze_routed_experts_leaves_the_rest_trainable` | `--freeze-experts` |
| `test_init_from_target_selects_parts` | `--init-from-target`, including `all` |
| `test_init_from_target_rejects_an_unknown_part` | an unknown part is refused, not ignored |

## Checklist

I have filled in:

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan/results, such as providing test command and pasting the results.
- [ ] (Optional) The necessary documentation update.
- [ ] I (a human) have written or reviewed the code in this pr to the best of my ability.
