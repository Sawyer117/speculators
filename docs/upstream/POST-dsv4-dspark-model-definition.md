# Design: DeepSeek-V4-Flash DSpark draft model

*Follow-up to #952, split as requested. Companion design: expert-parallel training.*

## Summary

Add a `dsv4_dspark` speculator: the DSpark algorithm (block drafting, a Markov transition
head, a confidence head) on a draft whose backbone mirrors DeepSeek-V4-Flash's decoder layer.

One new directory, **2039 lines across 11 files**, plus 354 lines of tests. Zero
deletions anywhere.

```
models/dsv4_dspark/
  __init__.py   19   config.py   130   core.py   789   checkpoint_mapping.py  242
  backbone/
    __init__.py 16   attention.py 171  moe.py    322   block.py   70
    hyper.py   119   rotary.py    112  norm.py    49
```

Three shared files are touched, by 34 lines between them and with no behaviour change
for any other model: three lines of registration in `models/__init__.py`, one flag in
`train/config/schema.py`, and in `train/checkpointer.py` an optional hook that lets a
model translate its own on-disk layout when a run resumes. A model that does not define
the hook gets back the object it was passed — that is a test, not a claim.

**Deliberately not in this proposal.** Our tree carries four more files under `backbone/`
— expert-parallel dispatch, a grouped GEMM over the stacked expert weights, a kernel
registry, and a `torch.compile` wrapper for the expert path — and none of them are here.
Not just the files: the dispatch points are gone too, so each heavy op calls its reference
implementation directly rather than looking one up. There is no plugin seam in this diff to
review, and no vendor call anywhere in it.

The expert-parallel dispatch belongs to the companion design. The rest is an accelerator
story with its own API surface and its own precedent question; folding it in here would be
a second proposal riding along inside the first. We would rather land the model on its own
merits and bring that up separately, if there is interest at all.

**Three environment switches, and two of them we are happy to move.** The model reads
`DSPARK_RECOMPUTE` (activation checkpointing per draft layer), `DSPARK_MOE_BALANCE_RATE`
(the aux-loss-free `noaux_tc` bias update — the rate is also the switch), and
`DSPARK_LOG_EXPERT_LOAD` (the expert-utilisation counters). The counters we would keep
here, since routing collapse is invisible in the loss curve and that is a property of this
model. The other two are training-recipe features that happen to live in the model file —
if you would rather see them as their own PR, or promoted to config fields rather than
environment reads, we will do either; tell us which you prefer. One thing to know either
way: the acceptance numbers below come from runs with both of them enabled.

## It no longer depends on the expert-parallel proposal

An earlier version of this said the model could not be trained upstream without expert
parallelism, and offered that as an honest limitation. @zihanlin-ai pointed out that a
first-class **experts-frozen** switch removes the dependency. They are right, and it is a
better design than the one we brought, so it is now part of this proposal:

```
--freeze-experts     routed experts require_grad=False and are excluded from the optimizer
                     => nothing to shard, no all-to-all, no new FSDP hooks
```

With the routed experts read-only, every rank holds the same weights and ordinary FSDP over
the trainable remainder (attention, mHC, Markov head, confidence head, projections) is enough.
The two designs can now be reviewed in either order, and "take the model, decide about EP
later" is a coherent choice rather than a broken one.

Two things we should be straight about:

- **The memory does not disappear, it stops being sharded.** ~21B parameters of frozen bf16
  experts is ~42 GB resident per rank. This is an 80 GB-class recipe.
- **We have not trained this way.** Every number below comes from full-expert EP training. We
  are not going to attach an acceptance-length claim to a regime we have not measured.

## Why this needs its own model definition

Nothing in the ecosystem fits the draft's attention. The per-head sink lives in the
softmax denominator, so no fused path takes it, and the access pattern is a block
drafter's — gamma-wide non-causal queries over a sliding window of target context, no KV
cache — rather than a decoder's. HF's DeepSeek-V4 module answers the other question.

The rest follows from the target: MLA with q/o dual LoRA, hyper-connections instead of a
residual stream, 256 routed + 1 shared expert per layer, full 129,280 vocabulary. We did
not try smaller — the released draft has this shape, and matching its acceptance length
was the bar.

Three layers, ~21B total parameters, ~1.5B active per token.

## Two more of @zihanlin-ai's model-side points

**Expert-utilization counters.** Asked for in the model definition from day one, because
routing collapse is invisible in the loss curve. We have them — per-layer used/dead counts,
normalized entropy, effective expert count, and the hot-expert set across steps so a *fixed*
collapsed subset can be told apart from a rotating one — and this proposal makes them a
documented feature rather than a debug flag. Their 13–22 of 256 matches what we measured on a
different verifier family (~14 effective of 256), which is worth more than either number
alone.

**Initialization.** `--init-from-target attn moe hc norm` (or `all`) stays opt-in beside a
from-scratch default. They audited the released weights and found MTP lineage only in
Flash-family layer 0; we found the released draft's router `gate.bias` balanced only in
layer 0 and untouched elsewhere, yet scoring 4.42. Neither collapse nor MTP inheritance is
the binding constraint, so neither should be the default.

One asymmetry inside it, in case you disagree: the MoE router is never warm-started, only
its experts. The verifier's `gate.weight` was fitted to a much wider hidden distribution
and concentrates the draft's routing on a few experts, and its balance bias solves for the
verifier's own load. That is an argument rather than a measurement — we have not A/B'd
inheriting the router — so it is stated here rather than left as a flag whose default we
could not defend.

Their other three points bear on expert-parallel training and are answered in the companion
design.

## Performance

The expert compute here is the reference implementation: correct, portable, not fast. We
are not quoting a speedup, because we have not benchmarked a fused grouped GEMM against
it — the one clean number we have is for `torch.compile` over the expert path, which is
a different thing.

If the project ever wants an in-tree way for accelerators to plug in, we have working
code and would bring it as its own proposal.

## Checkpoint layout — a question for you, not a decision we should make

Two conventions exist for a DSpark draft, and they disagree. We are the first
speculators-trained DSV4 draft, so this combination has not come up before.

| loader | reads |
|---|---|
| `vllm/models/deepseek_v4/nvidia/dspark.py` | `mtp.{0,1,2}.*` from the target checkpoint |
| `vllm_ascend/models/deepseek_v4/dspark.py` | `mtp.{i}.*` |
| `vllm/model_executor/models/qwen3_dspark.py` | speculators-native (`layers.*`, `d2t`/`t2d`) |

The split is not by hardware — GPU and Ascend agree for DSV4. It is by **provenance**: drafts
released by DeepSeek live in the target checkpoint's `mtp.*` namespace, and drafts trained by
speculators (Qwen3, Gemma4) use this library's own layout.

The `mtp.*` prefix is a release artifact rather than a statement about the algorithm, and it
is already causing real trouble upstream. vLLM #52165, open now, opens with: *"DeepSeek-V4-Flash-0731
and DeepSeek-V4-Pro-0813 do not have MTP heads — their `mtp.*` tensors are DSpark drafters.
vLLM routes them to `DeepSeekV4MTPModel` anyway and dies deep in the weight loader."*

**Three ways to go:**

| | today | cost |
|---|---|---|
| **A** emit `mtp.*` | loads on GPU and Ascend unchanged | a speculators checkpoint that does not look like one; the misleading prefix spreads |
| **B** emit speculators-native | nothing serves it | needs a loader branch in both vLLM and vllm-ascend |
| **C** emit native, ship an exporter | one documented step to serve | the export step exists until B lands |

**We lean to A**, and the reason is that speculators and vLLM are one project. A checkpoint
this library trains should be one the sibling inference engine can load; if it is not, that is
a defect in the pair, whichever layout is tidier. B does not exist yet, and C does not
actually fix that — under C the native checkpoint is just as unloadable, and the export step
is a tax every user pays until B lands, which it may never do.

So: emit `mtp.*`, and a draft trained here is a drop-in replacement for the released DeepSeek
draft on both GPU and Ascend, with nothing to convert and nothing to explain.

A is implemented as registered `WeightRenaming` / `WeightConverter` rules, the way
`transformers` handles Mixtral — no save override. Three tests, on a CPU-sized model: the
emitted key set equals the released draft's weight index, tensors survive a save/load
round trip bit-identically, and checkpoints written before the mapping still load.

One thing worth passing on: a round trip is not a sufficient check. A rule and its own
inverse agree whatever they emit, so a mapping can round-trip perfectly and still write a
layout no loader recognises — ours did, until the key set was compared against the
release. Whichever layout you choose, the test that has standing compares to the target
convention, not to itself.

A is also what keeps serving out of scope. DeepSeek ships the DSV4 draft *inside* the target checkpoint — one directory,
67,606 target tensors and 4,705 `mtp.*` draft tensors under one `config.json` that already
carries the `dspark_*` fields. A draft trained here writes exactly those 4,705 tensors, so
it is a drop-in for the ones already there and nothing has to describe it: the target's own
config still applies. Under B the checkpoint would instead need vLLM to grow a way to route
and configure it, and that work would belong to this proposal.

The line we would draw, since it is easier to agree on now than later: what this proposal
guarantees is the checkpoint, not the engine. A run that changes the recipe — a block
convolution over the draft positions, a selection head, a different block width — saves and
reloads losslessly, and whatever the released layout has no slot for keeps its module name
instead of being forced into `mtp.*` under an invented one. Whether a given inference engine
understands a recipe it has never seen is that engine's concern.

We would still rather you chose than we did, because it is a convention question about your
library. And one thing about A deserves saying out loud rather than being discovered later:
it propagates a prefix that upstream is currently fighting (#52165 above), into checkpoints
that are not MTP at all. If you would rather we emit the native layout and take the loader
changes as follow-up work, we will do that instead and are happy to write them.

One boundary, stated so it is not mistaken for an omission. Serving a *speculators-format*
DSV4 checkpoint is a separate matter: `algos.py::update_dspark` falls back to
`Qwen3DSparkModel`, so such a checkpoint reaches the wrong architecture. That is a vLLM-side
concern and not part of this proposal — and choosing A is precisely what means nothing here
depends on it. We are raising it because a reviewer will notice, not because this needs it.

## Evidence

Trained on 775,965 rows self-distilled from the target at temperature 0, on 8 Ascend NPUs
(EP8 + FSDP2), with hidden states supplied online by a live serving instance of the target.

Measured on the same serving stack, greedy, `num_speculative_tokens=5`:

| | gsm8k | math500 | humaneval | mbpp | mt-bench | 5-set mean | non-chat 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| released DeepSeek draft | 4.665 | 4.639 | 4.939 | 4.526 | 3.347 | **4.4232** | 4.6922 |
| this draft, 5 epochs | 4.845 | 4.565 | 4.971 | 4.546 | 3.154 | **4.4162** | **4.7317** |
| | | | | | | 99.84% | 100.84% |

Ahead of the released draft on the four non-chat datasets; essentially all of the remaining
gap is multi-turn chat, which we read as training-data coverage rather than architecture.

The run is fully annealed (cosine to zero at 5 epochs) and the last three checkpoints span
0.014 on the five-set mean.

## Open questions

1. **What would you want to see before taking ~2000 lines in-tree?** It is a new
   directory plus thirty-four lines in three shared files, and it reaches the released
   DeepSeek draft's
   acceptance length on the same serving stack — but it is still code someone has to own.
   What we commit to: maintaining it, keeping it working as the training stack moves, and
   testing it on the hardware we have. If there is a bar for a model this size — a test, a
   doc, a second-platform run — tell us what it is and we will meet it.
2. **Is the reference expert implementation acceptable in-tree**, given it is portable and
   correct but not fast? We think yes for a first landing, and that performance is a separate
   conversation, but it is your call whether a slow path is worth having.
3. **Whether `--freeze-experts` should be the documented default** for anyone reproducing
   this without expert-parallel hardware, given we have not measured its quality.
