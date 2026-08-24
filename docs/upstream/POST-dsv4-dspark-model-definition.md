# Design: DeepSeek-V4-Flash DSpark draft model

*Follow-up to #952, split as requested. Companion design: expert-parallel training.*

## Summary

Add a `dsv4_dspark` speculator: the DSpark algorithm (block drafting, a Markov transition
head, a confidence head) on a draft whose backbone mirrors DeepSeek-V4-Flash's decoder layer.

It is a pure addition — one new directory, **+2622 lines across 15 files, zero deletions**,
plus three lines of registration in `models/__init__.py`. No shared file is modified.

```
models/dsv4_dspark/
  config.py    139     core.py      750     weights.py   234
  backbone/
    attention.py 176   moe.py       268     block.py     102
    hyper.py     117   rotary.py    105     norm.py       48
    kernels.py    95   moe_ep.py    177     moe_grouped_gemm.py 248
    moe_compile.py 120
```

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

## Why the architecture is what it is

The draft's job is to guess what the target will say. Each choice below exists because the
target has it:

| | why |
|---|---|
| MLA with q/o dual LoRA, per-head attention sinks | the target's attention is MLA; a draft with ordinary MHA has to learn a different function of the same hidden states |
| manifold-constrained hyper-connections instead of a plain residual stream | same reason |
| 256 routed + 1 shared expert per layer | the released DeepSeek draft has this shape and reproducing its acceptance length was the bar |
| full 129,280 vocabulary, no draft-vocab reduction | the released draft does not reduce, and a reduced head trains a mapping the target never uses |

Three layers, ~21B total parameters, ~1.5B active per token.

## Expert-utilization counters, shipped as a feature

@zihanlin-ai asked for per-layer / per-slot expert-utilization counters in the model
definition from day one, because routing collapse is invisible in the loss curve. We agree,
we have them, and this proposal promotes them from a debug flag to a documented feature.

They report deep draft slots using 13–22 of 256 experts. We measured the same shape on a
different verifier family: normalized routing entropy falling to ~0.52 in the last layer,
about 14 effective experts of 256. Independent agreement across two verifier families is
worth more than either measurement alone.

What is logged per layer: used/dead expert counts, normalized entropy, effective expert count
(`E^entropy`, which is the honest number — "used" is misleading at a few hundred tokens per
rank per step), and the hot-expert set across steps. That last one matters: single-step
entropy cannot tell a *fixed* collapsed subset apart from a rotating one, and only the first
is a real collapse.

## Initialization: from scratch by default

`--init-layer-from-target` and `--init-{moe,attn,hc,norm}-from-target` initialize draft
layers from chosen verifier layers. They stay **opt-in beside a from-scratch default**.

@zihanlin-ai audited the released V4 DSpark weights and found detectable MTP lineage only in
Flash-family layer 0. We reached the same conclusion from a different direction: the released
draft's router `gate.bias` is balanced only in layer 0 and left untouched elsewhere, yet it
scores 4.42 acceptance length. Neither routing collapse nor MTP inheritance is the binding
constraint, so neither should be the default.

## The boundary we would like your call on

Three files under `backbone/` are parallelism and performance, not modelling:

```
moe_ep.py            177    expert-parallel dispatch / all-to-all
moe_grouped_gemm.py  248    grouped GEMM over the stacked expert weights
moe_compile.py       120    torch.compile wrapper for the expert path
```

Removing them leaves ~2077 lines of pure modelling code, a much easier diff. Our reading of
how this project already handles the question:

| | lines | proposed home |
|---|---:|---|
| `moe_ep.py` (expert-parallel all-to-all) | 177 | in-tree — zero `torch_npu`, plain `torch.distributed` |
| kernel registry + plugin point | ~110 | in-tree — no vendor name appears in it |
| `_grouped_matmul_torch` (reference implementation, and the parity oracle) | ~30 | in-tree |
| the NPU half of `moe_grouped_gemm.py` | ~200 | out-of-tree bridge |
| `moe_compile.py` | 120 | out-of-tree bridge |

The reasoning is your own precedent. #775 ("opt-in `transfer_to_npu`") was closed unmerged
for putting a vendor shim in `src/`; #589 ("selectable attention backend") merged because it
solved the same class of problem — flex attention unavailable on some accelerators — with a
portable option instead. `src/` today has zero direct `torch_npu` calls and zero `is_cuda`
branches, going through `torch.accelerator` throughout. We read that as: portable in, vendor
out.

So the model ships a registry with a pure-torch reference for every heavy op, resolved at
call time, and **is correct with zero accelerated kernels registered**. The reference is also
the parity oracle those kernels are validated against, which means nobody needs the hardware
to work on this code.

One request attached to that: shipping only an interface leaves an accelerator user having to
import a bridge by hand, and an install that forgets is correct but quietly slower with
nothing saying so. A ~15-line `discover_plugins()` scanning a `speculators.kernels` entry
point group makes `pip install` the whole story, with no vendor name in the scanning code and
no behaviour change when no such distribution is installed. Happy to fold it in here or send
it separately, whichever you prefer.

## Serving

`vllm/transformers_utils/configs/speculators/algos.py` already registers `dspark` and already
passes through exactly the fields this model needs — `markov_rank`, `markov_head_type`,
`block_size`, `enable_confidence_head`, `confidence_head_with_markov`. It maps them to
`Qwen3DSparkModel`, a hardcoded architecture.

A checkpoint trained by this proposal is therefore a standard speculators checkpoint that
vLLM *almost* knows how to serve; what is missing is an architecture branch, in the same
shape `eagle3` and `peagle` already have. We currently bridge that with a converter in our
fork. It is a stopgap and a vLLM-side change, so it is out of scope here — mentioned only so
the picture is complete.

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

1. **Does the project want a model this size in-tree at all?** It is a pure addition, but
   2622 lines is a maintenance commitment. A plausible alternative is that DSV4-scale drafts
   live out-of-tree and speculators owns only the mechanisms. We would not argue against it.
2. **The `backbone/` boundary above** — which of those three files, if any, belong in-tree.
3. **Whether `--freeze-experts` should be the documented default** for anyone reproducing
   this without expert-parallel hardware, given we have not measured its quality.
