# Design: expert-parallel training for MoE drafts

*Follow-up to #952, split as requested. Companion design: the DSV4-Flash DSpark draft model.*

## Summary

Three optional model-side hooks and one new parameter, so a model can say how it wants to be
sharded. A model that defines none of them gets exactly today's behaviour.

This is what makes **full** expert training practical. It is no longer a prerequisite for an
MoE draft to exist upstream — see "Scope narrowed" below.

## Problem

A draft that mirrors an MoE target is itself an MoE. Ours is three layers but ~21B total
parameters with ~1.5B active: 256 routed + 1 shared expert per layer. Pure FSDP over that
shape all-gathers every expert on every rank on every step, when each token needs 8 of them.

`apply_fully_sharded()` today is 20 lines and hardcodes the wrap granularity:

```python
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

There is no way for a model to say "shard these differently" or "leave these alone".

## What this proposes

| hook | returns | meaning |
|---|---|---|
| `fsdp_wrap_plan()` | `list[nn.Module]` | FSDP unit granularity, children before parents. Default `list(model.layers)` — today's behaviour. |
| `fsdp_ignored_params()` | `set[Parameter]` | parameters FSDP must leave alone (the rank-local experts). Default empty. |
| `ep_local_param_keys()` | `set[str]` | `state_dict` names that are rank-local, so the rank-0 broadcast skips them. Default empty. |

plus `apply_fully_sharded(model, mesh=None, ...)` — the `DeviceMesh` every `fully_shard` call
uses, so expert and non-expert parameters live on one mesh.

None of these exist upstream today.

## The decision that makes it cheap: experts are DTensors, not plain tensors

Under EP each rank physically holds only its `[n_local, ...]` slice. The obvious
implementation keeps those as ordinary tensors and excludes them from FSDP. **We built that
first and reverted it**: `clip_grad_norm_` broke, AdamW's `_foreach_mul_` broke, and
checkpointing broke, each needing its own plain-vs-DTensor special case. It was whack-a-mole,
and it would not have been upstreamable.

Instead the local slice is wrapped as a `Shard(0)` DTensor **on the same mesh as the
FSDP-sharded rest**. Every parameter in the model is then a uniform DTensor, and:

- the optimizer needs no special casing,
- gradient clipping needs no special casing,
- **checkpointing needs no changes at all** — `DistributedCheckpointer` already consolidates
  with `get_model_state_dict(full_state_dict=True, cpu_offload=True)` and writes ordinary
  safetensors on rank 0. We did not touch `checkpointer.py`. An EP-trained checkpoint is
  `config.json` + `model.safetensors`, the same as any other run.

The MoE forward reads `.to_local()` and moves tokens with an all-to-all rather than an FSDP
all-gather.

## Scope narrowed, thanks to @zihanlin-ai

They observed that there are two training regimes, and that only one of them needs a sharding
story: with routed experts frozen, every rank holds the same read-only weights, there is
nothing to shard, and plain FSDP over the trainable remainder is enough. The companion model
proposal now ships `--freeze-experts` and no longer depends on this design.

That makes the question here narrower and, we think, easier to answer: this is what makes
*full* expert training practical, not what makes an MoE draft trainable at all.

## Two more of their points

**The router dropping out of the autograd graph.** Their case is plain DDP, which hangs
without `find_unused_parameters=True`. We have not hit it, and we believe the reason is that
this recipe is FSDP2 (`fully_shard`), which has no all-parameters-must-be-used requirement —
so we would rather report "not observed on this path" than claim it cannot happen.

The related thing that *is* real under EP: on a given step a rank-local expert can receive
zero tokens and get no gradient. That is normal, and dropless routing does not prevent it.
What it means is that the optimizer and the clipping path must tolerate a `None` gradient on
an otherwise live parameter. Keeping the local slice as a `Shard(0)` DTensor is what makes
that a non-event here — there is no plain-vs-DTensor branch anywhere downstream to get it
wrong in.

**fp32 sharded originals (#711).** Agreed, and it matters more under EP than elsewhere. With
256 experts and top-8 routing an individual expert sees roughly 1/32 of the tokens, so its
gradients are correspondingly smaller — exactly the regime where a bf16 master weight
silently swallows the update. #711 is the right foundation; this adds one knob on top of it
rather than a second precision story.

## Mixed precision: one knob

`--bf16-experts` keeps the expert originals in bf16 instead of fp32. It is a memory/fidelity
trade, off by default, and it exists because on 64 GB-class devices the fp32 originals for
256 experts do not fit alongside everything else. On larger memory it should stay off, for
the reason in the paragraph above.

## Footprint

The change to shared code is small and additive: `apply_fully_sharded()` gains an optional
mesh argument and consults three optional hooks, defaulting to current behaviour when a model
defines none of them. Everything else lives in the model.

## Evidence

An EP8 + FSDP2 run on 8 Ascend NPUs, 256 experts per layer, trained to 5 epochs, producing a
draft at 4.4162 five-dataset mean acceptance length against 4.4232 for the released DeepSeek
draft on the same serving stack. Checkpointing, gradient clipping and optimizer state all
went through the unmodified upstream paths.

## What this does not include

- No pipeline or tensor parallelism.
- No opinion on how EP should compose with Ulysses SP (#871, #298) or with the data-parallel
  scale-out in #599. EP looks like a different axis to us, but if you have a broader picture
  of how parallelism should look in this library, we would much rather build into that than
  land something beside it.
- No vendor kernels. The MoE forward here is plain `torch.distributed` plus a reference GEMM;
  accelerated kernels are a separate, optional concern discussed in the model design.

## Open question

Where should this live? The hooks are model-agnostic and the mechanism is general, but
speculators has no MoE draft today, so there is nothing to shard until the companion model
lands. If you would rather see the model first and revisit this afterwards, that is a
reasonable order and we are happy to follow it.

We also have multi-node Ascend hardware available for testing, which may be useful for the
"needs testing" blocker on #599 — as a second-platform check, not a substitute for GPU
testing.
