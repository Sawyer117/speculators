# DRAFT — Design: expert-parallel training for MoE drafts (NOT SENT)

> 状态:草稿,给用户过目用。按 [[no-community-push]] 不外发。
> 事实依据均已对**真上游** `vllm-project/speculators@main` 核实(merge-base 2026-07-17)。
> 2026-08-24 修订:按 @shanjiaz(#952,拆成两份独立设计)与 @zihanlin-ai(#952)。

---

## Problem

A draft that mirrors an MoE target is itself an MoE. Ours is 3 layers but ~21B total
parameters with ~1.5B active per token: 256 routed experts + 1 shared per layer. Pure FSDP
over that shape all-gathers every expert on every rank on every step, when each token only
needs 8 of them. The training recipe that works is expert parallelism — each rank owns a
disjoint slice of the experts and tokens are routed to them with an all-to-all.

`apply_fully_sharded()` today is 20 lines and hardcodes the wrap granularity:

```python
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

There is no way for a model to say "shard these differently" or "leave these alone".

## What this proposes

Three optional model-side hooks and one new parameter. A model that defines none of them
gets exactly today's behaviour.

| hook | returns | meaning |
|---|---|---|
| `fsdp_wrap_plan()` | `list[nn.Module]` | FSDP unit granularity, children before parents. Default: `list(model.layers)` — today's behaviour. |
| `fsdp_ignored_params()` | `set[Parameter]` | params FSDP must leave alone (the rank-local experts). Default: empty. |
| `ep_local_param_keys()` | `set[str]` | `state_dict` names that are rank-local, so the rank-0 broadcast skips them. Default: empty. |

plus `apply_fully_sharded(model, mesh=None, ...)` — the `DeviceMesh` every `fully_shard`
uses, so expert and non-expert parameters live on one mesh.

None of these exist upstream today (checked: zero occurrences in `src/`).

## The part that makes it cheap: experts are DTensors, not plain tensors

Under EP each rank physically holds only its `[n_local, ...]` slice. The obvious
implementation keeps those as ordinary tensors and excludes them from FSDP. **We built that
first and reverted it**: `clip_grad_norm_` broke, AdamW's `_foreach_mul_` broke, and
checkpointing broke, each needing its own plain-vs-DTensor special case. It was whack-a-mole
and it would not have been upstreamable.

Instead the local slice is wrapped as a `Shard(0)` DTensor **on the same mesh as the
FSDP-sharded rest**. Every parameter in the model is then a uniform DTensor, and:

- the optimizer needs no special casing,
- gradient clipping needs no special casing,
- **checkpointing needs no changes at all** — `DistributedCheckpointer` already consolidates
  with `get_model_state_dict(full_state_dict=True, cpu_offload=True)` and writes ordinary
  safetensors on rank 0. We did not touch `checkpointer.py` and it works. A trained
  EP checkpoint is `config.json` + `model.safetensors`, same as any other run.

The MoE forward reads `.to_local()` and moves tokens with an all-to-all rather than an FSDP
all-gather.

## Mixed precision: one knob

FSDP2's `MixedPrecisionPolicy` upcasts every trainable parameter to an fp32 master. For the
experts that is ~15B parameters of master weights. A `bf16_experts` flag keeps the routed
experts in bf16 (no fp32 master) while everything else keeps its master copy; their gradients
are far less rounding-sensitive than the small tensors'. Default off = today's semantics.

This only matters when a rank must materialise more than its own expert slice; under EP each
rank upcasts `1/EP` of them and the default is fine. We would be happy to drop this from the
proposal if you would rather not carry the knob.

## ★ The design question we would like your opinion on

The two coupling points to the model are currently duck-typed:

```python
distributed.py   if type(module).__name__ != "GroupedExperts":   # a class-name string
trainer.py       _ep_local = getattr(self.model, "ep_local_param_keys", None)
```

Neither imports our model, and the whole thing is model-agnostic in mechanism — but a
class-name string is not a contract. Options we see:

1. **A protocol** — `ExpertParallelModule` with the stacked-weight layout as its interface,
   and `shard_experts_as_dtensor` walks for `isinstance`.
2. **An explicit predicate** — the model hands `shard_experts_as_dtensor` the modules to
   wrap, so `train/` never introspects types at all.
3. **Keep the hooks, drop the string** — `fsdp_ignored_params()` already returns exactly the
   parameters in question; the separate walk could be removed and driven from that set.

(3) is the smallest and we lean toward it, but you have more context on where this lands
relative to other parallelism work. If expert parallelism is already planned in some other
shape, we would rather adapt to it than land a second mechanism.

## Three points raised by @zihanlin-ai (#952)

They raised five observations from training an MoE drafter downstream; two bear on the model
definition and are answered there. These three are ours.

**Two training regimes — and this proposal is only needed for one of them.** They are right,
and it changed the shape of the pair: with routed experts frozen, every rank holds the same
read-only weights, there is nothing to shard, and plain FSDP over the trainable remainder is
enough. The companion model proposal now ships `--freeze-experts` and no longer depends on
this one. **So the question here is narrower than it was**: this is what makes *full* expert
training practical, not what makes an MoE draft trainable at all. If the project would rather
have the model first and decide about EP later, that is now a coherent choice.

**The router dropping out of the autograd graph.** Their case is plain DDP, which hangs
without `find_unused_parameters=True`. We have not hit it, and we think the reason is that
the recipe here is FSDP2 (`fully_shard`), which does not have DDP's all-params-must-be-used
requirement — so we would rather say "not observed on this path" than claim it cannot happen.
The related thing that *is* real under EP: on any given step a rank-local expert can receive
zero tokens and therefore get no gradient at all. That is normal and dropless routing does
not prevent it; what it means is that an optimizer or clipping implementation must tolerate a
`None` gradient on a parameter that is otherwise perfectly live. Keeping the local slice as a
`Shard(0)` DTensor rather than a plain tensor is what makes that a non-event here, because
nothing downstream has a plain-vs-DTensor branch to get it wrong in.

**fp32 sharded originals (#711).** Agreed, and more so under EP than they may realize. With
256 experts and top-8 routing, an individual expert sees roughly 1/32 of the tokens, so its
gradients are correspondingly smaller — exactly the regime where a bf16 master weight
silently swallows the update. #711 having already moved upstream to fp32 originals plus
autocast is the right foundation; this proposal adds one knob on top of it (below) rather
than a second precision story.

## Footprint

Against `main`, excluding `examples/` and `docs/`: `train/` is +510 / −85 across 6 files, and
that number includes things unrelated to this design (a `--no-validation` flag, a log mirror,
memory stats — we will carve those out). The EP mechanism itself is roughly 140 lines across
`distributed.py` and `trainer.py`.

## Evidence

8× Ascend NPU (EP8 + FSDP2), 3 layers × 256 experts, 21B total / 1.5B active, full 129,280
vocabulary, 775,965 self-distilled rows, 5 epochs. No NaNs; expert utilisation reaches
251/255/256 of 256 per layer. The resulting draft scores a 5-dataset mean acceptance length
of **4.4162 vs 4.4232 for the released DeepSeek draft (99.84%)** on the same serving stack,
and **4.7317 vs 4.6922 (100.84%) on the four non-chat datasets** — i.e. ahead everywhere
except multi-turn chat.

We would run whatever additional validation you want, including on a small dense model where
`fsdp_ignored_params()` returns empty and the path must be bit-identical to today.

## What this does NOT include

Model-specific code. This design is only the training mechanism; the DSV4-Flash DSpark draft
that motivated it is a separate proposal, and it cannot be trained upstream without this one.
