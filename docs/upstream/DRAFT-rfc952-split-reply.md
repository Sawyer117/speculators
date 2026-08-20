# DRAFT — reply to RFC #952 (NOT SENT; for review)

> 状态:草稿。按 [[no-community-push]],由用户本人决定发不发、何时发。
> 依据:worklog §10(侵入性预算)、§11(上游现状)。数字均为对**真上游** `vllm-project/speculators`
> `main` 取 merge-base(2026-07-17)后的净改动,已排除 `examples/` `docs/`。

---

@shanjiaz Thanks — splitting it up makes sense. Before proposing designs I measured the
actual footprint against upstream `main`, since "how intrusive" should be a number rather
than an adjective. Excluding `examples/` and `docs/`:

| area | files | +/- | note |
|---|---:|---:|---|
| `models/dsv4_dspark/` (new dir) | 15 | +2622 / **-0** | pure addition |
| `train/` | 6 | +510 / -85 | the only place with real deletions |
| other shared files | 6 | **+98 / -3** | see below |
| `models/dspark/` | 4 | +93 / -16 | |

The six shared-file touches in full: `models/metrics.py` +53, `data_generation/preprocessing.py`
+21, `model.py` +12, `models/utils.py` +5, `models/dflash/core.py` +4, `models/__init__.py` +3
(registration).

**The part I think is worth knowing up front:** inside `train/`, the code that is specific to
DeepSeek-V4 rather than to expert parallelism is **three lines**, and both sites are already
duck-typed — neither imports our model:

```python
distributed.py:250   if type(module).__name__ != "GroupedExperts":
trainer.py:341-342   _ep_local = getattr(self.model, "ep_local_param_keys", None)
                     _expert_keys = set(_ep_local()) if callable(_ep_local) else set()
```

So the expert-parallel work is essentially model-agnostic already. What a design would add is
a proper contract for those two hooks (an expert-module protocol, or an explicit predicate)
instead of a class-name string.

### Proposed split — three, not two

Measuring turned up a third group: ~100 lines in `train/` that relate to neither DSV4 nor EP.
Keeping them out of both designs keeps each one small.

1. **Generic training-loop improvements** — `--no-validation` (train on the full split instead
   of silently discarding the held-out slice), a plain-text log mirror (piping through `tee`
   makes stdout a non-tty and strips rich's colors and progress bar), and per-step device
   memory stats. Independent of everything else here; happy to send this as an ordinary PR
   rather than an RFC.

2. **Expert-parallel training (FSDP2 + EP)** — the general one. Routed experts become
   `Shard(0)` DTensors on the same mesh as the FSDP-sharded rest, so the optimizer, gradient
   clipping and checkpointing need no plain-vs-DTensor special casing; FSDP is told to ignore
   them and the MoE moves tokens with an all-to-all. Worth noting for the checkpoint question:
   this is *why* `DistributedCheckpointer` still consolidates to plain safetensors with no
   changes to it at all.

3. **DSV4-Flash DSpark draft model definition** — the new directory. Every part of it exists to
   keep the draft isomorphic to the target's decoder layer (MLA with q/o dual LoRA and per-head
   sinks, mHC in place of the residual, 256 routed experts + 1 shared); a smaller draft does not
   learn this target. It is a pure addition apart from three lines of registration.

(2) and (3) are separable but not independent: (3) alone cannot be trained upstream without
(2). I would rather state that plainly than pretend either stands alone.

Suggested order: (1) first since it is uncontroversial and unblocks nothing, then (2), then
(3). Happy to take any of it to Slack — whichever is easier for you.
