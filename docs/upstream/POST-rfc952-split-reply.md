# Reply to #952 — the split, and what changed while doing it

> English version, ready to post. The Chinese draft (`DRAFT-rfc952-split-reply.md`) is kept
> as the earlier state; do not send that one — several of its claims no longer hold.

---

@shanjiaz Splitting it made sense, and doing it changed the answer, so this is worth a
short note before the two designs rather than just links.

**The two designs are now independent.** My earlier framing was that the model could not be
trained upstream without the expert-parallel work — I said I would rather state that plainly
than pretend either stood alone. @zihanlin-ai pointed out that a first-class experts-frozen
switch removes the dependency, and they are right. With the routed experts read-only there is
nothing to shard, ordinary FSDP over the trainable remainder is enough, and the model
proposal now ships `--freeze-experts`. So the pair can be reviewed in either order, and
"take the model, decide about EP later" is a coherent choice rather than a broken one.

That also narrows the second proposal honestly: expert parallelism is what makes *full*
expert training practical, not what makes an MoE draft trainable at all.

**1. [DSV4-Flash DSpark draft model definition](#)** — one new directory, ~2090 lines across
11 files, zero deletions, plus twenty lines in two shared files: three lines of registration
in `models/__init__.py`, and an optional hook in `train/checkpointer.py` that lets a model
translate its own on-disk layout when a run resumes. A model that does not define the hook
gets back the object it was passed; that is a test, not a claim.

**2. [Expert-parallel training for MoE drafts](#)** — three optional model-side hooks and one
`DeviceMesh` parameter on `apply_fully_sharded()`. A model that defines none of them gets
exactly today's behaviour. The two coupling points to the model are currently duck-typed and
neither imports our model:

```python
distributed.py:250   if type(module).__name__ != "GroupedExperts":
trainer.py:341       _ep_local = getattr(self.model, "ep_local_param_keys", None)
```

The mechanism is model-agnostic already; what the design adds is a real contract in place of
a class-name string, and that is the part I would most like an opinion on.

**One decision I would rather you made than I did**, in the model design: which layout a
speculators-trained DSV4 draft should write. DeepSeek ships that draft inside the target
checkpoint under an `mtp.*` namespace with per-expert tensors, which every DSV4 loader on
both GPU and Ascend reads today; this library's own drafts use `layers.*` with stacked
experts. I have implemented the first and explained why, but it is a convention question
about your library and the alternative is laid out beside it.

**A carve-out so it does not muddy either diff.** Our `train/` also carries changes that
belong to neither proposal — a `--no-validation` flag, a plain-text log mirror, per-step
device memory stats, and data-pipeline work for online hidden states. Those stay out of both
and can come separately if they are wanted at all.

@zihanlin-ai — thank you, and not only for the point above. Two of your five observations we
had reached independently from a different verifier family, which is worth more than either
of us having reached them alone; they are answered in the model design. The DDP router
question and fp32 sharded originals (#711) bear on training and are answered in the EP one.

Happy to start with either, or to take it to Slack — whichever is easier.
