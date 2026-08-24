# WITHDRAWN — "fail a resume in which no checkpoint tensor matched a parameter"

> Drafted 2026-08-25 and withdrawn the same day. No PR was opened. The branch
> (`pr/resume-key-mismatch`) and its worktree are deleted, on the fork and locally; the
> change is small enough to rebuild from this page if a real case ever turns up.
>
> Filed under WITHDRAWN, not POST, so nobody picks it up and sends it. Kept as a record
> so the idea is not proposed again without the measurement below.

## What was proposed

Both resume paths in `train/checkpointer.py` load with `strict=False` and discard the
`_IncompatibleKeys` they get back. The draft argued that this makes a wholesale key
mismatch silent, so a resume can quietly train from the initial weights, and proposed
raising when no checkpoint tensor matched a parameter.

## Why it was withdrawn

The motivating scenario — "point `--resume-from-checkpoint` at a directory saved by a
different speculator configuration and nothing raises" — was asserted, not measured. When
measured on stock upstream models it does not hold:

```
eagle3 draft: 29 parameters   dspark draft: 35   shared names: 28
resume an eagle3 checkpoint into a dspark model:
  RuntimeError: size mismatch for lm_head.weight ...
```

Two things upstream already does that the draft did not account for:

- **The names agree.** `fc`, `norm`, `embed_tokens`, `lm_head`, `layers.*` are shared
  across the algorithms in this library. Switching algorithm and reusing `--save-path`
  produces a near-total *match*, not a mismatch.
- **`strict=False` does not cover shapes.** A size mismatch raises from
  `load_state_dict` regardless, so the cases where the names do line up but the tensors
  do not are already loud.

That leaves the raise branch reachable only for a checkpoint whose keys use a different
naming scheme *with compatible shapes*. Nothing upstream produces one. Our DSV4 draft
does, because it writes the released `mtp.*` layout — which is our design decision, so
the guard belongs in our model, which is where it now lives
(`DSV4DSparkDraftModel.state_dict_from_checkpoint`).

## What remains true

The observation that started this — a resume loading 0 of 110 tensors in silence — was
real, and it is fixed. It was our own layout that made it reachable, not an upstream
defect. Proposing it upstream would have been asking the project to carry a guard for a
situation only our checkpoints can create.

If upstream ever renames a parameter, this becomes reachable there too; the guard is
eight lines and can be raised then, with a real case attached.
