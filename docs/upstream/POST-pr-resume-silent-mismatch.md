# PR draft — fail a resume in which no checkpoint tensor matched a parameter

> Branch `pr/loud-resume-mismatch` (worktree `/workspace/pr_resume`), one commit on top of
> `upstream/main@2aec948`. **Not pushed, no PR opened.**
> Target: `vllm-project/speculators`. Independent of the DSV4 work — it was found there,
> but nothing in it is DSV4-specific.

---

## Title

`fix(train): fail a resume in which no checkpoint tensor matched a parameter`

## Body

Both resume paths in `train/checkpointer.py` load with `strict=False`:

```python
# SingleGPUCheckpointer.load_model_state_dict
# Note: `strict=False` because we don't load the verifier weights
model.load_state_dict(full_state_dict, strict=False)

# DistributedCheckpointer.load_model_state_dict
set_model_state_dict(model, full_state_dict, options=StateDictOptions(
    full_state_dict=True, broadcast_from_rank0=True, strict=False))
```

That is the right call — a speculator checkpoint deliberately omits the verifier weights,
so missing keys are normal. What it also does is make a *wholesale* mismatch silent. Both
calls return an `_IncompatibleKeys`, and both throw it away.

Point `--resume-from-checkpoint` at a directory saved by a different speculator
configuration and nothing raises, nothing is logged, and the trainer prints:

```
Found checkpoint at <path>.
Resuming training on epoch 3.
```

over a model still at its initial values. The damage is not just a lost resume:
`setup_trainer` sets `current_epoch = previous_epoch + 1`, so a five-epoch run that
"resumes" at epoch 3 now trains two epochs from scratch and stops. It produces a
plausible checkpoint, a plausible loss curve, and nothing anywhere says what happened.

We hit this for a real reason rather than a hypothetical one: our draft writes a
checkpoint whose keys are not its parameter names, and the first resume loaded 0 of 110
tensors without a word (85 missing / 110 unexpected). That part is ours to solve. But the
silence is not — any parameter rename in this library reaches the same place.

### The change

Check the value both paths already return:

```python
incompatible = model.load_state_dict(full_state_dict, strict=False)
check_resume_loaded(incompatible, checkpoint_keys, path)
```

- every checkpoint tensor unexpected ⟹ nothing loaded ⟹ **raise**
- some unexpected ⟹ **warn**, and continue — that is usually a renamed parameter, not the
  wrong checkpoint, and failing the run over it would be worse than saying so
- none unexpected ⟹ silent, as today

Two details worth flagging in review:

- The key count is taken **before** the load. `set_model_state_dict` merges the model's own
  entries into the dict it is handed, so counting afterwards can never equal the number of
  unexpected keys and the check would never fire.
- Every rank loads the same file and reaches the same verdict, so a failure raises on all
  of them rather than leaving some ranks waiting in the `dist.barrier()` below.

### Tests

Three, in `tests/unit/train/test_checkpoint.py`:

| test | on `main` |
|---|---|
| a normal resume loads its weights (verifier keys absent) | passes — a regression guard |
| a checkpoint matching nothing raises | **fails: DID NOT RAISE** |
| a partial match warns and continues | **fails: no warning** |

### Not in this PR

Whether `strict=False` is the right default at all, and whether missing keys deserve the
same treatment. Missing keys are load-bearing here (the verifier weights), so tightening
them needs a way to say which absences are expected — a larger question than this fix.
