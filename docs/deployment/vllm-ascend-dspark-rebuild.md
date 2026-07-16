# Rebuild vLLM-ascend DSpark: old fork (60365071) → rewrite (PR #12005)

**Why.** Our serve runs `Sawyer117/vllm-ascend @ dspark-dsv4 = 60365071`, which carries the
**pre-rewrite** DSpark (DSpark piggybacking on the DFlash proposer, `self.method="dflash"`).
PRs **#12004** (draft model) + **#12005** (eager spec-decode) + `a586569e` (noncausal DSA
attention) rewrote DSpark into a **dedicated** path. On this old build a **known-good released
draft scores accept_len 1.34** (should be ~3.94) — same collapse for our draft. The rewrite
deliberately separates DSpark from DFlash (its own query-length, `blk = 1 + num_spec`), **removes
the `num_spec % dspark_block_size` constraint**, and **removes** `VLLM_ASCEND_DSPARK_USE_STANDARD_DSA`
(the TND-NaN/PA_ND dilemma). Expectation: released draft on the new build → ~3.9.

**PR relationship:** `pr-12005 = pr-12004 + one commit`. Checking out **#12005 gets all three**
rewrite commits (`a586569e` noncausal DSA, `56f210c9` draft model, `431a64b1` eager decoding).
#11431 is a *separate* refactor of the same #11196 — **not needed** (12005 already self-contains the
noncausal DSA).

## What we DON'T need to port
All 15 of our fork's commits over its old base are the **old DSpark** (`07b2167e` + fixes, incl.
`db343667` = the `n_predict=dspark_block_size` patch that caused the ÷5 constraint). **Every one is
superseded by the rewrite — cherry-pick none.** The lone non-DSpark commit `60365071`
(w8a8 `o_proj` plain-matmul) is **w8a8-only → irrelevant for our bf16 serve.**

## Compatibility (de-risked)
- `requirements.txt` pins **`torch==2.10.0` / `torch-npu==2.10.0`** — **identical to the box**. No torch bump.
- Method now accepts **`"dspark"` (official) or `"mtp"`** (`spec_decode/__init__.py:47`).
- num_spec constraint gone → use official **7** or **5** freely.
- **One real cost:** the rewrite changes **83 csrc files** (`torch_binding.cpp/meta`, kernels) →
  **the CANN custom ops MUST be recompiled**, not just a python reinstall.

## Steps (in a branch + separate worktree/env — non-destructive, keep 60365071 for rollback)

```bash
# 0) in a00652497's vllm-ascend checkout that BUILT 60365071
cd /home/a00652497/dspark_2026/installation/vllm-ascend-v4     # (the source tree)

# 1) fetch the rewrite as a branch (from upstream vllm-project, or your fork mirror)
git fetch origin pull/12005/head:dspark-dsv4-v2

# 2) build in a SEPARATE worktree so the current serve tree/.so stay intact
git worktree add ../vllm-ascend-v2 dspark-dsv4-v2
cd ../vllm-ascend-v2

# 3) separate env so you can A/B against the working one
conda create -n dspark-dsv4-v2 --clone dspark-dsv4-base    # keep -base intact
conda activate dspark-dsv4-v2

# 4) RECOMPILE the CANN custom ops (this is the real work — 83 csrc files changed),
#    using the SAME toolchain/recipe a00652497 used for 60365071 (CANN 9.0.0 / torch_npu 2.10):
bash csrc/build.sh                    # + build_aclnn.sh / build_batch_invariant_ops.sh if your original build ran them
pip install -e . --no-build-isolation

# 5) sanity
python -c "import vllm_ascend, vllm; print('ok | vllm', vllm.__version__)"
```

> **Fallback if the ~250-commit main jump breaks the build:** cherry-pick just the 3 rewrite commits
> (`a586569e`, `56f210c9`, `431a64b1`) onto our current base. Higher conflict risk (they depend on
> newer-main infra: the `copy_and_expand_dflash_and_dspark_inputs_*` triton util, `is_deepseek_v4_dspark_config`,
> the new dsa framework) — try the clean #12005 build first.

## Serve on the new build (minimal changes to `serve_dsv4_bf16_dualnode.sh`)
- **method:** leave `"mtp"` (still routed) or switch to `"dspark"` (official).
- **num_spec:** constraint gone → try **7** (official) and **5** (apples-to-apples vs our 1.34 baseline).
- **`VLLM_ASCEND_DSPARK_USE_STANDARD_DSA`:** now a **no-op** (flag removed) — harmless to leave, or drop it.
- Everything else unchanged: bf16 target, `dspark_target_layer_ids=[40,41,42]`, released draft dir,
  `draft_sample_method=greedy`.
- Point env at the new env/build; run on **both** 115/116.

## Validate
Released draft (`/share/.../released_draft_fp8_standalone`) → `run_dspark_eval.sh` gsm8k 200.
- **~3.9** ⇒ the rewrite fixed it → then serve OUR draft and measure it fairly.
- still **~1.3** ⇒ deeper than the proposer rewrite (unlikely given the evidence) → back to the aux/attention audit in `HANDOFF_dspark_accept_collapse_2026-07-16.md`.

## Risks
- OPEN/WIP PR (#12005 unmerged) — may have rough edges.
- 250-commit main jump — unrelated API/behavior drift possible; that's why worktree + separate env.
- csrc recompile needs the CANN build env (the box has it — it built 60365071).
