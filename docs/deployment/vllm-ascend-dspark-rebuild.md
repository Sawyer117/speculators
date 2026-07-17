# Rebuild vLLM-ascend DSpark: old fork (60365071) → rewrite (PR #12005)

**Why.** Our serve runs `Sawyer117/vllm-ascend @ dspark-dsv4 = 60365071`, which carries the
**pre-rewrite** DSpark (DSpark piggybacking on the DFlash proposer, `self.method="dflash"`).
PRs **#12004** (draft model) + **#12005** (eager spec-decode) + `a586569e` (noncausal DSA
attention) rewrote DSpark into a **dedicated** path. On this old build a **known-good released
draft scores accept_len 1.34** (should be ~3.94) — same collapse for our draft. The rewrite
deliberately separates DSpark from DFlash (its own query-length, `blk = 1 + num_spec`), **removes
the `num_spec % dspark_block_size` constraint**, and **removes** `VLLM_ASCEND_DSPARK_USE_STANDARD_DSA`
(the TND-NaN/PA_ND dilemma). Expectation: released draft on the new build → ~3.9.

**PR relationship (tracking issue #11126) — a 4-PR STACK by @QwertyJack, each depends on the previous:**
`#12003` (attn, noncausal DSA/SAS) → `#12004` (draft model) → `#12005` (eager spec-decode, the
"correctness baseline") → `#12006` (FULL ACLGraph, perf-only). The `pr-12005` head branch already
**folds the whole stack** (its main-based diff contains PRs 1-2), and its extra dep `#11765` (generic
`AscendDsparkProposer`) is **already MERGED into main** — so **checking out `#12005` head is
self-contained** (verified: `dspark_proposer.py`, `deepseek_v4_dspark_proposer.py`,
`deepseek_v4_draft.py` all present). `#12006` (graph) is optional. `#11431` (@drslark) is a **separate,
competing** refactor — **do NOT mix it in**.

**⚠️ STATUS — these are WIP, not finished.** As of 2026-07-16: **#12004/#12005/#12006 are DRAFT**,
#12003 is ready, **none are merged, all `mergeable=False`** (conflicts with a moved main). Human review
is early (mostly bot reviews: gemini-code-assist, github-actions conflict warnings); the heavy human
review is on the original monolith #11196. So: **build a pinned snapshot of #12005 to VALIDATE the fix,
not as a final/production drop — it will change.** Record the exact head commit you build.

## What we DON'T need to port
All 15 of our fork's commits over its old base are the **old DSpark** (`07b2167e` + fixes, incl.
`db343667` = the `n_predict=dspark_block_size` patch that caused the ÷5 constraint). **Every one is
superseded by the rewrite — cherry-pick none.** The lone non-DSpark commit `60365071`
(w8a8 `o_proj` plain-matmul) is **w8a8-only → irrelevant for our bf16 serve.**

## Cherry-pick / merge? — NO. Just check out the PR branch.
The DSpark rewrite is a coherent branch on modern main; **cherry-picking our 15 old-DSpark commits onto
it (or its commits onto our old base) is the wrong move** — all 15 are superseded, and the rewrite depends
on newer-main infra. **Strategy = check out `pr-12005` head as a branch** (`dspark-dsv4-v2`), build it
as-is. Optionally push that branch to `Sawyer117/vllm-ascend` for the record. Pin the exact head commit
(`431a64b18b` today) since #12005 is a moving DRAFT.

## Compatibility (de-risked)
- **vllm: BOTH our HEAD and pr-12005 pin `VLLM_TAG=v0.23.0`** (Dockerfile) — **same vllm the box already has.
  Do NOT rebuild vllm; reuse `/home/a00652497/dspark_2026/installation/vllm-v0.23.0`.**
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

# 4) RECOMPILE the CANN custom ops (this is the real work — 83 csrc files changed:
#    torch_binding, sparse_attn/hc_post/causal_conv1d kernels — the new noncausal DSA needs them).
#    ⚠️ FIRST source the 9.0.0 nnal/atb set_env in a FRESH shell (NOT the 900env's stale 8.5.1 atb —
#       that causes libatb.so / "Mki::Dl undefined symbol" clashes). Then run the SAME recipe
#       a00652497 used to build 60365071 (same CANN 9.0.0 / torch_npu 2.10):
bash csrc/build.sh                    # builds ophost/opapi/opgraph/onnxplugin (+ ops_transformer install.sh)
pip install -e . --no-build-isolation # vllm 0.23.0 already present from the cloned env → no vllm rebuild

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
- **Run `EAGER=1`.** #12005 is the *eager correctness baseline*; FULL ACLGraph is the later WIP #12006.
  Validate correctness in eager first; only try graph mode once the accept number is confirmed.
- Everything else unchanged: bf16 target, `dspark_target_layer_ids=[40,41,42]`, released draft dir,
  `draft_sample_method=greedy`.
- Point env at the new env/build; run on **both** 115/116.

## Validate
Released draft (`/share/.../released_draft_fp8_standalone`) → `run_dspark_eval.sh` gsm8k 200.
- **~3.9** ⇒ the rewrite fixed it → then serve OUR draft and measure it fairly.
- still **~1.3** ⇒ deeper than the proposer rewrite (unlikely given the evidence) → back to the aux/attention audit in `HANDOFF_dspark_accept_collapse_2026-07-16.md`.

## Custom ops are PACKAGE-ISOLATED (no contamination — verified)
Rebuilding does NOT clobber the working 60365071 serve. `csrc/build_aclnn.sh` installs to
`custom_ops_install_dir="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"` (the checkout dir), and
`vllm_ascend/utils.py:323` prepends *this package's* ops to `ASCEND_CUSTOM_OPP_PATH` at import — each
env/checkout finds its own ops. Confirm on the box: (1) `grep -n custom_ops_install_dir= csrc/build_aclnn.sh`;
(2) after build, new ops under `vllm-ascend-serving/vllm_ascend/_cann_ops_custom/vendors/` while
`vllm-ascend-v4/...` is untouched; (3) `ls "$ASCEND_OPP_PATH/vendors/"` gains nothing; (4) in the serving
env `python -c "import vllm_ascend,os;print(os.environ.get('ASCEND_CUSTOM_OPP_PATH'))"` starts with the
serving path. Only gotcha: ensure nothing hard-codes `ASCEND_CUSTOM_OPP_PATH` at vllm-ascend-v4 in the
serve script / 900env (`env | grep OPP`).

## Risks
- OPEN/WIP PR (#12005 unmerged, DRAFT) — may have rough edges; pin the head commit.
- ~235-commit main jump — unrelated API/behavior drift possible; that's why separate clone + separate env.
- csrc recompile needs the CANN build env (the box has it — it built 60365071); source 9.0.0 nnal/atb fresh.
