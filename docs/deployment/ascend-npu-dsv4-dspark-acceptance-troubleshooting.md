# DSV4-DSpark acceptance troubleshooting — low peak vs low tail

_Last updated: 2026-08-06._

> **★★★ RESOLVED (2026-08-05): the dominant tail / train↔serve gap was DEGENERATE TRAINING-RoPE.** The
> draft's complex `freqs_cis` was cast to bf16 by the trainer (NPU can't index complex), silently dropping
> the imaginary part → `apply_rotary_emb`'s complex×real became a **scale-only** op (no rotation), while the
> serve rotated properly. Fixed to real cos/sin interleaved (`feb0066`/`8db8f75`); a from-scratch RoPE-fixed
> run evals at 0.5ep to **mean 3.84 = new best**, tail lifted, `train↑/eval↓` divergence gone (see
> `ascend-npu-dsv4-dspark-eval-results.md` `ep0p5-ropefix` row). **This tree is retained for *residual*
> diagnosis; RoPE is no longer a suspect — it's the found+fixed cause.** ⚠️ Note below where "RoPE
> byte-parity" was wrongly listed under *ruled out* — byte-parity passed but the runtime rotation was
> degenerate, which is the whole lesson.

When a trained DSV4-DSpark draft under-performs the released draft's acceptance, **the shape of the
per-position curve tells you which class of bug it is.** A low *peak* (pos0 itself is low) and a low
*tail* (pos0 fine, pos1..4 decay) have almost disjoint causes — diagnose them differently.

## The lever: released draft peaks on OUR serve

The released DSV4 draft reaches **gsm8k accept-len 4.658** on our own vllm-ascend serve
(`ascend-npu-dsv4-dspark-eval-results.md`). So if **our** draft is low on the **same** serve, the
fault is in **our draft (training or conversion), not the serve engine**. This single fact rules out
"the serve is broken" for every case below.

## Decisive first split: training pos0 vs serve pos0

Before anything else, compare our draft's **training pos0 accuracy** (teacher-forced) with its
**serve pos0 acceptance**:

| Observation | Meaning | Go to |
|---|---|---|
| both low | training / data / loss problem | Case A (training side) |
| training pos0 high, **serve** pos0 low | serve runs a task training never optimized | Case A (mismatch/conversion) |
| both pos0 fine, only tail low | forward + conversion are correct | **Case B** |

---

## Case A — low PEAK (pos0 itself is low)

A low peak rules out the data-tail story; it means our draft was optimized for a slightly different
function than the serve runs, OR the serve loads a subtly-wrong draft. Ranked:

**A1. Train↔serve forward-convention mismatch** (top suspect for a capped peak)
1. **HS / aux capture point** — the target hidden fed in training must be captured at the SAME layer
   (`[40,41,42]`), residual convention, and `mean(1)` reduction as the serve feeds the draft. A
   pre/post-residual or wrong-layer diff makes the draft learn a different input → peak capped.
2. **RoPE at runtime** — ★ **CONFIRMED ROOT CAUSE (fixed 2026-08-05, `feb0066`).** The training rotary was
   DEGENERATE: complex `freqs_cis` cast to bf16 dropped the imaginary part → scale-only, no rotation (serve
   rotated properly). Byte-parity of the cache was NOT sufficient to catch this. Now real cos/sin interleaved
   `x*cos + rotate_half(x)*sin`, matching the serve. If you see a low tail on an OLD ckpt, this is why — retrain.
3. **`main_proj` / `main_norm` input path + mHC Sinkhorn / attn_sink** — must bit-match serve.
4. **`sample_from_anchor`** — the fix is live and gated (`core.py:518/559`: True ⇒ no target-roll +
   slot0 trained ⇒ matches the vllm-ascend serve). Confirm the run's actual value is `True` via
   `DSPARK_DIAGNOSE=1`. A stray `False` reintroduces the off-by-one (slot0 untrained) → capped peak.

**A2. Conversion / weight remap** (training perfect, serve loads it wrong)
5. `weights.py` stage assignment — our model-level singletons (`main_proj`/`main_norm` on stage 0;
   `norm`/`markov`/`confidence`/`hc_head` on the last stage) must map to the correct `mtp.{i}` stage
   or the serve silently runs a broken draft. **Verify this before trusting any eval.**

**A3. Distribution / verifier parity**
6. **Verifier parity** — the draft must be served against the EXACT verifier it trained on
   (`DeepSeek-V4-Flash-bf16`); a different quant/checkpoint caps acceptance.
7. **Rollout↔eval distribution + chat-template** — draft trained on the open_perfectblend rollout; if
   eval domains are OOD vs that mix, or the train/serve chat-template differs, even the peak is lower.

---

## Case B — pos0 fine, TAIL low (pos1..4 decay)

Good pos0 proves input/forward/conversion/verifier are all correct (those would tank pos0 too). The
tail is a different, largely-structural set of causes. Ranked:

0. **★ DEGENERATE TRAINING-RoPE (was THE #1 tail cause — FIXED 2026-08-05, `feb0066`).** The complex→bf16 cast
   turned the draft's rotary into a scale-only op (no rotation); the further block positions lean hardest on
   correct positional rotation, so the tail was trained wrong while the serve rotated properly ⇒ tail collapsed
   at serve. RoPE-fixed retrain lifted gsm8k pos2/3/4 59.8/46.3/35.0 → 65.6/53.6/42.8 at 0.5ep and resolved the
   `train↑/eval↓` divergence. **Rule this out FIRST (is the ckpt post-`feb0066`?) before the residual causes below.**
1. **Exposure bias (teacher-forcing gap)** — a *residual* tail cause after RoPE. Training is teacher-forced (pos_k
   sees the ground-truth prefix); serve is autoregressive (pos_k sees the draft's OWN predicted
   prefix). Errors compound down the block: pos0 has no prefix dependency, pos3/4 depend on the
   draft's own pos0..2. Intrinsic to the parallel-draft block design.
2. **Data / convergence** — the tail is far more data-hungry than pos0. This was the `epoch4-17w`
   failure (17W too small → pos3/4 = 19/5) vs released's flat tail. **The released draft reaches a
   flat tail at the SAME γ=4 and SAME architecture → the tail is RECOVERABLE; the lever is data +
   convergence, not the architecture.** First check whether the tail is still rising in training
   (still climbing ⇒ just undertrained, keep going).
3. **Loss-decay weighting (γ)** — later positions are down-weighted `exp(-k/γ)` (γ=4 ⇒ [1,.78,.61,.47,.37])
   → less gradient → slower to converge. Raising γ (flatter) gives the tail more signal IF it is
   undertrained — but released proves γ=4 suffices with enough data, so treat γ as secondary to data.
4. **Markov head** — the mechanism meant to carry later positions; if undertrained or its serve-time
   coupling differs, the tail specifically suffers (pos0 barely uses it).
5. **Content-blind parallel backbone** — all block positions are predicted in parallel from the same
   anchor; later positions are intrinsically harder (weaker signal the further from the anchor).
6. **Serve sampling / AR-drift** — if the serve samples (temp>0), the draft's own prefix diverges from
   the greedy path it trained on → worse tail. Keep greedy train/serve consistency.

**Bottom line (revised 2026-08-05):** good pos0 + weak tail ⇒ **first confirm the ckpt was trained
post-`feb0066` (real-RoPE)** — degenerate RoPE was THE tail killer and is now fixed (ropefix 0.5ep tail
65.6/53.6/42.8). AFTER RoPE, the residual tail levers are data + convergence (released proves the
architecture flattens the tail at γ=4); exposure bias is the mechanism but data overcomes it. Confirm the
tail is still climbing before touching γ.

---

## What is already ruled out (do not re-chase)

- Architecture fidelity — verified module-by-module vs the released `mtp.{i}` draft (all diffs are
  naming/dtype); window aligned to 128; recipe aligned (cosine / warmup 0.04 / tv 1.8).
  ⚠️ **NOT "RoPE byte-parity" — that was wrongly on this list.** The cache bytes matched, but the bf16 cast
  dropped the complex imaginary part at RUNTIME → scale-only, no rotation. Byte-parity is NECESSARY, not
  SUFFICIENT; verify the actual rotation. Root-caused + fixed `feb0066` (see A1.2 / Case-B item 0).
- Serve engine — the released draft peaks at 4.658 on it.
- The expert forward is bf16 in every run and at serve (train/serve consistent); fp32 is master/optimizer
  only. See `ascend-npu-dsv4-dspark-run-comparison.md`.
