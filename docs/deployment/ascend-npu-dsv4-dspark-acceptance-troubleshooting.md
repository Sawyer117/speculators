# DSV4-DSpark acceptance troubleshooting — low peak vs low tail

_Last updated: 2026-07-21._

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
2. **RoPE at runtime** — re-verify (not just byte-parity) that the training rotary equals the serve's
   `get_cos_and_sin_dsa`.
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

1. **Exposure bias (teacher-forcing gap)** — the dominant tail cause. Training is teacher-forced (pos_k
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

**Bottom line:** good pos0 + weak tail ⇒ most likely **data + convergence** (released proves the
architecture can flatten the tail); exposure bias is the mechanism but data overcomes it. Confirm the
tail is still climbing before touching γ.

---

## What is already ruled out (do not re-chase)

- Architecture fidelity — verified module-by-module vs the released `mtp.{i}` draft (all diffs are
  naming/dtype); RoPE byte-parity; window aligned to 128; recipe aligned (cosine / warmup 0.04 / tv 1.8).
- Serve engine — the released draft peaks at 4.658 on it.
- The expert forward is bf16 in every run and at serve (train/serve consistent); fp32 is master/optimizer
  only. See `ascend-npu-dsv4-dspark-run-comparison.md`.
