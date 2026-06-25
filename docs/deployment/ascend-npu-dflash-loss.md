# DFlash loss — why the numbers differ (and what they mean)

A reference for the recurring "my DFlash loss starts at ~3.5 but my colleague's is
~12 — which is right?" confusion, plus the vocab-size red herring. **Both ~3.5 and
~12 are "correct" — they're the same training with different loss *reductions*.**

## TL;DR

| Question | Answer |
|---|---|
| Why ~12 vs ~3.5? | The reported loss is a **decay-weighted reduction**, and PR **#496** changed its **denominator** (`÷Σ(mask·decay)` → `÷Σ(mask)`), scaling the reported value (and the gradient) by **~3.4×**. Pre-#496 ≈ 12, post-#496 ≈ 3.5. |
| Is it `ce` vs `kl`? | **No.** Both `ce` and `kl_div` land ~3.5 after the current reduction. (The earlier "ce vs kl" theory was wrong.) |
| Is it vocab size (32000 vs 151936)? | **No.** vLLM doesn't set vocab; the run uses full **151,936**. ln(32000)=10.4 vs ln(151936)=11.9 — only ~1.5 apart, not 12 vs 3.5. |
| So which loss should I compare? | **Not the loss magnitude.** Compare **per-position acceptance / accept-length** — those are reduction- and loss-fn-independent. |

## 1. The reported loss is NOT a raw per-token loss

`src/speculators/models/metrics.py::loss_function`:

```
reported = Σ_p ( L_p · mask_p · decay_p ) / Σ_p mask_p
```

- `L_p` = per-position loss (`ce_loss` = hard CE vs teacher argmax, or `kl_div_loss` = soft KL).
- `decay_p` = `dflash_loss_decay` (γ=4): pos0→0, pos1→1, pos2→e^(-1/γ), … (later positions matter less).
- The **numerator is decay-weighted; the denominator is NOT**. So the reported value is a *weighted sum ÷ unweighted count*, i.e. raw × (Σdecay/Σmask) ≈ raw × **0.28**.

So at init (≈ uniform draft), raw per-token CE ≈ `ln(151936) ≈ 11.9` → **reported ≈ 11.9 × 0.28 ≈ 3.3** ✓ (not 12).

## 2. The #496 denominator change explains 12 vs 3.5

| Version | Reduction denominator | Reported at init |
|---|---|---|
| Original DFlash (#354, `f324209`) | `Σ(mask·decay)` → a **true weighted mean** | raw per-token CE ≈ `ln V` ≈ **12** |
| #496 "Refactor metrics" (`0d73ed0`) → today | `Σ(mask)` (decay kept in numerator only) | ≈ raw × (Σdecay/Σmask) ≈ **3.5** |

Quantitatively **12 / 3.5 ≈ 3.4 = Σmask/Σdecay** (γ=4, block 16) — exact, not a coincidence.
So a colleague seeing **~12 is running pre-#496 library code** (weighted-mean denominator);
post-#496 code reports **~3.5**.

> ⚠️ **The loss is back-propagated**, so post-#496 the gradient is scaled ~3.4× *down* too
> → **effective LR ~3.4× lower** for DFlash. To match the original/colleague dynamics:
> raise `lr` ~3.4× (`6e-4 → ~2e-3`) **or** restore the `÷Σ(mask·decay)` denominator. This
> is likely an unintended #496 regression.

## 3. The loss-fn default (`ce` vs `kl_div`) — separate issue (#542)

DFlash's hardcoded default was `ce_loss` (per [issue #541](https://github.com/vllm-project/speculators/issues/541)).
PR **#542** (`e99dadc`) added `--loss-fn` but set the CLI default to `kl_div` and flipped
DFlash's internal fallback `ce_loss → kl_div_loss` — i.e. it **silently changed DFlash's
default**, contrary to #541's "make it configurable, keep the default" intent and to #542's
own "backward compatible" description. We pin **`--loss-fn ce`** (DFlash's validated default)
via the `LOSS_FN` config knob. *(Note: `ce` ≈ 3.5 and `kl_div` ≈ 3.5 here — the reduction in
§1-2, not the loss fn, sets the magnitude.)*

## 4. vocab size is a red herring (the "32000" claim)

Claim heard: *"vLLM defaults vocab to 32000, it should be 151643."* **Incorrect:**

- `scripts/launch_vllm.py` **does not touch vocab**; vLLM serves Qwen3-4B with its real
  config vocab = **151,936**. vocab also doesn't enter hidden-state extraction at all.
- The DFlash **config dataclass DOES default** `draft_vocab_size = 32000`
  (`models/dflash/config.py`). **BUT `scripts/train.py` overrides it**: when
  `--draft-vocab-size` is unset (None), it falls back to the **full** verifier vocab
  (logs `Using full verifier vocab`) → **151,936**. To actually use 32000 you need a
  `token_freq.pt` **and** `--draft-vocab-size` (the flag alone falls back to full). So
  real training runs use **151,936** unless you opt into a reduced vocab.
- Qwen3-4B vocab is **151,936** (151,643 is a Qwen2-era number).
- Magnitude check: `ln(32000)=10.4` vs `ln(151936)=11.9` — only **~1.5 apart**. vocab moves
  the baseline by ~1.5; it is **not** the ~3.4× (12 vs 3.5) gap — that's the §2 denominator.
  A colleague reporting ~14 at full vocab is on the pre-#496 weighted-mean reduction (per-
  token loss ≈ lnV); the same full vocab post-#496 reports ~3.5.

(Reducing draft vocab via a `token_freq.pt` + `--draft-vocab-size` is a legitimate *speed*
optimization, not a loss-correctness issue, and is unrelated to the magnitude question.)

## 5. SpecForge reference, for comparison

`SpecForge/specforge/core/loss.py` (Eagle3): only **soft-CE/KL** (teacher probs; no hard CE),
reduced by **`.mean()` over B×T** (no decay, divides by total incl. masked). Its eagle3 recipe
actually uses the **LK loss** (`lk_loss.py`) — which is the source of a colleague's
`--loss-fn lk_hybrid --lk-eta 3.0`. So SpecForge's reduction differs from DFlash's on every
axis (soft vs hard, no-decay vs decay, ÷B×T vs ÷Σmask) → **loss magnitudes are not comparable
across the two frameworks** either.

## Practical takeaways

1. **Don't compare DFlash loss magnitude** across forks / loss-fns / reductions. Compare
   **per-position acceptance and expected accept-length** (`analyze_train_log.py`).
2. To reproduce a `~12` baseline (or match pre-#496 gradient scale): restore
   `÷Σ(mask·decay)`, or raise `lr ~3.4×`.
3. Your current `--loss-fn ce`, full-vocab (151,936), post-#496 run is **self-consistent**;
   just know its effective LR is ~3.4× lower than the original DFlash recipe at the same `lr`.

*Commits referenced: #354 `f324209` (original DFlash), #496 `0d73ed0` (metrics refactor /
denominator change), #542 `e99dadc` (configurable loss-fn + default flip), issue #541.*
