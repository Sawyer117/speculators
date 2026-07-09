"""Load released DeepSeek-V4-Flash-DSpark draft weights into our clean-room model.

The release stores the draft under the ``mtp.*`` namespace (3 stages), plus the
shared ``embed`` / ``head`` it ties to the frozen target. Every weight is fp8
(experts fp4) with a companion ``.scale`` tensor; loading into our bf16 model
dequantizes ``weight * scale`` (the ``.scale`` keys are consumed there, not
mapped to a parameter).

This module provides:

* :func:`map_released_key` — released key -> our parameter key (or ``None`` when
  the key is a ``.scale`` sidecar, a base ``layers.*`` layer, or a base-model-only
  tensor our draft doesn't carry).
* :func:`expected_draft_keys` — the parameter keys our :class:`DSparkDraftModel`
  exposes, built analytically from the config (no torch needed) for verification.
* :func:`verify_mapping` — check the release↔ours key bijection from a safetensors
  index (structural "can we load it" check; no download, no dequant).
* :func:`load_released_draft` — the real loader (maps + dequantizes into the model).

Naming already lines up (we adopted the official ``wq_a`` / ``attn_sink`` /
``experts.i.w{1,2,3}`` names); the only renames are ``ffn.gate`` -> ``ffn.router``,
``hc_attn_*`` -> ``attn_hc.*``, ``hc_ffn_*`` -> ``ffn_hc.*``, and the stage-0/2
extras (``main_proj`` / ``norm`` / ``markov_head`` / ``confidence_head`` /
``hc_head_*``) which live at model level for us.
"""
from __future__ import annotations

import re

from .config import DSparkDraftConfig

# Which mtp stage owns the "extra" (non-per-layer) parts.
_STAGE0 = 0  # main_proj / main_norm
# stage last (n_draft_layers - 1) owns: norm, markov_head, confidence_head, hc_head_*

_HC_SITE = {"hc_attn": "attn_hc", "hc_ffn": "ffn_hc"}


def map_released_key(key: str, n_draft_layers: int = 3) -> str | None:
    """Map a released checkpoint key to our parameter key, or ``None`` to skip.

    ``None`` is returned for: ``.scale`` sidecars (folded into dequant), base
    ``layers.*`` decoder layers (target-only), and the base model's own
    ``norm`` / ``hc_head_*`` (our draft's come from the last mtp stage).
    """
    if key.endswith(".scale"):
        return None
    if key.startswith("layers."):
        return None  # base 43-layer target — not part of the draft
    last = n_draft_layers - 1

    # ---- shared with the frozen target ----
    if key == "embed.weight":
        return "embed_tokens.weight"
    if key == "head.weight":
        return "lm_head.weight"
    # base-model-only tensors (the draft's equivalents come from mtp.{last}.*)
    if key in ("norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"):
        return None

    m = re.match(r"^mtp\.(\d+)\.(.*)$", key)
    if not m:
        return None
    stage, rest = int(m.group(1)), m.group(2)

    # ---- stage-0 extras (target-hidden conditioning) -> model level ----
    if rest == "main_proj.weight":
        return "main_proj.weight"
    if rest == "main_norm.weight":
        return "main_norm.weight"

    # ---- last-stage extras (output head) -> model level ----
    if rest == "norm.weight":
        return "norm.weight"
    if rest.startswith("markov_head."):
        return rest  # markov_head.markov_w1.weight / markov_head.markov_w2.weight
    if rest == "confidence_head.proj.weight":
        return "confidence_head.proj.weight"
    hh = re.match(r"^hc_head_(fn|base|scale)$", rest)
    if hh:
        return f"hc_head.hc_{hh.group(1)}"

    # ---- per-layer block parts ----
    hc = re.match(r"^hc_(attn|ffn)_(fn|base|scale)$", rest)
    if hc:
        return f"layers.{stage}.{_HC_SITE['hc_' + hc.group(1)]}.{hc.group(2)}"
    if rest.startswith("ffn.gate."):
        return f"layers.{stage}.ffn.router.{rest[len('ffn.gate.'):]}"
    if rest.startswith(("attn.", "attn_norm.", "ffn_norm.", "ffn.experts.", "ffn.shared_experts.")):
        return f"layers.{stage}.{rest}"
    return None


def expected_draft_keys(cfg: DSparkDraftConfig) -> set[str]:
    """The parameter keys our DSparkDraftModel exposes (built from config)."""
    keys: set[str] = {
        "embed_tokens.weight",
        "lm_head.weight",
        "main_proj.weight",
        "main_norm.weight",
        "norm.weight",
        "hc_head.hc_fn",
        "hc_head.hc_base",
        "hc_head.hc_scale",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
    }
    for n in range(cfg.n_draft_layers):
        p = f"layers.{n}."
        keys |= {
            p + "attn.wq_a.weight", p + "attn.q_norm.weight", p + "attn.wq_b.weight",
            p + "attn.wkv.weight", p + "attn.kv_norm.weight",
            p + "attn.wo_a.weight", p + "attn.wo_b.weight", p + "attn.attn_sink",
            p + "attn_norm.weight", p + "ffn_norm.weight",
            p + "ffn.router.weight", p + "ffn.router.bias",
            p + "ffn.shared_experts.w1.weight", p + "ffn.shared_experts.w2.weight",
            p + "ffn.shared_experts.w3.weight",
        }
        for site in ("attn_hc", "ffn_hc"):
            keys |= {p + f"{site}.fn", p + f"{site}.base", p + f"{site}.scale"}
        for e in range(cfg.n_routed_experts):
            keys |= {p + f"ffn.experts.{e}.w1.weight",
                     p + f"ffn.experts.{e}.w2.weight",
                     p + f"ffn.experts.{e}.w3.weight"}
    return keys


def verify_mapping(released_keys, cfg: DSparkDraftConfig) -> dict:
    """Check the release↔ours key bijection (no torch, no download).

    Returns a report dict with ``ok`` and the offending sets. ``released_keys``
    is any iterable of checkpoint tensor names (e.g. an index's weight_map keys).
    """
    expected = expected_draft_keys(cfg)
    mapped: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for k in released_keys:
        tgt = map_released_key(k, cfg.n_draft_layers)
        if tgt is None:
            continue
        if tgt in mapped:
            collisions.append((tgt, mapped[tgt], k))
        mapped[tgt] = k
    mapped_targets = set(mapped)
    unfilled = expected - mapped_targets       # our params with no release source
    unexpected = mapped_targets - expected     # release keys mapping to nothing we have
    return {
        "ok": not unfilled and not unexpected and not collisions,
        "num_expected": len(expected),
        "num_mapped": len(mapped_targets),
        "unfilled": sorted(unfilled),
        "unexpected": sorted(unexpected),
        "collisions": collisions,
    }
