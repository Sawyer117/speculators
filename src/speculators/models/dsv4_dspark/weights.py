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


def expected_draft_shapes(cfg: DSparkDraftConfig) -> dict[str, list[int]]:
    """Expected parameter shapes (as ``nn.Linear`` weight ``[out, in]`` etc.).

    Analytic (no torch) so it can be diffed against a released safetensors
    header. Matches the module definitions in :mod:`.backbone` / :mod:`.draft`.
    """
    H, V = cfg.hidden_size, cfg.vocab_size
    hd, nh = cfg.head_dim, cfg.num_heads
    qlr, olr, og = cfg.q_lora_rank, cfg.o_lora_rank, cfg.o_groups
    mi, ne, mr = cfg.moe_inter_dim, cfg.n_routed_experts, cfg.markov_rank
    hc = cfg.hc_mult
    mix = (2 + hc) * hc
    s: dict[str, list[int]] = {
        "embed_tokens.weight": [V, H],
        "lm_head.weight": [V, H],
        "main_proj.weight": [H, H * cfg.num_target_layers],
        "main_norm.weight": [H],
        "norm.weight": [H],
        "hc_head.hc_fn": [hc, hc * H],
        "hc_head.hc_base": [hc],
        "hc_head.hc_scale": [1],
        "markov_head.markov_w1.weight": [V, mr],
        "markov_head.markov_w2.weight": [V, mr],
        "confidence_head.proj.weight": [1, H + mr],
    }
    for n in range(cfg.n_draft_layers):
        p = f"layers.{n}."
        s |= {
            p + "attn.wq_a.weight": [qlr, H], p + "attn.q_norm.weight": [qlr],
            p + "attn.wq_b.weight": [nh * hd, qlr],
            p + "attn.wkv.weight": [hd, H], p + "attn.kv_norm.weight": [hd],
            p + "attn.wo_a.weight": [og * olr, nh * hd // og],
            p + "attn.wo_b.weight": [H, og * olr],
            p + "attn.attn_sink": [nh],
            p + "attn_norm.weight": [H], p + "ffn_norm.weight": [H],
            p + "ffn.router.weight": [ne, H], p + "ffn.router.bias": [ne],
        }
        for w, out in (("w1", mi), ("w2", H), ("w3", mi)):
            in_ = H if w != "w2" else mi
            s[p + f"ffn.shared_experts.{w}.weight"] = [out, in_]
            for e in range(ne):
                s[p + f"ffn.experts.{e}.{w}.weight"] = [out, in_]
        for site in ("attn_hc", "ffn_hc"):
            s[p + f"{site}.fn"] = [mix, hc * H]
            s[p + f"{site}.base"] = [mix]
            s[p + f"{site}.scale"] = [3]
    return s


# Released quant dtypes: attn/shared linears are fp8 (1 byte/value, unpacked);
# experts are fp4 packed 2-per-byte (stored as I8), so their last dim is halved.
_FP4_DTYPES = {"I8", "U8", "F4_E2M1", "F4", "FP4"}


def verify_shapes(released: dict, cfg: DSparkDraftConfig) -> dict:
    """Check released tensor shapes against ours (fp4 experts unpacked ×2).

    ``released`` maps checkpoint key -> ``{"shape": [...], "dtype": "..."}``
    (e.g. parsed from safetensors headers). Skips ``.scale`` sidecars. Returns
    a report with ``ok`` and any ``mismatches``.
    """
    exp = expected_draft_shapes(cfg)
    checked = 0
    mismatches: list[tuple] = []
    for rk, info in released.items():
        if rk.endswith(".scale"):
            continue
        tgt = map_released_key(rk, cfg.n_draft_layers)
        if tgt is None or tgt not in exp:
            continue
        shape = list(info["shape"])
        if info.get("dtype") in _FP4_DTYPES and len(shape) == 2:
            shape = [shape[0], shape[1] * 2]  # unpack fp4 nibble packing
        checked += 1
        if shape != list(exp[tgt]):
            mismatches.append((rk, tgt, info["shape"], info.get("dtype"), exp[tgt]))
    return {"ok": not mismatches, "checked": checked, "mismatches": mismatches}


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


# per-expert released key -> our layer/index for stacking into GroupedExperts.
_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")

# map_released_key predates a rename in the live model: the target-hidden conditioning
# projection/norm the release calls main_proj/main_norm are built by the shared
# DFlash/DSpark base as ``fc`` / ``hidden_norm`` (core.py: "fc(=main_proj role)"). Remap
# so the release loads onto the live parameters. (map_released_key + expected_draft_keys
# are also stale for this; fix them there when the convert path is next touched.)
_LIVE_RENAME = {
    "main_proj.weight": "fc.weight",
    "main_norm.weight": "hidden_norm.weight",
}
# our params with no release source that are NOT a problem: the release's confidence head
# has no bias (ours built one -> left at init; it does not feed routing/backbone).
_ALLOWED_MISSING = ("confidence_head.proj.bias",)


def load_released_draft(model, released_dir, *, verbose: bool = True) -> dict:
    """Load a RELEASED (or standalone-converted) DSpark draft into ``model`` in place.

    ``released_dir`` is a directory holding ``model.safetensors`` (or a sharded set +
    ``model.safetensors.index.json``) in the release ``mtp.*`` namespace — e.g. the
    ``released_draft_bf16_standalone`` dir that :mod:`scripts.build_released_draft_dir`
    produces (mtp.* dequant'd to bf16 + borrowed ``embed`` / ``head``).

    Every tensor is remapped with :func:`map_released_key`; the per-expert
    ``...ffn.experts.{e}.w{k}.weight`` tensors are STACKED into our
    :class:`GroupedExperts` layout (``layers.{n}.ffn.experts.w{k}`` shaped
    ``[E, out, in]``). Loaded non-strict: the frozen shared parts our model already
    filled from the verifier (``verifier_lm_head`` / ``verifier_norm``, and ``embed`` /
    ``lm_head`` when the release omits them) are left as-is. Returns the load report.

    Requires torch + safetensors (imported lazily so the analytic helpers above stay
    torch-free). Raises on an fp8/fp4 release — dequant to bf16 first with
    ``scripts/build_released_draft_dir.py --dequant-bf16``.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    d = Path(released_dir)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
    else:
        single = d / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"no model.safetensors[.index.json] in {d}")
        with safe_open(str(single), framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    n_draft_layers = len(model.layers)
    # group per-shard so each file opens once.
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    mapped: dict[str, "torch.Tensor"] = {}
    experts: dict[tuple[int, int], dict[int, "torch.Tensor"]] = {}
    n_skipped = 0
    for shard, keys in by_shard.items():
        with safe_open(str(d / shard), framework="pt") as f:
            for k in keys:
                tgt = map_released_key(k, n_draft_layers)
                if tgt is None:
                    n_skipped += 1
                    continue
                tgt = _LIVE_RENAME.get(tgt, tgt)  # main_proj/main_norm -> fc/hidden_norm
                t = f.get_tensor(k)
                if t.element_size() == 1:  # fp8 / fp4 block-quant weight (has a .scale)
                    raise ValueError(
                        f"'{k}' is fp8/fp4 ({t.dtype}); this loader wants a bf16 draft. "
                        "Build one first: scripts/build_released_draft_dir.py --dequant-bf16."
                    )
                em = _EXPERT_RE.match(tgt)
                if em:
                    layer, e, wk = int(em.group(1)), int(em.group(2)), int(em.group(3))
                    experts.setdefault((layer, wk), {})[e] = t
                else:
                    mapped[tgt] = t

    # stack per-expert -> GroupedExperts [E, out, in] (expert-id order).
    for (layer, wk), per_e in experts.items():
        E = len(per_e)
        stacked = torch.stack([per_e[e] for e in range(E)], dim=0)
        mapped[f"layers.{layer}.ffn.experts.w{wk}"] = stacked

    incompat = model.load_state_dict(mapped, strict=False)
    missing = [k for k in incompat.missing_keys]
    unexpected = list(incompat.unexpected_keys)
    # the verifier-filled frozen parts are expected to be "missing" from the release.
    frozen = tuple(
        s for s in ("verifier_lm_head", "verifier_norm", "embed_tokens", "lm_head")
    )
    surprising = [
        k for k in missing
        if not k.startswith(frozen) and "freqs_cis" not in k
        and k not in _ALLOWED_MISSING
    ]
    report = {
        "loaded": len(mapped),
        "skipped": n_skipped,
        "missing": missing,
        "unexpected": unexpected,
        "surprising_missing": surprising,
        "ok": not unexpected and not surprising,
    }
    if verbose:
        print(
            f"[load_released_draft] loaded {report['loaded']} tensors "
            f"({len(experts)} stacked expert groups), skipped {n_skipped}; "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"surprising_missing={surprising[:6]}"
            f"{' …' if len(surprising) > 6 else ''}  ok={report['ok']}",
            flush=True,
        )
    return report
