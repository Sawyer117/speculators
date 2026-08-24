"""Released-layout <-> module-name mapping, declared once and used in both directions.

WHY THIS EXISTS
A checkpoint this library trains should be one the sibling inference engine can load.
Every DSV4 DSpark loader in the ecosystem -- vLLM's
``vllm/models/deepseek_v4/nvidia/dspark.py`` and vllm-ascend's
``vllm_ascend/models/deepseek_v4/dspark.py`` -- reads the released ``mtp.*`` namespace
with per-expert tensors, because that is how DeepSeek shipped the draft. Emitting
anything else means every user runs a conversion script forever.

WHY IT IS DECLARATIVE RATHER THAN A SAVE OVERRIDE
``transformers`` already solves this, and the same way for every MoE model it ships: a
list of ``WeightRenaming`` / ``WeightConverter`` rules registered per model class.
``from_pretrained`` applies them, and ``save_pretrained(save_original_format=True)`` --
the default -- applies the REVERSE through ``revert_weight_conversion``. One declaration
covers both directions, and ``MergeModulelist(dim=0)`` is the same operation Mixtral and
Qwen-MoE use to bridge per-expert checkpoints to a stacked expert parameter.

DIRECTION: ``source_patterns`` are CHECKPOINT keys, ``target_patterns`` are MODULE
names.

THE MAPPING, verified key by key against the released draft's own weight index
(4711 keys quantized; 2382 once the ``.scale`` companions are dropped):

    embed.weight                          <-> embed_tokens.weight        [129280, 4096]
    head.weight                           <-> lm_head.weight             [129280, 4096]
    mtp.0.main_proj.weight                <-> fc.weight                  [4096, 12288]
    mtp.0.main_norm.weight                <-> hidden_norm.weight         [4096]
    mtp.{last}.norm.weight                <-> norm.weight                [4096]
    mtp.{last}.markov_head.*              <-> markov_head.*              [129280, 256]
    mtp.{last}.confidence_head.*          <-> confidence_head.*          [1, 4352]
    mtp.{last}.hc_head_{base,fn,scale}    <-> hc_head.hc_{base,fn,scale}
    mtp.{i}.hc_attn_{base,fn,scale}       <-> layers.{i}.attn_hc.{...}
    mtp.{i}.hc_ffn_{base,fn,scale}        <-> layers.{i}.ffn_hc.{...}
    mtp.{i}.ffn.gate.{weight,bias}        <-> layers.{i}.ffn.router.{...}
    mtp.{i}.ffn.experts.{e}.w{k}.weight   <-> layers.{i}.ffn.experts.w{k}   [256, ...]
    mtp.{i}.<everything else>             <-> layers.{i}.<same>   (attn.*, *_norm, ...)

WHY THE STAGE INDICES ARE PINNED RATHER THAN ``\\d+``
The conditioning projection sits on the FIRST stage and the output heads on the LAST
one, and that placement is not recoverable from the module name -- ``fc.weight`` carries
no stage. A pattern like ``mtp\\.\\d+\\.main_proj\\.`` matches fine when loading and
then reverses into the literal key ``mtp.\\d+.main_proj.weight`` when saving, because
``\\d+`` is not a capturing group. Only a concrete index round-trips, so the rules are
built for a given depth. For the same reason each ``{base,fn,scale}`` alternation is
spelled out: a transform may carry at most one capturing group, and the stage index has
to be it.

NOTE ON ``confidence_head.proj.bias`` -- present in our module when
``confidence_head_bias=True``, absent from the released layout, which has no slot for
it. The config defaults to False, so a normal run never creates it and save / resume /
serve all agree; enabling it is opting out of a byte-identical released layout, which is
the caller's choice.

The released top-level ``norm.weight`` and ``hc_head_{base,fn,scale}`` (duplicates of
the ``mtp.{last}`` ones, absent from the standalone bf16 draft) are deliberately
unmapped.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The number of decoder stages in the released DSV4-Flash draft. Pinned rules are built
# for this depth by default; ``build_mapping`` takes a depth so another is expressible.
RELEASED_N_LAYERS = 3

# These live on a semi-private path: not exported at the ``transformers`` top level, and
# the module moved between 4.x and 5.x. Failing to register must degrade to "keep the
# module names", never to an import error at model-registration time.
try:
    import torch
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import (
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
        dot_natural_key,
        rename_source_key,
    )

    _AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed transformers
    _AVAILABLE = False


def build_mapping(n_layers: int = RELEASED_N_LAYERS) -> list:
    """The rules. ORDER IS GENERIC-FIRST -- see the note above; this is not a typo."""
    last = n_layers - 1
    rules: list = [
        # --- per-stage ---
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.wq_a\.",
            target_patterns=r"layers.\1.attn.wq_a.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.wq_b\.",
            target_patterns=r"layers.\1.attn.wq_b.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.wkv\.",
            target_patterns=r"layers.\1.attn.wkv.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.wo_a\.",
            target_patterns=r"layers.\1.attn.wo_a.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.wo_b\.",
            target_patterns=r"layers.\1.attn.wo_b.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.q_norm\.",
            target_patterns=r"layers.\1.attn.q_norm.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.kv_norm\.",
            target_patterns=r"layers.\1.attn.kv_norm.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn\.attn_sink$",
            target_patterns=r"layers.\1.attn.attn_sink",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.attn_norm\.",
            target_patterns=r"layers.\1.attn_norm.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.ffn_norm\.",
            target_patterns=r"layers.\1.ffn_norm.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.ffn\.shared_experts\.",
            target_patterns=r"layers.\1.ffn.shared_experts.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.ffn\.experts\.",
            target_patterns=r"layers.\1.ffn.experts.",
        ),
        # the router is called `gate` in the release
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.ffn\.gate\.",
            target_patterns=r"layers.\1.ffn.router.",
        ),
    ]
    # --- per-stage: hyper-connections are flat in the release, nested in the module ---
    for flat, nested in (("hc_attn", "attn_hc"), ("hc_ffn", "ffn_hc")):
        for part in ("base", "fn", "scale"):
            rules.append(
                WeightRenaming(
                    source_patterns=rf"^mtp\.(\d+)\.{flat}_{part}$",
                    target_patterns=rf"layers.\1.{nested}.{part}",
                )
            )
    rules += [
        # --- top level (no stage prefix on the checkpoint side) ---
        WeightRenaming(
            source_patterns=r"^embed\.weight$", target_patterns="embed_tokens.weight"
        ),
        WeightRenaming(
            source_patterns=r"^head\.weight$", target_patterns="lm_head.weight"
        ),
        # --- first stage only: the target-hidden conditioning ---
        WeightRenaming(source_patterns=r"^mtp\.0\.main_proj\.", target_patterns="fc."),
        WeightRenaming(
            source_patterns=r"^mtp\.0\.main_norm\.", target_patterns="hidden_norm."
        ),
        # --- last stage only: heads that sit outside the layer stack ---
        WeightRenaming(
            source_patterns=rf"^mtp\.{last}\.norm\.", target_patterns="norm."
        ),
        WeightRenaming(
            source_patterns=rf"^mtp\.{last}\.markov_head\.",
            target_patterns="markov_head.",
        ),
        WeightRenaming(
            source_patterns=rf"^mtp\.{last}\.confidence_head\.",
            target_patterns="confidence_head.",
        ),
    ]
    for part in ("base", "fn", "scale"):
        rules.append(
            WeightRenaming(
                source_patterns=rf"^mtp\.{last}\.hc_head_{part}$",
                target_patterns=f"hc_head.hc_{part}",
            )
        )
    # --- the one real transformation: per-expert tensors -> one stacked parameter ---
    for w in ("w1", "w2", "w3"):
        rules.append(
            WeightConverter(
                source_patterns=f"ffn.experts.*.{w}.weight",
                target_patterns=f"ffn.experts.{w}",
                operations=[MergeModulelist(dim=0)],
            )
        )
    return rules


def register(
    class_name: str = "DSV4DSparkDraftModel", n_layers: int = RELEASED_N_LAYERS
) -> bool:
    """Register the mapping, reporting whether it took so a caller can log it."""
    if not _AVAILABLE:
        logger.warning(
            "transformers conversion-mapping API unavailable; DSV4-DSpark checkpoints "
            "be written in module-name layout and will need conversion before serving."
        )
        return False
    try:
        register_checkpoint_conversion_mapping(
            class_name, build_mapping(n_layers), overwrite=True
        )
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive; a bad rule must not break import
        logger.warning("could not register the DSV4-DSpark checkpoint mapping: %s", exc)
        return False
    return True


def is_released_layout(state_dict: dict) -> bool:
    """Released checkpoints put every stage under ``mtp.``; ours use ``layers.``."""
    return any(key.startswith("mtp.") for key in state_dict)


def to_module_layout(state_dict: dict, n_layers: int = RELEASED_N_LAYERS) -> dict:
    """Released layout -> module names, applying the SAME rules in the load direction.

    ``save_pretrained`` writes the released layout and ``from_pretrained`` reads it, but
    the trainer resumes by reading the safetensors directly and calling
    ``load_state_dict`` /
    ``set_model_state_dict`` on the raw keys. Those see ``mtp.*`` against a model whose
    parameters are ``layers.*``, and both are called with ``strict=False`` -- so without
    this a resume loads NOTHING and silently continues from random initialisation.

    This is not a second copy of the mapping: it runs the rule objects from
    :func:`build_mapping` through ``transformers``' own ``rename_source_key``.
    """
    if not _AVAILABLE:
        return state_dict
    rules = build_mapping(n_layers)
    renamings = [r for r in rules if not isinstance(r, WeightConverter)]
    converters = [r for r in rules if isinstance(r, WeightConverter)]

    converted: dict = {}
    to_stack: dict[str, list] = {}
    # dot_natural_key, not sorted(): experts must stack 0,1,2,...,10 not 0,1,10,2.
    for key in sorted(state_dict, key=dot_natural_key):
        new_key, matched_converter = rename_source_key(key, renamings, converters)
        if matched_converter is None:
            converted[new_key] = state_dict[key]
        else:
            to_stack.setdefault(new_key, []).append(state_dict[key])
    for key, tensors in to_stack.items():
        converted[key] = torch.stack(tensors, dim=0)  # what MergeModulelist(dim=0) does
    return converted
