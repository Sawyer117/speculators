"""Released-layout <-> module-name mapping, declared once and used in both directions.

WHY THIS EXISTS
A checkpoint this library trains should be one the sibling inference engine can load. Every
DSV4 DSpark loader in the ecosystem -- vLLM's ``vllm/models/deepseek_v4/nvidia/dspark.py`` and
vllm-ascend's ``vllm_ascend/models/deepseek_v4/dspark.py`` -- reads the released ``mtp.*``
namespace with per-expert tensors, because that is how DeepSeek shipped the draft. Emitting
anything else means every user runs a conversion script forever.

WHY IT IS DECLARATIVE RATHER THAN A SAVE OVERRIDE
``transformers`` already solves this, and the same way for every MoE model it ships: a list of
``WeightRenaming`` / ``WeightConverter`` rules registered per model class. ``from_pretrained``
applies them, and ``save_pretrained(save_original_format=True)`` -- the default -- applies the
REVERSE through ``revert_weight_conversion``. One declaration covers both directions, and
``MergeModulelist(dim=0)`` is the same operation Mixtral and Qwen-MoE use to bridge per-expert
checkpoints to a stacked expert parameter.

DIRECTION: ``source_patterns`` are CHECKPOINT keys, ``target_patterns`` are MODULE names.

THE MAPPING, verified key by key against the released standalone draft
(``released_draft_bf16_standalone``, 2378 tensors) and a 5-epoch reproduction checkpoint
(84 tensors). Every shape matches; the counts reconcile exactly:
``84 - 1 (confidence bias) - 9 (stacked) + 2304 (per-expert) = 2378``.

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
    mtp.{i}.<everything else>             <-> layers.{i}.<same>   (attn.*, *_norm, shared_experts)

NOTE ON ``{last}`` -- the rules below match any stage index rather than hardcoding 2. Those
suffixes exist on exactly one stage in the released layout, so matching ``\\d+`` is unambiguous
and keeps the mapping independent of ``n_draft_layers``.

NOTE ON ``confidence_head.proj.bias`` -- present in our module when
``confidence_head_bias=True``, absent from the released layout, which has no slot for it. The
config defaults to False, so it does not normally exist; enabling it is opting out of a
byte-identical released layout, which is the caller's choice to make.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# These live on a semi-private path: not exported at the ``transformers`` top level, and the
# module moved between 4.x and 5.x. Failing to register must degrade to "checkpoints stay in
# module-name layout", never to an import error at model-registration time.
try:
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import (
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
    )

    _AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed transformers
    _AVAILABLE = False


def build_mapping() -> list:
    """The rules, most specific first: a later catch-all must not shadow an earlier rename."""
    rules: list = [
        # --- top level (no stage prefix on the checkpoint side) ---
        WeightRenaming(source_patterns=r"^embed\.weight$", target_patterns="embed_tokens.weight"),
        WeightRenaming(source_patterns=r"^head\.weight$", target_patterns="lm_head.weight"),
        # --- stage-0 only: the target-hidden conditioning ---
        WeightRenaming(source_patterns=r"^mtp\.\d+\.main_proj\.", target_patterns="fc."),
        WeightRenaming(source_patterns=r"^mtp\.\d+\.main_norm\.", target_patterns="hidden_norm."),
        # --- last-stage only: heads that sit outside the layer stack ---
        WeightRenaming(source_patterns=r"^mtp\.\d+\.norm\.", target_patterns="norm."),
        WeightRenaming(source_patterns=r"^mtp\.\d+\.markov_head\.", target_patterns="markov_head."),
        WeightRenaming(source_patterns=r"^mtp\.\d+\.confidence_head\.", target_patterns="confidence_head."),
        WeightRenaming(source_patterns=r"^mtp\.\d+\.hc_head_(base|fn|scale)$", target_patterns=r"hc_head.hc_\1"),
        # --- per-stage: hyper-connections are flat in the release, nested in the module ---
        WeightRenaming(source_patterns=r"^mtp\.(\d+)\.hc_attn_(base|fn|scale)$", target_patterns=r"layers.\1.attn_hc.\2"),
        WeightRenaming(source_patterns=r"^mtp\.(\d+)\.hc_ffn_(base|fn|scale)$", target_patterns=r"layers.\1.ffn_hc.\2"),
        # --- per-stage: the router is called `gate` in the release ---
        WeightRenaming(source_patterns=r"^mtp\.(\d+)\.ffn\.gate\.", target_patterns=r"layers.\1.ffn.router."),
    ]
    # --- the one real transformation: per-expert tensors -> one stacked parameter ---
    for w in ("w1", "w2", "w3"):
        rules.append(
            WeightConverter(
                source_patterns=f"ffn.experts.*.{w}.weight",
                target_patterns=f"ffn.experts.{w}",
                operations=[MergeModulelist(dim=0)],
            )
        )
    # --- catch-all LAST: attn.*, attn_norm, ffn_norm, ffn.shared_experts keep their names ---
    rules.append(WeightRenaming(source_patterns=r"^mtp\.(\d+)\.", target_patterns=r"layers.\1."))
    return rules


def register(class_name: str = "DSV4DSparkDraftModel") -> bool:
    """Register the mapping. Returns whether it took, so a caller can log rather than guess."""
    if not _AVAILABLE:
        logger.warning(
            "transformers conversion-mapping API not available; DSV4-DSpark checkpoints will "
            "be written in module-name layout and will need conversion before serving."
        )
        return False
    try:
        register_checkpoint_conversion_mapping(class_name, build_mapping(), overwrite=True)
    except Exception as exc:  # pragma: no cover - defensive; a bad rule must not break import
        logger.warning("could not register the DSV4-DSpark checkpoint mapping: %s", exc)
        return False
    return True
