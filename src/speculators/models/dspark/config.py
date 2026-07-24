from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus sequential correction and confidence heads.

    The active sequential head lets each draft position condition on previously
    sampled tokens within the block; the confidence head predicts per-position
    acceptance probability. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["dspark"] = "dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
            "Default True matches DeepSeek/DeepSpec convention."
        ),
    )

    # Sequential (Markov) head.
    markov_rank: int = Field(
        default=256,
        description=(
            "Low-rank dimension of the Markov logit-bias factorization B = W1 @ W2. "
            "Set to 0 to disable the sequential head (pure DFlash drafting)."
        ),
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla",
        description=(
            "Sequential head variant: 'vanilla' (first-order Markov bias), 'gated' "
            "(hidden-gated bias), or 'rnn' (recurrent state over the block)."
        ),
    )

    # Pre-projection causal correction head. When enabled it replaces MarkovHead.
    enable_correction_head: bool = Field(
        default=False,
        description=(
            "Replace the Markov head with a causal head that predicts a gated "
            "hidden-space residual from previous-token, DFlash hidden, and "
            "block-position embeddings before the single LM-head projection."
        ),
    )
    correction_hidden_size: int = Field(
        default=512,
        gt=0,
        description="Hidden width of the causal correction head.",
    )
    correction_rank: int = Field(
        default=256,
        gt=0,
        description="Low-rank bottleneck used to produce the hidden residual.",
    )
    correction_num_layers: int = Field(
        default=1,
        gt=0,
        description="Number of causal Transformer layers in the correction head.",
    )
    correction_num_heads: int = Field(
        default=8,
        gt=0,
        description="Attention heads in each correction layer.",
    )
    correction_gate_bias: float = Field(
        default=0.0,
        description="Initial bias of the sigmoid hidden-residual gate.",
    )
    correction_generated_token_training: bool = Field(
        default=False,
        description=(
            "During training, feed each greedily generated Correction token into "
            "the next block position instead of teacher-forcing ground-truth tokens. "
            "Ignored when Correction is disabled."
        ),
    )
    correction_rollout_metrics: bool = Field(
        default=True,
        description=(
            "Measure greedy self-feedback correction metrics during validation. "
            "Disable to reduce validation compute."
        ),
    )
    correction_base_diagnostics: bool = Field(
        default=False,
        description=(
            "During validation only, run an extra base LM-head projection for "
            "change/gain diagnostics. It is never used by Correction itself."
        ),
    )

    # Confidence head.
    enable_confidence_head: bool = Field(
        default=True,
        description="Whether to attach the per-position acceptance-probability head.",
    )
    confidence_head_with_markov: bool = Field(
        default=True,
        description=(
            "Concatenate the active sequential state (Markov embedding or causal "
            "correction state) with the backbone hidden state for confidence."
        ),
    )
    confidence_detach_features: bool = Field(
        default=False,
        description=(
            "Detach backbone and sequential features before ConfidenceHead. "
            "False lets confidence loss train the active Markov/correction path; "
            "True makes confidence a fully auxiliary observer."
        ),
    )
