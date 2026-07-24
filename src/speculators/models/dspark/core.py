from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics, select_logged_metrics
from speculators.models.dspark.model_definitions import (
    CausalCorrectionHead,
    ConfidenceHead,
    MarkovHead,
)
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a sequential correction and confidence head.

    The legacy Markov path refines base logits. The causal Correction path refines
    DFlash hidden states from previous-token, hidden, and block-position features
    before the sole LM-head projection. An opt-in collaboration path gates a low-rank
    Markov bias from Correction state. The confidence head predicts each position's
    acceptance probability.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size
        if (
            config.correction_generated_token_ratio > 0.0
            and not config.enable_correction_head
        ):
            raise ValueError(
                "correction_generated_token_ratio > 0 requires Correction"
            )
        if (
            config.correction_generated_token_warmup
            + config.correction_generated_token_ramp
            > 1.0
        ):
            raise ValueError(
                "Correction generated-token warmup + ramp must be <= 1"
            )

        self.markov_head: MarkovHead | None = None
        self.correction_head: CausalCorrectionHead | None = None
        self.correction_markov_gate: torch.nn.Linear | None = None
        self.correction_markov_scale: torch.nn.Parameter | None = None
        if config.enable_correction_head:
            self.correction_head = CausalCorrectionHead(
                input_hidden_size=hidden_size,
                token_embedding_size=hidden_size,
                block_size=self.block_size,
                correction_hidden_size=config.correction_hidden_size,
                correction_rank=config.correction_rank,
                num_layers=config.correction_num_layers,
                num_heads=config.correction_num_heads,
                gate_bias=config.correction_gate_bias,
            )
            if config.correction_with_markov:
                if config.markov_rank <= 0:
                    raise ValueError(
                        "correction_with_markov=True requires markov_rank > 0"
                    )
                if config.markov_head_type == "rnn":
                    raise ValueError(
                        "Correction-Markov collaboration supports only vanilla "
                        "or gated Markov heads"
                    )
                self.markov_head = MarkovHead(
                    verifier_vocab_size=self.verifier_vocab_size,
                    draft_vocab_size=self.draft_vocab_size,
                    markov_rank=config.markov_rank,
                    hidden_size=hidden_size,
                    head_type=config.markov_head_type,
                )
                self.correction_markov_gate = torch.nn.Linear(
                    config.correction_hidden_size, 1
                )
                self.correction_markov_scale = torch.nn.Parameter(torch.zeros(()))
                torch.nn.init.zeros_(self.correction_markov_gate.weight)
                torch.nn.init.constant_(
                    self.correction_markov_gate.bias,
                    config.correction_markov_gate_bias,
                )
        elif config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if (
                config.confidence_head_with_markov
                and self.markov_head is None
                and self.correction_head is None
            ):
                raise ValueError(
                    "confidence_head_with_markov=True requires an enabled Markov "
                    "or correction head."
                )
            sequential_dim = 0
            if config.confidence_head_with_markov:
                sequential_dim = (
                    config.correction_hidden_size
                    if self.correction_head is not None
                    else config.markov_rank
                )
            input_dim = hidden_size + sequential_dim
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_correction_head=kwargs.get("enable_correction_head", False),
            correction_hidden_size=kwargs.get("correction_hidden_size", 512),
            correction_rank=kwargs.get("correction_rank", 256),
            correction_num_layers=kwargs.get("correction_num_layers", 1),
            correction_num_heads=kwargs.get("correction_num_heads", 8),
            correction_gate_bias=kwargs.get("correction_gate_bias", 0.0),
            correction_with_markov=kwargs.get("correction_with_markov", False),
            correction_markov_gate_bias=kwargs.get(
                "correction_markov_gate_bias", -2.0
            ),
            correction_generated_token_ratio=kwargs.get(
                "correction_generated_token_ratio", 0.0
            ),
            correction_generated_token_warmup=kwargs.get(
                "correction_generated_token_warmup", 0.2
            ),
            correction_generated_token_ramp=kwargs.get(
                "correction_generated_token_ramp", 0.4
            ),
            correction_rollout_metrics=kwargs.get(
                "correction_rollout_metrics", True
            ),
            correction_base_diagnostics=kwargs.get(
                "correction_base_diagnostics", False
            ),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            confidence_detach_features=kwargs.get(
                "confidence_detach_features", False
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        confidence_length_alpha = kwargs.get("confidence_length_alpha", 0.0)
        confidence_loss_weighting = kwargs.get(
            "confidence_loss_weighting", "uniform"
        )
        first_error_focal_alpha = kwargs.get("first_error_focal_alpha", 0.0)
        adaptive_loss = kwargs.get("adaptive_loss", "none")
        ssal_curriculum = kwargs.get("ssal_curriculum", False)
        ssal_curriculum_start = kwargs.get("ssal_curriculum_start", 0.1)
        ssal_curriculum_end = kwargs.get("ssal_curriculum_end", 0.6)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        generated_token_ratio = float(
            kwargs.get("correction_generated_token_ratio", 0.0)
        )
        generated_token_warmup = float(
            kwargs.get("correction_generated_token_warmup", 0.2)
        )
        generated_token_ramp = float(
            kwargs.get("correction_generated_token_ramp", 0.4)
        )
        if not 0.0 <= generated_token_ratio <= 1.0:
            raise ValueError("Generated-token ratio must be in [0, 1]")
        if not 0.0 <= generated_token_warmup <= 1.0:
            raise ValueError("Generated-token warmup must be in [0, 1]")
        if not 0.0 <= generated_token_ramp <= 1.0:
            raise ValueError("Generated-token ramp must be in [0, 1]")
        if generated_token_warmup + generated_token_ramp > 1.0:
            raise ValueError("Generated-token warmup + ramp must be <= 1")
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "confidence_length_alpha": confidence_length_alpha,
            "confidence_loss_weighting": confidence_loss_weighting,
            "first_error_focal_alpha": first_error_focal_alpha,
            "adaptive_loss": adaptive_loss,
            "ssal_decay_weight": 0.0,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        train_kw = dict(shared)
        if ssal_curriculum:
            train_kw["ssal_curriculum"] = True
            train_kw["ssal_curriculum_start"] = ssal_curriculum_start
            train_kw["ssal_curriculum_end"] = ssal_curriculum_end
        if generated_token_ratio > 0.0:
            train_kw["correction_generated_token_curriculum"] = True
            train_kw["correction_generated_token_target_ratio"] = (
                generated_token_ratio
            )
            train_kw["correction_generated_token_warmup"] = generated_token_warmup
            train_kw["correction_generated_token_ramp"] = generated_token_ramp
        return train_kw, dict(shared)

    @staticmethod
    def _confidence_features(
        hidden_states: torch.Tensor,
        sequential_states: torch.Tensor | None,
        *,
        detach: bool,
    ) -> torch.Tensor:
        """Build consistently coupled or detached confidence-head features."""
        if detach:
            hidden_states = hidden_states.detach()
            if sequential_states is not None:
                sequential_states = sequential_states.detach()
        if sequential_states is None:
            return hidden_states
        return torch.cat(
            [hidden_states, sequential_states.to(hidden_states.dtype)], dim=-1
        )

    def _apply_collaborative_markov(
        self,
        correction_logits: torch.Tensor,
        correction_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate a low-rank Markov bias with Correction's causal state."""
        if (
            self.markov_head is None
            or self.correction_markov_gate is None
            or self.correction_markov_scale is None
        ):
            raise RuntimeError("Correction-Markov collaboration is not enabled")
        prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
        markov_bias = self.markov_head.block_bias(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
            prev_emb=prev_emb,
        )
        gate_dtype = self.correction_markov_gate.weight.dtype
        local_gate = torch.sigmoid(
            self.correction_markov_gate(correction_states.to(gate_dtype))
        )
        gate = torch.tanh(self.correction_markov_scale) * local_gate
        collaborative_logits = correction_logits + (
            gate.to(markov_bias.dtype) * markov_bias
        )
        return collaborative_logits, gate, prev_emb

    @torch.compiler.disable
    def _generated_feedback_correction(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a differentiable Correction pass with greedy token self-feedback.

        Token selection is discrete, but the per-position logits and causal K/V
        states retain their autograd graph.  The first input is always the real
        anchor; every later input is generated by the current Correction model.
        """
        if self.correction_head is None:
            raise RuntimeError(
                "_generated_feedback_correction requires enable_correction_head=True"
            )
        if dflash_hidden.ndim != 3:
            raise ValueError("dflash_hidden must be rank-3")
        if dflash_hidden.shape[1] != self.block_size:
            raise ValueError(
                f"Expected block_size={self.block_size}, got {dflash_hidden.shape[1]}"
            )
        if anchor_token_ids.shape != (dflash_hidden.shape[0],):
            raise ValueError(
                f"Expected anchor_token_ids shape {(dflash_hidden.shape[0],)}, "
                f"got {anchor_token_ids.shape}"
            )

        previous_ids = anchor_token_ids.long()
        cache = None
        output_tokens: list[torch.Tensor] = []
        output_logits: list[torch.Tensor] = []
        output_states: list[torch.Tensor] = []
        start_position = 0 if self.config.sample_from_anchor else 1

        for position in range(self.block_size):
            current_hidden = dflash_hidden[:, position]
            if position < start_position:
                final_logits = self.lm_head(
                    current_hidden.to(self.lm_head.weight.dtype)
                )
                causal_states = current_hidden.new_zeros(
                    current_hidden.shape[0], self.config.correction_hidden_size
                )
            else:
                with torch.no_grad():
                    previous_emb = self.embed_tokens(previous_ids).unsqueeze(1)
                block_positions = torch.full(
                    (dflash_hidden.shape[0], 1),
                    position,
                    dtype=torch.long,
                    device=dflash_hidden.device,
                )
                delta_hidden, causal_states, cache = self.correction_head(
                    previous_emb,
                    dflash_hidden[:, position : position + 1],
                    block_positions,
                    cache=cache,
                    use_cache=True,
                )
                corrected_hidden = current_hidden + delta_hidden[:, 0].to(
                    current_hidden.dtype
                )
                final_logits = self.lm_head(
                    corrected_hidden.to(self.lm_head.weight.dtype)
                )
                causal_states = causal_states[:, 0]
                if getattr(self, "markov_head", None) is not None:
                    final_logits, _, _ = self._apply_collaborative_markov(
                        final_logits.unsqueeze(1),
                        causal_states.unsqueeze(1),
                        previous_ids.unsqueeze(1),
                        dflash_hidden[:, position : position + 1],
                    )
                    final_logits = final_logits[:, 0]

            with torch.no_grad():
                if temperature > 0:
                    probabilities = torch.softmax(
                        final_logits.float() / temperature, dim=-1
                    )
                    draft_ids = torch.multinomial(
                        probabilities, num_samples=1
                    ).squeeze(-1)
                else:
                    draft_ids = torch.argmax(final_logits, dim=-1)
            output_tokens.append(draft_ids)
            output_logits.append(final_logits)
            output_states.append(causal_states)

            if position < start_position:
                continue
            previous_ids = draft_ids
            if self.d2t is not None:
                previous_ids = previous_ids + self.d2t[previous_ids]

        return (
            torch.stack(output_tokens, dim=1),
            torch.stack(output_logits, dim=1),
            torch.stack(output_states, dim=1),
        )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        confidence_length_alpha: float = 0.0,
        confidence_loss_weighting: str = "uniform",
        first_error_focal_alpha: float = 0.0,
        adaptive_loss: str = "none",
        ssal_decay_weight: torch.Tensor | float = 0.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        correction_use_generated_tokens: bool = False,
        correction_generated_token_ratio: torch.Tensor | float = 0.0,
        correction_generated_token_curriculum_active: bool = False,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                project_logits=self.correction_head is None,
                **kwargs,
            )
        )

        # DSpark: add the active sequential correction and predict confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        base_logits = logits
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)
        block_positions = torch.arange(block, device=hidden.device).expand(
            num_blocks, -1
        )

        confidence_logits = None
        prev_emb = None
        correction_states = None
        rollout_logits = None
        collaboration_base_logits = None
        collaboration_gate = None
        if self.correction_head is not None:
            if self.training and correction_use_generated_tokens:
                _, logits_blocks, correction_states = (
                    self._generated_feedback_correction(
                        hidden_blocks,
                        anchor_token_ids=block_tokens[:, 0],
                    )
                )
                logits = logits_blocks.reshape(1, mask_tokens_size, -1)
            else:
                if self.config.sample_from_anchor:
                    with torch.no_grad():
                        prev_gt_emb = self.embed_tokens(block_tokens)
                    delta_hidden, correction_states, _ = self.correction_head(
                        prev_gt_emb,
                        hidden_blocks,
                        block_positions,
                    )
                    corrected_hidden = hidden_blocks + delta_hidden.to(
                        hidden_blocks.dtype
                    )
                else:
                    with torch.no_grad():
                        prev_gt_emb = self.embed_tokens(block_tokens[:, :-1])
                    delta_hidden, draft_states, _ = self.correction_head(
                        prev_gt_emb,
                        hidden_blocks[:, 1:],
                        block_positions[:, 1:],
                    )
                    corrected_hidden = torch.cat(
                        [
                            hidden_blocks[:, :1],
                            hidden_blocks[:, 1:]
                            + delta_hidden.to(hidden_blocks.dtype),
                        ],
                        dim=1,
                    )
                    correction_states = torch.cat(
                        [
                            draft_states.new_zeros(
                                num_blocks, 1, draft_states.shape[-1]
                            ),
                            draft_states,
                        ],
                        dim=1,
                    )

                # Teacher-forced training/validation can project the full block
                # together. Generated-token training projects each position once
                # inside its autoregressive self-feedback loop.
                logits = self.lm_head(
                    corrected_hidden.reshape(1, mask_tokens_size, -1).to(
                        self.lm_head.weight.dtype
                    )
                )
                if self.markov_head is not None:
                    collaboration_base_logits = logits
                    collaborative_blocks, collaboration_gate, prev_emb = (
                        self._apply_collaborative_markov(
                            logits.view(num_blocks, block, -1),
                            correction_states,
                            prev_token_ids,
                            hidden_blocks,
                        )
                    )
                    logits = collaborative_blocks.reshape(
                        1, mask_tokens_size, -1
                    )

            # Optional validation-only base projection for change/gain diagnostics.
            # It is never part of the training or inference correction path.
            if not self.training and self.config.correction_base_diagnostics:
                with torch.no_grad():
                    base_logits = self.lm_head(
                        hidden.detach().to(self.lm_head.weight.dtype)
                    )

            # Validation keeps the teacher-forced view for comparison and also
            # measures the actual generated-token feedback chain.
            if not self.training and self.config.correction_rollout_metrics:
                _, rollout_blocks = self.rollout_correction(
                    hidden_blocks.detach(),
                    anchor_token_ids=block_tokens[:, 0],
                )
                rollout_logits = rollout_blocks.reshape(1, mask_tokens_size, -1)
        elif self.markov_head is not None:
            if logits is None:
                raise RuntimeError("Markov correction requires base logits")
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if logits is None:
            raise RuntimeError("DSpark forward did not produce draft logits")

        if self.confidence_head is not None:
            sequential_states = None
            if self.config.confidence_head_with_markov:
                sequential_states = (
                    correction_states if correction_states is not None else prev_emb
                )
            conf_features = self._confidence_features(
                hidden_blocks,
                sequential_states,
                detach=self.config.confidence_detach_features,
            )
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            confidence_length_alpha=confidence_length_alpha,
            confidence_loss_weighting=confidence_loss_weighting,  # type: ignore[arg-type]
            first_error_focal_alpha=first_error_focal_alpha,
            adaptive_loss=adaptive_loss,  # type: ignore[arg-type]
            ssal_decay_weight=ssal_decay_weight,
            base_logits=(
                base_logits
                if self.markov_head is not None
                or (self.correction_head is not None and base_logits is not None)
                else None
            ),
            rollout_logits=rollout_logits,
            collaboration_base_logits=collaboration_base_logits,
            collaboration_gate=collaboration_gate,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        metrics = select_logged_metrics(
            metrics,
            include_diagnostics=(
                not self.training and self.config.correction_base_diagnostics
            ),
        )
        if (
            self.training
            and self.correction_head is not None
            and correction_generated_token_curriculum_active
        ):
            ratio = torch.as_tensor(
                correction_generated_token_ratio,
                device=loss.device,
                dtype=torch.float32,
            ).detach()
            one = torch.ones((), device=loss.device, dtype=torch.float32)
            metrics["correction_generated_token_ratio_sum"] = ratio
            metrics["correction_generated_token_ratio_total"] = one
        return None, loss, metrics

    @torch.compiler.disable
    @torch.no_grad()
    def rollout_correction(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively apply correction using generated-token feedback."""
        tokens, logits, _ = self._generated_feedback_correction(
            dflash_hidden,
            anchor_token_ids,
            temperature=temperature,
        )
        return tokens, logits
