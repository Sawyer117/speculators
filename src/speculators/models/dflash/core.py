import logging
from copy import deepcopy
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids

logger = logging.getLogger(__name__)

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
_compiled_create_block_mask = torch.compile(create_block_mask)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        # verifier_lm_head is reloaded from the verifier (see load_verifier_weights)
        # and excluded on save, so it is expected to be absent from checkpoints.
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            _compiled_create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        hidden_size = tl_config.hidden_size
        num_target_layers = len(self.target_layer_ids)
        self.block_size = config.block_size
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(
                    config.transformer_layer_config,  # type: ignore[arg-type]
                    layer_idx,
                    heterogeneous_kv_projections=(
                        config.dflash_heterogeneous_kv_projections
                    ),
                )
                for layer_idx in range(num_draft_layers)
            ]
        )
        self.sliding_window = tl_config.sliding_window
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(tl_config.layer_types)
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        rotary_config = config.transformer_layer_config
        rope_params = getattr(rotary_config, "rope_parameters", None)
        if rope_params and "sliding_attention" in rope_params:
            if self.uses_full_attn:
                logger.warning(
                    "Flattening nested rope_parameters to the sliding_attention "
                    "variant, but this model has %d full-attention layer(s). "
                    "Full-attention layers may use incorrect rope scaling.",
                    num_draft_layers - len(self.sliding_window_indices),
                )
            rotary_config = deepcopy(rotary_config)
            rotary_config.rope_parameters = rope_params["sliding_attention"]
        self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)  # type: ignore[arg-type]

        self.dflash_gated_layer_fusion = config.dflash_gated_layer_fusion
        self.dflash_dfly_layer_residual = config.dflash_dfly_layer_residual
        if self.dflash_dfly_layer_residual and not self.dflash_gated_layer_fusion:
            raise ValueError(
                "dflash_dfly_layer_residual requires dflash_gated_layer_fusion"
            )
        self.fc = nn.Linear(
            num_target_layers * hidden_size,
            hidden_size,
            bias=False,
        )
        self.layer_fusion_norms: nn.ModuleList | None = None
        self.layer_fusion_score: nn.Linear | None = None
        self.layer_fusion_proj: nn.Linear | None = None
        self.layer_fusion_gate: nn.Parameter | None = None
        if self.dflash_gated_layer_fusion:
            self.layer_fusion_norms = nn.ModuleList(
                [
                    Qwen3RMSNorm(
                        hidden_size,
                        eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
                    )
                    for _ in range(num_target_layers)
                ]
            )
            self.layer_fusion_score = nn.Linear(hidden_size, 1, bias=False)
            self.layer_fusion_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            # ⚠ 1-D with numel 1, NOT a 0-dim scalar: FSDP2's fully_shard rejects scalar
            # parameters outright ("doesn't support scalar parameters"), which is how this
            # whole family of gates killed the first DSV4 run at shard time. Every use is an
            # elementwise `tanh(gate) * tensor`, so shape (1,) broadcasts to exactly the same
            # result a 0-dim tensor gave. Do not "simplify" these back to torch.zeros(()).
            self.layer_fusion_gate = nn.Parameter(torch.zeros(1))

        self.dfly_layer_fusion_logits: nn.Parameter | None = None
        self.dfly_layer_residual_gate: nn.Parameter | None = None
        if self.dflash_dfly_layer_residual:
            self.dfly_layer_fusion_logits = nn.Parameter(
                torch.zeros(num_draft_layers, num_target_layers)
            )
            self.dfly_layer_residual_gate = nn.Parameter(torch.zeros(1))

        self.hidden_norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False

        self.context_hidden_proj: nn.Linear | None = None
        self.context_hidden_gate: nn.Parameter | None = None
        if config.dflash_context_residual:
            self.context_hidden_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.context_hidden_gate = nn.Parameter(torch.zeros(1))

        self.verifier_final_hidden_proj: nn.Linear | None = None
        self.verifier_final_hidden_gate: nn.Parameter | None = None
        if config.dflash_verifier_final_residual:
            self.verifier_final_hidden_proj = nn.Linear(
                hidden_size, hidden_size, bias=False
            )
            self.verifier_final_hidden_gate = nn.Parameter(torch.zeros(1))

        self.block_position_embedding: nn.Embedding | None = None
        if config.dflash_block_position_embedding:
            self.block_position_embedding = nn.Embedding(self.block_size, hidden_size)

        # Warn if using DFlash with sample_from_anchor=True (may not be supported)
        if type(self).__name__ == "DFlashDraftModel" and config.sample_from_anchor:
            logger.warning(
                "DFlash with sample_from_anchor=True may not be supported in "
                "all inference engines (e.g., vLLM). Verify compatibility with your "
                "deployment target."
            )

        self.post_init()
        if self.layer_fusion_score is not None:
            nn.init.zeros_(self.layer_fusion_score.weight)
        if self.dfly_layer_fusion_logits is not None:
            nn.init.zeros_(self.dfly_layer_fusion_logits)
        if self.dfly_layer_residual_gate is not None:
            nn.init.zeros_(self.dfly_layer_residual_gate)
        if self.block_position_embedding is not None:
            nn.init.zeros_(self.block_position_embedding.weight)

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        config = DFlashSpeculatorConfig(
            **cls._build_base_config_kwargs("dflash", verifier_config, **kwargs)
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _build_base_config_kwargs(
        algorithm: str,
        verifier_config: "PretrainedConfig",
        **kwargs,
    ) -> dict:
        """Shared DFlash-family config kwargs for ``from_training_args``.

        DSpark reuses this and appends its Markov/confidence/loss fields.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        block_size = kwargs.get("block_size", 8)

        # DSV4 NOTE: `"dspark" in algorithm` (not `== "dspark"`) so "dsv4_dspark" also
        # defaults sample_from_anchor=True — its serve samples every block slot.
        default_sample_from_anchor = "dspark" in algorithm
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        sample_from_anchor = (
            default_sample_from_anchor
            if sample_from_anchor_arg is None
            else sample_from_anchor_arg
        )

        # Calculate speculative tokens based on sample_from_anchor
        # False: anchor is bonus token (block_size - 1 tokens)
        # True: sample from anchor too (block_size tokens)
        speculative_tokens = block_size if sample_from_anchor else block_size - 1


        return {
            "transformer_layer_config": verifier_config,
            "draft_vocab_size": kwargs["draft_vocab_size"],
            "block_size": block_size,
            "aux_hidden_state_layer_ids": target_layer_ids,
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": kwargs.get("sliding_window_non_causal", False),
            "dflash_context_residual": kwargs.get(
                "dflash_context_residual", False
            ),
            "dflash_verifier_final_residual": kwargs.get(
                "dflash_verifier_final_residual", False
            ),
            "dflash_block_position_embedding": kwargs.get(
                "dflash_block_position_embedding", False
            ),
            "dflash_gated_layer_fusion": kwargs.get(
                "dflash_gated_layer_fusion", False
            ),
            "dflash_dfly_layer_residual": kwargs.get(
                "dflash_dfly_layer_residual", False
            ),
            "dflash_heterogeneous_kv_projections": kwargs.get(
                "dflash_heterogeneous_kv_projections", False
            ),
            "sample_from_anchor": sample_from_anchor,
            "speculators_config": SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        }

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
            sliding_window_non_causal=sliding_window_non_causal,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size
        )

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    def _prepare_target_hidden(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the shared target projection and expose raw per-layer states."""
        baseline_projection = self.fc(hidden_states)
        if not self.dflash_gated_layer_fusion:
            return baseline_projection, None

        if (
            self.layer_fusion_norms is None
            or self.layer_fusion_score is None
            or self.layer_fusion_proj is None
            or self.layer_fusion_gate is None
        ):
            raise RuntimeError("Gated layer fusion modules were not initialized")
        num_layers = len(self.layer_fusion_norms)
        hidden_size = self.config.transformer_layer_config.hidden_size
        expected_size = num_layers * hidden_size
        if hidden_states.shape[-1] != expected_size:
            raise ValueError(
                "Expected concatenated verifier hidden size "
                f"{expected_size}, got {hidden_states.shape[-1]}"
            )

        layer_states = hidden_states.reshape(
            *hidden_states.shape[:-1], num_layers, hidden_size
        )
        normalized = torch.stack(
            [
                norm(layer_states[..., layer_idx, :])
                for layer_idx, norm in enumerate(self.layer_fusion_norms)
            ],
            dim=-2,
        )
        scores = self.layer_fusion_score(normalized).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(normalized.dtype)
        fused = (normalized * weights.unsqueeze(-1)).sum(dim=-2)
        fusion_residual = self.layer_fusion_proj(fused)
        projected = baseline_projection + (
            torch.tanh(self.layer_fusion_gate) * fusion_residual
        )
        return projected, layer_states

    def _fuse_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the existing shared FC plus token-adaptive target context."""
        shared_projection, _ = self._prepare_target_hidden(hidden_states)
        return self.hidden_norm(shared_projection)

    def _add_dfly_layer_residual(
        self,
        shared_projection: torch.Tensor,
        target_layer_states: torch.Tensor | None,
        draft_layer_idx: int,
    ) -> torch.Tensor:
        """Add DFly's draft-layer-specific target view to the shared projection."""
        if not self.dflash_dfly_layer_residual:
            return self.hidden_norm(shared_projection)
        if (
            self.dfly_layer_fusion_logits is None
            or self.dfly_layer_residual_gate is None
            or target_layer_states is None
        ):
            raise RuntimeError("DFly layer residual modules were not initialized")

        weights = torch.softmax(
            self.dfly_layer_fusion_logits[draft_layer_idx].float(), dim=-1
        ).to(target_layer_states.dtype)
        weight_shape = [1] * (target_layer_states.ndim - 2) + [weights.shape[0], 1]
        layer_residual = (target_layer_states * weights.view(*weight_shape)).sum(dim=-2)
        return self.hidden_norm(
            shared_projection
            + self.dfly_layer_residual_gate.to(layer_residual.dtype) * layer_residual
        )

    def _prepare_missing_checkpoint_weights(self, loading_info: dict) -> None:
        """Safely initialize optional DFlash modules absent from a checkpoint."""
        missing_keys = tuple(loading_info.get("missing_keys", ()))
        if self.dflash_dfly_layer_residual:
            trained_dfly_fragments = (
                "dfly_layer_fusion_logits",
                "layer_fusion_norms",
                "layer_fusion_score",
                "layer_fusion_proj",
                "layer_fusion_gate",
            )
            missing_trained_dfly = [
                key
                for key in missing_keys
                if any(fragment in key for fragment in trained_dfly_fragments)
            ]
            if missing_trained_dfly:
                preview = ", ".join(missing_trained_dfly[:8])
                suffix = " ..." if len(missing_trained_dfly) > 8 else ""
                raise RuntimeError(
                    "The checkpoint enables DFly layer residuals but does not "
                    f"contain their trained weights: {preview}{suffix}. Do not "
                    "enable DFly by editing an older checkpoint config."
                )
            gate_missing = any(
                "dfly_layer_residual_gate" in key for key in missing_keys
            )
            if gate_missing:
                if self.dfly_layer_residual_gate is None:
                    raise RuntimeError("DFly residual gate was not constructed")
                with torch.no_grad():
                    # Checkpoints from the original ungated implementation used
                    # the full residual, so a scale of one preserves them exactly.
                    self.dfly_layer_residual_gate.fill_(1.0)
                logger.warning(
                    "Loaded a legacy ungated DFly checkpoint; initialized its "
                    "new residual gate to 1 for exact backward compatibility."
                )

        if self.config.dflash_heterogeneous_kv_projections:
            copied: list[str] = []
            for layer_idx, layer in enumerate(self.layers):
                attention = layer.self_attn
                for target_name, shared_name in (
                    ("target_k_proj", "k_proj"),
                    ("target_v_proj", "v_proj"),
                ):
                    target_proj = getattr(attention, target_name, None)
                    shared_proj = getattr(attention, shared_name, None)
                    if target_proj is None or shared_proj is None:
                        raise RuntimeError(
                            "Heterogeneous K/V is enabled but its projection "
                            "modules were not constructed"
                        )
                    key_fragment = f"layers.{layer_idx}.self_attn.{target_name}."
                    state_names = tuple(target_proj.state_dict())
                    missing_state_names = tuple(
                        name
                        for name in state_names
                        if any(
                            key_fragment in key
                            and key.endswith(f"{target_name}.{name}")
                            for key in missing_keys
                        )
                    )
                    if not missing_state_names:
                        continue
                    if len(missing_state_names) != len(state_names):
                        raise RuntimeError(
                            "Checkpoint contains only part of heterogeneous K/V "
                            f"projection {layer_idx}.{target_name}; refusing to "
                            "overwrite its loaded weights."
                        )
                    target_proj.load_state_dict(shared_proj.state_dict())
                    copied.append(f"layer {layer_idx} {target_name}")
            if copied:
                logger.warning(
                    "Checkpoint has no trained heterogeneous K/V weights; copied "
                    "the loaded shared K/V projections for baseline-equivalent "
                    "initialization: %s",
                    ", ".join(copied),
                )

    def _condition_noise_embedding(
        self,
        noise_embedding: torch.Tensor,
        fused_context: torch.Tensor,
        anchor_positions: torch.Tensor,
        document_ids: torch.Tensor,
        *,
        verifier_pre_lm_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply opt-in, inference-safe DFlash block conditioning."""
        if self.block_position_embedding is not None:
            slot_ids = torch.arange(
                self.block_size,
                dtype=torch.long,
                device=noise_embedding.device,
            ).repeat(anchor_positions.numel())
            noise_embedding = noise_embedding + self.block_position_embedding(
                slot_ids
            ).unsqueeze(0).to(noise_embedding.dtype)

        if self.context_hidden_proj is not None:
            if self.context_hidden_gate is None:
                raise RuntimeError("Context residual gate was not initialized")
            context_positions = (anchor_positions - 1).clamp_min(0)
            last_context = fused_context[:, context_positions, :]
            residual = self.context_hidden_proj(last_context)
            residual = residual.repeat_interleave(self.block_size, dim=1)

            anchor_docs = document_ids[:, anchor_positions]
            context_docs = document_ids[:, context_positions]
            valid_context = (
                (anchor_positions.unsqueeze(0) > 0)
                & (anchor_docs == context_docs)
                & (anchor_docs != -1)
            )
            valid_context = valid_context.repeat_interleave(
                self.block_size, dim=1
            ).unsqueeze(-1)
            residual = residual * valid_context.to(residual.dtype)
            noise_embedding = noise_embedding + (
                torch.tanh(self.context_hidden_gate)
                * residual.to(noise_embedding.dtype)
            )

        if self.verifier_final_hidden_proj is not None:
            if self.verifier_final_hidden_gate is None:
                raise RuntimeError(
                    "Verifier final-hidden residual gate was not initialized"
                )
            if verifier_pre_lm_hidden is None:
                raise ValueError(
                    "verifier_pre_lm_hidden is required when the verifier "
                    "final-hidden residual is enabled"
                )
            context_positions = (anchor_positions - 1).clamp_min(0)
            last_context = verifier_pre_lm_hidden[:, context_positions, :]
            residual = self.verifier_final_hidden_proj(last_context)
            residual = residual.repeat_interleave(self.block_size, dim=1)

            anchor_docs = document_ids[:, anchor_positions]
            context_docs = document_ids[:, context_positions]
            valid_context = (
                (anchor_positions.unsqueeze(0) > 0)
                & (anchor_docs == context_docs)
                & (anchor_docs != -1)
            )
            valid_context = valid_context.repeat_interleave(
                self.block_size, dim=1
            ).unsqueeze(-1)
            residual = residual * valid_context.to(residual.dtype)
            noise_embedding = noise_embedding + (
                torch.tanh(self.verifier_final_hidden_gate)
                * residual.to(noise_embedding.dtype)
            )

        return noise_embedding

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        *,
        project_logits: bool = True,
        **kwargs,
    ):
        """Run the anchored-block draft transformer and optionally project logits.

        Returns ``(hidden, logits, targets, aligned_loss_mask,
        anchored_block_indices)``. ``logits`` is ``None`` when
        ``project_logits=False`` so DSpark can correct hidden states before the
        single draft-vocabulary projection.
        """
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 3072)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        with torch.no_grad():
            verifier_pre_lm_hidden = self.verifier_norm(
                verifier_last_hidden_states.to(self.verifier_norm.weight.dtype)
            )

        shared_projection, target_layer_states = self._prepare_target_hidden(
            hidden_states
        )
        fc_output = self.hidden_norm(shared_projection)
        noise_embedding = self._condition_noise_embedding(
            noise_embedding,
            fc_output,
            anchor_positions,
            document_ids,
            verifier_pre_lm_hidden=verifier_pre_lm_hidden,
        )
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            # Upstream #832: when only a subset of positions is anchored (the normal
            # case -- 512 anchors out of total_seq_len), run the verifier LM head on
            # just those rows instead of the whole sequence. verifier_pre_lm_hidden is
            # already normed above, so indexing it is free (RMSNorm is per-token).
            if anchored_block_indices.numel() < total_seq_len:
                target_indices = (
                    anchored_block_indices
                    if self.config.sample_from_anchor
                    else (anchored_block_indices - 1) % total_seq_len
                )
                targets = self.verifier_lm_head(
                    verifier_pre_lm_hidden[:, target_indices]
                )
            else:
                verifier_logits = self.verifier_lm_head(verifier_pre_lm_hidden)
                if not self.config.sample_from_anchor:
                    # False: shift right by 1 so slot j predicts token at position j
                    verifier_logits = torch.roll(verifier_logits, 1, dims=1)
                # else: True, slot k predicts token at position k+1 (next), no shift
                targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        for layer_idx, layer in enumerate(self.layers):
            target_hidden = fc_output
            if self.dflash_dfly_layer_residual:
                target_hidden = self._add_dfly_layer_residual(
                    shared_projection,
                    target_layer_states,
                    layer_idx,
                )
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=target_hidden,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden) if project_logits else None
        # shape when projected: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        # For sample_from_anchor=False, mask slot 0 (anchor) since it's not trained
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        _, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )
        if logits is None:
            raise RuntimeError("DFlash forward requires projected draft logits")
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        return None, loss, metrics
