"""DSV4 DSpark draft = upstream dense-line DSpark method + our sparse backbone.

We subclass upstream's :class:`~speculators.models.dspark.core.DSparkDraftModel`
and override ONLY the decoder stack inside ``_backbone_forward``: the anchor
sampling, target-distribution computation, Markov + confidence heads, compound
loss, registration, ``from_training_args`` and the data contract are all
inherited verbatim. In place of the Qwen3 DFlash decoder layers we run our
clean-room DSV4 stack — multi-head latent attention + per-head sink + 256-expert
MoE + hyper-connections — with our interleaved DSV4 RoPE.

The block-attention contract is identical to DFlash's: the draft block queries
(``noise_embedding [1, TB, H]``) attend to ``[target-hidden context | block]``
under the additive ``attention_mask`` (block-diagonal + sliding window). Our
existing ``MhcDecoderBlock`` forward already implements exactly this shape
(``block_x``, ``context_x``, per-position freqs, ``attn_bias``); this module
only wires the RoPE positions and mask, and manages the mHC streams across the
stack (expand once, collapse with the HyperHead at the end).
"""
from __future__ import annotations

from typing import ClassVar, Literal

import torch
from torch import nn

from speculators import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel

from .backbone.block import MhcDecoderBlock
from .backbone.hyper import HyperHead
from .backbone.rotary import precompute_freqs_cis
from .config import DSparkDraftConfig

__all__ = ["DSV4DSparkConfig", "DSV4DSparkDraftModel"]


@SpeculatorModelConfig.register("dsv4_dspark")
class DSV4DSparkConfig(DSparkSpeculatorConfig):
    """Dense-line DSpark config + the DSV4 sparse-backbone hyperparameters.

    ``transformer_layer_config`` still carries the shared shape the SpeculatorModel
    machinery needs (hidden_size, vocab_size, num_hidden_layers = draft depth,
    rms_norm_eps); the fields below configure our MLA + MoE + mHC backbone.
    """

    speculators_model_type: Literal["dsv4_dspark"] = "dsv4_dspark"  # type: ignore[assignment]

    # multi-head latent attention
    num_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    # mixture of experts
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 2048
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def backbone_config(self) -> DSparkDraftConfig:
        """Build the plain backbone dataclass our modules consume."""
        tl = self.transformer_layer_config
        return DSparkDraftConfig(
            vocab_size=tl.vocab_size,
            hidden_size=tl.hidden_size,
            rms_norm_eps=tl.rms_norm_eps,
            n_draft_layers=tl.num_hidden_layers,
            block_size=self.block_size,
            noise_token_id=self.mask_token_id or 0,
            target_layer_ids=tuple(self.aux_hidden_state_layer_ids or (0, 1, 2)),
            markov_rank=self.markov_rank,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            o_groups=self.o_groups,
            window_size=self.window_size,
            rope_theta=self.rope_theta,
            rope_factor=self.rope_factor,
            original_seq_len=self.original_seq_len,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.n_activated_experts,
            moe_inter_dim=self.moe_inter_dim,
            score_func=self.score_func,
            route_scale=self.route_scale,
            swiglu_limit=self.swiglu_limit,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
        )


@SpeculatorModel.register("dsv4_dspark")
class DSV4DSparkDraftModel(DSparkDraftModel):
    """DSpark method (inherited) over the DSV4-native sparse backbone."""

    config_class: ClassVar[type[DSV4DSparkConfig]] = DSV4DSparkConfig  # type: ignore[assignment,misc]
    _no_split_modules: ClassVar[list[str]] = ["MhcDecoderBlock"]  # type: ignore[assignment]

    def __init__(self, config: DSV4DSparkConfig) -> None:
        # Force the additive (eager) float mask BEFORE super().__init__ reads it to
        # pick the mask builder — our sink attention consumes it as an additive bias.
        config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
        # DFlash/DSpark __init__ builds fc(=main_proj role)/hidden_norm/norm/embed/
        # lm_head/verifier + markov/confidence + a stack of Qwen3 layers we discard.
        super().__init__(config=config)
        bb = config.backbone_config()
        self.backbone_cfg = bb

        # Swap the decoder stack for our DSV4-native blocks + the mHC head.
        self.layers = nn.ModuleList(MhcDecoderBlock(bb) for _ in range(bb.n_draft_layers))
        self.hc_head = HyperHead(bb)

        # Our interleaved DSV4 RoPE frequencies, indexed by absolute position
        # (YaRN off on the sliding path -> original_seq_len=0).
        self._rope_dim = bb.rope_head_dim
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                bb.rope_head_dim, bb.original_seq_len or 1, 0, bb.rope_theta,
                bb.rope_factor, bb.beta_fast, bb.beta_slow,
            ),
            persistent=False,
        )
        self._init_backbone_params()

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSV4DSparkDraftModel":
        """Build a DSV4 DSpark model from CLI args (DSV4 backbone fields default to
        the released config; the DSpark method fields mirror upstream)."""
        config = DSV4DSparkConfig(
            **cls._build_base_config_kwargs("dsv4_dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=kwargs.get("enable_confidence_head", True),
            confidence_head_with_markov=kwargs.get("confidence_head_with_markov", True),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    def _init_backbone_params(self) -> None:
        """Initialize the freshly-built backbone params (post_init ran on the old
        Qwen3 layers). Uninitialized ``torch.empty`` params (mHC fn) would NaN."""
        std = 0.02
        for m in [*self.layers, self.hc_head]:
            for name, p in m.named_parameters():
                if p.dim() >= 2 and (".fn" in name or "weight" in name or "hc_fn" in name):
                    if torch.isnan(p).any() or not p.abs().sum().isfinite() or p.abs().sum() == 0:
                        nn.init.normal_(p, std=std)

    def _rope_at(self, positions: torch.Tensor) -> torch.Tensor:
        """freqs_cis at the given absolute positions -> [len, rope_dim//2]."""
        if positions.numel() and int(positions.max()) >= self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(
                self._rope_dim, int(positions.max()) + 1, 0, self.backbone_cfg.rope_theta,
                self.backbone_cfg.rope_factor, self.backbone_cfg.beta_fast,
                self.backbone_cfg.beta_slow,
            ).to(self.freqs_cis.device)
        return self.freqs_cis.to(positions.device)[positions]

    def _backbone_forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """DFlash scaffolding (copied) with the decoder stack swapped for our
        DSV4 sparse stack + DSV4 RoPE. Returns the same 5-tuple DSpark consumes."""
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
            (1, mask_tokens_size), self.mask_token_id, dtype=torch.long, device=device
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)  # [1, TB, H]

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)  # [1, T, H]  (main_x context)

        from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks

        block_positions = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )  # [TB]
        ctx_positions = position_ids[0]  # [T]

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

        # DSV4 RoPE freqs at the ctx and block absolute positions.
        ctx_freqs = self._rope_at(ctx_positions)
        block_freqs = self._rope_at(block_positions)

        # mHC streams across the stack; each block attends noise -> [ctx | block].
        hc = self.backbone_cfg.hc_mult
        streams = noise_embedding.unsqueeze(2).repeat(1, 1, hc, 1)  # [1, TB, hc, H]
        for layer_idx, layer in enumerate(self.layers):
            attn_bias = (
                sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask
            )
            streams = layer(
                streams,
                fc_output,
                block_freqs,
                ctx_freqs,
                self._mask_to_bias(attn_bias),
            )

        hidden = self.norm(self.hc_head(streams))  # [1, TB, H]
        logits = self.lm_head(hidden)

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size).unsqueeze(0).to(
                aligned_loss_mask.dtype
            )
        )
        aligned_loss_mask[:, :: self.block_size] = 0
        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @staticmethod
    def _mask_to_bias(mask: torch.Tensor | None) -> torch.Tensor | None:
        """Reshape the DFlash eager float mask to our sink attn_bias [1, TB, Sk]."""
        if mask is None:
            return None
        # eager float mask is [1, 1, TB, Sk] (or [1, TB, Sk]); collapse the head dim.
        while mask.dim() > 3:
            mask = mask.squeeze(1)
        return mask
