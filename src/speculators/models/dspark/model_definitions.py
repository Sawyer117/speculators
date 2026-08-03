"""Sequential correction, Markov, and confidence heads for DSpark."""

from typing import Literal

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention, silu

__all__ = [
    "CausalCorrectionHead",
    "ConfidenceHead",
    "MarkovHead",
]


CorrectionCache = list[tuple[torch.Tensor, torch.Tensor]]


class _TinyCausalAttention(nn.Module):
    """Small causal self-attention with an inference-friendly K/V cache."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = value.shape
        return value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        past_len = 0
        if cache is not None:
            past_key, past_value = cache
            past_len = past_key.shape[-2]
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        query_len = query.shape[-2]
        if past_len == 0:
            attn_mask = None
            is_causal = query_len > 1
        else:
            query_pos = past_len + torch.arange(
                query_len, device=query.device
            ).unsqueeze(-1)
            key_pos = torch.arange(key.shape[-2], device=query.device).unsqueeze(0)
            attn_mask = key_pos <= query_pos
            is_causal = False

        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).contiguous().view(
            hidden_states.shape[0], query_len, -1
        )
        next_cache = (key, value) if use_cache else None
        return self.o_proj(output), next_cache


class _TinyCausalLayer(nn.Module):
    """Pre-norm attention/MLP block used by :class:`CausalCorrectionHead`."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_size)
        self.attention = _TinyCausalAttention(hidden_size, num_heads)
        self.post_attention_norm = nn.RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention_output, next_cache = self.attention(
            self.input_norm(hidden_states), cache, use_cache=use_cache
        )
        hidden_states = hidden_states + attention_output
        normed = self.post_attention_norm(hidden_states)
        mlp_output = self.down_proj(
            silu(self.gate_proj(normed)) * self.up_proj(normed)
        )
        return hidden_states + mlp_output, next_cache


class _LowRankCorrectionExpert(nn.Module):
    """Low-rank expert that returns a hidden-sized correction residual."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        output_size: int,
        *,
        zero_init_output: bool,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, output_size, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(silu(self.down(hidden_states)))


class CausalCorrectionHead(nn.Module):
    """Predict a gated hidden or logit residual for the parallel DFlash output.

    Each block position combines the previous-token embedding, the current DFlash
    hidden state, and a learned block-position embedding.  Causal attention carries
    those features across the block.

    ``output_mode="hidden"`` preserves the baseline path: the returned residual is
    added to the DFlash hidden state before the model's single full LM-head
    projection. MoE hidden mode fuses an always-on shared expert and one Top-1
    expert after both return to the DFlash hidden width. MoE logits mode instead
    fuses them in a common low-rank space before one shared vocabulary projection.
    Detached previous-logit uncertainty statistics can optionally condition only
    the MoE router and residual gate. ``output_mode="logits"`` additionally
    encodes the previous position's target/generated logits and returns a low-rank
    vocabulary bias that is added to ordinary DFlash base logits, analogous to the
    Markov head. Optional
    auxiliary hidden output and corrected-hidden feedback let logit mode retain and
    propagate a trainable pre-LM representation without changing its token output
    path. An optional block memory is projected once per head invocation and
    broadcast across its causal slots. The caller may alternatively project that
    corrected hidden for the current token before adding the vocabulary residual.
    """

    def __init__(
        self,
        *,
        input_hidden_size: int,
        token_embedding_size: int,
        block_size: int,
        correction_hidden_size: int,
        correction_rank: int,
        num_layers: int = 1,
        num_heads: int = 8,
        gate_bias: float = 0.0,
        output_mode: Literal["hidden", "logits"] = "hidden",
        draft_vocab_size: int | None = None,
        enable_hidden_auxiliary: bool = False,
        enable_hidden_feedback: bool = False,
        block_memory_size: int | None = None,
        enable_moe: bool = False,
        moe_shared_rank: int = 128,
        moe_expert_rank: int = 64,
        moe_num_experts: int = 4,
        moe_logit_routing: bool = False,
    ) -> None:
        super().__init__()
        if correction_hidden_size <= 0:
            raise ValueError("correction_hidden_size must be positive")
        if correction_rank <= 0:
            raise ValueError("correction_rank must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if output_mode not in ("hidden", "logits"):
            raise ValueError(f"Unsupported correction output mode: {output_mode!r}")
        if (output_mode == "logits" or moe_logit_routing) and (
            draft_vocab_size is None or draft_vocab_size <= 0
        ):
            raise ValueError(
                "draft_vocab_size must be positive for logit-aware Correction"
            )
        if enable_moe and min(
            moe_shared_rank,
            moe_expert_rank,
            moe_num_experts,
        ) <= 0:
            raise ValueError("Correction MoE ranks and expert count must be positive")
        if moe_logit_routing and not enable_moe:
            raise ValueError("MoE logit routing requires Correction MoE")

        self.output_mode = output_mode
        self.draft_vocab_size = draft_vocab_size
        self.enable_hidden_feedback = enable_hidden_feedback
        self.enable_moe = enable_moe
        self.moe_num_experts = moe_num_experts if enable_moe else 0
        self.moe_logit_routing = moe_logit_routing
        self.hidden_proj = nn.Linear(
            input_hidden_size, correction_hidden_size, bias=False
        )
        self.token_proj = nn.Linear(
            token_embedding_size, correction_hidden_size, bias=False
        )
        self.position_embedding = nn.Embedding(block_size, correction_hidden_size)
        self.layers = nn.ModuleList(
            [
                _TinyCausalLayer(correction_hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.RMSNorm(correction_hidden_size)
        residual_rank = moe_shared_rank if enable_moe else correction_rank
        self.correction_down = nn.Linear(
            correction_hidden_size, residual_rank, bias=False
        )
        self.moe_shared_common_up: nn.Linear | None = None
        if enable_moe and output_mode == "logits":
            self.moe_shared_common_up = nn.Linear(
                moe_shared_rank, correction_rank, bias=False
            )
        correction_up_input_size = (
            correction_rank
            if self.moe_shared_common_up is not None
            else residual_rank
        )
        self.correction_up = nn.Linear(
            correction_up_input_size,
            input_hidden_size if output_mode == "hidden" else int(draft_vocab_size),
            bias=False,
        )
        self.moe_router: nn.Linear | None = None
        self.moe_experts: nn.ModuleList | None = None
        self.moe_selected_scale: nn.Parameter | None = None
        self.moe_logit_stats_proj: nn.Linear | None = None
        self.moe_logit_router: nn.Linear | None = None
        self.moe_logit_gate: nn.Linear | None = None
        if enable_moe:
            expert_output_size = (
                input_hidden_size if output_mode == "hidden" else correction_rank
            )
            self.moe_router = nn.Linear(
                correction_hidden_size, moe_num_experts, bias=True
            )
            self.moe_experts = nn.ModuleList(
                [
                    _LowRankCorrectionExpert(
                        correction_hidden_size,
                        moe_expert_rank,
                        expert_output_size,
                        zero_init_output=output_mode == "hidden",
                    )
                    for _ in range(moe_num_experts)
                ]
            )
            self.moe_selected_scale = nn.Parameter(torch.ones(()))
            if moe_logit_routing:
                logit_feature_size = 16
                self.moe_logit_stats_proj = nn.Linear(
                    3, logit_feature_size, bias=False
                )
                self.moe_logit_router = nn.Linear(
                    logit_feature_size, moe_num_experts, bias=False
                )
                self.moe_logit_gate = nn.Linear(
                    logit_feature_size, 1, bias=False
                )
        self.previous_logits_down: nn.Linear | None = None
        self.previous_logits_proj: nn.Linear | None = None
        if output_mode == "logits":
            # Markov-like W1/W2 factorization around the causal head:
            #   previous probs[V] -> rank -> correction state -> rank -> bias[V].
            # Keeping W1 separate from the zero-initialized output W2 makes the
            # previous-logit feature available from the first training step.
            self.previous_logits_down = nn.Linear(
                int(draft_vocab_size), correction_rank, bias=False
            )
            self.previous_logits_proj = nn.Linear(
                correction_rank, correction_hidden_size, bias=False
            )
        self.hidden_feedback_proj: nn.Linear | None = None
        if enable_hidden_feedback:
            self.hidden_feedback_proj = nn.Linear(
                input_hidden_size, correction_hidden_size, bias=False
            )
        self.block_memory_proj: nn.Linear | None = None
        if block_memory_size is not None:
            if block_memory_size <= 0:
                raise ValueError("block_memory_size must be positive")
            self.block_memory_proj = nn.Linear(
                block_memory_size, correction_hidden_size, bias=False
            )
        self.auxiliary_hidden_up: nn.Linear | None = None
        self.auxiliary_hidden_gate: nn.Linear | None = None
        if output_mode == "logits" and (
            enable_hidden_auxiliary or enable_hidden_feedback
        ):
            self.auxiliary_hidden_up = nn.Linear(
                correction_rank, input_hidden_size, bias=False
            )
            self.auxiliary_hidden_gate = nn.Linear(correction_hidden_size, 1)
        self.residual_gate = nn.Linear(correction_hidden_size, 1)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.correction_up.weight)
        if self.moe_router is not None:
            nn.init.normal_(self.moe_router.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.moe_router.bias)
        if self.moe_logit_router is not None:
            nn.init.zeros_(self.moe_logit_router.weight)
        if self.moe_logit_gate is not None:
            nn.init.zeros_(self.moe_logit_gate.weight)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, gate_bias)
        if self.auxiliary_hidden_up is not None:
            nn.init.zeros_(self.auxiliary_hidden_up.weight)
        if self.auxiliary_hidden_gate is not None:
            nn.init.zeros_(self.auxiliary_hidden_gate.weight)
            nn.init.constant_(self.auxiliary_hidden_gate.bias, gate_bias)

    def _moe_logit_features(
        self,
        previous_logits: torch.Tensor | None,
        previous_logits_mask: torch.Tensor | None,
        previous_probs: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Encode detached distribution uncertainty without a vocabulary map."""
        if not self.moe_logit_routing:
            return None
        if (
            previous_logits is None
            or previous_logits_mask is None
            or self.moe_logit_stats_proj is None
        ):
            raise RuntimeError("MoE logit routing requires previous logits and mask")

        if previous_probs is None:
            log_probs = torch.log_softmax(
                previous_logits.detach().float(), dim=-1
            )
            probs = log_probs.exp()
        else:
            probs = previous_probs.detach().float()
            log_probs = probs.clamp_min(1e-9).log()
        top2 = probs.topk(k=2, dim=-1).values
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = entropy / torch.log(
            probs.new_tensor(float(previous_logits.shape[-1]))
        )
        stats = torch.stack(
            [entropy, top2[..., 0], top2[..., 0] - top2[..., 1]], dim=-1
        )
        stats = stats * previous_logits_mask.to(stats.dtype).unsqueeze(-1)
        return silu(
            self.moe_logit_stats_proj(
                stats.to(self.moe_logit_stats_proj.weight.dtype)
            )
        )

    def _moe_router_logits(
        self,
        causal_states: torch.Tensor,
        logit_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.moe_router is None:
            raise RuntimeError("Correction MoE is not enabled")
        router_logits = self.moe_router(causal_states)
        if self.moe_logit_routing:
            if logit_features is None or self.moe_logit_router is None:
                raise RuntimeError("MoE logit routing features are missing")
            router_logits = router_logits + self.moe_logit_router(logit_features)
        return router_logits

    def _selected_expert_residual(
        self,
        causal_states: torch.Tensor,
        logit_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch every token to one selected expert and restore its shape."""
        if (
            self.moe_router is None
            or self.moe_experts is None
            or self.moe_selected_scale is None
        ):
            raise RuntimeError("Correction MoE is not enabled")

        router_probs = torch.softmax(
            self._moe_router_logits(causal_states, logit_features).float(), dim=-1
        )
        selected_ids = router_probs.argmax(dim=-1)
        selected_probs = router_probs.gather(
            -1, selected_ids.unsqueeze(-1)
        ).squeeze(-1)
        flat_states = causal_states.reshape(-1, causal_states.shape[-1])
        flat_ids = selected_ids.reshape(-1)
        flat_probs = selected_probs.reshape(-1)
        flat_output = causal_states.new_zeros(
            flat_states.shape[0], self.moe_experts[0].up.out_features
        )
        for expert_id, expert in enumerate(self.moe_experts):
            token_indices = torch.nonzero(
                flat_ids == expert_id, as_tuple=False
            ).flatten()
            expert_output = expert(flat_states.index_select(0, token_indices))
            expert_output = expert_output * flat_probs.index_select(
                0, token_indices
            ).to(expert_output.dtype).unsqueeze(-1)
            flat_output = flat_output.index_copy(
                0, token_indices, expert_output.to(flat_output.dtype)
            )
        return flat_output.view(
            *causal_states.shape[:-1], self.moe_experts[0].up.out_features
        )

    def _fused_moe_output(
        self,
        causal_states: torch.Tensor,
        logit_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fuse shared and Top-1 experts in hidden or common-rank space."""
        if not self.enable_moe or self.moe_selected_scale is None:
            raise RuntimeError("Correction MoE is not enabled")
        shared_rank = silu(self.correction_down(causal_states))
        if self.output_mode == "logits":
            if self.moe_shared_common_up is None:
                raise RuntimeError("Logit MoE common projection is missing")
            shared = self.moe_shared_common_up(shared_rank)
        else:
            shared = self.correction_up(shared_rank)
        return shared + self.moe_selected_scale.to(shared.dtype) * (
            self._selected_expert_residual(causal_states, logit_features)
        )

    def _residual_from_causal_states(
        self,
        causal_states: torch.Tensor,
        logit_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.enable_moe:
            residual = self._fused_moe_output(causal_states, logit_features)
        else:
            residual = self.correction_up(
                silu(self.correction_down(causal_states))
            )
        gate_logits = self.residual_gate(causal_states)
        if self.moe_logit_routing:
            if logit_features is None or self.moe_logit_gate is None:
                raise RuntimeError("MoE logit routing features are missing")
            gate_logits = gate_logits + self.moe_logit_gate(logit_features)
        residual = residual * torch.sigmoid(gate_logits)
        if self.enable_moe and self.output_mode == "logits":
            residual = self.correction_up(residual)
        return residual

    def moe_router_statistics(
        self,
        causal_states: torch.Tensor,
        valid_mask: torch.Tensor,
        previous_logits: torch.Tensor | None = None,
        previous_logits_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Switch-style load-balance loss and normalized entropy."""
        if self.moe_router is None:
            raise RuntimeError("Correction MoE is not enabled")
        if valid_mask.shape != causal_states.shape[:-1]:
            raise ValueError("MoE valid mask and causal states must align")

        logit_features = self._moe_logit_features(
            previous_logits, previous_logits_mask
        )
        router_probs = torch.softmax(
            self._moe_router_logits(causal_states, logit_features).float(), dim=-1
        ).reshape(-1, self.moe_num_experts)
        selected_ids = router_probs.argmax(dim=-1)
        mask = valid_mask.reshape(-1).to(router_probs.dtype)
        normalizer = mask.sum().clamp_min(1.0)
        importance = (router_probs * mask.unsqueeze(-1)).sum(dim=0) / normalizer
        load = router_probs.new_zeros(self.moe_num_experts)
        load.scatter_add_(0, selected_ids, mask)
        load = load / normalizer
        balance_loss = self.moe_num_experts * (
            importance * load.detach()
        ).sum()
        entropy = -(
            router_probs.clamp_min(1e-9).log() * router_probs
        ).sum(dim=-1)
        entropy = (entropy * mask).sum() / normalizer
        if self.moe_num_experts > 1:
            entropy = entropy / torch.log(
                router_probs.new_tensor(float(self.moe_num_experts))
            )
        return balance_loss, entropy

    def auxiliary_hidden_residual(
        self,
        causal_states: torch.Tensor,
        previous_logits: torch.Tensor | None = None,
        previous_logits_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project Correction states to an auxiliary DFlash-hidden residual."""
        logit_features = self._moe_logit_features(
            previous_logits, previous_logits_mask
        )
        if self.output_mode == "hidden":
            return self._residual_from_causal_states(
                causal_states, logit_features
            )
        if self.auxiliary_hidden_up is None or self.auxiliary_hidden_gate is None:
            raise RuntimeError(
                "Auxiliary hidden residual was not enabled for logit Correction"
            )
        if self.enable_moe:
            common_states = self._fused_moe_output(
                causal_states, logit_features
            )
        else:
            common_states = silu(self.correction_down(causal_states))
        delta_hidden = self.auxiliary_hidden_up(common_states)
        return delta_hidden * torch.sigmoid(
            self.auxiliary_hidden_gate(causal_states)
        )

    def forward(
        self,
        previous_token_embeddings: torch.Tensor,
        dflash_hidden: torch.Tensor,
        block_positions: torch.Tensor,
        previous_logits: torch.Tensor | None = None,
        previous_logits_mask: torch.Tensor | None = None,
        previous_corrected_hidden: torch.Tensor | None = None,
        previous_corrected_hidden_mask: torch.Tensor | None = None,
        block_memory: torch.Tensor | None = None,
        cache: CorrectionCache | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, CorrectionCache | None]:
        """Return ``(delta, causal_states, next_cache)``.

        ``delta`` is hidden-sized in hidden mode and draft-vocabulary-sized in
        logits mode. The latter requires previous logits plus a validity mask so
        block position zero can use an explicit no-history feature.
        """
        prefix_shape = dflash_hidden.shape[:-1]
        if previous_token_embeddings.shape[:-1] != prefix_shape:
            raise ValueError("previous-token embeddings and DFlash hidden must align")
        if block_positions.shape != prefix_shape:
            raise ValueError("block positions and DFlash hidden must align")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, got {len(cache)}"
            )
        if self.block_memory_proj is None:
            if block_memory is not None:
                raise ValueError(
                    "block memory is only valid when cross-block memory is enabled"
                )
        else:
            if block_memory is None:
                raise ValueError(
                    "cross-block-memory Correction requires a block memory"
                )
            expected_memory_shape = (
                dflash_hidden.shape[0],
                self.block_memory_proj.in_features,
            )
            if block_memory.shape != expected_memory_shape:
                raise ValueError(
                    "Expected block memory shape "
                    f"{expected_memory_shape}, got {tuple(block_memory.shape)}"
                )
        if self.hidden_feedback_proj is None:
            if (
                previous_corrected_hidden is not None
                or previous_corrected_hidden_mask is not None
            ):
                raise ValueError(
                    "previous corrected hidden is only valid when hidden feedback "
                    "is enabled"
                )
        else:
            if (
                previous_corrected_hidden is None
                or previous_corrected_hidden_mask is None
            ):
                raise ValueError(
                    "hidden-feedback Correction requires previous corrected hidden "
                    "and mask"
                )
            if previous_corrected_hidden.shape != dflash_hidden.shape:
                raise ValueError(
                    "previous corrected hidden and DFlash hidden must align"
                )
            if previous_corrected_hidden_mask.shape != prefix_shape:
                raise ValueError(
                    "previous corrected hidden mask and DFlash hidden must align"
                )
        needs_previous_logits = self.output_mode == "logits" or (
            self.moe_logit_routing
        )
        if not needs_previous_logits:
            if previous_logits is not None or previous_logits_mask is not None:
                raise ValueError(
                    "previous logits require logit-residual or MoE routing mode"
                )
        else:
            if previous_logits is None or previous_logits_mask is None:
                raise ValueError(
                    "logit-aware Correction requires previous logits and mask"
                )
            expected_logits_shape = (*prefix_shape, int(self.draft_vocab_size))
            if previous_logits.shape != expected_logits_shape:
                raise ValueError(
                    "Expected previous logits shape "
                    f"{expected_logits_shape}, got {tuple(previous_logits.shape)}"
                )
            if previous_logits_mask.shape != prefix_shape:
                raise ValueError(
                    "previous logits mask and DFlash hidden must align"
                )

        dtype = self.hidden_proj.weight.dtype
        hidden_states = (
            self.hidden_proj(dflash_hidden.to(dtype))
            + self.token_proj(previous_token_embeddings.to(dtype))
            + self.position_embedding(
                block_positions.to(device=dflash_hidden.device, dtype=torch.long)
            ).to(dtype)
        )
        if self.block_memory_proj is not None:
            assert block_memory is not None  # noqa: S101
            hidden_states = hidden_states + self.block_memory_proj(
                block_memory.to(dtype)
            ).unsqueeze(1)
        previous_probs = None
        if self.output_mode == "logits":
            assert previous_logits is not None  # noqa: S101
            assert previous_logits_mask is not None  # noqa: S101
            assert self.previous_logits_down is not None  # noqa: S101
            assert self.previous_logits_proj is not None  # noqa: S101
            previous_probs = torch.softmax(previous_logits.float(), dim=-1)
            previous_rank = self.previous_logits_down(previous_probs.to(dtype))
            previous_rank = previous_rank * previous_logits_mask.to(
                device=dflash_hidden.device, dtype=dtype
            ).unsqueeze(-1)
            hidden_states = hidden_states + self.previous_logits_proj(previous_rank)
        if self.hidden_feedback_proj is not None:
            assert previous_corrected_hidden is not None  # noqa: S101
            assert previous_corrected_hidden_mask is not None  # noqa: S101
            feedback_hidden = previous_corrected_hidden.to(dtype)
            feedback_hidden = feedback_hidden * previous_corrected_hidden_mask.to(
                device=dflash_hidden.device, dtype=dtype
            ).unsqueeze(-1)
            hidden_states = hidden_states + self.hidden_feedback_proj(feedback_hidden)

        next_cache: CorrectionCache = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[layer_idx]
            hidden_states, next_layer_cache = layer(
                hidden_states, layer_cache, use_cache=use_cache
            )
            if use_cache:
                assert next_layer_cache is not None  # noqa: S101
                next_cache.append(next_layer_cache)

        causal_states = self.output_norm(hidden_states)
        logit_features = self._moe_logit_features(
            previous_logits,
            previous_logits_mask,
            previous_probs=previous_probs,
        )
        delta = self._residual_from_causal_states(causal_states, logit_features)
        return delta, causal_states, next_cache if use_cache else None


class MarkovHead(nn.Module):
    """Low-rank sequential logit bias ``B = W1 @ W2``.

    ``W1`` indexes the verifier vocabulary (the previous token id); ``W2`` projects
    to the draft vocabulary so the bias adds onto the DFlash logits.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        head_type: str = "vanilla",
    ) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {markov_rank}")
        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        self.head_type = head_type
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        elif head_type == "rnn":
            # Joint [gate; candidate; output] projection over [state; prev_emb; hidden].
            self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up W1 embeddings for the given previous-token ids."""
        return self.markov_w1(token_ids.long())

    def block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [N, block_size]
        hidden_states: torch.Tensor,  # [N, block_size, hidden]
        prev_emb: torch.Tensor | None = None,  # [N, block_size, r]
    ) -> torch.Tensor:
        """Return the per-position logit bias, shape [N, block_size, draft_vocab]."""
        if prev_emb is None:
            prev_emb = self.prev_embeddings(prev_token_ids)
        prev_emb = prev_emb.to(self.markov_w2.weight.dtype)

        if self.head_type == "vanilla":
            return self.markov_w2(prev_emb)

        if self.head_type == "gated":
            hidden_states = hidden_states.to(prev_emb.dtype)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
            )
            return self.markov_w2(gate * prev_emb)

        # rnn: maintain a recurrent state across block positions.
        hidden_states = hidden_states.to(prev_emb.dtype)
        num_blocks, block_size, _ = prev_emb.shape
        state = prev_emb.new_zeros(num_blocks, self.markov_rank)
        outputs = []
        for k in range(block_size):
            z = torch.cat([state, prev_emb[:, k], hidden_states[:, k]], dim=-1)
            gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
            outputs.append(self.markov_w2(torch.tanh(out_raw)))
        return torch.stack(outputs, dim=1)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)
