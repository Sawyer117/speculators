"""The DeepSeek-V4-Flash DSpark draft model (faithful, teacher-forced training).

A semi-autoregressive drafter: from a target-hidden context it emits a block of
``block_size`` (gamma) token distributions in one forward. Structure (released
``inference/model.py``, clean-room):

* ``main_proj`` / ``main_norm`` project the target's [40,41,42] hidden into the
  draft residual (``main_x``, the sliding-window context the blocks attend to);
* ``block_size`` draft tokens ``[anchor, noise×(gamma-1)]`` are embedded and
  hyper-connection-expanded, then passed through ``n_draft_layers`` MoE blocks
  conditioned on ``main_x``;
* a Markov low-rank head adds an intra-block logit bias from the previous token,
  and a confidence head predicts each position's acceptance probability.

The embedding and lm_head are the **frozen target** ones (shared, loaded from
the verifier). Training is teacher-forced on cached target hidden states: the
Markov head is fed the ground-truth previous tokens (not samples), and block
position ``i`` predicts target token ``i+1``. Sampling (inference-parity) reuses
the same block-logit core; the loss lives in :mod:`.loss`.
"""
from __future__ import annotations

import torch
from torch import nn

from .backbone.block import MhcDecoderBlock
from .backbone.hyper import HyperHead
from .backbone.norm import RMSNorm
from .backbone.rotary import precompute_freqs_cis
from .config import DSparkDraftConfig


class MarkovHead(nn.Module):
    """Low-rank Markov logit-bias head.

    ``markov_w1`` embeds the previous token to a rank-``r`` vector; ``markov_w2``
    projects it back to a vocab-size logit bias. Returns ``(bias, embed)`` — the
    embed is also concatenated into the confidence-head input.
    """

    def __init__(self, cfg: DSparkDraftConfig) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(cfg.vocab_size, cfg.markov_rank)
        self.markov_w2 = nn.Linear(cfg.markov_rank, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.markov_w1(token_ids)
        return self.markov_w2(embed), embed


class ConfidenceHead(nn.Module):
    """Per-position acceptance predictor: ``Linear(dim + rank, 1)`` (fp32)."""

    def __init__(self, cfg: DSparkDraftConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.hidden_size + cfg.markov_rank, 1, bias=False)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([hidden, markov_embed], dim=-1).float()
        return self.proj(feats).squeeze(-1)


class DSparkDraftModel(nn.Module):
    """Faithful DSV4 DSpark draft (block-gamma, teacher-forced)."""

    def __init__(self, cfg: DSparkDraftConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hc_mult = cfg.hc_mult
        self.block_size = cfg.block_size

        # Frozen, shared with the target (loaded from the verifier).
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Target-hidden conditioning (stage 0).
        self.main_proj = nn.Linear(cfg.hidden_size * cfg.num_target_layers, cfg.hidden_size, bias=False)
        self.main_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.layers = nn.ModuleList(MhcDecoderBlock(cfg) for _ in range(cfg.n_draft_layers))

        # Output head (stage last).
        self.hc_head = HyperHead(cfg)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.markov_head = MarkovHead(cfg)
        self.confidence_head = ConfidenceHead(cfg)

        # Plain RoPE (YaRN off on the sliding path); enough positions for a
        # window + block. Rebuilt lazily if a longer window is requested.
        self._rope_len = cfg.window_size + cfg.block_size
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                cfg.rope_head_dim, self._rope_len, 0, cfg.rope_theta,
                cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
            ),
            persistent=False,
        )

    def freeze_target_weights(self) -> None:
        """Freeze the shared target embed + lm_head (never trained)."""
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)

    def _freqs(self, n: int, offset: int, device) -> torch.Tensor:
        if offset + n > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(
                self.cfg.rope_head_dim, offset + n, 0, self.cfg.rope_theta,
                self.cfg.rope_factor, self.cfg.beta_fast, self.cfg.beta_slow,
            ).to(self.freqs_cis.device)
        return self.freqs_cis[offset : offset + n].to(device)

    def context(self, context_main_hidden: torch.Tensor) -> torch.Tensor:
        """target [N, W, num_target*dim] -> main_x context [N, W, dim]."""
        return self.main_norm(self.main_proj(context_main_hidden))

    def forward(
        self,
        context_main_hidden: torch.Tensor,
        block_input_ids: torch.Tensor,
        markov_ids: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced block-gamma forward.

        Args:
            context_main_hidden: ``[N, W, num_target_layers*dim]`` — target
                [40,41,42] hidden over the anchor's context window.
            block_input_ids: ``[N, gamma]`` — decoder input tokens
                ``[anchor, noise×(gamma-1)]``.
            markov_ids: ``[N, gamma]`` — teacher tokens fed to the Markov head
                (``[anchor, true_{p+1}, …, true_{p+gamma-1}]``).
            attn_bias: optional ``[N, gamma, W+gamma]`` additive mask.

        Returns ``(block_logits [N, gamma, vocab], confidence [N, gamma],
        hidden [N, gamma, dim])``.
        """
        n, w, _ = context_main_hidden.shape
        gamma = self.block_size
        device = block_input_ids.device

        context_x = self.context(context_main_hidden)  # [N, W, dim]
        context_freqs = self._freqs(w, 0, device)
        block_freqs = self._freqs(gamma, w, device)

        x = self.embed_tokens(block_input_ids)  # [N, gamma, dim]
        streams = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [N, gamma, hc, dim]
        for layer in self.layers:
            streams = layer(streams, context_x, block_freqs, context_freqs, attn_bias)

        hidden = self.hc_head(streams)  # [N, gamma, dim]
        base_logits = self.lm_head(self.norm(hidden))  # [N, gamma, vocab]

        markov_bias, markov_embed = self.markov_head(markov_ids)  # [N, gamma, vocab], [N, gamma, rank]
        logits = base_logits + markov_bias
        confidence = self.confidence_head(hidden, markov_embed)  # [N, gamma]
        return logits, confidence, hidden
