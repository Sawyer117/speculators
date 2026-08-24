"""Sequential (Markov) and confidence heads for the DSpark draft model."""

import torch
from torch import nn

__all__ = [
    "ConfidenceHead",
    "MarkovHead",
]


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
        if head_type not in ("vanilla", "gated", "rnn", "dflash2"):
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
        elif head_type == "dflash2":
            # The released DFlash2 selector's score, verbatim:
            #     S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>
            # (vllm-ascend #14533, `qwen3_dflash2.py::_score_edges`). Its
            # `predecessor_codebook` IS our markov_w1 and its `successor_codebook` IS our
            # markov_w2.weight -- same shapes, same roles -- so the only piece we lack is H.
            #
            # ★ WHY NOT REUSE "gated". That variant computes sigmoid(...) * prev_emb, so its
            # modulation lives in (0, 1): it can only ATTENUATE a dimension, never amplify it
            # and never flip its sign. Every gated bias is therefore a shrunk, sign-preserving
            # copy of the vanilla one. Since the head is a rank-`r` approximation of a V x V
            # transition matrix, a (0,1) mask can only SUBTRACT from the unconditional bigram
            # statistics already occupying that basis, whereas a signed linear scale makes each
            # context a genuinely different rank-`r` matrix rather than a masked subset of one.
            #
            # Initialised to the exact identity (W=0, b=1 => H(h) == 1), so at step 0 this head
            # computes the same bias as `vanilla`. That keeps a paired A/B clean from the very
            # first step, and on a warm start it means the gate cannot scramble an already
            # trained bias before it has learned anything. Symmetry is not a concern: each rank
            # dimension receives a distinct gradient through its own A/B columns.
            self.hidden_projection = nn.Linear(hidden_size, markov_rank)
            nn.init.zeros_(self.hidden_projection.weight)
            nn.init.ones_(self.hidden_projection.bias)

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

        if self.head_type == "dflash2":
            hidden_states = hidden_states.to(prev_emb.dtype)
            return self.markov_w2(prev_emb * self.hidden_projection(hidden_states))

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


class SelectHead(nn.Module):
    """An ADDITIVE, context-dependent transition term, kept separate from the Markov head.

        S_t(a, b) = U_t(b) + <A(a), B(b)>            <- MarkovHead, unconditional bigram
                           + <A'(a) * H'(h_t), B'(b)>  <- this, the selection term

    WHY SEPARATE RATHER THAN FOLDED INTO MarkovHead (which `markov_head_type="dflash2"`
    does, matching the released DFlash2 selector exactly):

      * It DEGRADES GRACEFULLY. Fused, a converter that drops H leaves the serve computing
        an unconditioned bias from weights trained WITH conditioning -- silently wrong, and
        precisely how the Correction-head experiment became unevaluable. Additive, dropping
        this head falls back to today's exact behaviour: no gain, but no error.
      * It IS the ablation. `logits = base + markov_bias + select_bias`; zeroing the last
        term reproduces today's model bit for bit, so "what is selection alone worth" is
        directly measurable. A fused head can never be split that way, because the two arms
        train different w1/w2 in the first place.
      * The serve patch can land BEFORE the weights exist -- the term is identically zero
        until trained, so the port is verifiable as a no-op.

    Measured motivation (a 5-epoch `ep5p0-ropefix` checkpoint): the Markov head's own
    predecessor codebook already runs at 99.6% effective rank, with 99% of the composed
    transition matrix's energy spread over 252 of its 256 dimensions. There is no idle block
    of directions for a modulation to borrow, which is the case for giving selection its own.

    Init is the exact identity: B' = 0 makes the whole term vanish at step 0, so a paired A/B
    starts from the same point. B' moves first and A' unfreezes once it is nonzero -- the
    standard zero-init-output pattern, as in LoRA and in the zero-initialised --dflash-* flags.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        select_rank: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if select_rank <= 0:
            raise ValueError(f"select_rank must be > 0, got {select_rank}")
        self.select_rank = select_rank
        self.select_w1 = nn.Embedding(verifier_vocab_size, select_rank)  # A'
        self.select_w2 = nn.Linear(select_rank, draft_vocab_size, bias=False)  # B'
        self.select_hidden = nn.Linear(hidden_size, select_rank)  # H'
        nn.init.zeros_(self.select_w2.weight)
        nn.init.zeros_(self.select_hidden.weight)
        nn.init.ones_(self.select_hidden.bias)

    def block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [N, block_size]
        hidden_states: torch.Tensor,  # [N, block_size, hidden]
    ) -> torch.Tensor:
        """Return the per-position selection bias, shape [N, block_size, draft_vocab]."""
        prev_emb = self.select_w1(prev_token_ids.long())
        hidden_states = hidden_states.to(prev_emb.dtype)
        return self.select_w2(prev_emb * self.select_hidden(hidden_states))


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit).

    ``bias`` is family-dependent, so it is a config knob rather than a constant. The
    released DSV4-Flash draft layout carries only ``confidence_head.proj.weight`` (see
    ``dsv4_dspark/checkpoint_mapping.py``) — hence the ``False`` default; the
    Qwen3 DSpark draft does carry a bias. vLLM's own ``DSparkConfidenceHead`` makes the
    same split: its DSV4 construction takes the ``bias=False`` default while the Qwen3
    one passes ``bias=True``. A bias the serving layout cannot represent trains fine but
    is silently dropped at conversion.
    """

    def __init__(self, input_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)
