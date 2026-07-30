"""Configuration for the generic sparse Mixture-of-Experts (MoE) draft layer.

A speculator draft whose FFN is an MoE declares its expert geometry here. The
fields are plain architectural hyper-parameters (expert count, activated top-k,
inner dim, scoring); nothing model-family specific lives in this dataclass, so
the same :class:`~speculators.models.moe.layer.MoE` builds a draft MoE for any
verifier whose distribution an MoE draft should match.
"""

from __future__ import annotations

from dataclasses import dataclass

# Supported router score functions (applied to the pre-bias logits).
SCORE_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus")


@dataclass
class MoEConfig:
    """Geometry of a routed + shared MoE FFN.

    ``n_routed_experts`` is the GLOBAL expert count. Under expert-parallelism a
    rank owns ``n_routed_experts // ep_size`` whole experts (see
    :mod:`speculators.train.expert_parallel`); without EP one rank holds them all.
    """

    hidden_size: int
    moe_inter_dim: int
    n_routed_experts: int
    n_activated_experts: int
    n_shared_experts: int = 1
    swiglu_limit: float = 0.0
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.score_func not in SCORE_FUNCS:
            raise ValueError(
                f"score_func must be one of {SCORE_FUNCS}, got {self.score_func!r}"
            )
        if self.n_activated_experts > self.n_routed_experts:
            raise ValueError(
                "n_activated_experts cannot exceed n_routed_experts "
                f"({self.n_activated_experts} > {self.n_routed_experts})"
            )
        if self.n_shared_experts != 1:
            # The stacked-weight checkpoint layout carries exactly one shared expert
            # (key ``shared_experts.{w1,w2,w3}``); >1 is not yet supported.
            raise ValueError("only n_shared_experts == 1 is supported")
