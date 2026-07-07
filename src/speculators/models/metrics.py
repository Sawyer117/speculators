from collections.abc import Callable

import torch

_EPS = 1e-5


def compute_accuracy_single_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor | None,  # shape: [1, seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, seq_len]
):
    """Compute full and conditional accuracy counts for a single speculative step.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: If provided, restricts accuracy to masked positions.
        prev_correct: Boolean mask of positions correct so far. Updated in place
            via logical AND with the current step's correctness.

    Returns:
        Tuple of (full_correct, full_total, cond_correct, cond_total) as raw
        counts suitable for distributed reduction before computing ratios.
    """
    correct = pred_ids == target_ids
    cond_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)
    if prev_correct is not None:
        cond_total = prev_correct.sum().float()
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))

    correct_sum = correct.float().sum()
    full_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)

    return correct_sum, full_total, correct_sum, cond_total


@torch.no_grad()
def compute_accuracy_multi_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    num_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-position correct/total counts across multiple speculative steps.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: Boolean mask selecting positions to evaluate.
        pos_idx: Position index within each speculative block (e.g. 0,1,2,3,0,1,2,3).
        num_pos: Number of distinct positions (i.e. block size).

    Returns:
        Tuple of (correct_per_pos, total_per_pos) both with shape [num_pos].
        Overall counts can be derived by summing these.
    """
    correct = pred_ids == target_ids
    correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    pos_idx = torch.masked_select(pos_idx, loss_mask.to(torch.bool))

    correct_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    total_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    correct_per_pos.scatter_add_(0, pos_idx, correct.float())
    total_per_pos.scatter_add_(0, pos_idx, torch.ones_like(correct, dtype=torch.float))

    return correct_per_pos, total_per_pos  # shape: [num_pos], [num_pos]


def kl_div_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position KL divergence from draft logits to target logits.

    Args:
        logits: Draft model logits (log-softmax applied internally).
        targets: Target model logits (softmax applied internally).

    Returns:
        Per-position KL divergence with shape [1, seq_len].
    """
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    elementwise_loss = torch.nn.functional.kl_div(
        logits, target_p, reduction="none", log_target=False
    ).sum(dim=-1)  # shape: [1, seq_len]

    return elementwise_loss  # noqa: RET504


def ce_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position cross-entropy loss using argmax of target logits as labels.

    Args:
        logits: Draft model logits.
        targets: Target model logits (argmax taken to produce hard labels).

    Returns:
        Per-position cross-entropy loss with shape [1, seq_len].
    """
    batch_size, seq_len, draft_vocab_size = logits.shape
    target_ids = torch.argmax(targets, dim=-1)  # shape: [1, seq_len]

    elementwise_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, draft_vocab_size),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)

    return elementwise_loss  # noqa: RET504


def tv_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position total variation (TV) distance from draft to target.

    The rejection-sampling acceptance rate of speculative decoding equals the
    distributional overlap between target and draft,
    ``alpha = sum_v min(p_v, q_v) = 1 - d_TV(p, q)``. Minimizing this TV distance
    therefore directly optimizes the acceptance rate, whereas cross-entropy and
    KL only optimize it indirectly (KL is a loose upper bound on TV via Pinsker).

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position TV distance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # shape: [1, seq_len]
    elementwise_loss = 1.0 - overlap

    return elementwise_loss  # noqa: RET504


def neg_log_acceptance_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position negative log-acceptance (LK) loss.

    The speculative-decoding acceptance rate equals the draft/target distribution
    overlap, ``alpha = sum_v min(p_v, q_v)`` (the same quantity computed in
    ``tv_loss``). This loss is ``-log(alpha)``. Its gradient is
    ``(1 / alpha) * grad(TV)``: the ``1 / alpha`` factor amplifies the otherwise
    vanishing TV gradient when overlap is low (early training), giving TV's
    acceptance-optimal target a usable gradient from a cold start. When the target
    is a point mass, this loss reduces to cross-entropy.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position negative log-acceptance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # alpha, shape: [1, seq_len]
    elementwise_loss = -torch.log(overlap.clamp_min(_EPS))

    return elementwise_loss  # noqa: RET504


def confidence_loss(
    confidence_logits: torch.Tensor,  # shape: [1, seq_len]
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Per-position BCE loss for the DSpark accept-rate (confidence) head.

    The DSpark draft carries an ``AcceptRatePredictor`` (a ``Linear(hidden, 1)``)
    that predicts, from each draft position's hidden state, the probability the
    drafted token is accepted by the verifier. It is trained (BCE-with-logits) to
    regress the **soft acceptance rate** ``alpha = sum_v min(q_v, p_v) = 1 - d_TV``
    — the rejection-sampling accept probability, NOT a hard argmax match. The target
    is detached: the head learns to *predict* acceptance, it must not move the draft
    distribution.

    Mirrors DeepSpec ``deepspec/modeling/dspark/loss.py`` (``_compute_accept_rate_3d``
    forms ``alpha`` then ``F.binary_cross_entropy_with_logits`` against the detached
    ``alpha``). Cross-checked against that source.

    Args:
        confidence_logits: Raw (pre-sigmoid) accept-rate head output [1, seq_len].
        logits: Draft model logits [1, seq_len, V].
        targets: Verifier logits in draft-vocab space [1, seq_len, V].

    Returns:
        Per-position BCE loss with shape [1, seq_len].
    """
    with torch.no_grad():
        draft_p = torch.nn.functional.softmax(logits.float(), dim=-1)
        target_p = torch.nn.functional.softmax(targets.float(), dim=-1)
        accept_rate = (
            torch.minimum(draft_p, target_p).sum(dim=-1).clamp_(0.0, 1.0)
        )  # alpha = overlap = 1 - d_TV, shape: [1, seq_len]
    elementwise_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence_logits.float(), accept_rate, reduction="none"
    )  # shape: [1, seq_len]

    return elementwise_loss  # noqa: RET504


def l1_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Per-position L1 distance ``sum_v |q_v - p_v|`` between draft and target.

    This is the DSpark "l1" term (``deepspec/.../loss.py:_compute_local_l1_term``).
    Note it is exactly **twice** the total-variation distance — :func:`tv_loss`
    returns ``1 - overlap == 0.5 * L1`` — so DSpark's ``l1_loss_alpha`` (default 0.9)
    weights this full-L1 form directly.

    Returns:
        Per-position L1 distance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    elementwise_loss = (draft_p - target_p).abs().sum(dim=-1)  # shape: [1, seq_len]

    return elementwise_loss  # noqa: RET504


def combo_ce_l1_loss(ce_alpha: float = 0.1, l1_alpha: float = 0.9):
    """Return a weighted ``ce_alpha * CE + l1_alpha * L1`` per-position loss fn.

    Reproduces the DSpark distribution term (DeepSpec defaults ``0.1 * CE + 0.9 * L1``
    with L1 the full ``sum_v |q_v - p_v|`` from :func:`l1_loss`). ``speculators`` ships
    CE and TV (== L1/2) as *separate* selectable losses; this factory returns a single
    ``(logits, targets) -> [1, seq_len]`` callable so it drops into
    :func:`loss_function` unchanged (masking + decay applied there).

    Args:
        ce_alpha: Weight on the cross-entropy term (DSpark default 0.1).
        l1_alpha: Weight on the L1 term (DSpark default 0.9).

    Returns:
        A per-position loss callable ``(logits, targets) -> [1, seq_len]``.
    """

    def _combo(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ce_alpha * ce_loss(logits, targets) + l1_alpha * l1_loss(logits, targets)

    return _combo


def dflash_loss_decay(pos_idx: torch.Tensor, gamma: float):
    """Compute DFlash-style exponential decay weights per position.

    Position 0 gets weight 0, position 1 gets weight 1, and subsequent positions
    decay as exp(-(pos - 1) / gamma).

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Decay rate (higher = slower decay).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    # pos_idx = 0 1 2 3 0 1 2 3, block_size = 4
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    # decay_mult = e^-(0 0 1 2 0 0 1 2) / gamma
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    # w = 0 1 e^-1/gamma e^-2/gamma 0 1 e^-1/gamma e^-2/gamma
    return decay_mult  # noqa: RET504


def exp_loss_decay(pos_idx: torch.Tensor, gamma: float):
    """Compute simple exponential decay weights as gamma^pos_idx.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Base of the exponent (typically in (0, 1]).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    return gamma**pos_idx


def resolve_loss_fn(
    name: str,
) -> "Callable[[torch.Tensor, torch.Tensor], torch.Tensor]":
    """Resolves a loss function given its abbreviated name.

    Args:
        name: ``"kl_div"`` for KL-divergence, ``"ce"`` for cross-entropy,
            ``"tv"`` for total variation, or ``"nla"`` for negative
            log-acceptance.

    Returns:
        The corresponding loss function.

    Raises:
        ValueError: If *name* is not a recognised loss function.
    """
    loss_fn_map: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "kl_div": kl_div_loss,
        "ce": ce_loss,
        "tv": tv_loss,
        "nla": neg_log_acceptance_loss,
    }
    if name not in loss_fn_map:
        raise ValueError(
            f"Unknown loss function '{name}'. Choose from: {sorted(loss_fn_map.keys())}"
        )
    return loss_fn_map[name]


def loss_function(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    """Compute masked, optionally position-decayed training loss.

    Args:
        logits: Draft model logits.
        targets: Target model logits.
        loss_mask: Boolean mask selecting positions to include in the loss.
        pos_idx: Position indices within each speculative block.
        loss_fn: Per-position loss function (default: kl_div_loss).
        decay_fn: Optional position-dependent decay weighting function.

    Returns:
        Scalar mean loss across the batch.
    """
    elementwise_loss = loss_fn(logits, targets)  # shape: [1, seq_len]

    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if decay_fn is not None:
        decay_mult = decay_fn(pos_idx.to(elementwise_loss.dtype))
        elementwise_loss = elementwise_loss * decay_mult

    denominator = loss_mask.sum(dim=1) + _EPS

    batch_loss = torch.sum(elementwise_loss, dim=1) / denominator  # shape: [1]
    return batch_loss.mean()  # shape: []
