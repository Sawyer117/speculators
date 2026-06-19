import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
):
    """
    Build a flex-attention mask mod where each query block corresponds to one anchor.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries in block j:
        - may attend to base tokens in the same document with
          position < anchor_positions[j]
        - may attend to all tokens in their own synthetic block j
        - may not attend to other synthetic blocks or later base tokens

    Args:
        document_ids: [total_seq_len] maps each position to its doc index, pad -1
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block
        sliding_window: integer size of sliding window or None for full attn
        sliding_window_non_causal: Use non causal mask for sliding window attn

    Returns:
        mask_mod, q_len, kv_len
    """
    # Always use non_causal for full attn
    non_causal = sliding_window is None or sliding_window_non_causal

    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    # For each query position, which anchor does it belong to?
    # query q in [j*block_size, (j+1)*block_size) belongs to anchor_positions[j]
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may see base-sequence tokens in the same document before the anchor.
        """
        # absolute base position
        q_anchor = query_anchor_positions[q_idx]
        # doc id for this query block
        q_doc = document_ids[q_anchor]

        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)  # safe indexing
        kv_doc = document_ids[kv_base_pos]

        same_doc = (q_doc == kv_doc) & (q_doc != -1)
        before_anchor = kv_base_pos < q_anchor

        in_window = (
            (kv_base_pos >= q_anchor - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def same_block_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may attend to tokens in their own synthetic block.
        Non-causal unless non_causal=False,
        in which case only prior positions are attended.
        """
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        same = kv_is_block & (q_block == kv_block)
        if not non_causal:
            same = same & (kv_idx <= q_idx + total_seq_len)
        return same

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len


def build_anchor_block_dense_mask(
    lengths: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    device: torch.device,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
) -> torch.Tensor:
    """Vectorized dense boolean version of ``create_anchor_block_mask_mod``.

    Returns a ``[1, 1, q_len, kv_len]`` boolean mask (True = attend) for use with
    SDPA / eager attention on backends where ``flex_attention`` is unavailable
    (e.g. Ascend NPU). The semantics exactly match the flex ``mask_mod``:
    ``base_prefix_mod`` OR ``same_block_mod``. On NPU, SDPA dispatches to a fused
    FlashAttention kernel, so attention scores are not materialized — only this
    mask (q_len x kv_len) is.
    """
    non_causal = sliding_window is None or sliding_window_non_causal
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long)
    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=device, dtype=torch.long),
        lengths.to(device),
    )
    if document_ids.numel() < total_seq_len:
        document_ids = torch.cat(
            [
                document_ids,
                -torch.ones(
                    total_seq_len - document_ids.numel(),
                    device=device,
                    dtype=torch.long,
                ),
            ]
        )

    query_anchor = torch.repeat_interleave(anchor_positions, block_size)  # [q_len]
    q_doc = document_ids[query_anchor].view(q_len, 1)  # [q_len, 1]
    q_idx = torch.arange(q_len, device=device)
    kv_idx = torch.arange(kv_len, device=device)
    q_block = (q_idx // block_size).view(q_len, 1)

    # base-prefix branch: same-document base tokens strictly before the anchor
    kv_is_base = (kv_idx < total_seq_len).view(1, kv_len)
    kv_base_pos = torch.clamp(kv_idx, max=total_seq_len - 1)
    kv_doc = document_ids[kv_base_pos].view(1, kv_len)
    same_doc = (q_doc == kv_doc) & (q_doc != -1)
    before_anchor = kv_base_pos.view(1, kv_len) < query_anchor.view(q_len, 1)
    if sliding_window is not None:
        in_window = kv_base_pos.view(1, kv_len) >= (
            query_anchor.view(q_len, 1) - sliding_window
        )
    else:
        in_window = True
    base = kv_is_base & same_doc & before_anchor & in_window

    # same-block branch: queries attend within their own synthetic block
    kv_is_block = (kv_idx >= total_seq_len).view(1, kv_len)
    kv_block = ((kv_idx - total_seq_len) // block_size).view(1, kv_len)
    block = kv_is_block & (q_block == kv_block)
    if not non_causal:
        block = block & (kv_idx.view(1, kv_len) <= (q_idx.view(q_len, 1) + total_seq_len))

    allow = base | block  # [q_len, kv_len]
    return allow.view(1, 1, q_len, kv_len)
