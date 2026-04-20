from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    del max_seq_len

    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")
    if in_query_or_key.shape[-1] != d_k:
        raise ValueError(f"Expected last dimension {d_k}, got {in_query_or_key.shape[-1]}")

    half_dim = d_k // 2
    freq_indices = torch.arange(half_dim, device=in_query_or_key.device, dtype=in_query_or_key.dtype)
    inv_freq = theta ** (-(2 * freq_indices) / d_k)

    positions = token_positions.to(device=in_query_or_key.device, dtype=in_query_or_key.dtype)
    while positions.ndim < in_query_or_key.ndim - 1:
        positions = positions.unsqueeze(-2)

    angles = positions.unsqueeze(-1) * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = in_query_or_key[..., ::2]
    x_odd = in_query_or_key[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack((rotated_even, rotated_odd), dim=-1).reshape_as(in_query_or_key)
