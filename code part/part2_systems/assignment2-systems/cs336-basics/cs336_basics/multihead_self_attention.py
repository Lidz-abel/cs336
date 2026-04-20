from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from cs336_basics.linear import linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    input_dim = in_features.shape[-1]
    q_total_dim = q_proj_weight.shape[0]
    k_total_dim = k_proj_weight.shape[0]
    v_total_dim = v_proj_weight.shape[0]

    if q_total_dim % num_heads != 0:
        raise ValueError("Q projection dimension must be divisible by num_heads")
    if k_total_dim % num_heads != 0:
        raise ValueError("K projection dimension must be divisible by num_heads")
    if v_total_dim % num_heads != 0:
        raise ValueError("V projection dimension must be divisible by num_heads")

    q = linear(d_in=input_dim, d_out=q_total_dim, weights=q_proj_weight, in_features=in_features)
    k = linear(d_in=input_dim, d_out=k_total_dim, weights=k_proj_weight, in_features=in_features)
    v = linear(d_in=input_dim, d_out=v_total_dim, weights=v_proj_weight, in_features=in_features)

    q_head_dim = q_total_dim // num_heads
    k_head_dim = k_total_dim // num_heads
    v_head_dim = v_total_dim // num_heads

    q = q.reshape(*q.shape[:-1], num_heads, q_head_dim).transpose(-3, -2)
    k = k.reshape(*k.shape[:-1], num_heads, k_head_dim).transpose(-3, -2)
    v = v.reshape(*v.shape[:-1], num_heads, v_head_dim).transpose(-3, -2)

    seq_len = in_features.shape[-2]
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )

    attention_output = scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal_mask)
    attention_output = attention_output.transpose(-3, -2).reshape(*in_features.shape[:-1], v_total_dim)

    return linear(
        d_in=v_total_dim,
        d_out=d_model,
        weights=o_proj_weight,
        in_features=attention_output,
    )
