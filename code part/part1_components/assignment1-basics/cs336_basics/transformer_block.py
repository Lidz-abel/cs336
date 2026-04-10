from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from cs336_basics.multihead_self_attention_with_rope import multihead_self_attention_with_rope
from cs336_basics.rmsnorm import rmsnorm
from cs336_basics.swiglu import swiglu


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    seq_len = in_features.shape[-2]
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0)

    x_norm = rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln1.weight"],
        in_features=in_features,
    )
    attn_out = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x_norm,
        token_positions=token_positions,
    )
    residual = in_features + attn_out

    residual_norm = rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln2.weight"],
        in_features=residual,
    )
    ff_out = swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=residual_norm,
    )

    return residual + ff_out
