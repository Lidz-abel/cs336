from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_probs = softmax(scores, dim=-1)

    if mask is not None:
        attention_probs = attention_probs * mask.to(dtype=attention_probs.dtype)
        denom = attention_probs.sum(dim=-1, keepdim=True)
        attention_probs = torch.where(
            denom > 0,
            attention_probs / denom,
            torch.zeros_like(attention_probs),
        )

    return torch.matmul(attention_probs, V)
