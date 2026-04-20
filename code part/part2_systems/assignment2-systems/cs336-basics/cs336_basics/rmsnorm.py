from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    if weights.shape != (d_model,):
        raise ValueError(f"Expected weights shape {(d_model,)}, got {tuple(weights.shape)}")
    if in_features.shape[-1] != d_model:
        raise ValueError(f"Expected input last dimension {d_model}, got {in_features.shape[-1]}")

    rms = torch.sqrt(torch.mean(in_features * in_features, dim=-1, keepdim=True) + eps)
    normalized = in_features / rms
    return normalized * weights
