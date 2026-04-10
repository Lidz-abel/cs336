from __future__ import annotations

import torch

from jaxtyping import Float
from torch import Tensor

def linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor,"d_out d_in"],
    in_features: Float[Tensor,"... d_in"],
)->Float[Tensor, "... d_out"]:
    """
    Computes a linear transformation of the input features.

    Args:
        d_in: The number of input features.
        d_out: The number of output features.
        weights: A tensor of shape (d_out, d_in) containing the weights of the linear transformation.
        in_features: A tensor of shape (..., d_in) containing the input features.

    Returns:
        A tensor of shape (..., d_out) containing the output features.
    """
    if weights.shape!=(d_out,d_in):
        raise ValueError(f"Expected weights of shape {(d_out,d_in)},got{tuple(weights.shape)}")
    if in_features.shape[-1]!=d_in:
        raise ValueError(f"Expected input last dimensioin{d_in}, got {in_features.shape[-1]}")

    return in_features @ weights.transpose(-1,-2)
    