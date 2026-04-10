from __future__ import annotations

import torch
from jaxtyping import Float,Int
from torch import Tensor

def embedding(
    vocab_size:int,
    d_model:int,
    weights:Float[Tensor,"vocab_size d_model"],
    token_ids: Int[Tensor,"..."],
)->Float[Tensor,"... d_model"]:
    """
    Args:
        vocab_size: the size of the vocabulary
        d_model: the dimension of the embedding
        weights: a tensor of shape (vocab_size, d_model) containing the embedding weights
        token_ids: a tensor of shape (...) containing the token ids to be embedded

    Returns:
        A tensor of shape (..., d_model) containing the embedded token vectors
    """
    if weights.shape!=(vocab_size, d_model):
        raise ValueError(f"Expected weights shape {(vocab_size, d_model)}, got {tuple(weights.shape)}")

    return weights[token_ids]