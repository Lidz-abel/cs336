from __future__ import annotations

import torch
from torch import Tensor
from jaxtyping import Float, Int

def softmax(in_features: Float[Tensor, "..."],dim: int)-> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.
    """
    shifted = in_features - torch.amax(in_features , dim=dim, keepdim=True)
    exp_shifted = torch.exp(shifted)
    return exp_shifted/torch.sum(exp_shifted, dim=dim, keepdim=True)
