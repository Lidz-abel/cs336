from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

def cross_entropy(
    inputs:Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"],
) -> Float[Tensor, ""]:
    """
    Computes the cross-entropy loss between the input and the target.
    """
    max_values = torch.amax(inputs, dim=-1, keepdim = True)
    shifted = inputs - max_values
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted),dim=-1))+max_values.squeeze(-1)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    losses= log_sum_exp - target_logits
    return losses.mean()


