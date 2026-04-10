from __future__ import annotations

import torch

from collections.abc import Iterable

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return 
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(),p=2) for p in params_with_grad]),
        p=2,
    )
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    if clip_coef<1:
        for param in params_with_grad:
            param.grad.mul_(clip_coef)
