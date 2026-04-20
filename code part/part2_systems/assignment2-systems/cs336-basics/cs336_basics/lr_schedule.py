from __future__ import annotations

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
)-> float:
    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it >= cosine_cycle_iters:
        return min_learning_rate

    if cosine_cycle_iters <= warmup_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)
