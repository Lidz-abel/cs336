from __future__ import annotations

from jaxtyping import Float
from torch import Tensor

from cs336_basics.linear import linear
from cs336_basics.silu import silu


def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    w1_out = linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_features,
    )
    w3_out = linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w3_weight,
        in_features=in_features,
    )
    gated = silu(w1_out) * w3_out
    return linear(
        d_in=d_ff,
        d_out=d_model,
        weights=w2_weight,
        in_features=gated,
    )
