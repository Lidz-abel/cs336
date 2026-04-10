from __future__ import annotations

from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.embedding import embedding
from cs336_basics.linear import linear
from cs336_basics.rmsnorm import rmsnorm
from cs336_basics.transformer_block import transformer_block


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    sequence_length = in_indices.shape[-1]
    if sequence_length > context_length:
        raise ValueError(
            f"Input sequence length {sequence_length} exceeds context length {context_length}"
        )

    hidden = embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )

    for layer_idx in range(num_layers):
        layer_weights = {
            key.replace(f"layers.{layer_idx}.", ""): value
            for key, value in weights.items()
            if key.startswith(f"layers.{layer_idx}.")
        }
        hidden = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=hidden,
        )

    hidden = rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln_final.weight"],
        in_features=hidden,
    )

    return linear(
        d_in=d_model,
        d_out=vocab_size,
        weights=weights["lm_head.weight"],
        in_features=hidden,
    )
