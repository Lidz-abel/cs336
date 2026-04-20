from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.embedding import embedding as embedding_fn
from cs336_basics.linear import linear as linear_fn
from cs336_basics.multihead_self_attention_with_rope import (
    multihead_self_attention_with_rope as attention_with_rope_fn,
)
from cs336_basics.rmsnorm import rmsnorm as rmsnorm_fn
from cs336_basics.swiglu import swiglu as swiglu_fn
from cs336_basics.transformer_block import transformer_block as transformer_block_fn
from cs336_basics.transformer_lm import transformer_lm as transformer_lm_fn


def _trunc_normal_parameter(
    *shape: int,
    std: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Parameter:
    weight = torch.empty(*shape, device=device, dtype=dtype)
    nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)
    return nn.Parameter(weight)


class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        std = math.sqrt(2.0 / (d_in + d_out))
        self.weight = _trunc_normal_parameter(
            d_out,
            d_in,
            std=std,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_features: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return linear_fn(
            d_in=self.d_in,
            d_out=self.d_out,
            weights=self.weight,
            in_features=in_features,
        )


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = _trunc_normal_parameter(
            vocab_size,
            d_model,
            std=1.0,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return embedding_fn(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            weights=self.weight,
            token_ids=token_ids,
        )


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return rmsnorm_fn(
            d_model=self.d_model,
            eps=self.eps,
            weights=self.weight,
            in_features=in_features,
        )


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def weights_dict(self, prefix: str = "") -> dict[str, Tensor]:
        return {
            f"{prefix}w1.weight": self.w1.weight,
            f"{prefix}w2.weight": self.w2.weight,
            f"{prefix}w3.weight": self.w3.weight,
        }

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return swiglu_fn(
            d_model=self.d_model,
            d_ff=self.d_ff,
            w1_weight=self.w1.weight,
            w2_weight=self.w2.weight,
            w3_weight=self.w3.weight,
            in_features=in_features,
        )


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def weights_dict(self, prefix: str = "") -> dict[str, Tensor]:
        return {
            f"{prefix}q_proj.weight": self.q_proj.weight,
            f"{prefix}k_proj.weight": self.k_proj.weight,
            f"{prefix}v_proj.weight": self.v_proj.weight,
            f"{prefix}output_proj.weight": self.output_proj.weight,
        }

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        return attention_with_rope_fn(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            o_proj_weight=self.output_proj.weight,
            in_features=in_features,
            token_positions=token_positions,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def weights_dict(self, prefix: str = "") -> dict[str, Tensor]:
        return {
            f"{prefix}ln1.weight": self.ln1.weight,
            **self.attn.weights_dict(prefix=f"{prefix}attn."),
            f"{prefix}ln2.weight": self.ln2.weight,
            **self.ffn.weights_dict(prefix=f"{prefix}ffn."),
        }

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        return transformer_block_fn(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            weights=self.weights_dict(),
            in_features=in_features,
        )


class BasicsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
        }
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def weights_dict(self) -> dict[str, Tensor]:
        weights = {
            "token_embeddings.weight": self.token_embeddings.weight,
            "ln_final.weight": self.ln_final.weight,
            "lm_head.weight": self.lm_head.weight,
        }
        for layer_idx, layer in enumerate(self.layers):
            weights.update(layer.weights_dict(prefix=f"layers.{layer_idx}."))
        return weights

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(param.numel() for param in self.parameters())
        if non_embedding:
            n_params -= self.lm_head.weight.numel()
        return n_params

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        return transformer_lm_fn(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self.weights_dict(),
            in_indices=in_indices,
        )
