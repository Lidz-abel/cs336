import importlib.metadata

from cs336_basics.model import (
    BasicsTransformerLM,
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
)

__version__ = importlib.metadata.version("cs336_basics")

__all__ = [
    "BasicsTransformerLM",
    "CausalMultiHeadSelfAttention",
    "Embedding",
    "Linear",
    "RMSNorm",
    "SwiGLU",
    "TransformerBlock",
]
