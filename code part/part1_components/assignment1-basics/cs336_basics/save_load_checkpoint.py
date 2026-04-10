from __future__ import annotations

import os
from typing import BinaryIO,IO
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
)->None:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src:str|os.PathLike|BinaryIO|IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
)->int: 
    """
    Given a path to a checkpoint, load the model and optimizer state dicts, and return the iteration number.
    """
    checkpoint = torch.load(src,map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]