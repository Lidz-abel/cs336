from __future__ import annotations

import numpy as np
import torch
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray[np.integer], 
    batch_size: int,
    context_length: int,
    device: str,
)->tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a batch of input and target sequences from the dataset.
    
    Args:
        dataset: A 1D numpy array of integers representing the dataset.
        batch_size: The number of sequences in the batch.
        context_length: The length of each input sequence.
        device: The device to which the tensors will be moved (e.g., 'cpu' or 'cuda').
    """
    if dataset.ndim!=1:
        raise ValueError("Dataset must be 1Dnumpy array.")
    if batch_size <= 0:
        raise ValueError("Batch_size must nbe positive.")
    if context_length<=0:
        raise ValueError("Context_length must be positive.")
    
    num_possible_starts = len(dataset) - context_length

    if num_possible_starts<=0:
        raise ValueError("dataset is too short.")
    
    start_indices = np.random.randint(
        0,
        num_possible_starts,
        size=batch_size,
        dtype=np.int64,
    )
    offsets = np.arange(context_length+1,dtype=np.int64)

    batch_np=np.asarray(
        dataset[start_indices[:,None]+offsets[None, :]],
        dtype=np.int64,
    )
    batch_cpu=torch.from_numpy(batch_np) 
    target_device=torch.device(device)

    if target_device.type=="cuda":
        batch_cpu=batch_cpu.pin_memory()
        batch = batch_cpu.to(target_device, dtype=torch.long, non_blocking=True)
    else:
        batch = batch_cpu.to(target_device, dtype=torch.long)
    
    x=batch[:,:-1]
    y=batch[:,1:]
    return x,y
    