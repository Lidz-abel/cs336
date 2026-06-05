from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


class _CompletedWork:
    def wait(self) -> None:
        return None


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


def _rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _requires_cpu_collective(tensor: Tensor) -> bool:
    return _dist_ready() and dist.get_backend() == "gloo" and tensor.is_cuda


def _broadcast_(tensor: Tensor, src: int = 0) -> None:
    if not _dist_ready():
        return
    if _requires_cpu_collective(tensor):
        if _rank() == src:
            buffer = tensor.detach().cpu()
        else:
            buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        dist.broadcast(buffer, src=src)
        tensor.copy_(buffer.to(device=tensor.device))
    else:
        dist.broadcast(tensor, src=src)


def _all_reduce_sum_(tensor: Tensor, async_op: bool = False):
    if not _dist_ready():
        return _CompletedWork()
    if _requires_cpu_collective(tensor):
        buffer = tensor.detach().cpu()
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        tensor.copy_(buffer.to(device=tensor.device))
        return _CompletedWork()
    return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)


def _unique_trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


class DistributedDataParallelIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._world_size = _world_size()
        self._handles: list[tuple[object, Tensor]] = []
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._broadcast_module_state()
        for param in _unique_trainable_parameters(self.module):
            self._hook_handles.append(param.register_post_accumulate_grad_hook(self._make_hook(param)))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _broadcast_module_state(self) -> None:
        with torch.no_grad():
            for tensor in self.module.state_dict().values():
                _broadcast_(tensor, src=0)

    def _make_hook(self, param: torch.nn.Parameter):
        def hook(_param: torch.nn.Parameter) -> None:
            if param.grad is None or self._world_size == 1:
                return
            work = _all_reduce_sum_(param.grad, async_op=True)
            self._handles.append((work, param.grad))

        return hook

    def finish_gradient_synchronization(self) -> None:
        for work, grad in self._handles:
            work.wait()
            grad.div_(self._world_size)
        self._handles.clear()


@dataclass
class _Bucket:
    params: list[torch.nn.Parameter]
    ready: set[int]
    work: object | None = None
    flat_grad: Tensor | None = None


class DistributedDataParallelBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None):
        super().__init__()
        self.module = module
        self._world_size = _world_size()
        self._bucket_size_bytes = None if bucket_size_mb is None else int(bucket_size_mb * 1024 * 1024)
        self._buckets: list[_Bucket] = []
        self._param_to_bucket: dict[int, int] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._broadcast_module_state()
        self._build_buckets(_unique_trainable_parameters(self.module))
        for param in _unique_trainable_parameters(self.module):
            self._hook_handles.append(param.register_post_accumulate_grad_hook(self._make_hook(param)))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _broadcast_module_state(self) -> None:
        with torch.no_grad():
            for tensor in self.module.state_dict().values():
                _broadcast_(tensor, src=0)

    def _build_buckets(self, params: Iterable[torch.nn.Parameter]) -> None:
        current: list[torch.nn.Parameter] = []
        current_size = 0
        max_size = self._bucket_size_bytes

        for param in params:
            param_size = param.numel() * param.element_size()
            if current and max_size is not None and current_size + param_size > max_size:
                self._append_bucket(current)
                current = []
                current_size = 0
            current.append(param)
            current_size += param_size

        if current:
            self._append_bucket(current)

    def _append_bucket(self, params: list[torch.nn.Parameter]) -> None:
        bucket_idx = len(self._buckets)
        self._buckets.append(_Bucket(params=list(params), ready=set()))
        for param in params:
            self._param_to_bucket[id(param)] = bucket_idx

    def _make_hook(self, param: torch.nn.Parameter):
        def hook(_param: torch.nn.Parameter) -> None:
            if param.grad is None:
                return
            bucket = self._buckets[self._param_to_bucket[id(param)]]
            bucket.ready.add(id(param))
            if len(bucket.ready) == len(bucket.params) and bucket.work is None:
                self._launch_bucket(bucket)

        return hook

    def _launch_bucket(self, bucket: _Bucket) -> None:
        if self._world_size == 1:
            return
        grads = [param.grad.reshape(-1) for param in bucket.params if param.grad is not None]
        if len(grads) != len(bucket.params):
            return
        bucket.flat_grad = torch.cat(grads)
        bucket.work = _all_reduce_sum_(bucket.flat_grad, async_op=True)

    def start_gradient_synchronization(self) -> None:
        for bucket in self._buckets:
            bucket.ready.clear()
            bucket.work = None
            bucket.flat_grad = None

    def finish_gradient_synchronization(self) -> None:
        for bucket in self._buckets:
            if bucket.work is None:
                self._launch_bucket(bucket)
            if bucket.work is None or bucket.flat_grad is None:
                continue
            bucket.work.wait()
            bucket.flat_grad.div_(self._world_size)
            offset = 0
            for param in bucket.params:
                assert param.grad is not None
                numel = param.numel()
                param.grad.copy_(bucket.flat_grad[offset : offset + numel].view_as(param))
                offset += numel
        self.start_gradient_synchronization()


def finish_gradient_synchronization(ddp_model: torch.nn.Module) -> None:
    ddp_model.finish_gradient_synchronization()

