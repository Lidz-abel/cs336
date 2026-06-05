from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist

from cs336_systems.distributed import _broadcast_, _dist_ready, _rank, _world_size


def _materialize_param_groups(params: Iterable) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter]]:
    groups: list[dict[str, Any]] = []
    all_params: list[torch.nn.Parameter] = []

    for item in params:
        if isinstance(item, dict):
            group = dict(item)
            group_params = list(group["params"])
            group["params"] = group_params
        else:
            group_params = [item]
            group = {"params": group_params}
        groups.append(group)
        all_params.extend(group_params)

    return groups, all_params


class ShardedOptimizer:
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self.global_param_groups, self.global_params = _materialize_param_groups(params)
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.world_size = _world_size()
        self.rank = _rank()

        if self.world_size == 1:
            self.local_optimizer = optimizer_cls(self.global_param_groups, **kwargs)
            return

        self.param_to_index = {id(param): idx for idx, param in enumerate(self.global_params)}
        local_groups: list[dict[str, Any]] = []
        for group in self.global_param_groups:
            local_group = {key: value for key, value in group.items() if key != "params"}
            local_group["params"] = [
                param for param in group["params"] if self._owner(self.param_to_index[id(param)]) == self.rank
            ]
            if local_group["params"]:
                local_groups.append(local_group)

        self.local_optimizer = optimizer_cls(local_groups, **kwargs) if local_groups else None

    def _owner(self, param_index: int) -> int:
        return param_index % self.world_size

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.global_params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.local_optimizer is not None:
            if closure is None:
                self.local_optimizer.step()
            else:
                self.local_optimizer.step(lambda: loss)

        if _dist_ready():
            with torch.no_grad():
                for index, param in enumerate(self.global_params):
                    _broadcast_(param.data, src=self._owner(index))
        return loss

    def state_dict(self):
        if self.local_optimizer is None:
            return {"state": {}, "param_groups": []}
        return self.local_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        if self.local_optimizer is not None:
            self.local_optimizer.load_state_dict(state_dict)

