from __future__ import annotations

import torch
import torch.distributed as dist


def distributed_update(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
) -> None:
    local_gradient = gradient.detach().clone()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_gradient, op=dist.ReduceOp.SUM)

    parameter.add_(local_gradient, alpha=-learning_rate)
