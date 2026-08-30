from __future__ import annotations

import torch
import torch.distributed as dist


def train_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    optimizer.zero_grad(set_to_none=True)

    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    optimizer.step()
    return float(loss.detach().cpu())
