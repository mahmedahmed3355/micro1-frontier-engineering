from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class WorkerConfig:
    rank: int
    world_size: int
    device: str


def build_worker_config(rank: int, world_size: int) -> WorkerConfig:
    return WorkerConfig(
        rank=rank,
        world_size=world_size,
        device="cuda",
    )


def initialize_worker(rank: int, world_size: int) -> WorkerConfig:
    config = build_worker_config(rank, world_size)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    torch.cuda.set_device(config.device)

    return config


def run_step(config: WorkerConfig, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        tensor = tensor.to(config.device)

    dist.all_reduce(tensor)

    return tensor
