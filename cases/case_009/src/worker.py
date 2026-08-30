from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import torch


async def run_gpu_worker(
    work: Callable[[], Awaitable[torch.Tensor]],
) -> torch.Tensor:
    task = asyncio.create_task(work())
    result = await task
    return result.cpu()
