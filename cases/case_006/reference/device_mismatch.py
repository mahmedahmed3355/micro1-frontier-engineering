from __future__ import annotations

import torch


def run_device_mismatch(
    values: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    values = values.to(device)
    weight = weight.to(device)

    return values * weight
