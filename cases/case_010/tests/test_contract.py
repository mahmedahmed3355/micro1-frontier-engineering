from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "system.py"
REFERENCE = CASE_ROOT / "reference" / "system.py"


def test_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_exists() -> None:
    assert REFERENCE.is_file()


def test_source_builds_worker_config() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "WorkerConfig(" in text
    assert 'device="cuda"' in text


def test_reference_derives_device_from_rank() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert 'device = f"cuda:{rank}"' in text


def test_source_preserves_generic_cuda_device_defect() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert 'device="cuda"' in text
    assert 'f"cuda:{rank}"' not in text


def test_reference_initializes_nccl_with_worker_configuration() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    config_position = text.find("config = build_worker_config(rank, world_size)")
    init_position = text.find("dist.init_process_group(")
    init_end = text.find(")", init_position)
    init_block = text[init_position:init_end]

    rank_position = init_block.find("rank=rank")
    world_size_position = init_block.find("world_size=world_size")

    assert config_position >= 0
    assert init_position > config_position
    assert rank_position >= 0
    assert world_size_position >= 0


def test_source_assigns_configured_cuda_device() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "torch.cuda.set_device(config.device)" in text


def test_reference_assigns_configured_cuda_device() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "torch.cuda.set_device(config.device)" in text


def test_reference_moves_tensor_before_collective() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    placement_position = text.find("tensor = tensor.to(config.device)")
    contiguous_position = text.find("tensor = tensor.contiguous()")
    reduce_position = text.find("dist.all_reduce(tensor)")

    assert placement_position >= 0
    assert contiguous_position > placement_position
    assert reduce_position > contiguous_position


def test_reference_synchronizes_after_collective() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    reduce_position = text.find("dist.all_reduce(tensor)")
    sync_position = text.find("torch.cuda.synchronize(tensor.device)")

    assert reduce_position >= 0
    assert sync_position > reduce_position


def test_source_omits_post_collective_synchronization() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "torch.cuda.synchronize" not in text


def test_reference_uses_tensor_device_for_completion_boundary() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert re.search(
        r"torch\.cuda\.synchronize\s*\(\s*tensor\.device\s*\)",
        text,
    )


def test_reference_collective_precedes_completion_boundary() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    reduce_position = text.find("dist.all_reduce(tensor)")
    sync_position = text.find("torch.cuda.synchronize(tensor.device)")
    return_position = text.find("return tensor")

    assert reduce_position >= 0
    assert sync_position > reduce_position
    assert return_position > sync_position


def test_source_has_distributed_collective() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "dist.all_reduce(tensor)" in text


def test_reference_has_distributed_collective() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "dist.all_reduce(tensor)" in text
