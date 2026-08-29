from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CasePaths:
    root: Path

    @property
    def input(self) -> Path:
        return self.root / "input.cu"

    @property
    def reference(self) -> Path:
        return self.root / "reference.cu"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def candidate(self) -> Path:
        return self.root / "candidate.cu"

    @property
    def trajectory(self) -> Path:
        return self.root / "trajectory.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "run_summary.json"
