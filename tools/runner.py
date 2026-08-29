from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int


def run_executable(
    executable: str | Path,
    args: list[str] | None = None,
    *,
    timeout: int = 300,
) -> RunResult:
    executable_path = Path(executable)

    if not executable_path.is_file():
        raise FileNotFoundError(f"Executable not found: {executable_path}")

    command = [
        str(executable_path),
        *(args or []),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    return RunResult(
        success=process.returncode == 0,
        stdout=process.stdout,
        stderr=process.stderr,
        return_code=process.returncode,
    )
