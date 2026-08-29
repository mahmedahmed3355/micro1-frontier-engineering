from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileResult:
    success: bool
    executable: Path
    stdout: str
    stderr: str
    return_code: int


def compile_cuda(
    source: str | Path,
    output: str | Path,
    *,
    nvcc: str = "nvcc",
    timeout: int = 300,
    extra_args: list[str] | None = None,
) -> CompileResult:
    source_path = Path(source)
    output_path = Path(output)

    if not source_path.is_file():
        raise FileNotFoundError(f"CUDA source not found: {source_path}")

    compiler = shutil.which(nvcc)

    if compiler is None:
        return CompileResult(
            success=False,
            executable=output_path,
            stdout="",
            stderr=(f"CUDA compiler '{nvcc}' was not found on PATH."),
            return_code=127,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        compiler,
        str(source_path),
        "-o",
        str(output_path),
    ]

    if extra_args:
        command.extend(extra_args)

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    return CompileResult(
        success=process.returncode == 0,
        executable=output_path,
        stdout=process.stdout,
        stderr=process.stderr,
        return_code=process.returncode,
    )
