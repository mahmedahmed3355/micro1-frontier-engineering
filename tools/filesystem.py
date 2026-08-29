from __future__ import annotations

from pathlib import Path


def read_text_file(path: str | Path) -> str:
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path.read_text(encoding="utf-8")


def write_text_file(
    path: str | Path,
    content: str,
) -> Path:
    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    return file_path


def list_files(
    root: str | Path,
    pattern: str = "*",
) -> list[Path]:
    root_path = Path(root)

    if not root_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {root_path}")

    return sorted(path for path in root_path.rglob(pattern) if path.is_file())
