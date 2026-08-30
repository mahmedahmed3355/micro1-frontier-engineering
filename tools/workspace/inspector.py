from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    size_bytes: int
    content: str


class WorkspaceInspector:
    """
    Deterministically reads explicitly allowed workspace files.

    The LLM receives actual file contents rather than merely
    receiving file paths and guessing what those files contain.
    """

    ALLOWED_SUFFIXES = {
        ".cu",
        ".cuh",
        ".cpp",
        ".h",
        ".hpp",
        ".json",
        ".md",
        ".txt",
        ".py",
}

    def inspect(
        self,
        workspace: str | Path,
        files: list[str] | None = None,
    ) -> list[WorkspaceFile]:
        root = Path(workspace).resolve()

        if not root.exists():
            raise FileNotFoundError(f"Workspace does not exist: {root}")

        if not root.is_dir():
            raise NotADirectoryError(root)

        if files is None:
            paths = sorted(
                path
                for path in root.rglob("*")
                if (path.is_file() and path.suffix.lower() in self.ALLOWED_SUFFIXES)
            )
        else:
            paths = []

            for file_name in files:
                candidate = (
                    Path(file_name)
                    if Path(file_name).is_absolute()
                    else root / file_name
                )

                candidate = candidate.resolve()

                if not self._inside(
                    candidate,
                    root,
                ):
                    raise ValueError(f"File escapes workspace: {file_name}")

                paths.append(candidate)

            paths.sort()

        result: list[WorkspaceFile] = []

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)

            if not path.is_file():
                raise ValueError(f"Not a file: {path}")

            if path.suffix.lower() not in (self.ALLOWED_SUFFIXES):
                raise ValueError(f"Unsupported file type: {path}")

            content = path.read_text(encoding="utf-8")

            result.append(
                WorkspaceFile(
                    path=str(path.relative_to(root)),
                    size_bytes=len(content.encode("utf-8")),
                    content=content,
                )
            )

        return result

    @staticmethod
    def _inside(
        path: Path,
        root: Path,
    ) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def format_workspace_for_agent(
    files: list[WorkspaceFile],
) -> str:
    """
    Convert deterministic file inspection into grounded
    text that can be supplied to an LLM.
    """

    sections: list[str] = []

    for file in files:
        sections.append(
            "\n".join(
                [
                    f"FILE: {file.path}",
                    f"SIZE_BYTES: {file.size_bytes}",
                    "CONTENT:",
                    "```",
                    file.content,
                    "```",
                ]
            )
        )

    return "\n\n".join(sections)
