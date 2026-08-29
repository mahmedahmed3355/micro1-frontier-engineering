import pytest

from tools.workspace import (
    WorkspaceInspector,
    format_workspace_for_agent,
)


def test_inspector_reads_real_file_contents(tmp_path):
    workspace = tmp_path / "case"
    workspace.mkdir()

    source = workspace / "input.cu"

    source.write_text(
        "__global__ void matmul_naive() {}",
        encoding="utf-8",
    )

    inspector = WorkspaceInspector()

    files = inspector.inspect(
        workspace,
        ["input.cu"],
    )

    assert len(files) == 1
    assert files[0].path == "input.cu"
    assert "__global__ void matmul_naive() {}" in files[0].content


def test_inspector_sorts_files_deterministically(tmp_path):
    workspace = tmp_path / "case"
    workspace.mkdir()

    (workspace / "reference.cu").write_text(
        "reference",
        encoding="utf-8",
    )

    (workspace / "input.cu").write_text(
        "input",
        encoding="utf-8",
    )

    inspector = WorkspaceInspector()

    files = inspector.inspect(workspace)

    assert [file.path for file in files] == [
        "input.cu",
        "reference.cu",
    ]


def test_inspector_rejects_path_escape(tmp_path):
    workspace = tmp_path / "case"
    workspace.mkdir()

    outside = tmp_path / "secret.cu"

    outside.write_text(
        "secret",
        encoding="utf-8",
    )

    inspector = WorkspaceInspector()

    with pytest.raises(ValueError):
        inspector.inspect(
            workspace,
            ["../secret.cu"],
        )


def test_inspector_rejects_unsupported_file_type(
    tmp_path,
):
    workspace = tmp_path / "case"
    workspace.mkdir()

    source = workspace / "script.py"

    source.write_text(
        "print('not allowed')",
        encoding="utf-8",
    )

    inspector = WorkspaceInspector()

    with pytest.raises(ValueError):
        inspector.inspect(
            workspace,
            ["script.py"],
        )


def test_format_workspace_contains_content(
    tmp_path,
):
    workspace = tmp_path / "case"
    workspace.mkdir()

    (workspace / "input.cu").write_text(
        "CUDA_SOURCE",
        encoding="utf-8",
    )

    inspector = WorkspaceInspector()

    files = inspector.inspect(
        workspace,
        ["input.cu"],
    )

    formatted = format_workspace_for_agent(files)

    assert "FILE: input.cu" in formatted
    assert "CUDA_SOURCE" in formatted
    assert "CONTENT:" in formatted
