from __future__ import annotations

from typing import Any

from agents.base import LLMClient
from tools.workspace import (
    WorkspaceInspector,
    format_workspace_for_agent,
)


class ReaderAgent:
    """
    Reads the supplied workspace and asks the LLM to produce
    a structured understanding of the available source files.

    When a workspace directory is supplied, the deterministic
    WorkspaceInspector provides the actual file contents.

    When only input_files are supplied, the original lightweight
    behavior is preserved for compatibility with existing callers.
    """

    name = "reader"

    def __init__(
        self,
        llm: LLMClient,
        inspector: WorkspaceInspector | None = None,
    ) -> None:
        self.llm = llm
        self.inspector = inspector if inspector is not None else WorkspaceInspector()

    def run(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        input_files = state.get(
            "input_files",
            [],
        )

        workspace = state.get("workspace")

        inspected_files = []

        if workspace:
            inspected_files = self.inspector.inspect(
                workspace,
                input_files or None,
            )

            workspace_context = format_workspace_for_agent(inspected_files)

            prompt = (
                "Inspect the supplied CUDA workspace "
                "using the actual file contents below.\n\n"
                "Do not guess or invent source code.\n"
                "Do not describe hypothetical kernels.\n"
                "Every observation must be grounded in "
                "the supplied contents.\n\n"
                f"{workspace_context}\n\n"
                "Produce a structured understanding of:\n"
                "- source files\n"
                "- CUDA kernels\n"
                "- entry points\n"
                "- inputs\n"
                "- outputs\n"
                "- launch configuration\n"
                "- important dependencies\n"
                "- important execution details\n"
            )

        else:
            prompt = (
                "Inspect the supplied CUDA workspace.\n\n"
                f"Input files:\n{input_files}\n\n"
                "Identify source files, CUDA kernels, "
                "entry points, inputs, outputs, and "
                "important dependencies."
            )

        response = self.llm.invoke(
            prompt,
            system_prompt=(
                "You are a codebase reader. "
                "Do not modify files. "
                "When source contents are supplied, "
                "ground every observation in those contents. "
                "Do not invent or assume missing code."
            ),
            metadata={
                "agent": self.name,
                "files_inspected": [file.path for file in inspected_files],
            },
        )

        result = {
            "code_map": {
                "raw_response": response,
                "input_files": input_files,
            }
        }

        if inspected_files:
            result["code_map"]["inspected_files"] = [
                {
                    "path": file.path,
                    "size_bytes": file.size_bytes,
                }
                for file in inspected_files
            ]

        return result
