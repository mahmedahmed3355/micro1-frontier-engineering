from __future__ import annotations

from typing import Any

from agents.base import LLMClient
from tools.workspace import (
    WorkspaceInspector,
    format_workspace_for_agent,
)


class GeminiReaderAgent:
    """
    Compact fact-oriented CUDA reader.

    Uses three small grounded LLM requests and records
    each request through the trajectory recorder.
    """

    name = "reader"

    def __init__(
        self,
        llm: LLMClient,
        inspector: WorkspaceInspector | None = None,
        trajectory_recorder: Any | None = None,
    ) -> None:
        self.llm = llm
        self.inspector = inspector if inspector is not None else WorkspaceInspector()
        self.trajectory_recorder = trajectory_recorder

    def _ask(
        self,
        question: str,
        source: str,
        fact_type: str,
    ) -> str:
        prompt = f"{question}\n\nSOURCE:\n{source}"

        system_prompt = (
            "You are a CUDA code reader. "
            "Use only the supplied source. "
            "Return only the requested facts. "
            "Be concise. "
            "No Markdown. "
            "No explanation. "
            "Do not invent information."
        )

        response = self.llm.invoke(
            prompt,
            system_prompt=system_prompt,
            metadata={
                "agent": self.name,
                "fact_type": fact_type,
            },
        )

        response = (response or "").strip()

        if not response:
            raise RuntimeError(f"Empty Gemini response: {fact_type}")

        if self.trajectory_recorder is not None:
            self.trajectory_recorder.record_llm_call(
                agent=self.name,
                model=getattr(
                    self.llm,
                    "model",
                    "unknown",
                ),
                prompt=prompt,
                response=response,
                metadata={
                    "fact_type": fact_type,
                },
            )

        return response

    def run(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        workspace = state.get("workspace")

        if not workspace:
            raise RuntimeError("GeminiReaderAgent requires a workspace.")

        input_files = state.get(
            "input_files",
            [],
        )

        inspected_files = self.inspector.inspect(
            workspace,
            input_files or None,
        )

        context = format_workspace_for_agent(inspected_files)

        # ----------------------------------------------------
        # CALL 1
        # ----------------------------------------------------

        kernel_entry = self._ask(
            (
                "Identify the CUDA kernel name and the "
                "main program entry point. "
                "Return exactly two lines:\n"
                "KERNEL=<name>\n"
                "ENTRY=<name>"
            ),
            context,
            "kernel_and_entry_point",
        )

        # ----------------------------------------------------
        # CALL 2
        # ----------------------------------------------------

        io_launch = self._ask(
            (
                "Identify the CUDA kernel input buffers, "
                "output buffer, and launch block dimensions. "
                "Return exactly three lines:\n"
                "INPUTS=<names>\n"
                "OUTPUT=<name>\n"
                "BLOCK=<dimensions>"
            ),
            context,
            "inputs_outputs_launch",
        )

        # ----------------------------------------------------
        # CALL 3
        # ----------------------------------------------------

        benchmark = self._ask(
            (
                "Identify the recorded baseline median kernel "
                "time and the benchmark/correctness constraints. "
                "Return exactly three lines:\n"
                "BASELINE=<value>\n"
                "BENCHMARK=<brief value>\n"
                "CORRECTNESS=<brief value>"
            ),
            context,
            "baseline_benchmark_correctness",
        )

        code_map = {
            "reader_backend": "gemini",
            "kernel_and_entry_point": kernel_entry,
            "inputs_outputs_launch": io_launch,
            "benchmark_and_correctness": benchmark,
            "input_files": input_files,
            "inspected_files": [
                {
                    "path": file.path,
                    "size_bytes": file.size_bytes,
                }
                for file in inspected_files
            ],
        }

        return {
            "code_map": code_map,
        }
