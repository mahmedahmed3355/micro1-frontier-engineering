from __future__ import annotations

import json
from typing import Any

from agents.agent import BaseAgent


class AnalyzerAgent(BaseAgent):
    """
    Analyze CUDA code using the context produced by the Reader.

    The Analyzer does not modify source files.
    It produces performance observations and optimization hypotheses.
    """

    name = "analyzer"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        code_map = state.get("code_map", {})
        input_files = state.get("input_files", [])
        input_source = state.get("input_source", "")

        prompt = (
            "Analyze the supplied CUDA implementation for performance.\n\n"
            "Reader output:\n"
            f"{json.dumps(code_map, indent=2)}\n\n"
            "Input files:\n"
            f"{json.dumps(input_files, indent=2)}\n\n"
            "Canonical input.cu source:\n"
            f"```cuda\n{input_source}\n```\n\n"
            "Identify likely performance bottlenecks and propose "
            "optimization hypotheses.\n\n"
            "For every hypothesis, provide:\n"
            "1. The suspected bottleneck.\n"
            "2. Evidence from the implementation.\n"
            "3. The proposed optimization.\n"
            "4. Expected benefit.\n"
            "5. Potential correctness or performance risks."
        )

        response = self.llm.invoke(
            prompt,
            system_prompt=(
                "You are a CUDA performance analysis specialist. "
                "Analyze only the supplied evidence. "
                "Do not modify files. "
                "Do not claim measured performance without benchmark data. "
                "Separate observations from hypotheses."
            ),
            metadata={
                "agent": self.name,
            },
        )

        return {
            "performance_analysis": {
                "raw_response": response,
                "input_files": input_files,
                "reader_context": code_map,
            },
            "optimization_hypotheses": [
                {
                    "source": "analyzer",
                    "raw_response": response,
                }
            ],
        }
