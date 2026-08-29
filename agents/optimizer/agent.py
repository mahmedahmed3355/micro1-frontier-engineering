from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from agents.agent import BaseAgent


class OptimizerAgent(BaseAgent):
    """
    Generate an optimized CUDA candidate from Analyzer output.

    The optimizer proposes a complete candidate implementation.
    Correctness and performance are determined independently
    by the verifier.
    """

    name = "optimizer"

    _CUDA_BLOCK = re.compile(
        r"^\s*```(?:cuda|cpp|c\+\+)?\s*\n(?P<source>[\s\S]*?)\n```\s*$",
        re.IGNORECASE,
    )

    def run(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        input_files = state.get(
            "input_files",
            [],
        )

        code_map = state.get(
            "code_map",
            {},
        )

        analysis = state.get(
            "performance_analysis",
            {},
        )

        hypotheses = state.get(
            "optimization_hypotheses",
            [],
        )

        metadata = state.get(
            "metadata",
            {},
        )

        baseline_evidence = state.get(
            "baseline_evidence",
            {},
        )

        input_source = state.get("input_source", "")
        previous_candidate = {
            "sha256": state.get("previous_candidate_sha256"),
            "source": state.get("previous_candidate_source", ""),
        }
        latest_feedback = state.get(
            "latest_feedback",
            {},
        )
        rejected_candidate_hashes = state.get(
            "rejected_candidate_hashes",
            [],
        )

        prompt = (
            "You are optimizing an existing CUDA workload.\n\n"
            "Your task is to generate ONE complete replacement "
            "source file for input.cu.\n\n"
            "============================================================\n"
            "HARD WORKLOAD CONTRACT\n"
            "============================================================\n"
            "The candidate MUST remain a complete standalone CUDA "
            "executable.\n\n"
            "The candidate MUST:\n"
            "1. Contain a valid int main(int argc, char** argv) "
            "entry point.\n"
            "2. Compile directly with nvcc as an executable.\n"
            "3. Preserve the command-line matrix size argument.\n"
            "4. Preserve matrix multiplication semantics C = A * B.\n"
            "5. Preserve deterministic input generation.\n"
            "6. Preserve random seed 12345.\n"
            "7. Preserve input range [-1.0, 1.0].\n"
            "8. Preserve one warm-up kernel execution.\n"
            "9. Preserve 100 measured kernel executions.\n"
            "10. Report the MEDIAN kernel execution time.\n"
            "11. Print exactly the required benchmark field "
            "KERNEL_TIME_MS=<value>.\n"
            "12. Print SIZE=<N>.\n"
            "13. Print RESULT_SAMPLE=<10 comma-separated values>.\n"
            "14. Preserve numerical correctness within the supplied "
            "tolerance.\n"
            "15. Preserve the existing observable workload behavior.\n"
            "16. Do not replace matrix multiplication with element-wise "
            "multiplication.\n"
            "17. Do not use hypothetical, placeholder, or pseudocode "
            "implementations.\n"
            "18. Do not depend on a pre-existing candidate.cu.\n"
            "19. Optimize the CUDA kernel execution path.\n\n"
            "============================================================\n"
            "FORBIDDEN CHANGES\n"
            "============================================================\n"
            "Do NOT change the workload into a different benchmark.\n"
            "Do NOT change the matrix initialization semantics.\n"
            "Do NOT remove the benchmark output fields.\n"
            "Do NOT replace the median benchmark with one timing.\n"
            "Do NOT replace CUDA execution with CPU computation.\n"
            "Do NOT add an unrelated verification program.\n"
            "Do NOT return only a kernel fragment.\n"
            "Do NOT return only a function.\n"
            "Do NOT return an explanation instead of source code.\n\n"
            "============================================================\n"
            "PREVIOUS VERIFIER FEEDBACK\n"
            "============================================================\n"
            f"{json.dumps(latest_feedback, indent=2)}\n\n"
            "============================================================\n"
            "IMMEDIATELY PREVIOUS CANDIDATE\n"
            "============================================================\n"
            f"{json.dumps(previous_candidate, indent=2)}\n\n"
            "============================================================\n"
            "REJECTED CANDIDATE HASHES\n"
            "============================================================\n"
            f"{json.dumps(rejected_candidate_hashes, indent=2)}\n\n"
            "Never reproduce an identical candidate whose SHA-256 is "
            "in REJECTED CANDIDATE HASHES. Diversify the implementation "
            "when the verifier evidence supports a different approach.\n\n"
            "============================================================\n"
            "WORKLOAD METADATA\n"
            "============================================================\n"
            f"{json.dumps(metadata, indent=2)}\n\n"
            "============================================================\n"
            "BASELINE EVIDENCE\n"
            "============================================================\n"
            f"{json.dumps(baseline_evidence, indent=2)}\n\n"
            "============================================================\n"
            "READER CONTEXT\n"
            "============================================================\n"
            f"{json.dumps(code_map, indent=2)}\n\n"
            "============================================================\n"
            "PERFORMANCE ANALYSIS\n"
            "============================================================\n"
            f"{json.dumps(analysis, indent=2)}\n\n"
            "============================================================\n"
            "OPTIMIZATION HYPOTHESES\n"
            "============================================================\n"
            f"{json.dumps(hypotheses, indent=2)}\n\n"
            "============================================================\n"
            "ORIGINAL INPUT FILES\n"
            "============================================================\n"
            f"{json.dumps(input_files, indent=2)}\n\n"
            "============================================================\n"
            "CANONICAL ORIGINAL SOURCE\n"
            "============================================================\n"
            f"```cuda\n{input_source}\n```\n\n"
            "============================================================\n"
            "CURRENT ITERATION\n"
            "============================================================\n"
            f"{state.get('iteration', 0)}\n\n"
            "Treat CURRENT ITERATION as the current candidate attempt "
            "number.\n\n"
            "============================================================\n"
            "GENERATION REQUIREMENT\n"
            "============================================================\n"
            "Generate a complete replacement input.cu.\n"
            "Preserve the workload contract first.\n"
            "Apply only evidence-supported CUDA optimizations.\n"
            "The independent verifier will determine whether the "
            "optimization actually improves performance.\n\n"
            "Return the complete source inside exactly one "
            "```cuda``` code block.\n"
            "Do not return multiple candidates.\n"
            "Do not return pseudocode."
        )

        response = self.llm.invoke(
            prompt,
            system_prompt=(
                "You are an expert CUDA performance engineer. "
                "Generate a complete standalone executable source file. "
                "Preserve the exact workload contract and observable "
                "benchmark interface. "
                "Never sacrifice correctness for optimization. "
                "Never claim success before independent verification."
            ),
            metadata={
                "agent": self.name,
                "iteration": state.get("iteration", 0),
            },
        )

        iteration = state.get(
            "iteration",
            0,
        )

        history_entry = {
            "agent": self.name,
            "iteration": iteration,
            "response": response,
        }

        match = self._CUDA_BLOCK.match(response)
        candidate_source = match.group("source").strip() if match else ""
        candidate_sha256 = (
            hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
            if candidate_source
            else ""
        )

        return {
            "candidate_files": ["candidate.cu"],
            "optimization_history": [
                *state.get("optimization_history", []),
                history_entry,
            ],
            "candidate_source": candidate_source,
            "candidate_sha256": candidate_sha256,
            "generation_error": (
                "Optimizer response must contain exactly one fenced CUDA source block."
                if not candidate_source
                else ""
            ),
        }
