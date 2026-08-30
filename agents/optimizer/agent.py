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
            "You are optimizing the existing CUDA workload described by the "
            "Reader and Analyzer.\n\n"
            "Generate ONE complete replacement source file for the canonical "
            "input source.\n\n"
            "============================================================\n"
            "HARD WORKLOAD PRESERVATION CONTRACT\n"
            "============================================================\n"
            "The candidate MUST implement the SAME workload as the canonical "
            "input source.\n"
            "The Reader context, canonical source, case metadata, and tests "
            "are authoritative.\n\n"
            "Before generating code, identify the exact kernel, launcher or "
            "entry point, inputs, outputs, mathematical operation, launch "
            "configuration, and command-line interface from the supplied "
            "canonical source.\n\n"
            "Do NOT assume GEMM, matrix multiplication, vector addition, or "
            "any other workload unless it is explicitly present in the "
            "canonical source and Reader context.\n\n"
            "The candidate MUST preserve:\n"
            "1. The original workload semantics.\n"
            "2. The original public caller-visible interface.\n"
            "3. The original input and output meaning.\n"
            "4. The original mathematical operation.\n"
            "5. Required boundary and correctness behavior.\n"
            "6. Required command-line arguments, if present.\n"
            "7. Required benchmark output fields, if present.\n"
            "8. Deterministic behavior required by the case.\n\n"
            "The candidate MAY optimize the implementation only when the "
            "optimization preserves the complete workload contract.\n\n"
            "============================================================\n"
            "CASE-SPECIFIC EVIDENCE HAS PRIORITY\n"
            "============================================================\n"
            "Use CASE METADATA, READER CONTEXT, ORIGINAL INPUT FILES, and "
            "CANONICAL ORIGINAL SOURCE as the source of truth.\n\n"
            "If the case is a vector-add kernel, generate vector-add code.\n"
            "If the case is matrix multiplication, generate matrix "
            "multiplication code.\n"
            "If the case is another workload, preserve that workload.\n\n"
            "Never substitute a different benchmark merely because an "
            "optimization template suggests it.\n\n"
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
            "Never reproduce an identical rejected candidate.\n\n"
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
            f"{input_source}\n\n"
            "============================================================\n"
            "CURRENT ITERATION\n"
            "============================================================\n"
            f"{state.get('iteration', 0)}\n\n"
            "============================================================\n"
            "GENERATION REQUIREMENT\n"
            "============================================================\n"
            "Generate exactly ONE complete replacement source file matching "
            "the canonical workload.\n"
            "Preserve correctness and the caller-visible contract before "
            "applying any optimization.\n"
            "Use only evidence-supported optimizations.\n"
            "Do not invent a different workload.\n"
            "Do not return an explanation instead of source code.\n"
            "Do not return multiple candidates.\n"
            "Return the complete source inside exactly one CUDA code block."
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
