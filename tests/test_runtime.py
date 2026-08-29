import json

from runtime.result import CasePaths
from runtime.runner import load_case


def test_case_paths():
    paths = CasePaths(root=__import__("pathlib").Path("/tmp/case_001"))

    assert paths.input.name == "input.cu"

    assert paths.reference.name == "reference.cu"

    assert paths.metadata.name == "metadata.json"

    assert paths.candidate.name == "candidate.cu"

    assert paths.trajectory.name == "trajectory.jsonl"


def test_load_case(tmp_path):
    case = tmp_path / "case_001"
    case.mkdir()

    (case / "input.cu").write_text(
        "__global__ void kernel() {}",
        encoding="utf-8",
    )

    (case / "reference.cu").write_text(
        "reference",
        encoding="utf-8",
    )

    (case / "metadata.json").write_text(
        json.dumps(
            {
                "case_id": "test_case",
                "workload": "cuda",
            }
        ),
        encoding="utf-8",
    )

    paths, metadata = load_case(case)

    assert paths.input.exists()
    assert paths.reference.exists()
    assert paths.metadata.exists()

    assert metadata["case_id"] == "test_case"
