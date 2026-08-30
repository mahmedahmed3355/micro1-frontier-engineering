from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools" / "batch" / "case_registry.json"

registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

cases = registry["cases"]

assert len(cases) == 10
assert [case["case_id"] for case in cases] == [
    f"case_{index:03d}" for index in range(1, 11)
]

for case in cases:
    case_path = ROOT / case["path"]

    assert case_path.is_dir(), case["case_id"]
    assert (case_path / "metadata.json").is_file(), case["case_id"]
    assert (case_path / "README.md").is_file(), case["case_id"]
    assert (case_path / "src").is_dir(), case["case_id"]
    assert (case_path / "reference").is_dir(), case["case_id"]
    assert (case_path / "tests").is_dir(), case["case_id"]
    assert (case_path / "trajectory").is_dir(), case["case_id"]

print("REGISTRY=VALID")
print(f"CASES={len(cases)}")
print("ORDER=001-010")
