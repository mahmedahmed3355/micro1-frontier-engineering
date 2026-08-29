import json

from agents import MockLLM
from llm.recording import RecordingLLM
from trajectories.recorder import TrajectoryRecorder


def test_recording_llm_records_agent_interaction(tmp_path):
    trajectory_path = tmp_path / "trajectory.jsonl"

    recorder = TrajectoryRecorder(trajectory_path)

    llm = RecordingLLM(
        MockLLM(response="analysis result"),
        recorder,
    )

    response = llm.invoke(
        "Analyze this CUDA kernel.",
        system_prompt=("You are a CUDA analyst."),
        metadata={
            "agent": "analyzer",
            "iteration": 0,
        },
    )

    assert response == "analysis result"

    lines = trajectory_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    event = json.loads(lines[0])

    assert event["agent"] == "analyzer"
    assert event["prompt"] == "Analyze this CUDA kernel."
    assert event["response"] == "analysis result"
    assert event["metadata"]["iteration"] == 0
    assert event["metadata"]["has_system_prompt"] is True


def test_trajectory_recorder_appends_events(tmp_path):
    path = tmp_path / "trajectory.jsonl"

    recorder = TrajectoryRecorder(path)

    recorder.record(
        agent="reader",
        prompt="read",
        response="done",
    )

    recorder.record(
        agent="analyzer",
        prompt="analyze",
        response="done",
    )

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    assert json.loads(lines[0])["agent"] == "reader"

    assert json.loads(lines[1])["agent"] == "analyzer"
