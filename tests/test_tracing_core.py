from __future__ import annotations

import json

import pytest

from AgentAttrPrune.llm import HFChat, generation_observer
from AgentAttrPrune.llm.hf_chat import _generation_finish_reasons
from AgentAttrPrune.tracing import GenerationTraceRecorder, load_trace


def test_generation_trace_schema_and_exact_token_ids(tmp_path):
    path = tmp_path / "sample" / "trace.json"
    recorder = GenerationTraceRecorder(
        path, config={"phase": "eval"}, sample={"task": "question"}
    )
    invocation = recorder.new_invocation_id()
    metadata = {
        "invocation_id": invocation,
        "phase": "eval",
        "round": 0,
        "agent_id": "a",
        "agent_type": "AnalyzeAgent",
        "role": "Critic",
        "incoming": [],
    }
    with recorder.observe(metadata):
        generation_observer.get()(
            {
                "model_name": "fake",
                "messages": [{"role": "user", "content": "question"}],
                "rendered_prompt": "prompt",
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "generated_ids": [[4, 5]],
                "output_text": ["answer"],
                "generation_kwargs": {"max_tokens": 2, "temperature": 0.0},
            }
        )

    recorder.write({"output": "answer"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["token_ids_complete"] is True
    assert payload["generation_complete"] is True
    assert payload["truncated_event_ids"] == []
    assert payload["events"][0]["input_ids"] == [[1, 2, 3]]
    assert payload["events"][0]["generated_ids"] == [[4, 5]]
    assert payload["events"][0]["finish_reason"] == "unknown"
    assert payload["events"][0]["hit_max_new_tokens"] is False
    assert payload["usage"] == {
        "llm_calls": 1,
        "prompt_tokens": 3,
        "generated_tokens": 2,
        "truncated_events": 0,
    }


def test_generation_trace_marks_length_truncation():
    recorder = GenerationTraceRecorder()
    invocation = recorder.new_invocation_id()
    recorder.record_generation(
        invocation,
        {
            "invocation_id": invocation,
            "round": 0,
            "agent_id": "a",
            "agent_type": "AnalyzeAgent",
            "role": "Critic",
            "incoming": [],
        },
        {
            "input_ids": [[1]],
            "generated_ids": [[2, 3]],
            "output_text": ["unfinished"],
            "finish_reasons": ["length"],
            "hit_max_new_tokens": [True],
            "generation_kwargs": {"max_tokens": 2},
        },
    )

    payload = recorder.payload()

    assert payload["generation_complete"] is False
    assert payload["truncated_event_ids"] == [0]
    assert payload["events"][0]["finish_reason"] == "length"
    assert payload["events"][0]["hit_max_new_tokens"] is True
    assert payload["usage"]["truncated_events"] == 1


def test_hf_generation_finish_reason_distinguishes_eos_length_and_stop():
    assert _generation_finish_reasons(
        [[1, 2, 9], [1, 2, 3], [1, 2]],
        eos_token_id=9,
        max_tokens=3,
    ) == ["eos", "length", "stopped"]


def test_legacy_trace_is_debug_only(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"config": {}, "sample": {}, "events": [], "final": {}}),
        encoding="utf-8",
    )
    payload = load_trace(path)
    assert payload["schema_version"] == 0
    assert payload["legacy_schema"] is True
    with pytest.raises(ValueError, match="debug-only"):
        load_trace(path, for_aggregation=True)
    with pytest.raises(ValueError, match="no schema_version"):
        load_trace(path, allow_legacy=False)


def test_source_event_provenance_is_preserved():
    recorder = GenerationTraceRecorder()
    source = {
        "invocation_id": recorder.new_invocation_id(),
        "phase": "calibration",
        "round": 0,
        "agent_id": "a",
        "agent_type": "AnalyzeAgent",
        "role": "Critic",
        "incoming": [],
    }
    target = {
        "invocation_id": recorder.new_invocation_id(),
        "phase": "calibration",
        "round": 0,
        "agent_id": "b",
        "agent_type": "AnalyzeAgent",
        "role": "Mathematician",
        "incoming": [
            {
                "event_id": 0,
                "round": 0,
                "agent_id": "a",
                "agent_type": "AnalyzeAgent",
                "role": "Critic",
                "edge_type": "spatial",
                "source_generated_ids": [[9]],
                "source_output": "x",
            }
        ],
    }
    for metadata in (source, target):
        recorder.record_fallback(
            metadata["invocation_id"],
            metadata,
            messages=[],
            output="x",
        )
    assert recorder.event(1)["incoming"][0]["event_id"] == 0
    assert recorder.event(1)["spatial_incoming_from"] == ["a"]
    assert recorder.payload()["token_ids_complete"] is False


@pytest.mark.parametrize("name", ["Qwen3-8B", "/models/Qwen3-8B", "openai:gpt"])
def test_hf_chat_requires_an_explicit_local_or_hf_prefix(name):
    with pytest.raises(ValueError, match="local:"):
        HFChat.resolve_model_name(name)


def test_hf_chat_resolves_supported_names_without_loading_model():
    assert HFChat.resolve_model_name("local:/models/Qwen3-8B") == "/models/Qwen3-8B"
    assert HFChat.resolve_model_name("hf:Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"
