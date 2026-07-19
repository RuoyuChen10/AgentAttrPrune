from __future__ import annotations

import asyncio
import threading
import time

import pytest

from AgentAttrPrune.agents import WorkerSpec
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder


SPECS = [
    WorkerSpec("a", "Critic"),
    WorkerSpec("b", "Mathematician"),
    WorkerSpec("c", "Historian"),
]


class RecordingStub:
    model_name = "stub"

    def __init__(self):
        self.calls = []

    async def agen(
        self, messages, *, max_tokens=256, temperature=0.0, num_comps=1
    ):
        copied = [dict(message) for message in messages]
        self.calls.append(copied)
        is_reviewer = "top decision-maker" in copied[0]["content"]
        output = "A" if is_reviewer else "Reasoning: stub\nDecision: A"
        observer = generation_observer.get()
        if observer is not None:
            observer(
                {
                    "model_name": self.model_name,
                    "messages": copied,
                    "rendered_prompt": "rendered",
                    "input_ids": [[1, 2, 3]],
                    "attention_mask": [[1, 1, 1]],
                    "generated_ids": [[4, 5]],
                    "output_text": [output],
                    "generation_kwargs": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "num_completions": num_comps,
                    },
                }
            )
        return output


class SlowFirstAttemptStub:
    """Emit one observer callback after its attempt has already timed out."""

    model_name = "stub"

    def __init__(self):
        self._next_attempt = 0
        self._lock = threading.Lock()

    def _generate(self, messages, max_tokens, temperature, num_comps):
        with self._lock:
            attempt = self._next_attempt
            self._next_attempt += 1
        if attempt == 0:
            time.sleep(0.08)
        output = f"attempt-{attempt}"
        observer = generation_observer.get()
        if observer is not None:
            observer(
                {
                    "model_name": self.model_name,
                    "messages": [dict(message) for message in messages],
                    "rendered_prompt": f"prompt-{attempt}",
                    "input_ids": [[attempt + 1]],
                    "attention_mask": [[1]],
                    "generated_ids": [[attempt + 10]],
                    "output_text": [output],
                    "generation_kwargs": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "num_completions": num_comps,
                    },
                }
            )
        return output

    async def agen(
        self, messages, *, max_tokens=256, temperature=0.0, num_comps=1
    ):
        return await asyncio.to_thread(
            self._generate, messages, max_tokens, temperature, num_comps
        )


def test_dense_topology_matches_full_connected_candidate_definition():
    graph = MultiAgentGraph(RecordingStub(), SPECS, num_rounds=2)

    assert graph.topology_for_round(0) == {
        "spatial": [("a", "b"), ("a", "c"), ("b", "c")],
        "temporal": [],
    }
    assert graph.topology_for_round(1) == {
        "spatial": [("a", "b"), ("a", "c"), ("b", "c")],
        "temporal": [
            (source, target)
            for source in ("a", "b", "c")
            for target in ("a", "b", "c")
        ],
    }


def test_dense_and_explicit_full_policy_share_prompts_and_outputs():
    dense_llm = RecordingStub()
    explicit_llm = RecordingStub()
    dense = MultiAgentGraph(dense_llm, SPECS, num_rounds=2)
    full_policy = RoundPolicy.dense(
        [spec.id for spec in SPECS],
        num_rounds=2,
        agent_roles=[spec.role for spec in SPECS],
        agent_types=["AnalyzeAgent"] * 3,
    )
    explicit = MultiAgentGraph(
        explicit_llm, SPECS, num_rounds=2, policy=full_policy
    )

    dense_result = asyncio.run(dense.run_sample({"task": "Question"}))
    explicit_result = asyncio.run(explicit.run_sample({"task": "Question"}))

    assert dense_result.final_answer == explicit_result.final_answer == "A"
    assert dense_result.worker_outputs == explicit_result.worker_outputs
    assert dense_llm.calls == explicit_llm.calls
    assert dense_result.edge_counts["worker_retention"] == 1.0


def test_round_provenance_and_reviewer_are_recorded_dense(tmp_path):
    recorder = GenerationTraceRecorder(
        tmp_path / "trace.json", config={"split": "dev"}, sample={"task": "Q"}
    )
    graph = MultiAgentGraph(
        RecordingStub(), SPECS, num_rounds=2, max_new_tokens=12
    )

    result = asyncio.run(
        graph.run_sample(
            {"task": "Q"}, recorder=recorder, phase="calibration", sample_index=4
        )
    )

    assert len(recorder.events) == 7
    round_zero = recorder.events[:3]
    round_one = recorder.events[3:6]
    reviewer = recorder.events[6]
    assert round_zero[0]["incoming_from"] == []
    assert round_zero[1]["spatial_incoming_from"] == ["a"]
    assert round_zero[2]["spatial_incoming_from"] == ["a", "b"]
    assert round_one[0]["temporal_incoming_from"] == ["a", "b", "c"]
    assert round_one[1]["spatial_incoming_from"] == ["a"]
    assert reviewer["agent_type"] == "FinalRefer"
    assert reviewer["spatial_incoming_from"] == ["a", "b", "c"]
    assert reviewer["temporal_incoming_from"] == []
    assert result.metrics["llm_calls"] == 7
    assert result.metrics["prompt_tokens"] == 21
    assert result.metrics["generated_tokens"] == 14
    assert result.edge_counts["reviewer_dense_edges"] == 3

    path = recorder.write(result.to_trace_final())
    assert path.exists()


def test_task_local_observer_does_not_mix_concurrent_sample_recorders():
    graph = MultiAgentGraph(RecordingStub(), SPECS[:2], num_rounds=1)
    first = GenerationTraceRecorder(sample={"task": "first"})
    second = GenerationTraceRecorder(sample={"task": "second"})

    async def run_both():
        return await asyncio.gather(
            graph.run_sample(
                {"task": "first"}, recorder=first, sample_index=10
            ),
            graph.run_sample(
                {"task": "second"}, recorder=second, sample_index=20
            ),
        )

    asyncio.run(run_both())
    assert len(first.events) == len(second.events) == 3
    assert {event["sample_index"] for event in first.events} == {10}
    assert {event["sample_index"] for event in second.events} == {20}
    assert all("first" in event["input_messages"][-1]["content"] for event in first.events)
    assert all("second" in event["input_messages"][-1]["content"] for event in second.events)


def test_timed_out_thread_generation_is_not_committed_over_successful_retry():
    recorder = GenerationTraceRecorder(sample={"task": "retry"})
    graph = MultiAgentGraph(
        SlowFirstAttemptStub(),
        SPECS[:1],
        num_rounds=1,
        timeout=0.02,
        max_tries=2,
    )

    async def run_and_drain_timed_out_thread():
        result = await graph.run_sample({"task": "retry"}, recorder=recorder)
        await asyncio.sleep(0.10)
        return result

    result = asyncio.run(run_and_drain_timed_out_thread())

    assert result.worker_outputs[0][0]["output"] == "attempt-1"
    assert len(recorder.events) == 2
    assert recorder.events[0]["output"] == "attempt-1"
    assert recorder.events[0]["generated_ids"] == [[11]]
    assert recorder.events[1]["incoming"][0]["source_generated_ids"] == [[11]]


def test_pruned_policy_is_round_specific():
    policy = RoundPolicy.from_dict(
        {
            "agents": ["a", "b", "c"],
            "agent_roles": ["Critic", "Mathematician", "Historian"],
            "num_rounds": 2,
            "matrix_orientation": "[round][source][target]",
            "masks": {
                "spatial": [
                    [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                ],
                "temporal": [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[1, 0, 0], [0, 0, 1], [0, 0, 0]],
                ],
            },
        }
    )
    graph = MultiAgentGraph(RecordingStub(), SPECS, 2, policy=policy)

    assert graph.topology_for_round(0) == {
        "spatial": [("a", "b")],
        "temporal": [],
    }
    assert graph.topology_for_round(1) == {
        "spatial": [("a", "c")],
        "temporal": [("a", "a"), ("b", "c")],
    }


def test_graph_rejects_policy_outside_ordered_spatial_candidates():
    policy = RoundPolicy.from_dict(
        {
            "agents": ["a", "b", "c"],
            "num_rounds": 1,
            "masks": {
                "spatial": [[[0, 0, 0], [1, 0, 0], [0, 0, 0]]],
                "temporal": [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            },
        }
    )
    with pytest.raises(ValueError, match="source < target"):
        MultiAgentGraph(RecordingStub(), SPECS, 1, policy=policy)


def test_worker_spec_and_generation_validation():
    with pytest.raises(ValueError, match="unique"):
        MultiAgentGraph(
            RecordingStub(),
            [WorkerSpec("a", "Critic"), WorkerSpec("a", "Historian")],
        )
    with pytest.raises(ValueError, match="positive"):
        MultiAgentGraph(RecordingStub(), SPECS, num_rounds=0)
