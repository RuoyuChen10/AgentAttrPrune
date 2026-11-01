from __future__ import annotations

import asyncio
import threading
import time

import pytest
from torch.cuda import OutOfMemoryError

from AgentAttrPrune.agents import WorkerSpec
from AgentAttrPrune.graph import GraphExecutionError, MultiAgentGraph
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder


SPECS = [
    WorkerSpec("a", "Critic"),
    WorkerSpec("b", "Mathematician"),
    WorkerSpec("c", "Historian"),
]
REVIEWER_AGENT_IDS = {"a": "7vWW", "b": "5h3K", "c": "4rrd"}


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
    dense = MultiAgentGraph(
        dense_llm,
        SPECS,
        num_rounds=2,
        reviewer_agent_ids=REVIEWER_AGENT_IDS,
    )
    full_policy = RoundPolicy.dense(
        [spec.id for spec in SPECS],
        num_rounds=2,
        agent_roles=[spec.role for spec in SPECS],
        agent_types=["AnalyzeAgent"] * 3,
    )
    explicit = MultiAgentGraph(
        explicit_llm,
        SPECS,
        num_rounds=2,
        policy=full_policy,
        reviewer_agent_ids=REVIEWER_AGENT_IDS,
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
        RecordingStub(),
        SPECS,
        num_rounds=2,
        max_new_tokens=12,
        reviewer_agent_ids=REVIEWER_AGENT_IDS,
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
    reviewer_prompt = reviewer["input_messages"][-1]["content"]
    assert "7vWW: Reasoning: stub" in reviewer_prompt
    assert "5h3K: Reasoning: stub" in reviewer_prompt
    assert "4rrd: Reasoning: stub" in reviewer_prompt
    assert "\n\na: Reasoning: stub" not in reviewer_prompt
    assert "\n\nb: Reasoning: stub" not in reviewer_prompt
    assert "\n\nc: Reasoning: stub" not in reviewer_prompt
    assert [
        incoming["reviewer_agent_id"] for incoming in reviewer["incoming"]
    ] == ["7vWW", "5h3K", "4rrd"]
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


def test_batch_provenance_is_preserved_and_counts_only_current_sample():
    metadata = [
        {"effective_batch_size": 1, "oom_fallback": True, "oom_split_depth": 3},
        {"effective_batch_size": 1, "oom_fallback": True, "oom_split_depth": 3},
        {"effective_batch_size": 1, "oom_fallback": True, "oom_split_depth": 3},
        {"effective_batch_size": 2, "oom_fallback": True, "oom_split_depth": 1},
        {"effective_batch_size": 4, "oom_fallback": False, "oom_split_depth": 0},
        {},
    ]

    class BatchMetadataStub(RecordingStub):
        async def agen(self, messages, **kwargs):
            observer = generation_observer.get()
            call_metadata = metadata[len(self.calls)]

            def add_metadata(generation):
                generation["generation_kwargs"].update(call_metadata)
                observer(generation)

            token = generation_observer.set(add_metadata)
            try:
                return await super().agen(messages, **kwargs)
            finally:
                generation_observer.reset(token)

    graph = MultiAgentGraph(BatchMetadataStub(), SPECS[:2], num_rounds=1)
    recorder = GenerationTraceRecorder(sample={"task": "Q"})
    asyncio.run(graph.run_sample({"task": "earlier"}, recorder=recorder))
    result = asyncio.run(graph.run_sample({"task": "current"}, recorder=recorder))

    assert len(recorder.events) == 6
    assert result.metrics["llm_calls"] == 3
    assert result.metrics["batch_metadata_calls"] == 2
    assert result.metrics["oom_fallback_calls"] == 1
    assert result.metrics["oom_max_split_depth"] == 1
    for event, expected in zip(recorder.events[3:], metadata[3:]):
        assert {key: event["llm_call"][key] for key in expected} == expected
    assert "effective_batch_size" not in recorder.events[-1]["llm_call"]

    legacy_graph = MultiAgentGraph(RecordingStub(), SPECS[:2], num_rounds=1)
    legacy_result = asyncio.run(
        legacy_graph.run_sample({"task": "legacy backend"}, recorder=recorder)
    )
    assert legacy_result.metrics["llm_calls"] == 3
    assert not {
        "batch_metadata_calls", "oom_fallback_calls", "oom_max_split_depth"
    }.intersection(legacy_result.metrics)


def test_cuda_oom_is_not_retried_and_preserves_original_cause():
    error = OutOfMemoryError("synthetic CUDA out of memory")

    class OOMStub:
        calls = 0

        async def agen(self, messages, **kwargs):
            self.calls += 1
            raise error

    llm = OOMStub()
    graph = MultiAgentGraph(llm, SPECS[:1], num_rounds=1, max_tries=3)
    recorder = GenerationTraceRecorder(sample={"task": "Q"})

    with pytest.raises(GraphExecutionError, match="failed after 1 attempts") as raised:
        asyncio.run(graph.run_sample({"task": "Q"}, recorder=recorder))

    assert llm.calls == 1
    assert raised.value.__cause__ is error
    assert recorder.events == []


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
