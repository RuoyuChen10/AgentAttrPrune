from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from AgentAttrPrune.agents import WorkerSpec
from AgentAttrPrune.attribution.edge import infer_worker_agents
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.graph.random_order import RandomOrderDenseGraph
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.tracing import GenerationTraceRecorder


SPECS = [
    WorkerSpec("a", "Critic"),
    WorkerSpec("b", "Mathematician"),
    WorkerSpec("c", "Historian"),
]
ALIASES = {"a": "7vWW", "b": "5h3K", "c": "4rrd"}


class RecordingStub:
    model_name = "stub"

    def __init__(self, *, emit_observer=True):
        self.calls = []
        self.emit_observer = emit_observer

    async def agen(self, messages, *, max_tokens=256, temperature=0.0, num_comps=1):
        copied = deepcopy(messages)
        call_index = len(self.calls)
        self.calls.append(copied)
        output = (
            "A" if "top decision-maker" in copied[0]["content"]
            else f"Reasoning: call {call_index}\nDecision: A"
        )
        observer = generation_observer.get()
        if self.emit_observer and observer is not None:
            observer({
                "model_name": self.model_name,
                "messages": copied,
                "rendered_prompt": f"prompt-{call_index}",
                "input_ids": [[1, 2, call_index + 3]],
                "attention_mask": [[1, 1, 1]],
                "generated_ids": [[100 + call_index]],
                "output_text": [output],
                "generation_kwargs": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "num_completions": num_comps,
                },
            })
        return output


@pytest.mark.parametrize("emit_observer", [True, False])
def test_original_orders_match_existing_dense_execution(emit_observer):
    dense_llm = RecordingStub(emit_observer=emit_observer)
    random_llm = RecordingStub(emit_observer=emit_observer)
    dense = MultiAgentGraph(dense_llm, SPECS, reviewer_agent_ids=ALIASES)
    randomized = RandomOrderDenseGraph(
        random_llm, SPECS, round_orders=[["a", "b", "c"]] * 2,
        reviewer_agent_ids=ALIASES,
    )
    dense_trace = GenerationTraceRecorder(sample={"task": "Q"})
    random_trace = GenerationTraceRecorder(sample={"task": "Q"})
    kwargs = {"phase": "calibration", "sample_index": 7, "iteration": 3}
    first = asyncio.run(dense.run_sample("Q", recorder=dense_trace, **kwargs))
    second = asyncio.run(randomized.run_sample("Q", recorder=random_trace, **kwargs))

    assert dense_llm.calls == random_llm.calls
    assert dense_trace.events == random_trace.events
    assert first.worker_outputs == second.worker_outputs
    assert first.reviewer_output == second.reviewer_output
    assert first.final_answer == second.final_answer
    assert first.edge_counts == second.edge_counts
    assert {
        key: value for key, value in first.metrics.items() if key != "elapsed_seconds"
    } == {
        key: value for key, value in second.metrics.items() if key != "elapsed_seconds"
    }


def test_different_round_orders_keep_identity_and_exact_provenance():
    orders = [["c", "a", "b"], ["b", "c", "a"]]
    llm = RecordingStub()
    graph = RandomOrderDenseGraph(
        llm, SPECS, round_orders=orders, reviewer_agent_ids=ALIASES,
    )
    recorder = GenerationTraceRecorder(sample={"task": "Q"})
    result = asyncio.run(graph.run_sample("Q", recorder=recorder, sample_index=2))
    events = recorder.events
    assert [event["agent_id"] for event in events] == [
        "c", "a", "b", "b", "c", "a", "dense_reviewer",
    ]
    assert [event["spatial_incoming_from"] for event in events[:6]] == [
        [], ["c"], ["c", "a"], [], ["b"], ["b", "c"],
    ]
    assert all(event["temporal_incoming_from"] == [] for event in events[:3])
    assert all(event["temporal_incoming_from"] == ["a", "b", "c"] for event in events[3:6])
    for event in events[:6]:
        for incoming in event["incoming"]:
            source = events[incoming["event_id"]]
            assert source["agent_id"] == incoming["agent_id"]
            assert source["output"] == incoming["source_output"]
            assert source["generated_ids"] == incoming["source_generated_ids"]
            assert source["event_id"] < event["event_id"]
            if incoming["edge_type"] == "spatial":
                assert source["round"] == event["round"]
                assert source["agent_id"] != event["agent_id"]
            else:
                assert source["round"] == event["round"] - 1

    for outputs in result.worker_outputs:
        assert [output["agent_id"] for output in outputs] == ["a", "b", "c"]
    assert [output["event_id"] for output in result.worker_outputs[0]] == [1, 2, 0]
    assert [output["event_id"] for output in result.worker_outputs[1]] == [5, 3, 4]
    reviewer = events[-1]
    assert reviewer["spatial_incoming_from"] == ["a", "b", "c"]
    assert [incoming["event_id"] for incoming in reviewer["incoming"]] == [5, 3, 4]
    assert [incoming["reviewer_agent_id"] for incoming in reviewer["incoming"]] == [
        ALIASES[agent_id] for agent_id in ("a", "b", "c")
    ]
    prompt = reviewer["input_messages"][-1]["content"]
    assert prompt.index("7vWW:") < prompt.index("5h3K:") < prompt.index("4rrd:")
    assert recorder.config["round_orders"] == orders
    assert infer_worker_agents(recorder.payload()) == ["a", "b", "c"]
    assert result.edge_counts["spatial"] == {"active": 6, "candidate": 6}
    assert result.edge_counts["temporal"] == {"active": 9, "candidate": 9}
    assert graph.topology_for_round(0)["spatial"] == [("a", "b"), ("c", "a"), ("c", "b")]
    assert graph.topology_for_round(1)["spatial"] == [("b", "a"), ("b", "c"), ("c", "a")]
    assert graph.topology_for_round(0)["temporal"] == []
    assert graph.topology_for_round(1)["temporal"] == [
        (source, target) for source in ("a", "b", "c") for target in ("a", "b", "c")
    ]


@pytest.mark.parametrize("orders", [
    [], [["a", "b", "c"]], [["a", "b", "c"]] * 3,
    [["a", "a", "c"], ["a", "b", "c"]],
    [["a", "b"], ["a", "b", "c"]],
    [["a", "b", "d"], ["a", "b", "c"]],
    [["a", "b", 3], ["a", "b", "c"]],
    ["abc", "abc"], "abc",
])
def test_invalid_round_orders_are_rejected(orders):
    with pytest.raises(ValueError, match="round_orders"):
        RandomOrderDenseGraph(RecordingStub(), SPECS, round_orders=orders)


def test_external_policy_is_not_accepted():
    with pytest.raises(TypeError, match="policy"):
        RandomOrderDenseGraph(
            RecordingStub(), SPECS, round_orders=[["a", "b", "c"]] * 2,
            policy=None,
        )


@pytest.mark.parametrize("config", [
    {"round_orders": [["a", "b", "c"]] * 2},
    {"agents": ["c", "b", "a"]},
    {"num_rounds": 1},
])
def test_conflicting_trace_identity_is_rejected_before_generation(config):
    llm = RecordingStub()
    graph = RandomOrderDenseGraph(
        llm, SPECS, round_orders=[["c", "a", "b"], ["b", "c", "a"]],
    )
    recorder = GenerationTraceRecorder(config=config)
    before = deepcopy(recorder.config)
    with pytest.raises(ValueError, match="Trace config"):
        asyncio.run(graph.run_sample("Q", recorder=recorder))
    assert not llm.calls
    assert not recorder.events
    assert recorder.config == before


def test_caller_cannot_change_saved_orders_through_input_lists():
    orders = [["c", "a", "b"], ["b", "c", "a"]]
    graph = RandomOrderDenseGraph(RecordingStub(), SPECS, round_orders=orders)
    orders[0].reverse()
    assert graph.round_orders == (("c", "a", "b"), ("b", "c", "a"))


def test_one_worker_has_only_cross_round_self_message():
    graph = RandomOrderDenseGraph(
        RecordingStub(), SPECS[:1], round_orders=[["a"], ["a"]],
        reviewer_agent_ids={"a": ALIASES["a"]},
    )
    recorder = GenerationTraceRecorder()
    result = graph.run("Q", recorder=recorder)
    assert recorder.events[0]["incoming"] == []
    assert recorder.events[1]["spatial_incoming_from"] == []
    assert recorder.events[1]["temporal_incoming_from"] == ["a"]
    assert result.edge_counts["worker_active"] == 1
