"""A realized sample must execute exactly its edges, orders and identities."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from AgentAttrPrune.attribution.edge import infer_worker_agents
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.graph.sampled_dag import SampledDAGGraph
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder
from test_random_order_graph import ALIASES, RecordingStub, SPECS


def dense_realization():
    return {
        "agents": ["a", "b", "c"], "num_rounds": 2,
        "masks": RoundPolicy.dense(["a", "b", "c"], num_rounds=2).to_dict()["masks"],
        "round_orders": [["a", "b", "c"], ["a", "b", "c"]],
        "seed": 19,
    }


def sparse_realization():
    return {
        "agents": ["a", "b", "c"], "num_rounds": 2,
        "masks": {
            "spatial": [
                [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
            ],
            "temporal": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
            ],
        },
        "round_orders": [["c", "b", "a"], ["b", "c", "a"]],
        "seed": 42,
    }


@pytest.mark.parametrize("observer", [True, False])
def test_canonical_dense_matches_existing_prompts_outputs_and_trace(observer):
    dense_llm, sampled_llm = RecordingStub(emit_observer=observer), RecordingStub(emit_observer=observer)
    dense = MultiAgentGraph(dense_llm, SPECS, reviewer_agent_ids=ALIASES)
    sampled = SampledDAGGraph(sampled_llm, SPECS, realized_graph=dense_realization(), reviewer_agent_ids=ALIASES)
    dense_trace, sampled_trace = GenerationTraceRecorder(), GenerationTraceRecorder()
    kwargs = {"phase": "evaluation", "sample_index": 8, "iteration": 2, "batch_index": 1}
    first = asyncio.run(dense.run_sample("Q", recorder=dense_trace, **kwargs))
    second = asyncio.run(sampled.run_sample("Q", recorder=sampled_trace, **kwargs))
    assert dense_llm.calls == sampled_llm.calls
    assert dense_trace.events == sampled_trace.events
    assert first.worker_outputs == second.worker_outputs
    assert first.reviewer_output == second.reviewer_output
    assert first.final_answer == second.final_answer
    for key in ("llm_calls", "prompt_tokens", "generated_tokens", "truncated_events"):
        assert first.metrics[key] == second.metrics[key]
    assert first.edge_counts["worker_active"] == second.edge_counts["worker_active"] == 15
    # Sampled graphs count all directed candidate slots, including both directions.
    assert second.edge_counts["worker_candidate"] == 21
    assert second.to_trace_final()["edge_counts"] == second.edge_counts


def test_reverse_edges_keep_identity_and_canonical_input_order():
    realized = sparse_realization()
    llm = RecordingStub()
    graph = SampledDAGGraph(llm, SPECS, realized_graph=realized, reviewer_agent_ids=ALIASES)
    recorder = GenerationTraceRecorder()
    result = graph.run("Q", recorder=recorder, sample_index=3)
    events = recorder.events
    assert [event["agent_id"] for event in events] == ["c", "b", "a", "b", "c", "a", "dense_reviewer"]
    # c executed first, but AgentPrune source-major predecessor order is b, c.
    assert events[2]["spatial_incoming_from"] == ["b", "c"]
    assert events[4]["spatial_incoming_from"] == ["b"]
    assert events[5]["spatial_incoming_from"] == ["b"]
    assert events[3]["temporal_incoming_from"] == ["b"]
    assert events[4]["temporal_incoming_from"] == ["c"]
    assert events[5]["temporal_incoming_from"] == ["a", "c"]
    for event in events[:6]:
        for incoming in event["incoming"]:
            source = events[incoming["event_id"]]
            assert source["agent_id"] == incoming["agent_id"]
            assert source["event_id"] < event["event_id"]
            assert source["output"] == incoming["source_output"]
            assert source["generated_ids"] == incoming["source_generated_ids"]
            assert source["round"] == event["round"] - (incoming["edge_type"] == "temporal")
    spatial_text = events[2]["input_messages"][-1]["content"]
    assert spatial_text.index("Agent b,") < spatial_text.index("Agent c,")
    assert [output["agent_id"] for output in result.worker_outputs[0]] == ["a", "b", "c"]
    assert [output["event_id"] for output in result.worker_outputs[0]] == [2, 1, 0]
    assert events[-1]["spatial_incoming_from"] == ["a", "b", "c"]
    assert [item["reviewer_agent_id"] for item in events[-1]["incoming"]] == [ALIASES[a] for a in ("a", "b", "c")]
    assert recorder.config["realized_graph"] == realized
    assert infer_worker_agents(recorder.payload()) == ["a", "b", "c"]
    assert result.edge_counts["spatial"] == {"active": 4, "candidate": 12}
    assert result.edge_counts["temporal"] == {"active": 4, "candidate": 9}
    assert result.edge_counts["reviewer_dense_edges"] == 3
    assert result.edge_counts["worker_retention"] == 8 / 21


def test_recorded_incoming_edges_equal_realized_masks():
    realized = sparse_realization()
    graph = SampledDAGGraph(RecordingStub(), SPECS, realized_graph=realized, reviewer_agent_ids=ALIASES)
    recorder = GenerationTraceRecorder()
    graph.run("Q", recorder=recorder)
    for round_index in range(2):
        for edge_type in ("spatial", "temporal"):
            actual = {
                (incoming["agent_id"], event["agent_id"])
                for event in recorder.events if event["round"] == round_index
                for incoming in event["incoming"] if incoming["edge_type"] == edge_type
            }
            expected = {
                (source, target)
                for i, source in enumerate(graph.agent_ids)
                for j, target in enumerate(graph.agent_ids)
                if realized["masks"][edge_type][round_index][i][j]
            }
            assert actual == expected == set(graph.topology_for_round(round_index)[edge_type])


@pytest.mark.parametrize("mutation, error", [
    ("cycle", "cyclic"), ("self", "self edges"), ("temporal_zero", "round zero"),
    ("dependency", "violates"), ("duplicate_order", "exactly once"),
    ("missing_order", "one permutation"), ("agent_order", "identity order"),
    ("round_count", "num_rounds"), ("nonbinary", "0/1"),
])
def test_invalid_realizations_are_rejected_before_generation(mutation, error):
    realized = dense_realization()
    if mutation == "cycle":
        realized["masks"]["spatial"][0][1][0] = 1
    elif mutation == "self":
        realized["masks"]["spatial"][0][0][0] = 1
    elif mutation == "temporal_zero":
        realized["masks"]["temporal"][0][0][0] = 1
    elif mutation == "dependency":
        realized["round_orders"][0].reverse()
    elif mutation == "duplicate_order":
        realized["round_orders"][0] = ["a", "a", "c"]
    elif mutation == "missing_order":
        realized["round_orders"].pop()
    elif mutation == "agent_order":
        realized["agents"].reverse()
    elif mutation == "round_count":
        realized["num_rounds"] = 3
    else:
        realized["masks"]["spatial"][0][0][1] = 0.5
    llm = RecordingStub()
    with pytest.raises(ValueError, match=error):
        SampledDAGGraph(llm, SPECS, realized_graph=realized)
    assert not llm.calls


def test_conflicting_trace_masks_are_rejected_without_modification():
    llm = RecordingStub()
    graph = SampledDAGGraph(llm, SPECS, realized_graph=sparse_realization())
    recorder = GenerationTraceRecorder(config={"realized_graph": dense_realization()})
    before = deepcopy(recorder.config)
    with pytest.raises(ValueError, match="realized_graph"):
        graph.run("Q", recorder=recorder)
    assert recorder.config == before
    assert not recorder.events
    assert not llm.calls


def test_caller_mutation_cannot_change_masks_orders_or_trace_metadata():
    realized = sparse_realization()
    original = deepcopy(realized)
    graph = SampledDAGGraph(RecordingStub(), SPECS, realized_graph=realized)
    realized["masks"]["spatial"][0][0][0] = 1
    realized["round_orders"][0].reverse()
    exported = graph.realized_graph
    exported["round_orders"][0].reverse()
    recorder = GenerationTraceRecorder()
    graph.run("Q", recorder=recorder)
    assert recorder.config["realized_graph"] == original


def test_disconnected_workers_and_temporal_self_still_execute_once():
    realized = dense_realization()
    realized["masks"]["spatial"] = [[[0] * 3 for _ in range(3)] for _ in range(2)]
    realized["masks"]["temporal"][1] = [[int(i == j) for j in range(3)] for i in range(3)]
    # This supplied topological order must be honored, without extra shuffling.
    realized["round_orders"] = [["b", "a", "c"], ["c", "b", "a"]]
    recorder = GenerationTraceRecorder()
    graph = SampledDAGGraph(RecordingStub(), SPECS, realized_graph=realized)
    result = graph.run("Q", recorder=recorder)
    assert [e["agent_id"] for e in recorder.events[:6]] == ["b", "a", "c", "c", "b", "a"]
    assert all(not e["spatial_incoming_from"] for e in recorder.events[:6])
    assert all(e["temporal_incoming_from"] == [e["agent_id"]] for e in recorder.events[3:6])
    assert result.metrics["llm_calls"] == 7
    assert result.edge_counts["worker_active"] == 3
