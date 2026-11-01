"""A frozen per-round order must survive every static pruning pattern."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from AgentAttrPrune.graph.fixed_order import FixedOrderGraph
from AgentAttrPrune.graph.random_order import RandomOrderDenseGraph
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.tracing import GenerationTraceRecorder
from test_random_order_graph import ALIASES, RecordingStub, SPECS
from test_sampled_dag import sparse_realization


@pytest.mark.parametrize("orders", [
    [["a", "b", "c"], ["a", "b", "c"]],
    [["c", "a", "b"], ["b", "c", "a"]],
    [["c", "b", "a"], ["b", "a", "c"]],
])
@pytest.mark.parametrize("observer", [True, False])
def test_dense_matches_existing_prescribed_order_exactly(orders, observer):
    old_llm, new_llm = RecordingStub(emit_observer=observer), RecordingStub(emit_observer=observer)
    old = RandomOrderDenseGraph(old_llm, SPECS, round_orders=orders, reviewer_agent_ids=ALIASES)
    new = FixedOrderGraph(new_llm, SPECS, round_orders=orders, reviewer_agent_ids=ALIASES)
    a, b = GenerationTraceRecorder(), GenerationTraceRecorder()
    first = old.run("Q", recorder=a, sample_index=9)
    second = new.run("Q", recorder=b, sample_index=9)
    assert old_llm.calls == new_llm.calls
    assert a.events == b.events
    assert first.worker_outputs == second.worker_outputs
    assert first.reviewer_output == second.reviewer_output
    assert first.final_answer == second.final_answer
    assert first.edge_counts == second.edge_counts
    assert first.edge_counts["worker_candidate"] == 15
    for metric in ("llm_calls", "prompt_tokens", "generated_tokens", "truncated_events"):
        assert first.metrics[metric] == second.metrics[metric]


def test_sparse_prefix_inputs_keep_canonical_output_provenance():
    policy = sparse_realization()
    graph = FixedOrderGraph(RecordingStub(), SPECS, policy=policy,
                            round_orders=policy["round_orders"], reviewer_agent_ids=ALIASES)
    recorder = GenerationTraceRecorder()
    result = graph.run("Q", recorder=recorder)
    assert [e["agent_id"] for e in recorder.events] == ["c", "b", "a", "b", "c", "a", "dense_reviewer"]
    assert recorder.events[2]["spatial_incoming_from"] == ["c", "b"]
    prompt = recorder.events[2]["input_messages"][-1]["content"]
    assert prompt.index("Agent c,") < prompt.index("Agent b,")
    assert recorder.events[5]["temporal_incoming_from"] == ["a", "c"]
    assert recorder.events[-1]["spatial_incoming_from"] == ["a", "b", "c"]
    for event in recorder.events:
        for incoming in event["incoming"]:
            source = recorder.events[incoming["event_id"]]
            assert source["agent_id"] == incoming["agent_id"]
            assert source["output"] == incoming["source_output"]
            assert source["generated_ids"] == incoming["source_generated_ids"]
            assert source["event_id"] < event["event_id"]
            if event["agent_id"] != "dense_reviewer":
                assert source["round"] == event["round"] - (incoming["edge_type"] == "temporal")
    assert [o["agent_id"] for o in result.worker_outputs[0]] == ["a", "b", "c"]
    assert [o["event_id"] for o in result.worker_outputs[0]] == [2, 1, 0]
    assert [item["reviewer_agent_id"] for item in recorder.events[-1]["incoming"]] == [ALIASES[a] for a in ("a", "b", "c")]
    assert result.edge_counts["worker_candidate"] == 15
    assert result.edge_counts["worker_active"] == 8


def test_empty_spatial_policy_does_not_reorder_disconnected_workers():
    policy = sparse_realization()
    policy["masks"]["spatial"] = [[[0] * 3 for _ in range(3)] for _ in range(2)]
    policy["masks"]["temporal"][1] = [[int(i == j) for j in range(3)] for i in range(3)]
    recorder = GenerationTraceRecorder()
    result = FixedOrderGraph(RecordingStub(), SPECS, policy=policy,
                            round_orders=policy["round_orders"]).run("Q", recorder=recorder)
    assert [e["agent_id"] for e in recorder.events[:6]] == ["c", "b", "a", "b", "c", "a"]
    assert all(not e["spatial_incoming_from"] for e in recorder.events[:6])
    assert all(e["temporal_incoming_from"] == [e["agent_id"]] for e in recorder.events[3:6])
    assert result.metrics["llm_calls"] == 7
    assert result.edge_counts["worker_active"] == 3


@pytest.mark.parametrize("mutation,match", [
    ("order", "conflict"), ("reverse", "violates"), ("self", "self edges"),
    ("temporal", "round zero"), ("identity", "agent"),
])
def test_invalid_policy_is_rejected_before_generation(mutation, match):
    policy = sparse_realization()
    orders = deepcopy(policy["round_orders"])
    if mutation == "order":
        policy["round_orders"][0].reverse()
    elif mutation == "reverse":
        policy["masks"]["spatial"][0] = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
    elif mutation == "self":
        policy["masks"]["spatial"][0][0][0] = 1
    elif mutation == "temporal":
        policy["masks"]["temporal"][0][0][0] = 1
    else:
        policy["agents"].reverse()
    llm = RecordingStub()
    with pytest.raises(ValueError, match=match):
        FixedOrderGraph(llm, SPECS, policy=policy, round_orders=orders)
    assert not llm.calls


def test_input_mutation_does_not_change_graph_and_trace_conflict_is_atomic():
    policy = sparse_realization()
    original = deepcopy(policy)
    graph = FixedOrderGraph(RecordingStub(), SPECS, policy=policy, round_orders=policy["round_orders"])
    policy["round_orders"][0].reverse()
    policy["masks"]["spatial"][0][0][0] = 1
    trace = GenerationTraceRecorder()
    graph.run("Q", recorder=trace)
    assert trace.config["round_orders"] == original["round_orders"]
    assert trace.config["fixed_order_masks"] == original["masks"]
    bad = GenerationTraceRecorder(config={"fixed_order_masks": {}})
    before = deepcopy(bad.config)
    with pytest.raises(ValueError, match="fixed_order_masks"):
        graph.run("Q", recorder=bad)
    assert bad.config == before and not bad.events


def test_batch_and_oom_metrics_preserve_current_sample_provenance():
    class BatchStub(RecordingStub):
        async def agen(self, messages, **kwargs):
            observer = generation_observer.get()
            def annotated(generation):
                generation["generation_kwargs"].update(effective_batch_size=4, oom_fallback=True, oom_split_depth=2)
                observer(generation)
            token = generation_observer.set(annotated)
            try:
                return await super().agen(messages, **kwargs)
            finally:
                generation_observer.reset(token)
    graph = FixedOrderGraph(BatchStub(), SPECS, round_orders=[["c", "b", "a"]] * 2)
    recorder = GenerationTraceRecorder()
    graph.run("first", recorder=recorder)
    result = graph.run("second", recorder=recorder)
    assert len(recorder.events) == 14
    assert result.metrics["llm_calls"] == result.metrics["batch_metadata_calls"] == 7
    assert result.metrics["oom_fallback_calls"] == 7
    assert result.metrics["oom_max_split_depth"] == 2


def test_concurrent_samples_do_not_mix_sources_or_identity():
    graph = FixedOrderGraph(RecordingStub(), SPECS, round_orders=[["c", "a", "b"], ["b", "c", "a"]])
    first, second = GenerationTraceRecorder(), GenerationTraceRecorder()
    async def both():
        await asyncio.gather(graph.run_sample("first", recorder=first, sample_index=4),
                             graph.run_sample("second", recorder=second, sample_index=5))
    asyncio.run(both())
    for recorder, index, task in ((first, 4, "first"), (second, 5, "second")):
        assert len(recorder.events) == 7
        assert {e["sample_index"] for e in recorder.events} == {index}
        assert all(task in e["input_messages"][-1]["content"] for e in recorder.events)
