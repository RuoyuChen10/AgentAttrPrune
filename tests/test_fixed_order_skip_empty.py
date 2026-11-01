"""The opt-in empty-round rule changes only the final source of review."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from AgentAttrPrune.agents import WorkerSpec
from AgentAttrPrune.graph.fixed_order import FixedOrderGraph
from AgentAttrPrune.graph.fixed_order_skip_empty import (
    SkipEmptyFinalRoundGraph, execution_contract_for_policy,
)
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder
from test_random_order_graph import ALIASES, RecordingStub, SPECS
from test_sampled_dag import sparse_realization


def empty_final_policy():
    policy = sparse_realization()
    for edge in ("spatial", "temporal"):
        policy["masks"][edge][1] = [[0] * 3 for _ in range(3)]
    return policy


@pytest.mark.parametrize("observer", [True, False])
def test_skip_keeps_round_zero_and_reviews_its_canonical_outputs(observer):
    policy = empty_final_policy()
    old_llm, new_llm = RecordingStub(emit_observer=observer), RecordingStub(emit_observer=observer)
    kwargs = {"policy": policy, "round_orders": policy["round_orders"], "reviewer_agent_ids": ALIASES}
    old = FixedOrderGraph(old_llm, SPECS, **kwargs)
    new = SkipEmptyFinalRoundGraph(new_llm, SPECS, **kwargs)
    old_trace, new_trace = GenerationTraceRecorder(), GenerationTraceRecorder()
    first = old.run("Q", recorder=old_trace, sample_index=4)
    second = new.run("Q", recorder=new_trace, sample_index=4)
    assert old_trace.events[:3] == new_trace.events[:3]
    assert old_llm.calls[:3] == new_llm.calls[:3]
    assert second.worker_outputs == first.worker_outputs[:1]
    assert len(old_trace.events) == 7 and len(new_trace.events) == 4
    assert [event["round"] for event in new_trace.events] == [0, 0, 0, 2]
    reviewer = new_trace.events[-1]
    assert reviewer["spatial_incoming_from"] == ["a", "b", "c"]
    assert [incoming["event_id"] for incoming in reviewer["incoming"]] == [2, 1, 0]
    assert [incoming["reviewer_agent_id"] for incoming in reviewer["incoming"]] == [ALIASES[a] for a in ("a", "b", "c")]
    for incoming in reviewer["incoming"]:
        source = new_trace.events[incoming["event_id"]]
        assert source["round"] == 0
        assert source["output"] == incoming["source_output"]
        assert source["generated_ids"] == incoming["source_generated_ids"]
    assert new_trace.config["num_rounds"] == 2
    assert new_trace.config["round_orders"] == policy["round_orders"]
    assert new_trace.config["fixed_order_masks"] == policy["masks"]
    assert new_trace.config["execution_contract"] == {
        "rule_version": 1, "rule": "skip_empty_final_round",
        "executed_worker_rounds": [0], "skipped_worker_rounds": [1], "reviewer_source_round": 0,
    }
    assert first.edge_counts == second.edge_counts
    assert first.metrics["llm_calls"] == 7 and second.metrics["llm_calls"] == 4
    assert "execution_contract" not in old_trace.config


@pytest.mark.parametrize("remaining", ["spatial", "temporal", "both"])
def test_any_final_round_edge_preserves_original_execution(remaining):
    policy = empty_final_policy()
    if remaining in ("spatial", "both"):
        policy["masks"]["spatial"][1][1][2] = 1
    if remaining in ("temporal", "both"):
        policy["masks"]["temporal"][1][2][2] = 1
    a, b = GenerationTraceRecorder(), GenerationTraceRecorder()
    first_llm, second_llm = RecordingStub(), RecordingStub()
    kwargs = {"policy": policy, "round_orders": policy["round_orders"], "reviewer_agent_ids": ALIASES}
    first = FixedOrderGraph(first_llm, SPECS, **kwargs).run("Q", recorder=a)
    second = SkipEmptyFinalRoundGraph(second_llm, SPECS, **kwargs).run("Q", recorder=b)
    assert a.events == b.events
    assert first_llm.calls == second_llm.calls
    assert first.worker_outputs == second.worker_outputs
    assert first.edge_counts == second.edge_counts
    assert b.config["execution_contract"]["executed_worker_rounds"] == [0, 1]
    assert b.config["execution_contract"]["skipped_worker_rounds"] == []
    assert b.config["execution_contract"]["reviewer_source_round"] == 1


def test_round_zero_still_executes_when_all_worker_edges_are_empty():
    policy = empty_final_policy()
    policy["masks"]["spatial"][0] = [[0] * 3 for _ in range(3)]
    trace = GenerationTraceRecorder()
    result = SkipEmptyFinalRoundGraph(RecordingStub(), SPECS, policy=policy,
        round_orders=policy["round_orders"]).run("Q", recorder=trace)
    assert len(trace.events) == 4
    assert all(event["round"] == 0 and not event["incoming"] for event in trace.events[:-1])
    assert result.edge_counts["worker_active"] == 0
    assert len(trace.events[-1]["incoming"]) == 3


def test_five_workers_use_six_calls_and_preserve_nine_static_edges():
    specs = SPECS + [WorkerSpec("d", "Psychologist"), WorkerSpec("e", "Knowlegable Expert")]
    agents = [spec.id for spec in specs]
    policy = RoundPolicy.dense(agents, num_rounds=2).to_dict()
    policy["masks"]["spatial"][0][0][1] = 0
    for edge in ("spatial", "temporal"):
        policy["masks"][edge][1] = [[0] * 5 for _ in range(5)]
    trace = GenerationTraceRecorder()
    result = SkipEmptyFinalRoundGraph(RecordingStub(), specs, policy=policy,
        round_orders=[agents, agents]).run("Q", recorder=trace)
    assert result.metrics["llm_calls"] == len(trace.events) == 6
    assert result.edge_counts["worker_active"] == 9
    assert result.edge_counts["worker_candidate"] == 45
    assert sum(len(event["incoming"]) for event in trace.events[:-1]) == 9
    assert trace.events[-1]["spatial_incoming_from"] == agents


@pytest.mark.parametrize("key,value", [
    ("execution_contract", {"executed_worker_rounds": [0, 1]}),
    ("fixed_order_masks", {}),
    ("round_orders", [["a", "b", "c"]] * 2),
])
def test_conflicting_trace_is_rejected_without_calls_or_metadata_changes(key, value):
    policy = empty_final_policy()
    llm = RecordingStub()
    graph = SkipEmptyFinalRoundGraph(llm, SPECS, policy=policy, round_orders=policy["round_orders"])
    trace = GenerationTraceRecorder(config={key: value})
    before = deepcopy(trace.config)
    with pytest.raises(ValueError, match="Trace config"):
        graph.run("Q", recorder=trace)
    assert trace.config == before and not trace.events and not llm.calls


@pytest.mark.parametrize("rounds", [1, 3])
def test_other_round_counts_require_an_explicit_different_rule(rounds):
    llm = RecordingStub()
    with pytest.raises(ValueError, match="exactly two"):
        SkipEmptyFinalRoundGraph(llm, SPECS, num_rounds=rounds,
                                round_orders=[["a", "b", "c"]] * rounds)
    assert not llm.calls


def test_contract_and_caller_policy_are_not_mutable_execution_state():
    policy = empty_final_policy()
    original = deepcopy(policy)
    graph = SkipEmptyFinalRoundGraph(RecordingStub(), SPECS, policy=policy, round_orders=policy["round_orders"])
    policy["masks"]["temporal"][1][0][0] = 1
    contract = graph.execution_contract
    contract["executed_worker_rounds"].append(1)
    assert graph.execution_contract == execution_contract_for_policy(original)
    trace = GenerationTraceRecorder()
    graph.run("Q", recorder=trace)
    assert len(trace.events) == 4


def test_concurrent_samples_keep_round_zero_sources_separate():
    policy = empty_final_policy()
    graph = SkipEmptyFinalRoundGraph(RecordingStub(), SPECS, policy=policy, round_orders=policy["round_orders"])
    a, b = GenerationTraceRecorder(), GenerationTraceRecorder()
    async def both():
        await asyncio.gather(graph.run_sample("first", recorder=a, sample_index=4),
                             graph.run_sample("second", recorder=b, sample_index=5))
    asyncio.run(both())
    for trace, index, task in ((a, 4, "first"), (b, 5, "second")):
        assert len(trace.events) == 4
        assert {event["sample_index"] for event in trace.events} == {index}
        for incoming in trace.events[-1]["incoming"]:
            event = trace.events[incoming["event_id"]]
            assert event["output"] == incoming["source_output"] and event["round"] == 0
            assert task in event["input_messages"][-1]["content"]


def test_skip_metrics_cover_only_current_sample_batch_calls():
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
    policy = empty_final_policy()
    graph = SkipEmptyFinalRoundGraph(BatchStub(), SPECS, policy=policy, round_orders=policy["round_orders"])
    trace = GenerationTraceRecorder()
    graph.run("first", recorder=trace)
    result = graph.run("second", recorder=trace)
    assert len(trace.events) == 8
    assert result.metrics["llm_calls"] == result.metrics["batch_metadata_calls"] == 4
    assert result.metrics["oom_fallback_calls"] == 4
    assert result.metrics["oom_max_split_depth"] == 2
