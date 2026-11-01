"""Opt-in execution rule for an empty second round of a fixed ordered graph.

The original policy and two-round coordinates remain frozen. When both masks
in round one are empty, only round zero executes and its outputs go straight
to the reviewer. The existing fixed-order executor remains unchanged.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from AgentAttrPrune.graph.fixed_order import FixedOrderGraph
from AgentAttrPrune.graph.multi_agent import GraphResult
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder


def execution_contract_for_policy(policy: RoundPolicy | Mapping[str, Any]) -> dict:
    """Describe the worker rounds actually executed by this two-round rule."""
    resolved = policy if isinstance(policy, RoundPolicy) else RoundPolicy.from_dict(dict(policy))
    if resolved.num_rounds != 2:
        raise ValueError("skip_empty_final_round requires exactly two worker rounds")
    skip = not bool(resolved.mask("spatial", 1).any().item()) and not bool(
        resolved.mask("temporal", 1).any().item()
    )
    return {
        "rule_version": 1,
        "rule": "skip_empty_final_round",
        "executed_worker_rounds": [0] if skip else [0, 1],
        "skipped_worker_rounds": [1] if skip else [],
        "reviewer_source_round": 0 if skip else 1,
    }


class SkipEmptyFinalRoundGraph(FixedOrderGraph):
    """Skip round one iff it has neither spatial nor temporal messages.

    Round zero always executes. The reviewer retains its original stage index
    (two), canonical input identity order and anonymous worker IDs. Its actual
    source round is recorded explicitly in ``execution_contract``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        execution_contract_for_policy(self.policy)

    @property
    def execution_contract(self) -> dict:
        return execution_contract_for_policy(self.policy)

    def _configure_recorder(self, recorder: GenerationTraceRecorder) -> None:
        contract = self.execution_contract
        if "execution_contract" in recorder.config and recorder.config["execution_contract"] != contract:
            raise ValueError("Trace config execution_contract conflicts with the graph")
        super()._configure_recorder(recorder)
        recorder.config["execution_contract"] = contract

    async def run_sample(
        self,
        sample: str | Mapping[str, Any],
        *,
        recorder: GenerationTraceRecorder | None = None,
        phase: str = "eval",
        sample_index: int | None = None,
        iteration: int | None = None,
        batch_index: int | None = None,
    ) -> GraphResult:
        if not self.execution_contract["skipped_worker_rounds"]:
            return await super().run_sample(
                sample, recorder=recorder, phase=phase, sample_index=sample_index,
                iteration=iteration, batch_index=batch_index,
            )

        task = self._task_from_sample(sample)
        sample_payload = {"task": task} if isinstance(sample, str) else dict(sample)
        active_recorder = recorder or GenerationTraceRecorder(
            config=dict(self.generation_config), sample=sample_payload,
        )
        self._configure_recorder(active_recorder)
        started_at = time.perf_counter()
        usage_before = active_recorder.summary
        common_metadata = {
            "phase": phase, "sample_index": sample_index,
            "iteration": iteration, "batch_index": batch_index,
        }
        workers_by_id = {worker.spec.id: worker for worker in self.workers}
        agent_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}
        order = self.round_orders[0]
        spatial_mask = self.policy.mask("spatial", 0)
        current: dict[str, dict[str, Any]] = {}
        for target_position, target_id in enumerate(order):
            target_index = agent_index[target_id]
            worker = workers_by_id[target_id]
            spatial = [
                self._incoming_record(current[source_id], edge_type="spatial", recorder=active_recorder)
                for source_id in order[:target_position]
                if bool(spatial_mask[agent_index[source_id], target_index].item())
            ]
            metadata = {
                **common_metadata, "invocation_id": active_recorder.new_invocation_id(),
                "round": 0, "agent_id": target_id,
                "agent_type": worker.spec.agent_type, "role": worker.spec.role,
                "incoming": spatial,
            }
            response = await self._invoke_recorded(
                lambda worker=worker, spatial=spatial: worker.arun(
                    task=task, spatial_incoming=spatial, temporal_incoming=[],
                    max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                ),
                active_recorder, metadata,
            )
            current[target_id] = self._output_record(response, metadata, active_recorder)

        reviewer_incoming = [
            self._incoming_record(current[agent_id], edge_type="spatial", recorder=active_recorder)
            for agent_id in self.agent_ids
        ]
        for incoming in reviewer_incoming:
            if incoming["agent_id"] in self.reviewer_agent_ids:
                incoming["reviewer_agent_id"] = self.reviewer_agent_ids[incoming["agent_id"]]
        reviewer_metadata = {
            **common_metadata, "invocation_id": active_recorder.new_invocation_id(),
            "round": self.num_rounds, "agent_id": self.reviewer.id,
            "agent_type": self.reviewer.agent_type, "role": self.reviewer.role,
            "incoming": reviewer_incoming,
        }
        reviewer_response = await self._invoke_recorded(
            lambda: self.reviewer.arun(
                task=task, worker_outputs=reviewer_incoming,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            ),
            active_recorder, reviewer_metadata,
        )
        usage_after = active_recorder.summary
        metrics: dict[str, int | float] = {
            key: int(usage_after[key] - usage_before[key])
            for key in ("llm_calls", "prompt_tokens", "generated_tokens", "truncated_events")
        }
        metrics["elapsed_seconds"] = time.perf_counter() - started_at
        batch_calls = [
            event["llm_call"]
            for event in active_recorder.events[usage_before["llm_calls"] :]
            if "effective_batch_size" in event.get("llm_call", {})
        ]
        if batch_calls:
            metrics["batch_metadata_calls"] = len(batch_calls)
            metrics["oom_fallback_calls"] = sum(bool(call.get("oom_fallback")) for call in batch_calls)
            metrics["oom_max_split_depth"] = max(int(call.get("oom_split_depth", 0)) for call in batch_calls)
        return GraphResult(
            final_answer=reviewer_response.output,
            worker_outputs=[[current[agent_id] for agent_id in self.agent_ids]],
            reviewer_output=self._output_record(reviewer_response, reviewer_metadata, active_recorder),
            metrics=metrics, edge_counts=self._edge_counts(),
            answer_normalizer=self.prompt_set.normalize_answer,
        )
