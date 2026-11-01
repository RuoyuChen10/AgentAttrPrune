"""Static edge masks with a permanently prescribed execution order per round."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from AgentAttrPrune.graph.multi_agent import GraphResult
from AgentAttrPrune.graph.random_order import RandomOrderDenseGraph
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder


class FixedOrderGraph(RandomOrderDenseGraph):
    """Keep identities canonical and spatial inputs in the execution prefix.

    With no policy, execute the prescribed dense forward graph. With a policy,
    only remove edges; neither disconnected workers nor missing dependencies
    may change the specified order. Temporal and reviewer inputs always use
    canonical identity order, independent of either round's execution order.
    """

    def __init__(
        self,
        llm: Any,
        worker_specs: Sequence[Any],
        num_rounds: int = 2,
        policy: RoundPolicy | Mapping[str, Any] | str | Path | None = None,
        prompt_set: Any = None,
        *,
        round_orders: Sequence[Sequence[str]],
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        timeout: float = 600.0,
        max_tries: int = 3,
        reviewer_id: str = "dense_reviewer",
        reviewer_agent_ids: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            llm, worker_specs, num_rounds=num_rounds, prompt_set=prompt_set,
            round_orders=round_orders, temperature=temperature,
            max_new_tokens=max_new_tokens, timeout=timeout, max_tries=max_tries,
            reviewer_id=reviewer_id, reviewer_agent_ids=reviewer_agent_ids,
        )
        if policy is not None:
            if isinstance(policy, (str, Path)):
                resolved = RoundPolicy.from_json(policy)
            elif isinstance(policy, Mapping):
                resolved = RoundPolicy.from_dict(copy.deepcopy(dict(policy)))
            elif isinstance(policy, RoundPolicy):
                resolved = RoundPolicy.from_dict(copy.deepcopy(policy.to_dict()))
            else:
                raise TypeError("policy must be a RoundPolicy, mapping, JSON path or None")
            declared = resolved.metadata.get("round_orders")
            if declared is not None and declared != [list(order) for order in self.round_orders]:
                raise ValueError("Policy round_orders conflict with the prescribed execution order")
            self.policy = self._resolve_policy(resolved)
            self._validate_candidate_topology()

    def _validate_candidate_topology(self) -> None:
        # RoundPolicy validates spatial DAGs/self-links and round-zero temporal.
        # Our only additional restriction is the exact prescribed spatial order.
        for r, order in enumerate(self.round_orders):
            positions = {agent: i for i, agent in enumerate(order)}
            mask = self.policy.mask("spatial", r)
            for i, source in enumerate(self.agent_ids):
                for j, target in enumerate(self.agent_ids):
                    if bool(mask[i, j]) and positions[source] >= positions[target]:
                        raise ValueError(f"Spatial edge {source} -> {target} violates round_orders[{r}]")

    def _configure_recorder(self, recorder: GenerationTraceRecorder) -> None:
        masks = self.policy.to_dict()["masks"]
        if "fixed_order_masks" in recorder.config and recorder.config["fixed_order_masks"] != masks:
            raise ValueError("Trace config fixed_order_masks conflicts with the graph")
        super()._configure_recorder(recorder)
        recorder.config["fixed_order_masks"] = masks

    async def _invoke_recorded(
        self,
        call: Callable[[], Awaitable[Any]],
        recorder: GenerationTraceRecorder,
        metadata: dict[str, Any],
    ) -> Any:
        response = await self._invoke(call, recorder=recorder, metadata=metadata)
        invocation_id = metadata["invocation_id"]
        if recorder.event_id_for(invocation_id) is None:
            recorder.record_fallback(
                invocation_id, metadata, messages=response.messages,
                output=response.output, model_name=getattr(self.llm, "model_name", None),
                generation_kwargs={
                    "max_tokens": self.max_new_tokens, "temperature": self.temperature,
                    "num_completions": 1,
                },
            )
        return response

    def _output_record(self, response: Any, metadata: dict, recorder: GenerationTraceRecorder) -> dict:
        return {
            "round": metadata["round"], "agent_id": metadata["agent_id"],
            "role": metadata["role"], "output": response.output,
            "parsed_answer": self.prompt_set.normalize_answer(response.output),
            "event_id": recorder.event_id_for(metadata["invocation_id"]),
        }

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
        previous: dict[str, dict[str, Any]] = {}
        all_outputs: list[list[dict[str, Any]]] = []
        for round_index, order in enumerate(self.round_orders):
            current: dict[str, dict[str, Any]] = {}
            spatial_mask = self.policy.mask("spatial", round_index)
            temporal_mask = self.policy.mask("temporal", round_index)
            for target_position, target_id in enumerate(order):
                target_index = agent_index[target_id]
                worker = workers_by_id[target_id]
                spatial = [
                    self._incoming_record(current[source_id], edge_type="spatial", recorder=active_recorder)
                    for source_id in order[:target_position]
                    if bool(spatial_mask[agent_index[source_id], target_index].item())
                ]
                temporal = [
                    self._incoming_record(previous[source_id], edge_type="temporal", recorder=active_recorder)
                    for source_index, source_id in enumerate(self.agent_ids)
                    if previous and bool(temporal_mask[source_index, target_index].item())
                ]
                metadata = {
                    **common_metadata, "invocation_id": active_recorder.new_invocation_id(),
                    "round": round_index, "agent_id": target_id,
                    "agent_type": worker.spec.agent_type, "role": worker.spec.role,
                    "incoming": spatial + temporal,
                }
                response = await self._invoke_recorded(
                    lambda worker=worker, spatial=spatial, temporal=temporal: worker.arun(
                        task=task, spatial_incoming=spatial, temporal_incoming=temporal,
                        max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                    ),
                    active_recorder, metadata,
                )
                current[target_id] = self._output_record(response, metadata, active_recorder)
            all_outputs.append([current[agent_id] for agent_id in self.agent_ids])
            previous = current

        reviewer_incoming = [
            self._incoming_record(previous[agent_id], edge_type="spatial", recorder=active_recorder)
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
            final_answer=reviewer_response.output, worker_outputs=all_outputs,
            reviewer_output=self._output_record(reviewer_response, reviewer_metadata, active_recorder),
            metrics=metrics, edge_counts=self._edge_counts(),
            answer_normalizer=self.prompt_set.normalize_answer,
        )
