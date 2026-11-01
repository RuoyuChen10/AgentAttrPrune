"""Execute an already sampled DAG without changing or resampling its edges."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from AgentAttrPrune.agents import WorkerSpec
from AgentAttrPrune.graph.multi_agent import GraphResult, MultiAgentGraph
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder
from AgentAttrPrune.tracing.recorder import to_jsonable


class SampledDAGGraph(MultiAgentGraph):
    """Run a sampled graph in its supplied topological order.

    Masks and output arrays use the permanent ``worker_specs`` identity order.
    Incoming messages use canonical source order, matching AgentPrune's
    source-major edge construction. This class never draws random numbers.
    """

    def __init__(
        self,
        llm: Any,
        worker_specs: Sequence[Any],
        num_rounds: int = 2,
        policy: RoundPolicy | Mapping[str, Any] | str | Path | None = None,
        prompt_set: Any = None,
        *,
        realized_graph: Mapping[str, Any],
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        timeout: float = 600.0,
        max_tries: int = 3,
        reviewer_id: str = "dense_reviewer",
        reviewer_agent_ids: Mapping[str, str] | None = None,
    ) -> None:
        if policy is not None:
            raise ValueError("Pass realized_graph instead of an external policy")
        if not isinstance(realized_graph, Mapping):
            raise TypeError("realized_graph must be a mapping")
        specs = tuple(
            WorkerSpec.from_value(value, index)
            for index, value in enumerate(worker_specs)
        )
        agent_ids = tuple(spec.id for spec in specs)
        if "agents" in realized_graph and tuple(realized_graph["agents"]) != agent_ids:
            raise ValueError("realized_graph agents must follow worker identity order")
        if "num_rounds" in realized_graph and realized_graph["num_rounds"] != num_rounds:
            raise ValueError("realized_graph num_rounds differs from the executor")
        raw_orders = realized_graph.get("round_orders")
        if not isinstance(raw_orders, (list, tuple)) or len(raw_orders) != num_rounds:
            raise ValueError("round_orders must contain one permutation per round")
        orders = []
        for round_index, order in enumerate(raw_orders):
            if (not isinstance(order, (list, tuple)) or len(order) != len(agent_ids)
                    or not all(isinstance(value, str) for value in order)
                    or set(order) != set(agent_ids)):
                raise ValueError(f"round_orders[{round_index}] must cover every worker exactly once")
            orders.append(tuple(order))
        self.round_orders = tuple(orders)
        resolved_policy = RoundPolicy.from_dict({
            "agents": list(agent_ids),
            "agent_roles": [spec.role for spec in specs],
            "agent_types": [spec.agent_type for spec in specs],
            "num_rounds": num_rounds,
            "matrix_orientation": realized_graph.get(
                "matrix_orientation", "[round][source][target]"
            ),
            "masks": realized_graph.get("masks", {}),
            "kind": "realized_sampled_dag",
        })
        self._realized_graph = copy.deepcopy(to_jsonable(dict(realized_graph)))
        self._realized_graph["masks"] = resolved_policy.to_dict()["masks"]
        self._realized_graph["round_orders"] = [list(order) for order in self.round_orders]
        super().__init__(
            llm, specs, num_rounds=num_rounds, policy=resolved_policy,
            prompt_set=prompt_set, temperature=temperature,
            max_new_tokens=max_new_tokens, timeout=timeout, max_tries=max_tries,
            reviewer_id=reviewer_id, reviewer_agent_ids=reviewer_agent_ids,
        )

    @property
    def realized_graph(self) -> dict[str, Any]:
        return copy.deepcopy(self._realized_graph)

    def _validate_candidate_topology(self) -> None:
        # RoundPolicy already rejects cycles, spatial self edges and round-zero
        # temporal edges. Also ensure the supplied execution order honors every
        # realized dependency; the executor must never silently reorder it.
        for round_index, order in enumerate(self.round_orders):
            positions = {agent_id: position for position, agent_id in enumerate(order)}
            spatial = self.policy.mask("spatial", round_index)
            for source_index, source in enumerate(self.agent_ids):
                for target_index, target in enumerate(self.agent_ids):
                    if (bool(spatial[source_index, target_index].item())
                            and positions[source] >= positions[target]):
                        raise ValueError(
                            f"round_orders[{round_index}] violates the sampled dependency "
                            f"{source} -> {target}"
                        )

    def _configure_recorder(self, recorder: GenerationTraceRecorder) -> None:
        expected = {
            "realized_graph": self.realized_graph,
            "round_orders": [list(order) for order in self.round_orders],
            "num_rounds": self.num_rounds,
            "agents": list(self.agent_ids),
            "agent_roles": list(self.agent_roles),
            "workers": [
                {"agent_id": spec.id, "role": spec.role, "agent_type": spec.agent_type,
                 "node_index": index}
                for index, spec in enumerate(self.worker_specs)
            ],
        }
        for key, value in expected.items():
            if key in recorder.config and recorder.config[key] != value:
                raise ValueError(f"Trace config {key} conflicts with the sampled graph")
        recorder.config.update(expected)

    def _edge_counts(self) -> dict[str, Any]:
        # Count all directed candidate slots, not the capacity of one complete
        # ordered DAG. Five workers/two rounds have 40 spatial + 25 temporal
        # slots; the sampled DAG generally realizes fewer than 45 worker edges.
        count = len(self.agent_ids)
        spatial_candidate = self.num_rounds * count * (count - 1)
        temporal_candidate = max(0, self.num_rounds - 1) * count * count
        active_spatial = int(self.policy.spatial_masks.sum().item())
        active_temporal = int(self.policy.temporal_masks.sum().item())
        candidate = spatial_candidate + temporal_candidate
        active = active_spatial + active_temporal
        return {
            "matrix_orientation": "[round][source][target]",
            "spatial": {"active": active_spatial, "candidate": spatial_candidate},
            "temporal": {"active": active_temporal, "candidate": temporal_candidate},
            "worker_active": active, "worker_candidate": candidate,
            "worker_retention": active / candidate if candidate else 1.0,
            "reviewer_dense_edges": count,
        }

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
            for target_id in order:
                target_index = agent_index[target_id]
                worker = workers_by_id[target_id]
                spatial = [
                    self._incoming_record(current[source_id], edge_type="spatial", recorder=active_recorder)
                    for source_index, source_id in enumerate(self.agent_ids)
                    if bool(spatial_mask[source_index, target_index].item())
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
