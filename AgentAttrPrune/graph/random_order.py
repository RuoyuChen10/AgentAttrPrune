"""Dense rollouts with an explicit, independently chosen order in each round.

This executor is separate from the fixed-order graph. Agent identities and
matrix coordinates stay canonical even when generation events run out of order.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from AgentAttrPrune.agents import WorkerSpec
from AgentAttrPrune.graph.multi_agent import GraphResult, MultiAgentGraph
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder


class RandomOrderDenseGraph(MultiAgentGraph):
    """Run all earlier-to-later spatial edges under supplied round orders.

    ``worker_specs`` defines the permanent identity/matrix order;
    ``round_orders`` defines generation order only. Spatial inputs follow the
    current round's execution prefix, while temporal and reviewer inputs use
    the permanent identity order. There is no external pruning-policy input.
    """

    def __init__(
        self,
        llm: Any,
        worker_specs: Sequence[Any],
        num_rounds: int = 2,
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
        if num_rounds < 1:
            raise ValueError("num_rounds must be positive")
        specs = tuple(
            WorkerSpec.from_value(value, index)
            for index, value in enumerate(worker_specs)
        )
        agent_ids = tuple(spec.id for spec in specs)
        if not specs:
            raise ValueError("worker_specs must contain at least one worker")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("worker ids must be unique")
        if isinstance(round_orders, (str, bytes)):
            raise ValueError("round_orders must contain one permutation per round")
        orders = []
        for round_index, order in enumerate(round_orders):
            if isinstance(order, (str, bytes)):
                raise ValueError(f"round_orders[{round_index}] must be a permutation")
            values = tuple(order)
            if (
                len(values) != len(agent_ids)
                or not all(isinstance(value, str) for value in values)
                or set(values) != set(agent_ids)
            ):
                raise ValueError(
                    f"round_orders[{round_index}] must be a permutation of "
                    f"{list(agent_ids)!r}"
                )
            orders.append(values)
        if len(orders) != num_rounds:
            raise ValueError("round_orders must contain one permutation per round")
        self.round_orders = tuple(orders)

        spatial = []
        temporal = []
        for round_index, order in enumerate(self.round_orders):
            position = {agent_id: index for index, agent_id in enumerate(order)}
            spatial.append([
                [position[source] < position[target] for target in agent_ids]
                for source in agent_ids
            ])
            temporal.append([
                [round_index > 0 for _ in agent_ids] for _ in agent_ids
            ])
        policy = RoundPolicy.from_dict({
            "agents": list(agent_ids),
            "agent_roles": [spec.role for spec in specs],
            "agent_types": [spec.agent_type for spec in specs],
            "num_rounds": num_rounds,
            "kind": "random_order_dense",
            "round_orders": [list(order) for order in self.round_orders],
            "masks": {"spatial": spatial, "temporal": temporal},
        })
        super().__init__(
            llm,
            specs,
            num_rounds=num_rounds,
            policy=policy,
            prompt_set=prompt_set,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            timeout=timeout,
            max_tries=max_tries,
            reviewer_id=reviewer_id,
            reviewer_agent_ids=reviewer_agent_ids,
        )

    def _validate_candidate_topology(self) -> None:
        for round_index, order in enumerate(self.round_orders):
            position = {agent_id: index for index, agent_id in enumerate(order)}
            spatial = self.policy.mask("spatial", round_index)
            temporal = self.policy.mask("temporal", round_index)
            for source_index, source in enumerate(self.agent_ids):
                for target_index, target in enumerate(self.agent_ids):
                    if bool(spatial[source_index, target_index].item()) != (
                        position[source] < position[target]
                    ):
                        raise ValueError("Spatial policy does not match the dense round order")
                    if bool(temporal[source_index, target_index].item()) != (round_index > 0):
                        raise ValueError("Temporal policy must be dense after round zero")

    def _configure_recorder(self, recorder: GenerationTraceRecorder) -> None:
        expected = {
            "round_orders": [list(order) for order in self.round_orders],
            "num_rounds": self.num_rounds,
            "agents": list(self.agent_ids),
            "agent_roles": list(self.agent_roles),
            "workers": [
                {
                    "agent_id": spec.id,
                    "role": spec.role,
                    "agent_type": spec.agent_type,
                    "node_index": index,
                }
                for index, spec in enumerate(self.worker_specs)
            ],
        }
        # Validate everything before adding metadata, so a conflicting recorder
        # is rejected before generation and remains unmodified.
        for key, value in expected.items():
            if key in recorder.config and recorder.config[key] != value:
                raise ValueError(f"Trace config {key} conflicts with the graph")
        recorder.config.update(expected)

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
            config=dict(self.generation_config), sample=sample_payload
        )
        self._configure_recorder(active_recorder)
        usage_before = active_recorder.summary
        started_at = time.perf_counter()
        workers_by_id = {worker.spec.id: worker for worker in self.workers}
        previous_outputs: dict[str, dict[str, Any]] = {}
        all_round_outputs: list[list[dict[str, Any]]] = []

        for round_index, order in enumerate(self.round_orders):
            current_outputs: dict[str, dict[str, Any]] = {}
            for target_position, target_id in enumerate(order):
                worker = workers_by_id[target_id]
                spatial_incoming = [
                    self._incoming_record(
                        current_outputs[source_id],
                        edge_type="spatial",
                        recorder=active_recorder,
                    )
                    for source_id in order[:target_position]
                ]
                temporal_incoming = [
                    self._incoming_record(
                        previous_outputs[source_id],
                        edge_type="temporal",
                        recorder=active_recorder,
                    )
                    for source_id in self.agent_ids
                    if previous_outputs
                ]
                invocation_id = active_recorder.new_invocation_id()
                metadata = {
                    "invocation_id": invocation_id,
                    "phase": phase,
                    "iteration": iteration,
                    "batch_index": batch_index,
                    "sample_index": sample_index,
                    "round": round_index,
                    "agent_id": worker.spec.id,
                    "agent_type": worker.spec.agent_type,
                    "role": worker.spec.role,
                    "incoming": spatial_incoming + temporal_incoming,
                }
                response = await self._invoke(
                    lambda worker=worker, spatial_incoming=spatial_incoming,
                    temporal_incoming=temporal_incoming: worker.arun(
                        task=task,
                        spatial_incoming=spatial_incoming,
                        temporal_incoming=temporal_incoming,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                    ),
                    recorder=active_recorder,
                    metadata=metadata,
                )
                if active_recorder.event_id_for(invocation_id) is None:
                    active_recorder.record_fallback(
                        invocation_id,
                        metadata,
                        messages=response.messages,
                        output=response.output,
                        model_name=getattr(self.llm, "model_name", None),
                        generation_kwargs={
                            "max_tokens": self.max_new_tokens,
                            "temperature": self.temperature,
                            "num_completions": 1,
                        },
                    )
                current_outputs[target_id] = {
                    "round": round_index,
                    "agent_id": worker.spec.id,
                    "role": worker.spec.role,
                    "output": response.output,
                    "parsed_answer": self.prompt_set.normalize_answer(response.output),
                    "event_id": active_recorder.event_id_for(invocation_id),
                }
            all_round_outputs.append([
                current_outputs[agent_id] for agent_id in self.agent_ids
            ])
            previous_outputs = current_outputs

        reviewer_incoming = [
            self._incoming_record(
                previous_outputs[agent_id],
                edge_type="spatial",
                recorder=active_recorder,
            )
            for agent_id in self.agent_ids
        ]
        for incoming in reviewer_incoming:
            agent_id = str(incoming["agent_id"])
            if agent_id in self.reviewer_agent_ids:
                incoming["reviewer_agent_id"] = self.reviewer_agent_ids[agent_id]
        reviewer_invocation = active_recorder.new_invocation_id()
        reviewer_metadata = {
            "invocation_id": reviewer_invocation,
            "phase": phase,
            "iteration": iteration,
            "batch_index": batch_index,
            "sample_index": sample_index,
            "round": self.num_rounds,
            "agent_id": self.reviewer.id,
            "agent_type": self.reviewer.agent_type,
            "role": self.reviewer.role,
            "incoming": reviewer_incoming,
        }
        reviewer_response = await self._invoke(
            lambda: self.reviewer.arun(
                task=task,
                worker_outputs=reviewer_incoming,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            ),
            recorder=active_recorder,
            metadata=reviewer_metadata,
        )
        if active_recorder.event_id_for(reviewer_invocation) is None:
            active_recorder.record_fallback(
                reviewer_invocation,
                reviewer_metadata,
                messages=reviewer_response.messages,
                output=reviewer_response.output,
                model_name=getattr(self.llm, "model_name", None),
                generation_kwargs={
                    "max_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "num_completions": 1,
                },
            )
        usage_after = active_recorder.summary
        metrics: dict[str, int | float] = {
            key: int(usage_after[key] - usage_before[key])
            for key in (
                "llm_calls", "prompt_tokens", "generated_tokens", "truncated_events"
            )
        }
        metrics["elapsed_seconds"] = time.perf_counter() - started_at
        reviewer_output = {
            "round": self.num_rounds,
            "agent_id": self.reviewer.id,
            "role": self.reviewer.role,
            "output": reviewer_response.output,
            "parsed_answer": self.prompt_set.normalize_answer(reviewer_response.output),
            "event_id": active_recorder.event_id_for(reviewer_invocation),
        }
        return GraphResult(
            final_answer=reviewer_response.output,
            worker_outputs=all_round_outputs,
            reviewer_output=reviewer_output,
            metrics=metrics,
            edge_counts=self._edge_counts(),
            answer_normalizer=self.prompt_set.normalize_answer,
        )
