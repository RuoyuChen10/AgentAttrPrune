from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from AgentAttrPrune.agents import AnalyzeAgent, FinalRefer, WorkerSpec
from AgentAttrPrune.prompt import MMLUPromptSet
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder


class GraphExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphResult:
    final_answer: str
    worker_outputs: list[list[dict[str, Any]]]
    reviewer_output: dict[str, Any]
    metrics: dict[str, int | float]
    edge_counts: dict[str, Any]
    answer_normalizer: Callable[[Any], str] = field(
        default=MMLUPromptSet.normalize_answer,
        repr=False,
        compare=False,
    )

    @property
    def parsed_answer(self) -> str:
        return self.answer_normalizer(self.final_answer)

    def to_trace_final(self) -> dict[str, Any]:
        return {
            "output": self.final_answer,
            "parsed_answer": self.parsed_answer,
            "reviewer": self.reviewer_output,
            "metrics": self.metrics,
            "edge_counts": self.edge_counts,
        }


class MultiAgentGraph:
    """One execution engine for both dense and deterministic pruned MAS runs.

    Spatial edges are constrained to the dense candidate DAG: an earlier
    worker may send to a later worker. Temporal edges carry the previous round
    into the current round and may include all ``N x N`` pairs. The reviewer is
    always connected densely to every final-round worker and is outside both
    policy matrices.
    """

    def __init__(
        self,
        llm: Any,
        worker_specs: Sequence[Any],
        num_rounds: int = 2,
        policy: RoundPolicy | Mapping[str, Any] | str | Path | None = None,
        prompt_set: Any = None,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        timeout: float = 600.0,
        max_tries: int = 3,
        reviewer_id: str = "dense_reviewer",
    ) -> None:
        if num_rounds < 1:
            raise ValueError("num_rounds must be positive")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_tries < 1:
            raise ValueError("max_tries must be positive")

        specs = tuple(
            WorkerSpec.from_value(value, index)
            for index, value in enumerate(worker_specs)
        )
        if not specs:
            raise ValueError("worker_specs must contain at least one worker")
        agent_ids = tuple(spec.id for spec in specs)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("worker ids must be unique")
        if reviewer_id in agent_ids:
            raise ValueError("reviewer_id cannot duplicate a worker id")

        self.llm = llm
        self.worker_specs = specs
        self.num_rounds = int(num_rounds)
        self.prompt_set = prompt_set or MMLUPromptSet
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = float(timeout)
        self.max_tries = int(max_tries)
        self.reviewer_id = reviewer_id
        self.workers = tuple(
            AnalyzeAgent(spec, llm, self.prompt_set) for spec in self.worker_specs
        )
        self.reviewer = FinalRefer(
            llm, self.prompt_set, agent_id=self.reviewer_id
        )
        self.policy = self._resolve_policy(policy)
        self._validate_candidate_topology()

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(spec.id for spec in self.worker_specs)

    @property
    def agent_roles(self) -> tuple[str, ...]:
        return tuple(spec.role for spec in self.worker_specs)

    @property
    def generation_config(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "timeout": self.timeout,
            "max_tries": self.max_tries,
        }

    def _resolve_policy(
        self,
        policy: RoundPolicy | Mapping[str, Any] | str | Path | None,
    ) -> RoundPolicy:
        if policy is None:
            resolved = RoundPolicy.dense(
                self.agent_ids,
                num_rounds=self.num_rounds,
                agent_roles=self.agent_roles,
                agent_types=["AnalyzeAgent"] * len(self.worker_specs),
            )
        elif isinstance(policy, (str, Path)):
            resolved = RoundPolicy.from_json(policy)
        elif isinstance(policy, Mapping):
            resolved = RoundPolicy.from_dict(policy)
        else:
            resolved = policy
        if not hasattr(resolved, "mask"):
            raise TypeError("policy must be a RoundPolicy-compatible object")
        if int(resolved.num_rounds) != self.num_rounds:
            raise ValueError(
                f"policy has {resolved.num_rounds} rounds, graph has {self.num_rounds}"
            )
        resolved.validate_agents(self.agent_ids)
        policy_roles = tuple(getattr(resolved, "agent_roles", ()))
        if policy_roles and policy_roles != self.agent_roles:
            raise ValueError(
                "Policy agent roles do not match graph worker roles: "
                f"policy={list(policy_roles)!r}, graph={list(self.agent_roles)!r}"
            )
        return resolved

    def _validate_candidate_topology(self) -> None:
        """Reject policy edges that do not exist in the dense spatial DAG."""
        for round_index in range(self.num_rounds):
            mask = self.policy.mask("spatial", round_index)
            for source in range(len(self.worker_specs)):
                for target in range(source + 1):
                    if bool(mask[source, target].item()):
                        raise ValueError(
                            "Spatial policies may only select dense candidate edges "
                            f"with source < target; got round {round_index}, "
                            f"{self.agent_ids[source]} -> {self.agent_ids[target]}"
                        )

    def topology_for_round(self, round_index: int) -> dict[str, list[tuple[str, str]]]:
        if not 0 <= round_index < self.num_rounds:
            raise IndexError("round_index is outside the configured policy")
        result: dict[str, list[tuple[str, str]]] = {
            "spatial": [],
            "temporal": [],
        }
        count = len(self.worker_specs)
        for edge_type in ("spatial", "temporal"):
            mask = self.policy.mask(edge_type, round_index)
            for source in range(count):
                for target in range(count):
                    if bool(mask[source, target].item()):
                        result[edge_type].append(
                            (self.agent_ids[source], self.agent_ids[target])
                        )
        return result

    def _edge_counts(self) -> dict[str, Any]:
        count = len(self.worker_specs)
        candidate_spatial = self.num_rounds * count * (count - 1) // 2
        candidate_temporal = max(0, self.num_rounds - 1) * count * count
        active_spatial = 0
        active_temporal = 0
        for round_index in range(self.num_rounds):
            active_spatial += int(
                self.policy.mask("spatial", round_index).sum().item()
            )
            active_temporal += int(
                self.policy.mask("temporal", round_index).sum().item()
            )
        candidate = candidate_spatial + candidate_temporal
        active = active_spatial + active_temporal
        return {
            "matrix_orientation": "[round][source][target]",
            "spatial": {
                "active": active_spatial,
                "candidate": candidate_spatial,
            },
            "temporal": {
                "active": active_temporal,
                "candidate": candidate_temporal,
            },
            "worker_active": active,
            "worker_candidate": candidate,
            "worker_retention": active / candidate if candidate else 1.0,
            "reviewer_dense_edges": count,
        }

    @staticmethod
    def _task_from_sample(sample: str | Mapping[str, Any]) -> str:
        if isinstance(sample, str):
            if not sample:
                raise ValueError("sample task cannot be empty")
            return sample
        if "task" not in sample:
            raise KeyError("sample must contain a 'task' field")
        task = str(sample["task"])
        if not task:
            raise ValueError("sample task cannot be empty")
        return task

    def _incoming_record(
        self,
        output: Mapping[str, Any],
        *,
        edge_type: str,
        recorder: GenerationTraceRecorder,
    ) -> dict[str, Any]:
        event_id = output.get("event_id")
        source_generated_ids = None
        if event_id is not None:
            source_generated_ids = recorder.event(int(event_id)).get("generated_ids")
        return {
            "event_id": event_id,
            "round": output["round"],
            "agent_id": output["agent_id"],
            "agent_type": "AnalyzeAgent",
            "role": output["role"],
            "edge_type": edge_type,
            "source_generated_ids": source_generated_ids,
            "source_output": output["output"],
            "output": output["output"],
        }

    async def _invoke(
        self,
        call: Callable[[], Awaitable[Any]],
        *,
        recorder: GenerationTraceRecorder,
        metadata: dict[str, Any],
    ) -> Any:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_tries + 1):
            try:
                with recorder.observe(metadata):
                    return await asyncio.wait_for(call(), timeout=self.timeout)
            except Exception as error:
                last_error = error
                if attempt == self.max_tries:
                    break
        raise GraphExecutionError(
            f"{metadata['agent_id']} failed after {self.max_tries} attempts"
        ) from last_error

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
            config={
                "num_rounds": self.num_rounds,
                "agents": list(self.agent_ids),
                "agent_roles": list(self.agent_roles),
                **self.generation_config,
            },
            sample=sample_payload,
        )
        usage_before = active_recorder.summary
        started_at = time.perf_counter()
        previous_outputs: list[dict[str, Any]] = []
        all_round_outputs: list[list[dict[str, Any]]] = []

        for round_index in range(self.num_rounds):
            current_outputs: list[dict[str, Any]] = []
            spatial_mask = self.policy.mask("spatial", round_index)
            temporal_mask = self.policy.mask("temporal", round_index)
            for target_index, worker in enumerate(self.workers):
                spatial_incoming = [
                    self._incoming_record(
                        current_outputs[source_index],
                        edge_type="spatial",
                        recorder=active_recorder,
                    )
                    for source_index in range(target_index)
                    if bool(spatial_mask[source_index, target_index].item())
                ]
                temporal_incoming = [
                    self._incoming_record(
                        previous_outputs[source_index],
                        edge_type="temporal",
                        recorder=active_recorder,
                    )
                    for source_index in range(len(self.workers))
                    if previous_outputs
                    and bool(temporal_mask[source_index, target_index].item())
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
                current_outputs.append(
                    {
                        "round": round_index,
                        "agent_id": worker.spec.id,
                        "role": worker.spec.role,
                        "output": response.output,
                        "parsed_answer": self.prompt_set.normalize_answer(
                            response.output
                        ),
                        "event_id": active_recorder.event_id_for(invocation_id),
                    }
                )
            all_round_outputs.append(current_outputs)
            previous_outputs = current_outputs

        reviewer_incoming = [
            self._incoming_record(
                output, edge_type="spatial", recorder=active_recorder
            )
            for output in previous_outputs
        ]
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
                "llm_calls",
                "prompt_tokens",
                "generated_tokens",
                "truncated_events",
            )
        }
        metrics["elapsed_seconds"] = time.perf_counter() - started_at
        reviewer_output = {
            "round": self.num_rounds,
            "agent_id": self.reviewer.id,
            "role": self.reviewer.role,
            "output": reviewer_response.output,
            "parsed_answer": self.prompt_set.normalize_answer(
                reviewer_response.output
            ),
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

    async def arun_sample(self, *args: Any, **kwargs: Any) -> GraphResult:
        return await self.run_sample(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> GraphResult:
        return await self.run_sample(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> GraphResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_sample(*args, **kwargs))
        raise RuntimeError(
            "MultiAgentGraph.run() cannot be called inside an event loop; "
            "await run_sample() instead"
        )
