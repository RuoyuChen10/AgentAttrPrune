from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from AgentAttrPrune.prompt import MMLUPromptSet


@dataclass(frozen=True)
class WorkerSpec:
    id: str
    role: str
    agent_type: str = "AnalyzeAgent"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("WorkerSpec.id cannot be empty")
        if not self.role.strip():
            raise ValueError("WorkerSpec.role cannot be empty")
        if self.agent_type != "AnalyzeAgent":
            raise ValueError("v1 supports only AnalyzeAgent workers")

    @classmethod
    def from_value(cls, value: Any, index: int) -> "WorkerSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            slug = value.strip().lower().replace(" ", "_")
            return cls(id=slug or f"agent_{index}", role=value)
        if isinstance(value, Mapping):
            worker_id = value.get("id", value.get("agent_id", f"agent_{index}"))
            return cls(
                id=str(worker_id),
                role=str(value["role"]),
                agent_type=str(value.get("agent_type", "AnalyzeAgent")),
            )
        if isinstance(value, Sequence) and len(value) == 2:
            return cls(id=str(value[0]), role=str(value[1]))
        raise TypeError(
            "worker_specs entries must be WorkerSpec, mapping, role string, or (id, role)"
        )


@dataclass(frozen=True)
class AgentResponse:
    output: str
    messages: list[dict[str, str]]


class AnalyzeAgent:
    def __init__(self, spec: WorkerSpec, llm: Any, prompt_set: Any) -> None:
        self.spec = spec
        self.llm = llm
        self.prompt_set = prompt_set

    async def arun(
        self,
        *,
        task: str,
        spatial_incoming: Sequence[Mapping[str, Any]],
        temporal_incoming: Sequence[Mapping[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> AgentResponse:
        messages = self.prompt_set.worker_messages(
            role=self.spec.role,
            task=task,
            spatial_incoming=spatial_incoming,
            temporal_incoming=temporal_incoming,
        )
        output = await self.llm.agen(
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if isinstance(output, list):
            output = output[0] if output else ""
        return AgentResponse(output=str(output), messages=messages)


class FinalRefer:
    def __init__(
        self,
        llm: Any,
        prompt_set: Any = MMLUPromptSet,
        *,
        agent_id: str = "dense_reviewer",
    ) -> None:
        self.id = agent_id
        self.role = "FinalRefer"
        self.agent_type = "FinalRefer"
        self.llm = llm
        self.prompt_set = prompt_set

    async def arun(
        self,
        *,
        task: str,
        worker_outputs: Sequence[Mapping[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> AgentResponse:
        messages = self.prompt_set.reviewer_messages(
            task=task, worker_outputs=worker_outputs
        )
        output = await self.llm.agen(
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if isinstance(output, list):
            output = output[0] if output else ""
        return AgentResponse(output=str(output), messages=messages)
