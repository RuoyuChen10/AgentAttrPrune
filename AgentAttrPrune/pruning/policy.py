"""Validated deterministic round-specific communication policies.

Every mask in this module uses ``[round][source][target]`` orientation.
Spatial masks describe within-round DAG edges. Temporal masks describe
previous-round-to-current-round edges and are always empty in round zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import torch


MATRIX_ORIENTATION = "[round][source][target]"
EDGE_TYPES = ("spatial", "temporal")


def normalize_round_masks(
    masks: Sequence,
    *,
    name: str,
    num_agents: int,
    num_rounds: Optional[int] = None,
) -> torch.Tensor:
    """Return a cloned CPU bool tensor after strict rank/shape validation."""

    try:
        tensor = torch.as_tensor(masks)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular rank-3 array") from error
    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape [round, source, target]; got {tuple(tensor.shape)}"
        )
    if tuple(tensor.shape[1:]) != (num_agents, num_agents):
        raise ValueError(
            f"{name} must have shape [round, {num_agents}, {num_agents}]; "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one round")
    if num_rounds is not None and tensor.shape[0] != num_rounds:
        raise ValueError(f"{name} has {tensor.shape[0]} rounds, expected {num_rounds}")
    if tensor.dtype != torch.bool:
        is_binary = torch.logical_or(tensor == 0, tensor == 1)
        if not bool(is_binary.all().item()):
            raise ValueError(f"{name} must contain only deterministic 0/1 values")
    return tensor.to(dtype=torch.bool, device="cpu").clone()


def _is_dag(mask: torch.Tensor) -> bool:
    indegree = mask.sum(dim=0).to(torch.long).tolist()
    queue = [index for index, value in enumerate(indegree) if value == 0]
    visited = 0
    while queue:
        source = queue.pop()
        visited += 1
        for target in torch.nonzero(mask[source], as_tuple=False).flatten().tolist():
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == mask.shape[0]


@dataclass(frozen=True)
class RoundPolicy:
    """An immutable, validated deterministic edge policy.

    Construct policies with :meth:`from_dict`, :meth:`from_json`, or
    :meth:`dense`; direct construction is supported but performs the same
    validation. Tensor-returning properties clone their values so callers
    cannot mutate policy state accidentally.
    """

    agents: tuple[str, ...]
    _spatial_masks: torch.Tensor = field(repr=False)
    _temporal_masks: torch.Tensor = field(repr=False)
    agent_roles: tuple[str, ...] = ()
    agent_types: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        agents = tuple(str(agent) for agent in self.agents)
        if not agents or any(not agent for agent in agents):
            raise ValueError("agents must be a non-empty sequence of non-empty ids")
        if len(set(agents)) != len(agents):
            raise ValueError("agents must contain unique ids in matrix order")
        spatial = normalize_round_masks(
            self._spatial_masks,
            name="spatial masks",
            num_agents=len(agents),
        )
        temporal = normalize_round_masks(
            self._temporal_masks,
            name="temporal masks",
            num_agents=len(agents),
            num_rounds=spatial.shape[0],
        )
        if bool(torch.diagonal(spatial, dim1=1, dim2=2).any().item()):
            raise ValueError("spatial masks cannot contain self edges")
        for round_index, mask in enumerate(spatial):
            if not _is_dag(mask):
                raise ValueError(f"spatial mask for round {round_index} is cyclic")
        if bool(temporal[0].any().item()):
            raise ValueError("temporal mask for round zero must be empty")
        roles = tuple(str(value) for value in self.agent_roles)
        types = tuple(str(value) for value in self.agent_types)
        if roles and len(roles) != len(agents):
            raise ValueError("agent_roles must follow the policy agent order")
        if types and len(types) != len(agents):
            raise ValueError("agent_types must follow the policy agent order")
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "_spatial_masks", spatial)
        object.__setattr__(self, "_temporal_masks", temporal)
        object.__setattr__(self, "agent_roles", roles)
        object.__setattr__(self, "agent_types", types)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def num_rounds(self) -> int:
        return int(self._spatial_masks.shape[0])

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def spatial_masks(self) -> torch.Tensor:
        return self._spatial_masks.clone()

    @property
    def temporal_masks(self) -> torch.Tensor:
        return self._temporal_masks.clone()

    # Compatibility names make the orientation explicit at graph call sites.
    @property
    def round_spatial_masks(self) -> torch.Tensor:
        return self.spatial_masks

    @property
    def round_temporal_masks(self) -> torch.Tensor:
        return self.temporal_masks

    def mask(self, edge_type: str, round_index: int) -> torch.Tensor:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"edge_type must be one of {EDGE_TYPES}")
        if not 0 <= int(round_index) < self.num_rounds:
            raise IndexError(
                f"round_index {round_index} is outside [0, {self.num_rounds - 1}]"
            )
        tensor = self._spatial_masks if edge_type == "spatial" else self._temporal_masks
        return tensor[int(round_index)].clone()

    def validate_agents(self, agents: Sequence[str]) -> None:
        actual = tuple(str(agent) for agent in agents)
        if actual != self.agents:
            raise ValueError(
                "Policy agent order does not match graph agent order: "
                f"policy={list(self.agents)!r}, graph={list(actual)!r}"
            )

    @classmethod
    def dense(
        cls,
        agent_ids: Optional[Sequence[str]] = None,
        *,
        num_workers: Optional[int] = None,
        num_rounds: int,
        agent_roles: Optional[Sequence[str]] = None,
        agent_types: Optional[Sequence[str]] = None,
    ) -> "RoundPolicy":
        """Build AgentPrune FullConnected candidate topology deterministically.

        Spatial edges follow insertion order (earlier source to later target),
        which yields a DAG. Every round after zero has the complete temporal
        ``N x N`` matrix, including self edges.
        """

        if num_rounds < 1:
            raise ValueError("num_rounds must be positive")
        if agent_ids is None:
            if num_workers is None or num_workers < 1:
                raise ValueError("provide agent_ids or a positive num_workers")
            agents = tuple(f"agent_{index}" for index in range(num_workers))
        else:
            agents = tuple(str(agent) for agent in agent_ids)
            if num_workers is not None and num_workers != len(agents):
                raise ValueError("num_workers disagrees with len(agent_ids)")
        count = len(agents)
        spatial = torch.zeros((num_rounds, count, count), dtype=torch.bool)
        for source in range(count):
            for target in range(source + 1, count):
                spatial[:, source, target] = True
        temporal = torch.ones((num_rounds, count, count), dtype=torch.bool)
        temporal[0] = False
        return cls(
            agents=agents,
            _spatial_masks=spatial,
            _temporal_masks=temporal,
            agent_roles=tuple(agent_roles or ()),
            agent_types=tuple(agent_types or ()),
            metadata={"kind": "dense", "matrix_orientation": MATRIX_ORIENTATION},
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoundPolicy":
        schema = payload.get("schema_version", 1)
        if int(schema) != 1:
            raise ValueError(f"Unsupported policy schema_version {schema!r}")
        orientation = payload.get("matrix_orientation", MATRIX_ORIENTATION)
        if orientation != MATRIX_ORIENTATION:
            raise ValueError(
                f"Policy matrix_orientation must be {MATRIX_ORIENTATION!r}"
            )
        agents = tuple(str(value) for value in payload.get("agents") or ())
        masks = payload.get("masks") or {}
        if "spatial" not in masks or "temporal" not in masks:
            raise ValueError("Policy must contain spatial and temporal masks")
        policy = cls(
            agents=agents,
            _spatial_masks=masks["spatial"],
            _temporal_masks=masks["temporal"],
            agent_roles=tuple(payload.get("agent_roles") or ()),
            agent_types=tuple(payload.get("agent_types") or ()),
            metadata={
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "agents",
                    "agent_roles",
                    "agent_types",
                    "masks",
                }
            },
        )
        declared_rounds = payload.get("num_rounds")
        if declared_rounds is not None and int(declared_rounds) != policy.num_rounds:
            raise ValueError("Policy num_rounds disagrees with its mask tensors")
        return policy

    @classmethod
    def from_json(cls, path: str | Path) -> "RoundPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Policy JSON must contain an object")
        return cls.from_dict(payload)

    @classmethod
    def load(cls, path: str | Path) -> "RoundPolicy":
        """Load a policy JSON file (public convenience alias)."""

        return cls.from_json(path)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "schema_version": 1,
                "agents": list(self.agents),
                "agent_roles": list(self.agent_roles),
                "agent_types": list(self.agent_types),
                "num_rounds": self.num_rounds,
                "matrix_orientation": MATRIX_ORIENTATION,
                "masks": {
                    "spatial": self._spatial_masks.to(torch.int8).tolist(),
                    "temporal": self._temporal_masks.to(torch.int8).tolist(),
                },
            }
        )
        return payload

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination
