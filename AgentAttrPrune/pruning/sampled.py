"""CPU policies and AgentPrune-style sampling from per-round attribution scores.

Importance is deliberately used as a logit, not as a Bernoulli probability.
Only sampling and cycle rejection follow AgentPrune: the logits are attribution
scores, and each round has its own parameters rather than learned shared ones.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from typing import Any

import torch


EDGE_TYPES = ("spatial", "temporal")
POLICY_VERSION = 1
SAMPLER_VERSION = 1


def _tensor(value: Any, shape: tuple[int, ...], label: str) -> torch.Tensor:
    try:
        result = torch.tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} must be a numeric matrix") from exc
    if tuple(result.shape) != shape or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must have shape {shape} and finite values")
    return result


def _dimensions(data: dict) -> tuple[list[str], int, int]:
    if data.get("matrix_orientation") != "[round][source][target]":
        raise ValueError("matrix_orientation must be [round][source][target]")
    agents = data.get("agents")
    if (
        not isinstance(agents, list) or not agents
        or any(not isinstance(agent, str) or not agent for agent in agents)
        or len(set(agents)) != len(agents)
    ):
        raise ValueError("agents must be unique nonempty canonical IDs")
    rounds = data.get("num_rounds")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("num_rounds must be a positive integer")
    return list(agents), rounds, len(agents)


def _admissible(rounds: int, count: int) -> dict[str, torch.Tensor]:
    spatial = torch.ones((rounds, count, count), dtype=torch.bool)
    for matrix in spatial:
        matrix.fill_diagonal_(False)
    temporal = torch.ones_like(spatial)
    temporal[0] = False
    return {"spatial": spatial, "temporal": temporal}


def _prune_group(
    logits: torch.Tensor, candidates: torch.Tensor, pruning_rate: float,
) -> tuple[torch.Tensor, int]:
    """Match torch rounding/ranking used by AgentPrune's update_masks.

    A requested zero rate is a true no-op; otherwise at least one active edge
    is removed, as in AgentPrune. Inactive slots sort below every active logit.
    """
    flat = logits.flatten().clone()
    active = candidates.flatten()
    available = int(active.sum())
    remove = int(torch.round(torch.tensor(available * pruning_rate, dtype=torch.float32)).item())
    if available and pruning_rate > 0:
        remove = max(1, remove)
    retained = active.clone()
    if available and remove:
        flat[~active] = flat.min() - 1.0
        order = torch.argsort(flat)
        retained[order[:int((~active).sum()) + remove]] = False
    return retained.reshape(candidates.shape), remove


def prepare_sampling_policy(
    aggregate: dict,
    pruning_rate: float,
    pruning_scope: str,
    round_policy: str = "per_round",
) -> dict:
    """Prune observed per-round token-ratio logits without projecting a DAG.

    ``global`` ranks all round/edge-type slots together. ``uniform`` ranks each
    round and edge type separately. Spatial reverse directions remain eligible;
    DAG cycle rejection occurs only while sampling an individual inference.
    """
    if round_policy != "per_round":
        raise ValueError("round_policy must be per_round; shared parameters are unsupported")
    if pruning_scope not in {"global", "uniform"}:
        raise ValueError("pruning_scope must be global or uniform")
    if isinstance(pruning_rate, bool) or not isinstance(pruning_rate, (int, float)):
        raise ValueError("pruning_rate must be a finite number in [0, 1]")
    pruning_rate = float(pruning_rate)
    if not math.isfinite(pruning_rate) or not 0 <= pruning_rate <= 1:
        raise ValueError("pruning_rate must be a finite number in [0, 1]")
    agents, rounds, count = _dimensions(aggregate)
    shape = (rounds, count, count)
    allowed = _admissible(rounds, count)
    parameters, raw = {}, {}
    logits, candidates = {}, {}
    for edge_type in EDGE_TYPES:
        try:
            metric = aggregate["metrics"]["token_ratio"][edge_type]
            sums = _tensor(metric["sum"], shape, f"{edge_type}.sum")
            counts = _tensor(metric["opportunity_count"], shape, f"{edge_type}.opportunity_count")
            importance = _tensor(metric["conditional_mean"], shape, f"{edge_type}.conditional_mean")
        except KeyError as exc:
            raise ValueError("aggregate requires token_ratio sum, opportunity_count and conditional_mean") from exc
        if bool((counts < 0).any()) or not bool(torch.equal(counts, counts.round())):
            raise ValueError(f"{edge_type} opportunity counts must be nonnegative integers")
        if bool((sums < 0).any()) or bool((sums > counts + 1e-7).any()):
            raise ValueError(f"{edge_type} token-ratio sums must lie between zero and counts")
        if bool((importance < 0).any()) or bool((importance > 1 + 1e-7).any()):
            raise ValueError(f"{edge_type} importance must lie in [0, 1]")
        observed = counts > 0
        if not torch.equal(observed, allowed[edge_type]):
            raise ValueError(
                f"{edge_type} coverage must include every admissible per-round edge "
                "and exclude spatial self edges / round-zero temporal edges"
            )
        expected = torch.where(observed, sums / counts.clamp_min(1), 0.0)
        if not torch.allclose(importance, expected, rtol=1e-7, atol=1e-9):
            raise ValueError(f"{edge_type} conditional_mean does not equal sum / opportunity_count")
        candidates[edge_type] = observed
        logits[edge_type] = importance.to(torch.float32)
        parameters[edge_type] = {
            "sum": sums.tolist(),
            "opportunity_count": counts.tolist(),
            "importance": importance.tolist(),
            "logits": logits[edge_type].tolist(),
            "probabilities": torch.sigmoid(logits[edge_type]).tolist(),
            "candidate_mask": observed.to(torch.int64).tolist(),
        }
        raw[edge_type] = {"sum": sums.tolist(), "opportunity_count": counts.tolist()}

    masks = {edge_type: candidates[edge_type].clone() for edge_type in EDGE_TYPES}
    groups = []
    if pruning_scope == "global":
        all_logits = torch.stack([logits[edge_type] for edge_type in EDGE_TYPES])
        all_candidates = torch.stack([candidates[edge_type] for edge_type in EDGE_TYPES])
        retained, removed = _prune_group(all_logits, all_candidates, pruning_rate)
        for index, edge_type in enumerate(EDGE_TYPES):
            masks[edge_type] = retained[index]
        groups.append({
            "name": "global", "round": None, "edge_type": None,
            "candidate_count": int(all_candidates.sum()), "pruned_count": removed,
            "retained_count": int(retained.sum()),
        })
    else:
        for round_index in range(rounds):
            for edge_type in EDGE_TYPES:
                available = int(candidates[edge_type][round_index].sum())
                if not available:
                    continue
                retained, removed = _prune_group(
                    logits[edge_type][round_index], candidates[edge_type][round_index], pruning_rate,
                )
                masks[edge_type][round_index] = retained
                groups.append({
                    "name": f"round_{round_index}_{edge_type}", "round": round_index,
                    "edge_type": edge_type, "candidate_count": available,
                    "pruned_count": removed, "retained_count": int(retained.sum()),
                })
    available = sum(int(value.sum()) for value in candidates.values())
    retained_count = sum(int(value.sum()) for value in masks.values())
    for edge_type in EDGE_TYPES:
        parameters[edge_type]["effective_probabilities"] = (
            torch.sigmoid(logits[edge_type]) * masks[edge_type]
        ).tolist()
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "policy_type": "attribution_logits_agentprune_sampling",
        "round_policy": round_policy,
        "agents": agents,
        "num_rounds": rounds,
        "matrix_orientation": "[round][source][target]",
        "metric": "token_ratio",
        "statistic": "conditional_mean",
        "pruning_scope": pruning_scope,
        "pruning_rate": pruning_rate,
        "probability_mapping": {
            "name": "importance_as_logit", "formula": "sigmoid(importance / 1.0)",
            "temperature": 1.0, "torch_dtype": "float32", "device": "cpu",
            "normalization": "none", "pruned_probability": 0.0,
        },
        "sampling_rules": {
            "candidate_order": "canonical source-major; spatial then temporal each round",
            "spatial": "skip masked, reject cycles, then draw Bernoulli",
            "temporal": "round zero inactive; later rounds include self and have no cycle rejection",
            "round_orders": "FIFO Kahn with canonical initial queue and successor order",
            "reviewer": "all final-round workers; excluded from pruning and sampling",
        },
        "source": {
            key: aggregate[key] for key in (
                "run_fingerprint", "num_aggregated_samples", "num_correct_samples",
                "num_questions", "num_rollouts", "selection_sha256", "aggregation_scope",
            ) if key in aggregate
        },
        "raw_round_statistics": raw,
        "parameters": parameters,
        "masks": {key: value.to(torch.int64).tolist() for key, value in masks.items()},
        "pruning": {
            "candidate_parameter_count": available,
            "pruned_parameter_count": available - retained_count,
            "retained_parameter_count": retained_count,
            "effective_pruning_rate": (available - retained_count) / available if available else 0.0,
            "rounding": "torch.round(float32(count * rate)); minimum one for positive rate",
            "ranking": "torch.argsort(float32 logits); inactive slots set below minimum; no extra tie-break",
            "global_flatten_order": "edge_type(spatial,temporal), round, source, target",
            "groups": groups,
        },
        "active_slots": {
            "candidate": available, "retained": retained_count,
            "candidate_by_round": {
                key: value.sum(dim=(1, 2)).tolist() for key, value in candidates.items()
            },
            "retained_by_round": {
                key: value.sum(dim=(1, 2)).tolist() for key, value in masks.items()
            },
        },
    }


def _cycle_reachable(successors: list[list[int]], start: int, target: int) -> bool:
    pending = [start]
    visited = set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node not in visited:
            visited.add(node)
            pending.extend(successors[node])
    return False


def _round_order(matrix: list[list[int]], agents: list[str]) -> list[str]:
    count = len(agents)
    indegree = [sum(matrix[source][target] for source in range(count)) for target in range(count)]
    pending = deque(index for index, degree in enumerate(indegree) if degree == 0)
    order = []
    while pending:
        source = pending.popleft()
        order.append(agents[source])
        for target, edge in enumerate(matrix[source]):
            if edge:
                indegree[target] -= 1
                if indegree[target] == 0:
                    pending.append(target)
    if len(order) != count:
        raise ValueError("sampled spatial graph unexpectedly contains a cycle")
    return order


def sample_graph(policy: dict, *, seed: int) -> dict:
    """Sample one rollout with a private CPU RNG, without loading a model."""
    if policy.get("policy_version") != POLICY_VERSION or policy.get("sampler_version") != SAMPLER_VERSION:
        raise ValueError("unsupported sampling policy or sampler version")
    if policy.get("round_policy") != "per_round":
        raise ValueError("sampling requires per_round parameters")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    agents, rounds, count = _dimensions(policy)
    shape = (rounds, count, count)
    admissible = _admissible(rounds, count)
    logits, masks = {}, {}
    for edge_type in EDGE_TYPES:
        logits[edge_type] = _tensor(policy["parameters"][edge_type]["logits"], shape, f"{edge_type}.logits").float()
        mask = _tensor(policy["masks"][edge_type], shape, f"{edge_type}.masks")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("sampling masks must contain only zero or one")
        if bool(((mask > 0) & ~admissible[edge_type]).any()):
            raise ValueError("sampling masks cannot include spatial self / round-zero temporal edges")
        masks[edge_type] = mask.bool()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    realized = {key: torch.zeros(shape, dtype=torch.int64).tolist() for key in EDGE_TYPES}
    orders, per_round = [], []
    categories = ("accepted", "masked", "cycle_rejected", "bernoulli_rejected", "draws", "inactive")
    totals = {key: 0 for key in categories}
    totals_by_type = {edge_type: dict(totals) for edge_type in EDGE_TYPES}
    for round_index in range(rounds):
        successors: list[list[int]] = [[] for _ in agents]
        record = {"round": round_index}
        for edge_type in EDGE_TYPES:
            stats = {key: 0 for key in categories}
            if edge_type == "temporal" and round_index == 0:
                stats["inactive"] = count * count
            else:
                for source in range(count):
                    for target in range(count):
                        if not masks[edge_type][round_index, source, target]:
                            stats["masked"] += 1
                            continue
                        if edge_type == "spatial" and _cycle_reachable(successors, target, source):
                            stats["cycle_rejected"] += 1
                            continue
                        probability = torch.sigmoid(logits[edge_type][round_index, source, target])
                        stats["draws"] += 1
                        if torch.rand(1, generator=generator, dtype=torch.float32) < probability:
                            realized[edge_type][round_index][source][target] = 1
                            if edge_type == "spatial":
                                successors[source].append(target)
                            stats["accepted"] += 1
                        else:
                            stats["bernoulli_rejected"] += 1
            record[edge_type] = stats
            for key in categories:
                totals[key] += stats[key]
                totals_by_type[edge_type][key] += stats[key]
        orders.append(_round_order(realized["spatial"][round_index], agents))
        per_round.append(record)
    state = generator.get_state().numpy().tobytes()
    policy_sha256 = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1, "sampler_version": SAMPLER_VERSION,
        "policy_sha256": policy_sha256,
        "seed": seed, "agents": agents, "num_rounds": rounds,
        "round_policy": "per_round", "matrix_orientation": "[round][source][target]",
        "masks": realized, "round_orders": orders,
        "diagnostics": {
            "per_round": per_round, "totals": totals, "totals_by_edge_type": totals_by_type,
            "rng_state_sha256": hashlib.sha256(state).hexdigest(),
            "rng": "private torch.Generator(cpu); torch.rand(1) float32",
            "masked_includes_spatial_diagonal": True,
        },
    }
