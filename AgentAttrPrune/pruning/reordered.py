"""Deterministic per-round ordering and static projection of attribution data."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from AgentAttrPrune.attribution.edge import build_pruned_masks
from AgentAttrPrune.pruning.policy import RoundPolicy


ORIENTATION = "[round][source][target]"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _dimensions(aggregate: Mapping[str, Any]) -> tuple[list[str], int, tuple[int, int, int]]:
    agents = aggregate.get("agents")
    rounds = aggregate.get("num_rounds")
    if (not isinstance(agents, list) or not agents
            or any(not isinstance(a, str) or not a for a in agents)
            or len(set(agents)) != len(agents)):
        raise ValueError("agents must be unique nonempty canonical IDs")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("num_rounds must be a positive integer")
    if aggregate.get("matrix_orientation") != ORIENTATION:
        raise ValueError(f"matrix_orientation must be {ORIENTATION}")
    return list(agents), rounds, (rounds, len(agents), len(agents))


def _matrix(value: Any, shape: tuple[int, int, int], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric matrix") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return result


def _candidate_masks(agents: list[str], orders: list[list[str]]) -> dict[str, np.ndarray]:
    shape = (len(orders), len(agents), len(agents))
    spatial = np.zeros(shape, dtype=bool)
    temporal = np.ones(shape, dtype=bool)
    temporal[0] = False
    for r, order in enumerate(orders):
        if (not isinstance(order, (list, tuple)) or len(order) != len(agents)
                or any(not isinstance(a, str) for a in order) or set(order) != set(agents)):
            raise ValueError("Every round order must be a permutation of canonical agent IDs")
        positions = {agent: i for i, agent in enumerate(order)}
        spatial[r] = [[positions[source] < positions[target] for target in agents] for source in agents]
    return {"spatial": spatial, "temporal": temporal}


def _checked_order_plan(aggregate: Mapping[str, Any], order_plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    agents, rounds, shape = _dimensions(aggregate)
    if (order_plan.get("agents") != agents or order_plan.get("num_rounds") != rounds
            or order_plan.get("matrix_orientation") != ORIENTATION):
        raise ValueError("Order plan identity, dimensions or orientation changed")
    orders = order_plan.get("round_orders")
    if not isinstance(orders, (list, tuple)) or len(orders) != rounds:
        raise ValueError("round_orders must contain one permutation per round")
    candidates = _candidate_masks(agents, orders)
    for edge, expected in candidates.items():
        declared = _matrix(order_plan.get("candidate_masks", {}).get(edge), shape, f"{edge} candidate mask")
        if not np.array_equal(declared, expected):
            raise ValueError("Order plan candidate masks do not match its round orders")
    return candidates


def _validate_token_statistics(aggregate: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    _, _, shape = _dimensions(aggregate)
    result = {}
    for edge in ("spatial", "temporal"):
        try:
            payload = aggregate["metrics"]["token_ratio"][edge]
            values = {key: _matrix(payload[key], shape, f"{edge}.{key}") for key in
                      ("conditional_mean", "sum", "opportunity_count", "candidate_count")}
        except KeyError as error:
            raise ValueError("Complete token_ratio statistics are required") from error
        counts, candidates = values["opportunity_count"], values["candidate_count"]
        if (np.any(counts < 0) or np.any(candidates < 0)
                or not np.array_equal(counts, np.round(counts))
                or not np.array_equal(candidates, np.round(candidates))
                or np.any(counts > candidates)):
            raise ValueError("Observation/candidate counts must be nonnegative integers with observations <= candidates")
        expected = np.divide(values["sum"], counts, out=np.zeros_like(counts), where=counts > 0)
        if (np.any(values["sum"] < 0) or np.any(values["sum"] > counts + 1e-7)
                or not np.allclose(values["conditional_mean"], expected, rtol=1e-7, atol=1e-9)):
            raise ValueError("Token ratio conditional_mean must equal sum / opportunity_count")
        result[edge] = values
    return result


def make_order_plan(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Sort raw spatial row-minus-column sums once, independently per round."""
    agents, rounds, shape = _dimensions(aggregate)
    stats = _validate_token_statistics(aggregate)
    spatial = stats["spatial"]["conditional_mean"]
    off = np.broadcast_to(~np.eye(len(agents), dtype=bool), shape)
    if not np.array_equal(stats["spatial"]["opportunity_count"] > 0, off):
        raise ValueError("Ordering requires every non-diagonal spatial direction to be observed")
    temporal_allowed = np.ones(shape, dtype=bool)
    temporal_allowed[0] = False
    if not np.array_equal(stats["temporal"]["opportunity_count"] > 0, temporal_allowed):
        raise ValueError("Ordering requires complete temporal coverage after round zero")
    orders, scores = [], []
    for r in range(rounds):
        rows, columns = spatial[r].sum(axis=1), spatial[r].sum(axis=0)
        score = rows - columns
        indices = sorted(range(len(agents)), key=lambda i: (-float(score[i]), i))
        orders.append([agents[i] for i in indices])
        scores.append({"round": r, "row_sum": rows.tolist(), "column_sum": columns.tolist(),
                       "score": score.tolist()})
    candidates = _candidate_masks(agents, orders)
    return {"schema_version": 1, "method": "spatial_row_sum_minus_column_sum",
            "metric": "token_ratio", "statistic": "conditional_mean", "apply_sigmoid": False,
            "tie_break": "canonical agent index ascending", "agents": agents,
            "num_rounds": rounds, "matrix_orientation": ORIENTATION, "round_orders": orders,
            "scores": scores, "source_run_fingerprint": aggregate.get("run_fingerprint"),
            "source_spatial_sha256": _digest(aggregate["metrics"]["token_ratio"]["spatial"]),
            "candidate_masks": {edge: value.astype(int).tolist() for edge, value in candidates.items()}}


def project_aggregate(aggregate: Mapping[str, Any], order_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Filter a copy, retaining original denominators and canonical coordinates.

    The frozen order is not recomputed from this aggregate, so recalibrated
    matrices can be evaluated under the same original order.
    """
    _, _, shape = _dimensions(aggregate)
    candidates = _checked_order_plan(aggregate, order_plan)
    stats = _validate_token_statistics(aggregate)
    for edge, allowed in candidates.items():
        if np.any(stats[edge]["opportunity_count"][allowed] <= 0):
            raise ValueError("Every allowed edge needs a completed attribution observation")
        if edge == "temporal" and np.any(stats[edge]["candidate_count"][~allowed] != 0):
            raise ValueError("Round zero temporal candidates must be absent")
    result = copy.deepcopy(dict(aggregate))
    for metric, payload in result["metrics"].items():
        for name, value in payload["spatial"].items():
            values = _matrix(value, shape, f"{metric}.spatial.{name}").copy()
            values[~candidates["spatial"]] = 0
            payload["spatial"][name] = values.tolist()
    result["ordering_projection"] = {
        "kind": "derived_order_projection", "round_orders": copy.deepcopy(order_plan["round_orders"]),
        "order_plan_sha256": _digest(order_plan), "source_aggregate_sha256": _digest(aggregate),
        "statistics": "allowed edges retain their original sums and observation counts; no inference or attribution performed",
        "matrix_orientation": ORIENTATION,
        "worker_candidate_edges": sum(int(value.sum()) for value in candidates.values()),
    }
    # These fields describe the unprojected source coverage, not this derivative.
    result.pop("spatial_off_diagonal_coverage_complete", None)
    result.pop("unobserved_spatial_edges", None)
    return result


def build_reordered_policy(aggregate: Mapping[str, Any], order_plan: Mapping[str, Any],
                           pruning_percent: int) -> dict[str, Any]:
    """Globally prune the ordered candidate graph without further sampling."""
    if isinstance(pruning_percent, bool) or not isinstance(pruning_percent, int) or not 0 <= pruning_percent <= 100:
        raise ValueError("pruning_percent must be an integer in [0, 100]")
    projected = project_aggregate(aggregate, order_plan)
    payload = build_pruned_masks(projected, metric="token_ratio", statistic="conditional_mean",
                                 pruning_rate=pruning_percent / 100, threshold=None,
                                 pruning_scope="global", unobserved="drop")
    candidates = _checked_order_plan(projected, order_plan)
    count = sum(int(value.sum()) for value in candidates.values())
    removed = round(count * pruning_percent / 100)
    if count and pruning_percent:
        removed = max(1, removed)
    actual_candidates = sum(value["candidate_edges"] for value in payload["retention"].values())
    actual_retained = sum(value["retained_candidate_edges"] for value in payload["retention"].values())
    if (actual_candidates != count or actual_retained != count - removed
            or any(value["dag_projection_dropped_edges"] for value in payload["retention"].values())):
        raise ValueError("Ordered global pruning changed the candidate budget or introduced a DAG projection")
    payload.update({"kind": "fixed_order_static_pruning", "execution_mode": "fixed_edges",
                    "round_orders": copy.deepcopy(order_plan["round_orders"]),
                    "order_plan_sha256": _digest(order_plan), "requested_pruning_percent": pruning_percent,
                    "worker_candidate_edges": count, "worker_retained_edges": actual_retained})
    RoundPolicy.from_dict(payload)
    return payload
