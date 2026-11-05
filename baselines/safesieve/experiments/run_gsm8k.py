"""Unchanged native SafeSieve GSM8K scoring and node-removal helpers."""
import math
import re
from decimal import Decimal, InvalidOperation

def score_answer(response, target, predictor):
    # Keep baseline extraction; malformed answers are wrong, not execution failures.
    if not re.search(r"\d", response):
        return None, False
    try:
        prediction = predictor(response)
        correct = Decimal(str(prediction).replace(",", "")) == Decimal(target)
    except (ValueError, TypeError, IndexError, InvalidOperation):
        prediction, correct = None, False
    return prediction, bool(correct)


def remove_isolated_nodes(graph, edge_manager, min_agents):
    import torch
    old_order = list(graph.node_order)
    size = len(old_order)
    temporal = {name: getattr(graph, name).detach().reshape(size, size).clone()
                for name in ("temporal_logits", "temporal_masks")}
    old_kwargs = dict(zip(old_order, graph.node_kwargs))
    removed = graph.check_and_remove_isolated_nodes(min_agents)
    if removed:
        kept_indices = [old_order.index(node_id) for node_id in graph.node_order]
        removed_indices = [old_order.index(node_id) for node_id in removed]
        edge_manager.remove_nodes_and_update_matrix(removed_indices, kept_indices)
        # Graph's generic removal does not resize unoptimized temporal tensors,
        # and its positional node_kwargs reconstruction can select wrong roles.
        for name, matrix in temporal.items():
            value = matrix[kept_indices][:, kept_indices].flatten()
            setattr(graph, name, torch.nn.Parameter(value, requires_grad=False))
        graph.node_kwargs = [old_kwargs[node_id] for node_id in graph.node_order]
        graph.agent_llm_assignments = [graph.nodes[node_id].llm_name for node_id in graph.node_order]
    return removed


def parse_similarity_score(response, maximum):
    """Prefer a labeled score; never infer it from an arbitrary reasoning number."""
    number = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    text = str(response).strip()
    patterns = (
        r"(?:score|points|rating)\s*[*`]*\s*[:：]\s*[*`]*\s*" + number,
        number + r"\s*(?:/|out of)\s*" + re.escape(str(maximum).removesuffix(".0")) + r"(?:\.0)?\b",
        number + r"\s*points\b",
        r"^[*`\s]*" + number + r"[*`\s.]*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if math.isfinite(score) and 0 <= score <= maximum:
                return score
            raise ValueError(f"Similarity score must be in [0, {maximum}]: {response!r}")
    raise ValueError(f"Cannot extract a labeled or standalone similarity score: {response!r}")
