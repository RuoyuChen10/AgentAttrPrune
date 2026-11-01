from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest
import torch

from AgentAttrPrune.pruning.sampled import prepare_sampling_policy, sample_graph


def aggregate_fixture(count=5, rounds=2, *, tied=False):
    metrics = {}
    for kind_index, edge_type in enumerate(("spatial", "temporal")):
        sums = torch.zeros((rounds, count, count), dtype=torch.float64)
        counts = torch.zeros_like(sums)
        for round_index in range(rounds):
            for source in range(count):
                for target in range(count):
                    if (edge_type == "spatial" and source == target) or (edge_type == "temporal" and round_index == 0):
                        continue
                    observations = 7 + round_index + source + target
                    value = 0.0 if tied else ((round_index * 31 + source * count + target + kind_index * 17) % 100) / 100
                    counts[round_index, source, target] = observations
                    sums[round_index, source, target] = value * observations
        metrics[edge_type] = {
            "sum": sums.tolist(), "opportunity_count": counts.tolist(),
            "conditional_mean": (sums / counts.clamp_min(1)).tolist(),
        }
    return {
        "agents": [f"agent_{index}" for index in range(count)],
        "matrix_orientation": "[round][source][target]",
        "num_rounds": rounds, "metrics": {"token_ratio": metrics},
        "num_aggregated_samples": 40, "num_correct_samples": 34,
    }


def baseline_oracle(policy, seed):
    """Small independent translation of AgentPrune's optimized Graph path.

    Use the baseline's global torch.rand rather than the production generator,
    recursive cycle check, successor lists, and FIFO topological execution.
    Returning RNG state also verifies skipped cycles and round-zero timing.
    """
    names = policy["agents"]
    size = len(names)
    masks = {kind: torch.tensor(policy["masks"][kind]) for kind in ("spatial", "temporal")}
    logits = {kind: torch.tensor(policy["parameters"][kind]["logits"]) for kind in masks}
    result = {kind: [] for kind in masks}
    orders = []

    def check_cycle(new_node, target_nodes, spatial_successors):
        if new_node in target_nodes:
            return True
        for successor in spatial_successors[new_node]:
            if check_cycle(successor, target_nodes, spatial_successors):
                return True
        return False

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for round_index in range(policy["num_rounds"]):
            successors = [[] for _ in names]
            for kind in ("spatial", "temporal"):
                matrix = [[0] * size for _ in names]
                if kind != "temporal" or round_index != 0:
                    for out_node in range(size):
                        for in_node in range(size):
                            if masks[kind][round_index, out_node, in_node] == 0.0:
                                continue
                            if kind == "spatial" and check_cycle(in_node, {out_node}, successors):
                                continue
                            edge_prob = torch.sigmoid(logits[kind][round_index, out_node, in_node] / 1.0)
                            if torch.rand(1) < edge_prob:
                                matrix[out_node][in_node] = 1
                                if kind == "spatial":
                                    successors[out_node].append(in_node)
                result[kind].append(matrix)
            in_degree = {node: sum(result["spatial"][-1][source][node] for source in range(size)) for node in range(size)}
            queue = [node for node, degree in in_degree.items() if degree == 0]
            order = []
            while queue:
                node = queue.pop(0)
                order.append(names[node])
                for successor in successors[node]:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)
            orders.append(order)
        rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    return result, orders, rng_hash


@pytest.mark.parametrize("scope", ["global", "uniform"])
@pytest.mark.parametrize("rate,removed", [(0.5, 32), (0.6, 39)])
def test_pruning_budget_and_per_round_parameters(scope, rate, removed):
    source = aggregate_fixture()
    before = deepcopy(source)
    policy = prepare_sampling_policy(source, rate, scope)
    assert source == before
    assert policy["round_policy"] == "per_round"
    assert policy["pruning"]["candidate_parameter_count"] == 65
    assert policy["pruning"]["pruned_parameter_count"] == removed
    assert policy["pruning"]["retained_parameter_count"] == 65 - removed
    assert policy["active_slots"]["candidate_by_round"] == {"spatial": [20, 20], "temporal": [0, 25]}
    if scope == "uniform":
        expected = [10, 10, 12] if rate == 0.5 else [12, 12, 15]
        assert [group["pruned_count"] for group in policy["pruning"]["groups"]] == expected
    assert policy["parameters"]["spatial"]["importance"][0][0][1] != policy["parameters"]["spatial"]["importance"][1][0][1]
    for kind in ("spatial", "temporal"):
        assert policy["parameters"][kind]["importance"] == source["metrics"]["token_ratio"][kind]["conditional_mean"]


@pytest.mark.parametrize("scope,rate", [("global", 0.0), ("global", 0.5), ("uniform", 0.6), ("uniform", 1.0)])
@pytest.mark.parametrize("seed", [0, 1, 7, 999, 2**40 + 33])
def test_sampler_matches_baseline_every_edge_order_and_rng(scope, rate, seed):
    policy = prepare_sampling_policy(aggregate_fixture(4, 3), rate, scope)
    before = deepcopy(policy)
    expected_masks, expected_orders, expected_rng = baseline_oracle(policy, seed)
    actual = sample_graph(policy, seed=seed)
    assert policy == before
    assert actual["masks"] == expected_masks
    assert actual["round_orders"] == expected_orders
    assert actual["diagnostics"]["rng_state_sha256"] == expected_rng
    diagnostics = actual["diagnostics"]
    assert diagnostics["per_round"][0]["temporal"]["draws"] == 0
    assert diagnostics["per_round"][0]["temporal"]["inactive"] == 16
    totals = diagnostics["totals"]
    assert totals["draws"] == totals["accepted"] + totals["bernoulli_rejected"]
    assert totals["accepted"] + totals["masked"] + totals["cycle_rejected"] + totals["bernoulli_rejected"] + totals["inactive"] == 3 * 2 * 16


@pytest.mark.parametrize("scope", ["global", "uniform"])
def test_tied_ranking_uses_baseline_full_tensor_argsort(scope):
    policy = prepare_sampling_policy(aggregate_fixture(tied=True), 0.5, scope)
    original = {kind: torch.tensor(policy["parameters"][kind]["candidate_mask"]) for kind in ("spatial", "temporal")}

    def reference(mask):
        flat = mask.flatten().clone()
        count = (flat > 0).sum()
        zeros = (flat == 0).sum()
        prune_count = torch.round(count * 0.5)
        logits = torch.zeros(flat.shape)
        logits[flat == 0] = logits.min() - 1.0
        flat[torch.argsort(logits)[:int(prune_count + zeros)]] = 0
        return flat.reshape(mask.shape)

    if scope == "global":
        expected = reference(torch.stack(list(original.values())))
        assert policy["masks"] == {kind: expected[index].tolist() for index, kind in enumerate(original)}
    else:
        for kind, mask in original.items():
            for round_index in range(2):
                assert policy["masks"][kind][round_index] == reference(mask[round_index]).tolist()


def test_no_static_dag_projection_zero_scores_still_sample_and_rng_is_private():
    policy = prepare_sampling_policy(aggregate_fixture(tied=True), 0, "global")
    assert policy["pruning"]["retained_parameter_count"] == 65
    assert policy["parameters"]["spatial"]["probabilities"][0][4][0] == 0.5
    assert policy["masks"]["spatial"][0][4][0] == 1
    initial_state = torch.get_rng_state().clone()
    rollouts = [sample_graph(policy, seed=seed) for seed in range(16)]
    assert torch.equal(initial_state, torch.get_rng_state())
    assert any(run["masks"]["spatial"][0][source][target] for run in rollouts for source in range(5) for target in range(source))
    assert any(run["diagnostics"]["totals"]["cycle_rejected"] for run in rollouts)
    assert any(run["masks"]["spatial"][0] != run["masks"]["spatial"][1] for run in rollouts)
    assert any(run["masks"]["temporal"][1][index][index] for run in rollouts for index in range(5))
    assert all(not run["masks"]["spatial"][round_index][index][index] for run in rollouts for round_index in range(2) for index in range(5))
    assert sample_graph(policy, seed=7) == sample_graph(policy, seed=7)


def test_scores_are_logits_not_direct_probabilities():
    source = aggregate_fixture()
    source["metrics"]["token_ratio"]["spatial"]["conditional_mean"][0][0][1] = 0.8
    source["metrics"]["token_ratio"]["spatial"]["sum"][0][0][1] = 0.8 * source["metrics"]["token_ratio"]["spatial"]["opportunity_count"][0][0][1]
    policy = prepare_sampling_policy(source, 0, "uniform")
    parameters = policy["parameters"]["spatial"]
    assert parameters["logits"][0][0][1] == pytest.approx(0.8)
    assert parameters["probabilities"][0][0][1] == torch.sigmoid(torch.tensor(0.8)).item()
    assert parameters["probabilities"][0][0][1] != pytest.approx(0.8)


@pytest.mark.parametrize("bad_rate", [-0.1, 1.1, float("nan"), True])
def test_invalid_rates_rejected(bad_rate):
    with pytest.raises(ValueError, match="pruning_rate"):
        prepare_sampling_policy(aggregate_fixture(), bad_rate, "global")


def test_missing_observations_wrong_denominator_and_shared_rejected():
    source = aggregate_fixture()
    with pytest.raises(ValueError, match="per_round"):
        prepare_sampling_policy(source, 0.5, "global", round_policy="shared")
    source["metrics"]["token_ratio"]["spatial"]["conditional_mean"][0][0][1] = 0.9
    with pytest.raises(ValueError, match="sum / opportunity_count"):
        prepare_sampling_policy(source, 0.5, "global")
    source = aggregate_fixture()
    for field in ("sum", "opportunity_count", "conditional_mean"):
        source["metrics"]["token_ratio"]["spatial"][field][0][0][1] = 0
    with pytest.raises(ValueError, match="coverage"):
        prepare_sampling_policy(source, 0.5, "global")


def test_invalid_spatial_self_and_round_zero_temporal_masks_rejected():
    policy = prepare_sampling_policy(aggregate_fixture(), 0, "global")
    for edge_type in ("spatial", "temporal"):
        invalid = deepcopy(policy)
        invalid["masks"][edge_type][0][0][0] = 1
        with pytest.raises(ValueError, match="spatial self / round-zero temporal"):
            sample_graph(invalid, seed=0)


@pytest.mark.parametrize("orientation", [None, "[round][target][source]"])
def test_missing_or_transposed_orientation_is_rejected(orientation):
    source = aggregate_fixture()
    source["matrix_orientation"] = orientation
    with pytest.raises(ValueError, match="matrix_orientation"):
        prepare_sampling_policy(source, 0.5, "global")
