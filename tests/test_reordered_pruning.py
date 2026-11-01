"""Order projection preserves conditional denominators and static budgets."""
from copy import deepcopy

import numpy as np
import pytest

from AgentAttrPrune.pruning.reordered import make_order_plan, project_aggregate, build_reordered_policy


def aggregate_fixture():
    agents = ["a", "b", "c", "d", "e"]
    result = {"agents": agents, "num_rounds": 2, "matrix_orientation": "[round][source][target]",
              "run_fingerprint": "test", "num_aggregated_samples": 40, "aggregation_scope": "all", "metrics": {}}
    spatial = np.array([[(i + 1) * .02 + (j + 1) * .005 for j in range(5)] for i in range(5)])
    np.fill_diagonal(spatial, 0)
    weights = {"spatial": np.stack([spatial, spatial.T]),
               "temporal": np.stack([np.zeros((5, 5)), np.full((5, 5), .1)])}
    counts = np.array([[20 + i - j for j in range(5)] for i in range(5)], dtype=float)
    np.fill_diagonal(counts, 0)
    opportunities = {"spatial": np.stack([counts, counts]),
                     "temporal": np.stack([np.zeros((5, 5)), np.full((5, 5), 40)])}
    for metric in ("binary", "word_ratio", "token_ratio"):
        result["metrics"][metric] = {}
        for edge in ("spatial", "temporal"):
            w, count = weights[edge], opportunities[edge]
            result["metrics"][metric][edge] = {
                "conditional_mean": w.tolist(), "sum": (w * count).tolist(),
                "sample_mean": (w * count / 40).tolist(), "opportunity_count": count.tolist(),
                "candidate_count": count.tolist()}
    return result


def test_per_round_raw_scores_have_different_orders_and_stable_ties():
    source = aggregate_fixture()
    saved = deepcopy(source)
    plan = make_order_plan(source)
    assert plan["round_orders"] == [["e", "d", "c", "b", "a"], ["a", "b", "c", "d", "e"]]
    for r, record in enumerate(plan["scores"]):
        assert sum(record["score"]) == pytest.approx(0)
        assert np.asarray(record["row_sum"]) - record["column_sum"] == pytest.approx(record["score"])
        assert np.asarray(plan["candidate_masks"]["spatial"][r]).sum() == 10
    assert np.asarray(plan["candidate_masks"]["temporal"]).sum() == 25
    assert source == saved
    for values in source["metrics"]["token_ratio"].values():
        values["sum"] = np.zeros((2,5,5)).tolist()
        values["conditional_mean"] = np.zeros((2,5,5)).tolist()
    tied = make_order_plan(source)
    assert tied["round_orders"] == [source["agents"]] * 2
    assert tied["apply_sigmoid"] is False


def test_projection_retains_source_denominators_and_temporal_verbatim():
    source = aggregate_fixture()
    before = deepcopy(source)
    plan = make_order_plan(source)
    projected = project_aggregate(source, plan)
    allowed = np.asarray(plan["candidate_masks"]["spatial"], dtype=bool)
    for metric, payload in projected["metrics"].items():
        assert payload["temporal"] == source["metrics"][metric]["temporal"]
        for statistic, values in payload["spatial"].items():
            actual = np.asarray(values)
            expected = np.asarray(source["metrics"][metric]["spatial"][statistic])
            assert np.array_equal(actual[allowed], expected[allowed])
            assert not actual[~allowed].any()
    observed = np.asarray(projected["metrics"]["token_ratio"]["spatial"]["opportunity_count"])
    assert set(observed[allowed]) != {40}
    assert projected["ordering_projection"]["worker_candidate_edges"] == 45
    assert projected["ordering_projection"]["kind"] == "derived_order_projection"
    assert source == before


@pytest.mark.parametrize("percentage,retained", [(0,45), (50,23), (60,18), (80,9), (100,0)])
def test_static_global_budget_counts_ordered_candidates(percentage, retained):
    source = aggregate_fixture()
    plan = make_order_plan(source)
    policy = build_reordered_policy(source, plan, percentage)
    assert policy["worker_candidate_edges"] == 45
    assert policy["worker_retained_edges"] == retained
    assert policy["execution_mode"] == "fixed_edges"
    assert policy["round_orders"] == plan["round_orders"]
    assert policy["pruning_allocation"] == "global"
    for edge in ("spatial", "temporal"):
        assert policy["retention"][edge]["dag_projection_dropped_edges"] == 0
        assert not np.any(np.asarray(policy["masks"][edge]) > plan["candidate_masks"][edge])


def test_rates_start_from_original_and_masks_are_nested():
    source = aggregate_fixture()
    original = deepcopy(source)
    plan = make_order_plan(source)
    policies = [build_reordered_policy(source, plan, p) for p in (50,60,80)]
    for left, right in zip(policies, policies[1:]):
        for edge in ("spatial", "temporal"):
            assert np.all(np.asarray(right["masks"][edge]) <= left["masks"][edge])
    assert source == original


def test_recalibration_uses_frozen_order_even_when_new_scores_reverse():
    source = aggregate_fixture()
    plan = make_order_plan(source)
    recalibrated = deepcopy(source)
    for metric in recalibrated["metrics"].values():
        for key, values in metric["spatial"].items():
            metric["spatial"][key] = np.asarray(values)[::-1].tolist()
    assert make_order_plan(recalibrated)["round_orders"] != plan["round_orders"]
    projected = project_aggregate(recalibrated, plan)
    # A second projection is allowed: the frozen plan must not be inferred anew.
    policy = build_reordered_policy(projected, plan, 60)
    assert policy["round_orders"] == plan["round_orders"]
    assert policy["worker_retained_edges"] == 18


@pytest.mark.parametrize("mutation", ["missing_observation", "wrong_sum", "orientation", "nan"])
def test_invalid_source_is_rejected(mutation):
    source = aggregate_fixture()
    values = source["metrics"]["token_ratio"]["spatial"]
    if mutation == "missing_observation":
        values["opportunity_count"][0][0][1] = 0
        values["sum"][0][0][1] = 0
        values["conditional_mean"][0][0][1] = 0
    elif mutation == "wrong_sum":
        values["sum"][0][0][1] += 1
    elif mutation == "orientation":
        source["matrix_orientation"] = "[round][target][source]"
    else:
        values["conditional_mean"][0][0][1] = float("nan")
    with pytest.raises(ValueError):
        make_order_plan(source)


def test_changed_order_or_missing_allowed_edge_is_rejected():
    source = aggregate_fixture()
    plan = make_order_plan(source)
    changed = deepcopy(plan)
    changed["round_orders"][0].reverse()
    with pytest.raises(ValueError, match="candidate masks"):
        project_aggregate(source, changed)
    for key in ("opportunity_count", "conditional_mean", "sum"):
        source["metrics"]["token_ratio"]["spatial"][key][0][4][0] = 0
    with pytest.raises(ValueError, match="allowed edge"):
        project_aggregate(source, plan)
