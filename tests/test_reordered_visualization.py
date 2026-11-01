from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from AgentAttrPrune.pruning.reordered import make_order_plan, project_aggregate, build_reordered_policy
from experiments_attrprune.reordered_visualization import ordered_block, render_matrices


def fixture():
    spatial = np.array([[[0,.7,.4],[.1,0,.2],[.8,.2,0]],
                        [[0,.1,.2],[.8,0,.9],[.1,.2,0]]])
    counts = np.full((2,3,3), 4.)
    for r in range(2):
        np.fill_diagonal(counts[r], 0)
    counts[0,2,0] = 3
    temporal = np.zeros((2,3,3))
    temporal[1] = np.arange(1,10).reshape(3,3) / 20
    temporal_counts = np.zeros_like(temporal)
    temporal_counts[1] = 4
    aggregate = {"agents": ["knowledgeable_expert", "critic", "historian"],
                 "num_rounds": 2, "matrix_orientation": "[round][source][target]",
                 "run_fingerprint": "source", "num_aggregated_samples": 4,
                 "metrics": {"token_ratio": {}}}
    for edge, means, count in (("spatial",spatial,counts), ("temporal",temporal,temporal_counts)):
        aggregate["metrics"]["token_ratio"][edge] = {
            "sum": (means * count).tolist(), "opportunity_count": count.tolist(),
            "candidate_count": count.tolist(), "conditional_mean": means.tolist()}
    plan = make_order_plan(aggregate)
    projected = project_aggregate(aggregate, plan)
    return aggregate, plan, projected


def test_temporal_display_rows_and_columns_use_different_round_orders():
    source, plan, projected = fixture()
    assert plan["round_orders"][0] != plan["round_orders"][1]
    data = ordered_block(projected, plan, "temporal", 1)
    assert data["source_agents"] == plan["round_orders"][0]
    assert data["target_agents"] == plan["round_orders"][1]
    for i, sender in enumerate(data["source_agents"]):
        for j, receiver in enumerate(data["target_agents"]):
            assert data["conditional_mean"][i,j] == source["metrics"]["token_ratio"]["temporal"]["conditional_mean"][1][source["agents"].index(sender)][source["agents"].index(receiver)]
    for r in range(2):
        spatial = ordered_block(projected, plan, "spatial", r)
        assert np.array_equal(spatial["opportunity_count"] > 0, np.triu(np.ones((3,3),dtype=bool),1))


def test_rendering_preserves_counts_and_exports_unambiguous_identity_axes(tmp_path):
    source, plan, aggregate = fixture()
    policies = {50: build_reordered_policy(aggregate, plan, 50)}
    before = copy.deepcopy((aggregate, plan, policies))
    result = render_matrices(aggregate, plan, policies, tmp_path, branch="direct")
    assert (aggregate, plan, policies) == before
    assert result["importance_color_limits"] == [0,1]
    payload = json.loads(Path(result["matrices_json"]).read_text())
    assert payload["canonical_metrics"] == aggregate["metrics"]
    temporal = payload["display_blocks"]["round_1_temporal"]
    assert temporal["source_agents"] == plan["round_orders"][0]
    assert temporal["target_agents"] == plan["round_orders"][1]
    with Path(result["edges_csv"]).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 27
    for row in rows:
        r, edge = int(row["round"]), row["type"]
        i, j = source["agents"].index(row["source"]), source["agents"].index(row["target"])
        assert float(row["opportunity_count"]) == aggregate["metrics"]["token_ratio"][edge]["opportunity_count"][r][i][j]
        assert int(row["retained_50pct"]) == policies[50]["masks"][edge][r][i][j]
    for paths in result["files"].values():
        assert Path(paths["png"]).read_bytes().startswith(b"\x89PNG")
        text = Path(paths["svg"]).read_text()
        assert "Knowl." in text and "source · round 0" in text and "target · round 1" in text
    assert "Colored 0 = observed zero" in text


def test_foreign_order_and_forbidden_policy_fail_before_output(tmp_path):
    _, plan, aggregate = fixture()
    broken = copy.deepcopy(plan)
    broken["agents"].reverse()
    with pytest.raises(ValueError, match="identity"):
        ordered_block(aggregate, broken, "spatial", 0)
    policies = {50: build_reordered_policy(aggregate, plan, 50)}
    policies[50]["masks"]["spatial"][0][0][0] = 1
    with pytest.raises(ValueError, match="forbidden"):
        render_matrices(aggregate, plan, policies, tmp_path / "unused", branch="direct")
    assert not (tmp_path / "unused").exists()
