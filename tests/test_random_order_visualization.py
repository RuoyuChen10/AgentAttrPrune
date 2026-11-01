from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from experiments_attrprune.random_order_visualization import render_random_order_visualizations


def _aggregate(*, reference=False):
    spatial_counts = np.zeros((2, 3, 3), dtype=float)
    spatial_sums = spatial_counts.copy()
    spatial_counts[0, 0, 1] = 3
    spatial_sums[0, 0, 1] = 0.9
    spatial_counts[1, 0, 1] = 1
    spatial_sums[1, 0, 1] = 0.9
    if not reference:
        # An observed zero in the reverse direction must remain visible.
        spatial_counts[0, 1, 0] = 1
    temporal_counts = np.zeros((2, 3, 3), dtype=float)
    temporal_counts[1] = 4
    temporal_sums = temporal_counts * 0.25
    metrics = {}
    for edge_type, sums, counts in (
        ("spatial", spatial_sums, spatial_counts),
        ("temporal", temporal_sums, temporal_counts),
    ):
        metrics[edge_type] = {
            "sum": sums.tolist(), "opportunity_count": counts.tolist(),
            "candidate_count": counts.tolist(),
            "conditional_mean": np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).tolist(),
        }
    return {
        "agents": ["expert", "critic", "historian"],
        "agent_roles": ["Expert", "Critic", "Historian"],
        "num_rounds": 2, "matrix_orientation": "[round][source][target]",
        "num_aggregated_samples": 4 if reference else 8,
        "run_fingerprint": "fixed" if reference else "random",
        "metrics": {"token_ratio": metrics},
    }


def test_visualizations_preserve_observed_zeros_and_pool_with_counts(tmp_path):
    aggregate, reference = _aggregate(), _aggregate(reference=True)
    original = copy.deepcopy((aggregate, reference))
    manifest = render_random_order_visualizations(aggregate, tmp_path, reference)
    assert (aggregate, reference) == original
    assert manifest["num_random_rollouts"] == 8
    assert manifest["num_reference_rollouts"] == 4
    assert manifest["intensity_color_limits"] == [0.0, 1.0]
    assert json.loads(Path(manifest["manifest_json"]).read_text()) == manifest

    pooled = json.loads(Path(manifest["files"]["pooled_spatial_json"]).read_text())
    assert pooled["sum"][0][1] == pytest.approx(1.8)
    assert pooled["opportunity_count"][0][1] == 4
    assert pooled["conditional_mean"][0][1] == pytest.approx(0.45)  # Not (0.3 + 0.9) / 2.
    assert pooled["conditional_mean"][1][0] == 0.0
    assert pooled["observed"][1][0] is True
    assert pooled["conditional_mean"][0][2] is None
    assert pooled["observed"][0][2] is False
    assert pooled["structural_allowed"][0][2] is True
    assert pooled["conditional_mean"][0][0] == 0.0
    assert pooled["structural_allowed"][0][0] is False

    with Path(manifest["files"]["edges_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 * 3 * 3
    keyed = {(row["round"], row["type"], row["source"], row["target"]): row for row in rows}
    observed_zero = keyed["0", "spatial", "critic", "expert"]
    assert observed_zero["observed"] == "True"
    assert observed_zero["conditional_mean"] == "0.0"
    absent = keyed["0", "spatial", "expert", "historian"]
    assert absent["conditional_mean"] == ""
    assert absent["observed"] == "False" and absent["structural_allowed"] == "True"
    forbidden = keyed["0", "spatial", "expert", "expert"]
    assert forbidden["conditional_mean"] == "0.0" and forbidden["structural_allowed"] == "False"
    temporal_self = keyed["1", "temporal", "expert", "expert"]
    assert temporal_self["count"] == "4" and temporal_self["structural_allowed"] == "True"

    for kind in ("conditional_mean", "opportunity_count", "pooled_spatial", "reference_comparison"):
        paths = manifest["files"][kind]
        assert Path(paths["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert ET.parse(paths["svg"]).getroot().tag.endswith("svg")
    svg = Path(manifest["files"]["reference_comparison"]["svg"]).read_text()
    assert "N/A" in svg and "forbidden" in svg
    assert "Fixed order · 4 rollouts" in svg and "Random order · 8 rollouts" in svg
    assert "Hist." in svg and "Critic" in svg
    assert manifest["display_labels"] == ["expert", "Critic", "Hist."]
    assert manifest["agents"] == pooled["agents"] == aggregate["agents"]
    assert any(row["target"] == "historian" for row in rows)


@pytest.mark.parametrize("field,replacement", [
    ("agents", ["critic", "expert", "historian"]),
    ("agent_roles", ["Changed", "Critic", "Historian"]),
    ("num_rounds", 3),
])
def test_reference_identity_and_round_mismatch_is_rejected_without_outputs(tmp_path, field, replacement):
    reference = _aggregate(reference=True)
    reference[field] = replacement
    with pytest.raises(ValueError, match="must match"):
        render_random_order_visualizations(_aggregate(), tmp_path / "unused", reference)
    assert not (tmp_path / "unused").exists()


def test_inconsistent_conditional_mean_is_rejected(tmp_path):
    aggregate = _aggregate()
    aggregate["metrics"]["token_ratio"]["spatial"]["conditional_mean"][0][0][1] = 0.9
    with pytest.raises(ValueError, match="sum / opportunity_count"):
        render_random_order_visualizations(aggregate, tmp_path / "unused", _aggregate(reference=True))
    assert not (tmp_path / "unused").exists()
