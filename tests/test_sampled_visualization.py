import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from AgentAttrPrune.pruning.sampled import prepare_sampling_policy
from experiments_attrprune.visualize_sampled_mmlu import _digest, load_policies, render


@pytest.fixture
def prepared(tmp_path):
    agents = ["knowledgeable_expert", "critic", "mathematician"]
    metrics = {}
    for edge in ("spatial", "temporal"):
        counts = np.ones((2,3,3))*40
        if edge == "spatial": counts[:,np.arange(3),np.arange(3)] = 0
        else: counts[0] = 0
        means = np.arange(18).reshape(2,3,3)/20
        means[counts == 0] = 0
        means[0,0,1] = 0
        metrics[edge] = {"sum":(means*counts).tolist(),"opportunity_count":counts.tolist(),
                         "conditional_mean":means.tolist()}
    aggregate = {"agents":agents,"num_rounds":2,"matrix_orientation":"[round][source][target]",
                 "metrics":{"token_ratio":metrics},"num_aggregated_samples":40,"run_fingerprint":"same-source"}
    for scope in ("global","uniform"):
        for rate in (50,60):
            directory = tmp_path/scope/f"prune_{rate}pct"; directory.mkdir(parents=True)
            policy = prepare_sampling_policy(aggregate,rate/100,scope)
            (directory/"sampling_policy.json").write_text(json.dumps(policy))
            (directory/"evaluation_config.json").write_text(json.dumps({"sampling_policy_sha256":_digest(policy)}))
    return tmp_path


def test_probability_figures_keep_source_identity_and_frozen_inputs(prepared):
    inputs = {p:p.read_bytes() for p in prepared.rglob("*.json")}
    result = render(prepared,prepared/"figures",dpi=45)
    assert inputs == {p:p.read_bytes() for p in inputs}
    assert len(result["figures"]) == 5
    assert result["color_limits"] == [.5,.75]
    values = json.loads(Path(result["json"]).read_text())
    assert values["agents"] == ["knowledgeable_expert","critic","mathematician"]
    assert values["display_labels"] == ["Knowl.","Critic","Math."]
    assert "not realized" in values["semantics"]
    with Path(result["csv"]).open() as handle: rows = list(csv.DictReader(handle))
    assert len(rows) == 5*2*2*3*3
    assert all(float(row["effective_probability"]) == 0 for row in rows if row["status"] != "retained")
    zero = next(row for row in rows if row["scope"] == "unpruned" and row["round"] == "0"
                and row["edge_type"] == "spatial" and row["source"] == "knowledgeable_expert" and row["target"] == "critic")
    assert float(zero["importance"]) == 0 and float(zero["probability_if_eligible"]) == .5
    assert all(row["probability_if_eligible"] == "" for row in rows if row["round"] == "0" and row["edge_type"] == "temporal")
    for files in result["figures"].values():
        assert Path(files["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert ET.parse(files["svg"]).getroot().tag.endswith("svg")
    svg = Path(result["figures"]["sigmoid_50pct_global_vs_uniform"]["svg"]).read_text()
    assert "Knowl." in svg and "Retained only in this policy" in svg


@pytest.mark.parametrize("kind",["hash","probability","mask","source"])
def test_modified_or_incompatible_policies_are_rejected_before_plotting(prepared,kind):
    path = prepared/"global/prune_50pct/sampling_policy.json"
    policy = json.loads(path.read_text())
    if kind == "hash": policy["pruning_scope"] = "uniform"
    elif kind == "probability": policy["parameters"]["spatial"]["probabilities"][0][0][1] = .3
    elif kind == "mask": policy["masks"]["spatial"][0][0][0] = 1
    else: policy["source"]["run_fingerprint"] = "another-source"
    path.write_text(json.dumps(policy))
    if kind != "hash":
        path.with_name("evaluation_config.json").write_text(json.dumps({"sampling_policy_sha256":_digest(policy)}))
    with pytest.raises(ValueError): render(prepared,prepared/"invalid")
    assert not (prepared/"invalid").exists()


def test_subset_and_unique_selection(prepared):
    policies,_ = load_policies(prepared,["uniform"],[60])
    assert list(policies) == [("uniform",60)]
    with pytest.raises(ValueError,match="unique"): load_policies(prepared,["global","global"],[50])
