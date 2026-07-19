import json

import pytest

from AgentAttrPrune.pruning import MATRIX_ORIENTATION, RoundPolicy


def test_dense_policy_matches_fullconnected_round_topology():
    policy = RoundPolicy.dense(["a", "b", "c"], num_rounds=2)

    assert policy.agents == ("a", "b", "c")
    assert policy.spatial_masks.to(int).tolist() == [
        [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
    ]
    assert not policy.temporal_masks[0].any()
    assert policy.temporal_masks[1].all()
    assert policy.temporal_masks[1, 1, 1]


def test_dense_policy_supports_num_workers_and_validates_agent_order():
    policy = RoundPolicy.dense(num_workers=2, num_rounds=1)

    assert policy.agents == ("agent_0", "agent_1")
    policy.validate_agents(["agent_0", "agent_1"])
    with pytest.raises(ValueError, match="agent order"):
        policy.validate_agents(["agent_1", "agent_0"])


def test_policy_json_round_trip_and_clone_safety(tmp_path):
    original = RoundPolicy.dense(["a", "b"], num_rounds=2)
    path = original.to_json(tmp_path / "policy.json")
    loaded = RoundPolicy.load(path)

    assert json.loads(path.read_text())["matrix_orientation"] == MATRIX_ORIENTATION
    assert loaded.to_dict() == original.to_dict()
    clone = loaded.spatial_masks
    clone[0, 0, 1] = False
    assert loaded.spatial_masks[0, 0, 1]


def test_policy_rejects_cycles_nonbinary_values_and_round_zero_temporal_edges():
    base = {
        "schema_version": 1,
        "agents": ["a", "b"],
        "num_rounds": 1,
        "matrix_orientation": MATRIX_ORIENTATION,
        "masks": {
            "spatial": [[[0, 1], [0, 0]]],
            "temporal": [[[0, 0], [0, 0]]],
        },
    }

    cyclic = json.loads(json.dumps(base))
    cyclic["masks"]["spatial"][0][1][0] = 1
    with pytest.raises(ValueError, match="cyclic"):
        RoundPolicy.from_dict(cyclic)

    nonbinary = json.loads(json.dumps(base))
    nonbinary["masks"]["spatial"][0][0][1] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        RoundPolicy.from_dict(nonbinary)

    temporal = json.loads(json.dumps(base))
    temporal["masks"]["temporal"][0][0][0] = 1
    with pytest.raises(ValueError, match="round zero"):
        RoundPolicy.from_dict(temporal)
