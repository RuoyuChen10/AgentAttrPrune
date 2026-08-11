from __future__ import annotations

import json

from AgentAttrPrune.datasets import (
    GSM8KDataset,
    HumanEvalDataset,
    dataset_identity,
    evaluate_humaneval_prediction,
)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_gsm8k_dataset_builds_samples_and_normalizes_numeric_answers(tmp_path):
    _write_jsonl(
        tmp_path / "test.jsonl",
        [
            {
                "question": "What is half of one?",
                "answer": "One divided by two is one half.\n#### 0.5",
            }
        ],
    )

    dataset = GSM8KDataset("test", tmp_path, shuffle_seed=None)
    sample = dataset.record_to_sample(dataset[0])

    assert sample == {
        "task": "What is half of one?",
        "benchmark": "gsm8k",
        "question": "What is half of one?",
        "answer": "0.5",
    }
    assert dataset.normalize_answer("The answer is 1/2") == "0.5"
    assert dataset.normalize_answer(r"Therefore \boxed{1,200}") == "1200"


def test_humaneval_partition_is_disjoint_and_identity_locks_selection(tmp_path):
    records = [
        {
            "name": f"task_{index}",
            "prompt": f"def f{index}():\n    pass\n",
            "entry_point": f"f{index}",
            "test": "assert True\n",
        }
        for index in range(6)
    ]
    _write_jsonl(tmp_path / "humaneval-py.jsonl", records)

    dev = HumanEvalDataset("dev", tmp_path, shuffle_seed=None)
    test = HumanEvalDataset("test", tmp_path, shuffle_seed=None)
    dev_ids = {record["name"] for record in dev}
    test_ids = {record["name"] for record in test}

    assert len(dev) == len(test) == 3
    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == {record["name"] for record in records}

    identity = dataset_identity(
        "humaneval",
        tmp_path,
        splits=("dev", "test"),
    )
    assert identity["partition"]["strategy"] == "seeded_disjoint_half"
    assert identity["splits"]["dev"]["record_count"] == 3
    assert (
        identity["splits"]["dev"]["selection_sha256"]
        != identity["splits"]["test"]["selection_sha256"]
    )


def test_humaneval_prediction_runs_complete_function_in_limited_subprocess():
    sample = {
        "task": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
        "entry_point": "add",
        "test": "assert add(2, 3) == 5\n",
    }
    result = evaluate_humaneval_prediction(
        "```python\ndef add(a, b):\n    return a + b\n```",
        sample,
        timeout=2.0,
    )

    assert result["correct"] is True
    assert result["parsed_answer"] == result["gold_answer"] == "pass"
    assert result["details"]["status"] == "passed"
