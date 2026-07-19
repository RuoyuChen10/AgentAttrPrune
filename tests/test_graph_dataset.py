from __future__ import annotations

import csv

import pytest

from AgentAttrPrune.datasets import MMLUDataset


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_mmlu_dataset_reads_original_layout_and_builds_model_sample(tmp_path):
    _write_csv(
        tmp_path / "data" / "dev" / "abstract_algebra_dev.csv",
        [["Question?", "one", "two", "three", "four", "B"]],
    )
    dataset = MMLUDataset("dev", tmp_path, shuffle_seed=None)

    assert len(dataset) == 1
    record = dataset[0]
    assert record["subject"] == "abstract_algebra"
    sample = dataset.record_to_sample(record)
    assert sample["answer"] == "B"
    assert "Option A: one" in sample["task"]
    assert sample["options"]["D"] == "four"


def test_mmlu_dataset_errors_cleanly_when_split_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="split directory"):
        MMLUDataset("val", tmp_path, download=False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Decision: C", "C"), ("A\nreason", "A"), ("Option D", "D")],
)
def test_mmlu_answer_normalization(raw, expected):
    assert MMLUDataset.normalize_answer(raw) == expected
