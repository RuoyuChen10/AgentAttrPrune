from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import numpy as np

from AgentAttrPrune.prompt import MMLUPromptSet


MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLUSplit = Literal["dev", "val", "test"]


def _data_root(path: Path) -> Path:
    if (path / "data").is_dir():
        return path / "data"
    return path


def download_mmlu(data_dir: str | Path, *, url: str = MMLU_URL) -> Path:
    """Download and safely unpack the original MMLU archive."""
    import requests

    destination = Path(data_dir).expanduser().resolve()
    root = _data_root(destination)
    if all((root / split).is_dir() for split in ("dev", "val", "test")):
        return root

    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:*") as archive:
        destination_resolved = destination.resolve()
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination_resolved not in member_path.parents and (
                member_path != destination_resolved
            ):
                raise ValueError(f"Unsafe path in MMLU archive: {member.name}")
        archive.extractall(destination)
    root = _data_root(destination)
    if not (root / "dev").is_dir():
        raise FileNotFoundError("Downloaded archive did not contain MMLU data/dev")
    return root


class MMLUDataset:
    """Deterministic, dependency-light reader for the original MMLU CSVs."""

    def __init__(
        self,
        split: MMLUSplit = "dev",
        data_dir: str | Path = "data/MMLU",
        *,
        shuffle_seed: int | None = 888,
        download: bool = False,
    ) -> None:
        if split not in {"dev", "val", "test"}:
            raise ValueError("split must be one of: dev, val, test")
        self.split = split
        self.data_dir = Path(data_dir).expanduser()
        root = _data_root(self.data_dir)
        split_dir = root / split
        if not split_dir.is_dir() and download:
            root = download_mmlu(self.data_dir)
            split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"MMLU split directory not found: {split_dir}. "
                "Pass download=True or run the dataset download command."
            )

        records: list[dict[str, str]] = []
        for csv_path in sorted(split_dir.glob("*.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    if len(row) < 6:
                        raise ValueError(
                            f"Expected 6 MMLU columns in {csv_path}, got {len(row)}"
                        )
                    records.append(
                        {
                            "question": row[0],
                            "A": row[1],
                            "B": row[2],
                            "C": row[3],
                            "D": row[4],
                            "correct_answer": row[5],
                            "subject": csv_path.stem.removesuffix(f"_{split}"),
                        }
                    )
        if not records:
            raise FileNotFoundError(f"No CSV files found under {split_dir}")
        if shuffle_seed is not None:
            permutation = np.random.default_rng(shuffle_seed).permutation(len(records))
            records = [records[int(index)] for index in permutation]
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self._records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return dict(self._records[index])

    @staticmethod
    def record_to_input(record: Mapping[str, Any]) -> dict[str, str]:
        return {
            "task": (
                f"{record['question']}\n"
                f"Option A: {record['A']}\n"
                f"Option B: {record['B']}\n"
                f"Option C: {record['C']}\n"
                f"Option D: {record['D']}\n"
            )
        }

    @classmethod
    def record_to_sample(cls, record: Mapping[str, Any]) -> dict[str, Any]:
        sample: dict[str, Any] = cls.record_to_input(record)
        sample.update(
            {
                "question": str(record["question"]),
                "options": {
                    label: str(record[label]) for label in ("A", "B", "C", "D")
                },
                "answer": cls.record_to_target_answer(record),
                "subject": str(record.get("subject", "")),
            }
        )
        return sample

    @staticmethod
    def record_to_target_answer(record: Mapping[str, Any]) -> str:
        return str(record["correct_answer"]).strip().upper()

    @staticmethod
    def normalize_answer(answer: Any) -> str:
        return MMLUPromptSet.normalize_answer(answer)

    def postprocess_answer(self, answer: Any) -> str:
        return self.normalize_answer(answer)
