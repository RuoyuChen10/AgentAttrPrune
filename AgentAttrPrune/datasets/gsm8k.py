from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import numpy as np

from AgentAttrPrune.prompt import GSM8KPromptSet


GSM8K_URLS = {
    "train": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/train.jsonl"
    ),
    "test": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/test.jsonl"
    ),
}
GSM8KSplit = Literal["train", "test"]


def _download_file(url: str, destination: Path) -> Path:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    return destination


def download_gsm8k(
    data_dir: str | Path,
    *,
    splits: tuple[GSM8KSplit, ...] = ("train", "test"),
) -> Path:
    """Download the original OpenAI GSM8K JSONL files."""

    root = Path(data_dir).expanduser().resolve()
    for split in splits:
        if split not in GSM8K_URLS:
            raise ValueError(f"Unsupported GSM8K split: {split!r}")
        path = root / f"{split}.jsonl"
        if not path.is_file():
            _download_file(GSM8K_URLS[split], path)
    return root


class GSM8KDataset:
    """Deterministic reader for the original OpenAI GSM8K train/test files."""

    def __init__(
        self,
        split: GSM8KSplit = "train",
        data_dir: str | Path = "data/GSM8K",
        *,
        shuffle_seed: int | None = 888,
        download: bool = False,
    ) -> None:
        if split not in GSM8K_URLS:
            raise ValueError("split must be one of: train, test")
        self.split = split
        self.data_dir = Path(data_dir).expanduser()
        path = self.data_dir / f"{split}.jsonl"
        if not path.is_file() and download:
            download_gsm8k(self.data_dir, splits=(split,))
        if not path.is_file():
            raise FileNotFoundError(
                f"GSM8K split file not found: {path}. "
                "Pass download=True or run the dataset download command."
            )

        records: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or not {
                    "question",
                    "answer",
                }.issubset(value):
                    raise ValueError(
                        f"Invalid GSM8K record at {path}:{line_number}"
                    )
                reasoning, answer = self._split_answer(str(value["answer"]))
                records.append(
                    {
                        "question": str(value["question"]),
                        "reasoning": reasoning,
                        "correct_answer": answer,
                    }
                )
        if not records:
            raise FileNotFoundError(f"No GSM8K records found in {path}")
        if shuffle_seed is not None:
            permutation = np.random.default_rng(shuffle_seed).permutation(len(records))
            records = [records[int(index)] for index in permutation]
        self._records = records

    @staticmethod
    def _split_answer(value: str) -> tuple[str, str]:
        if "####" not in value:
            raise ValueError("GSM8K answer is missing the '####' final-answer marker")
        reasoning, answer = value.rsplit("####", 1)
        return reasoning.strip(), GSM8KPromptSet.normalize_answer(answer)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self._records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return dict(self._records[index])

    @staticmethod
    def record_to_input(record: Mapping[str, Any]) -> dict[str, str]:
        return {"task": str(record["question"])}

    @classmethod
    def record_to_sample(cls, record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **cls.record_to_input(record),
            "benchmark": "gsm8k",
            "question": str(record["question"]),
            "answer": cls.record_to_target_answer(record),
        }

    @staticmethod
    def record_to_target_answer(record: Mapping[str, Any]) -> str:
        return GSM8KPromptSet.normalize_answer(record["correct_answer"])

    @staticmethod
    def normalize_answer(answer: Any) -> str:
        return GSM8KPromptSet.normalize_answer(answer)

    def postprocess_answer(self, answer: Any) -> str:
        return self.normalize_answer(answer)
