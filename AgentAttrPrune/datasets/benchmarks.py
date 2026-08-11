from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from AgentAttrPrune.prompt import (
    GSM8KPromptSet,
    GSM8K_WORKER_ROLES,
    HUMANEVAL_WORKER_ROLES,
    HumanEvalPromptSet,
    MMLUPromptSet,
    MMLU_WORKER_ROLES,
)

from .gsm8k import GSM8KDataset
from .humaneval import HumanEvalDataset, evaluate_humaneval_prediction
from .mmlu import MMLUDataset


BENCHMARKS = ("mmlu", "gsm8k", "humaneval")


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    dataset_class: type
    prompt_set: type
    worker_roles: tuple[tuple[str, str], ...]
    calibration_split: str
    evaluation_split: str
    calibration_limit: int
    evaluation_limit: int
    default_num_workers: int


DEFINITIONS = {
    "mmlu": BenchmarkDefinition(
        "mmlu",
        MMLUDataset,
        MMLUPromptSet,
        MMLU_WORKER_ROLES,
        "dev",
        "val",
        285,
        1531,
        5,
    ),
    "gsm8k": BenchmarkDefinition(
        "gsm8k",
        GSM8KDataset,
        GSM8KPromptSet,
        GSM8K_WORKER_ROLES,
        "train",
        "test",
        7473,
        1319,
        4,
    ),
    "humaneval": BenchmarkDefinition(
        "humaneval",
        HumanEvalDataset,
        HumanEvalPromptSet,
        HUMANEVAL_WORKER_ROLES,
        "dev",
        "test",
        80,
        81,
        5,
    ),
}


def benchmark_definition(name: str) -> BenchmarkDefinition:
    key = str(name).strip().lower()
    try:
        return DEFINITIONS[key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported benchmark {name!r}; choose from {', '.join(BENCHMARKS)}"
        ) from error


def create_dataset(
    benchmark: str,
    *,
    split: str,
    data_dir: str | Path,
    download: bool,
    shuffle_seed: int | None = 888,
):
    definition = benchmark_definition(benchmark)
    return definition.dataset_class(
        split=split,
        data_dir=data_dir,
        download=download,
        shuffle_seed=shuffle_seed,
    )


def evaluate_prediction(
    benchmark: str,
    output: Any,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    definition = benchmark_definition(benchmark)
    if definition.name == "humaneval":
        return evaluate_humaneval_prediction(output, sample)
    parsed = definition.prompt_set.normalize_answer(output)
    gold = definition.prompt_set.normalize_answer(sample["answer"])
    return {
        "parsed_answer": parsed,
        "gold_answer": gold,
        "correct": parsed == gold,
        "details": {},
    }


def _hash_files(root: Path, paths: Iterable[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    record_count = 0
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                record_count += sum(1 for _ in csv.reader(handle))
        else:
            with path.open("r", encoding="utf-8") as handle:
                record_count += sum(1 for line in handle if line.strip())
    return digest.hexdigest(), record_count


def dataset_identity(
    benchmark: str,
    data_dir: str | Path,
    *,
    splits: Iterable[str],
    shuffle_seed: int = 888,
) -> dict[str, Any]:
    """Fingerprint the exact files and logical selections used by a run."""

    definition = benchmark_definition(benchmark)
    requested = Path(data_dir).expanduser().resolve()
    root = (
        requested / "data"
        if definition.name == "mmlu" and (requested / "data").is_dir()
        else requested
    )
    split_payload: dict[str, Any] = {}
    for split in sorted({str(value) for value in splits}):
        if definition.name == "mmlu":
            paths = sorted((root / split).glob("*.csv"))
        elif definition.name == "gsm8k":
            paths = [root / f"{split}.jsonl"]
        else:
            paths = [root / "humaneval-py.jsonl"]
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError(
                f"No {definition.name} data files found for split {split!r} "
                f"under {root}"
            )
        digest, source_count = _hash_files(root, paths)
        dataset = create_dataset(
            definition.name,
            split=split,
            data_dir=requested,
            download=False,
            shuffle_seed=shuffle_seed,
        )
        selected_ids: list[str] = []
        if definition.name == "humaneval":
            selected_ids = [str(record["name"]) for record in dataset]
        split_payload[split] = {
            "file_count": len(paths),
            "record_count": len(dataset),
            "source_record_count": source_count,
            "sha256": digest,
        }
        if selected_ids:
            split_payload[split]["selection_sha256"] = hashlib.sha256(
                json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
    payload: dict[str, Any] = {
        "benchmark": definition.name,
        "requested_data_dir": str(requested),
        "data_root": str(root.resolve()),
        "shuffle_seed": int(shuffle_seed),
        "splits": split_payload,
    }
    if definition.name == "humaneval":
        payload["partition"] = {
            "strategy": "seeded_disjoint_half",
            "partition_seed": 888,
        }
    return payload
