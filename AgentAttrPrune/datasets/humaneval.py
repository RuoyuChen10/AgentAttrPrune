from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Literal, Mapping

import numpy as np

from AgentAttrPrune.prompt import HumanEvalPromptSet, extract_python_code


HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/yanweiyue/AgentPrune/"
    "main/dataset/humaneval/humaneval-py.jsonl"
)
HumanEvalSplit = Literal["dev", "test", "all"]
DEFAULT_EXECUTION_TIMEOUT = 5.0


def download_humaneval(
    data_dir: str | Path,
    *,
    url: str = HUMANEVAL_URL,
) -> Path:
    """Download the Python HumanEval variant used by the AgentPrune repository."""

    import requests

    root = Path(data_dir).expanduser().resolve()
    destination = root / "humaneval-py.jsonl"
    if destination.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    return root


def _stable_partition(
    records: list[dict[str, Any]],
    *,
    split: HumanEvalSplit,
    partition_seed: int,
) -> list[dict[str, Any]]:
    if split == "all":
        return records
    order = np.random.default_rng(partition_seed).permutation(len(records))
    midpoint = len(order) // 2
    selected = order[:midpoint] if split == "dev" else order[midpoint:]
    return [records[int(index)] for index in selected]


class HumanEvalDataset:
    """Reader for AgentPrune's HumanEval-Python file.

    The upstream file has no calibration split. For leakage-free attribution
    experiments, ``dev`` and ``test`` are a fixed, disjoint 50/50 partition
    controlled by ``partition_seed``.
    """

    def __init__(
        self,
        split: HumanEvalSplit = "dev",
        data_dir: str | Path = "data/HumanEval",
        *,
        shuffle_seed: int | None = 888,
        partition_seed: int = 888,
        download: bool = False,
    ) -> None:
        if split not in {"dev", "test", "all"}:
            raise ValueError("split must be one of: dev, test, all")
        self.split = split
        self.data_dir = Path(data_dir).expanduser()
        self.partition_seed = int(partition_seed)
        path = self.data_dir / "humaneval-py.jsonl"
        if not path.is_file() and download:
            download_humaneval(self.data_dir)
        if not path.is_file():
            raise FileNotFoundError(
                f"HumanEval file not found: {path}. "
                "Pass download=True or run the dataset download command."
            )

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                required = {"name", "prompt", "entry_point", "test"}
                if not isinstance(value, dict) or not required.issubset(value):
                    raise ValueError(
                        f"Invalid HumanEval record at {path}:{line_number}"
                    )
                records.append(dict(value))
        if not records:
            raise FileNotFoundError(f"No HumanEval records found in {path}")
        records = _stable_partition(
            records,
            split=split,
            partition_seed=self.partition_seed,
        )
        if shuffle_seed is not None:
            permutation = np.random.default_rng(shuffle_seed).permutation(len(records))
            records = [records[int(index)] for index in permutation]
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._records[index])

    @staticmethod
    def record_to_input(record: Mapping[str, Any]) -> dict[str, str]:
        return {"task": str(record["prompt"])}

    @classmethod
    def record_to_sample(cls, record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **cls.record_to_input(record),
            "benchmark": "humaneval",
            "task_id": str(record["name"]),
            "entry_point": str(record["entry_point"]),
            "test": str(record["test"]),
            "answer": "pass",
        }

    @staticmethod
    def record_to_target_answer(record: Mapping[str, Any]) -> str:
        del record
        return "pass"

    @staticmethod
    def normalize_answer(answer: Any) -> str:
        return HumanEvalPromptSet.normalize_answer(answer)

    def postprocess_answer(self, answer: Any) -> str:
        return self.normalize_answer(answer)


def _limit_child_resources(timeout: float) -> None:
    cpu_seconds = max(1, int(math.ceil(timeout)))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.setsid()


def evaluate_humaneval_prediction(
    output: Any,
    sample: Mapping[str, Any],
    *,
    timeout: float = DEFAULT_EXECUTION_TIMEOUT,
) -> dict[str, Any]:
    """Run one generated implementation and its tests in a limited subprocess.

    This limits time, memory, output-file size, and open files. It is not a
    security boundary against deliberately malicious model output.
    """

    code = extract_python_code(output)
    entry_point = str(sample["entry_point"])
    prompt = str(sample["task"])
    test = str(sample["test"])
    if not code:
        return {
            "parsed_answer": "fail",
            "gold_answer": "pass",
            "correct": False,
            "details": {"status": "empty_code"},
        }
    if f"def {entry_point}" not in code:
        code = prompt + code
    program = f"{code.rstrip()}\n\n{test.rstrip()}\n"
    program_sha256 = hashlib.sha256(program.encode("utf-8")).hexdigest()

    with tempfile.TemporaryDirectory(prefix="agentattrprune-humaneval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(path)],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                preexec_fn=lambda: _limit_child_resources(timeout),
            )
        except subprocess.TimeoutExpired as error:
            return {
                "parsed_answer": "fail",
                "gold_answer": "pass",
                "correct": False,
                "details": {
                    "status": "timeout",
                    "timeout_seconds": timeout,
                    "program_sha256": program_sha256,
                    "stdout": (error.stdout or "")[-4000:],
                    "stderr": (error.stderr or "")[-4000:],
                },
            }
    passed = completed.returncode == 0
    return {
        "parsed_answer": "pass" if passed else "fail",
        "gold_answer": "pass",
        "correct": passed,
        "details": {
            "status": "passed" if passed else "failed",
            "returncode": completed.returncode,
            "program_sha256": program_sha256,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        },
    }
