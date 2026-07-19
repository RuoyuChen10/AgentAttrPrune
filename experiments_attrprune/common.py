"""Shared configuration and result-path helpers for MMLU experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/egr/research-optml/ruoyu.chen/Models/Qwen3-8B")
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "AgentAttrPrune"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "MMLU"

WORKER_ROLES = (
    ("knowledgeable_expert", "Knowlegable Expert"),
    ("critic", "Critic"),
    ("mathematician", "Mathematician"),
    ("psychologist", "Psychologist"),
    ("historian", "Historian"),
    ("doctor", "Doctor"),
    ("lawyer", "Lawyer"),
    ("economist", "Economist"),
    ("programmer", "Programmer"),
)

ANSWER_CHOICES = frozenset("ABCD")


def mmlu_data_identity(
    data_dir: str | Path,
    *,
    splits: Iterable[str] = ("dev", "val"),
    shuffle_seed: int = 888,
) -> dict[str, Any]:
    """Fingerprint the exact MMLU CSV selection used by a formal run."""

    requested = Path(data_dir).expanduser().resolve()
    root = requested / "data" if (requested / "data").is_dir() else requested
    split_payload: dict[str, Any] = {}
    for split in sorted({str(value) for value in splits}):
        split_dir = root / split
        csv_paths = sorted(split_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No MMLU CSV files found under {split_dir}")
        digest = hashlib.sha256()
        record_count = 0
        for path in csv_paths:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            with path.open("r", encoding="utf-8", newline="") as handle:
                record_count += sum(1 for _ in csv.reader(handle))
        split_payload[split] = {
            "file_count": len(csv_paths),
            "record_count": record_count,
            "sha256": digest.hexdigest(),
        }
    return {
        "requested_data_dir": str(requested),
        "data_root": str(root.resolve()),
        "shuffle_seed": int(shuffle_seed),
        "splits": split_payload,
    }


@dataclass(frozen=True)
class ExperimentPaths:
    run_dir: Path

    @property
    def run_config(self) -> Path:
        return self.run_dir / "run_config.json"

    @property
    def calibration(self) -> Path:
        return self.run_dir / "calibration"

    @property
    def traces(self) -> Path:
        return self.calibration / "traces"

    @property
    def attribution(self) -> Path:
        return self.run_dir / "attribution"

    @property
    def evaluation(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def comparison(self) -> Path:
        return self.run_dir / "comparison_summary.json"


def model_slug(model_name: str | Path) -> str:
    value = str(model_name).rstrip("/").split("/")[-1] or "model"
    # Keep result-directory names stable across common model spellings such as
    # ``Qwen3-8B`` and the documented ``qwen3_8b`` run name.
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value.lower() or "model"


def resolve_run_dir(
    *,
    model_name: str | Path,
    seed: int,
    results_root: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    root = Path(results_root or os.environ.get("RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    if run_name:
        return root / run_name
    return root / "mmlu" / model_slug(model_name) / f"seed_{seed}"


def worker_metadata(num_workers: int) -> list[dict[str, Any]]:
    if not 1 <= num_workers <= len(WORKER_ROLES):
        raise ValueError(f"num_workers must be between 1 and {len(WORKER_ROLES)}")
    return [
        {
            "agent_id": agent_id,
            "role": role,
            "agent_type": "AnalyzeAgent",
            "node_index": index,
        }
        for index, (agent_id, role) in enumerate(WORKER_ROLES[:num_workers])
    ]


def canonical_model_name(value: str | Path) -> str:
    value = str(value)
    for prefix in ("local:", "hf:"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def hf_model_reference(value: str | Path) -> str:
    value = str(value)
    if value.startswith(("local:", "hf:")):
        return value
    return f"local:{value}"


def build_run_config(
    *,
    model_name: str | Path,
    num_workers: int,
    num_rounds: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    calibration_split: str = "dev",
    evaluation_split: str = "val",
    dataset_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    config: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "mmlu",
        "model_name": canonical_model_name(model_name),
        "workers": worker_metadata(num_workers),
        "num_rounds": num_rounds,
        "decoding": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": False,
        },
        "seed": seed,
        "splits": {
            "calibration": calibration_split,
            "evaluation": evaluation_split,
        },
        "topology": {
            "matrix_orientation": "[round][source][target]",
            "worker_spatial": "earlier-agent-to-later-agent DAG",
            "worker_temporal": "round>0 previous-to-current all N x N including self",
            "reviewer": "all final-round workers to dense FinalRefer",
        },
    }
    if dataset_identity is not None:
        config["dataset"] = dict(dataset_identity)
    config["fingerprint"] = config_fingerprint(config)
    return config


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "fingerprint"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_run_config(path: str | Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    expected = dict(expected)
    expected_fingerprint = expected.get("fingerprint") or config_fingerprint(expected)
    expected["fingerprint"] = expected_fingerprint
    if path.exists():
        current = load_json(path)
        current_fingerprint = current.get("fingerprint")
        actual_fingerprint = config_fingerprint(current)
        if not current_fingerprint or current_fingerprint != actual_fingerprint:
            raise ValueError(
                f"Existing run configuration at {path} has a missing or invalid "
                "fingerprint; refusing to resume."
            )
        if current_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Run configuration mismatch at {path}: "
                f"existing={current_fingerprint}, requested={expected_fingerprint}. "
                "Choose another RUN_NAME or results directory."
            )
        return current
    write_json(path, expected)
    return expected


def normalize_answer(value: Any) -> str:
    text = str(value).strip().upper()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    if first_line in ANSWER_CHOICES:
        return first_line
    patterns = (
        r"\bDECISION\s*:?\s*(?:OPTION\s+)?([ABCD])\b",
        r"\bANSWER\s+IS\s*:?\s*(?:OPTION\s+)?([ABCD])\b",
        r"\bOPTION\s+([ABCD])\b",
        r"\b([ABCD])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return text[:1]


def trace_is_correct(trace: Mapping[str, Any]) -> bool:
    final = trace.get("final") or {}
    parsed = final.get("parsed_answer", final.get("answer", final.get("output", "")))
    gold = final.get("gold_answer", (trace.get("sample") or {}).get("answer", ""))
    return normalize_answer(parsed) == normalize_answer(gold)


def completed_attempt_indices(path: str | Path) -> set[int]:
    return {
        int(item["sample_index"])
        for item in load_jsonl(path)
        if item.get("status") == "completed" and "sample_index" in item
    }


def worker_specs_from_config(config: Mapping[str, Any]):
    """Return core WorkerSpec values without importing the heavy model backend."""
    from AgentAttrPrune.agents import WorkerSpec

    return [WorkerSpec(item["agent_id"], item["role"]) for item in config["workers"]]


def count_correct(records: Iterable[Mapping[str, Any]]) -> int:
    return sum(bool(item.get("correct")) for item in records)
