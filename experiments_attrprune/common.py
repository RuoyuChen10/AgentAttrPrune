"""Shared configuration and result-path helpers for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from AgentAttrPrune.datasets import (
    BENCHMARKS,
    benchmark_definition,
    dataset_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/egr/research-optml/ruoyu.chen/Models/Qwen3-8B")
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "AgentAttrPrune"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "MMLU"
DEFAULT_DATA_DIRS = {
    "mmlu": DEFAULT_DATA_DIR,
    "gsm8k": REPO_ROOT / "data" / "GSM8K",
    "humaneval": REPO_ROOT / "data" / "HumanEval",
}

WORKER_ROLES = benchmark_definition("mmlu").worker_roles

ANSWER_CHOICES = frozenset("ABCD")


def mmlu_data_identity(
    data_dir: str | Path,
    *,
    splits: Iterable[str] = ("dev", "val"),
    shuffle_seed: int = 888,
) -> dict[str, Any]:
    """Fingerprint the exact MMLU CSV selection used by a formal run."""

    return dataset_identity(
        "mmlu",
        data_dir,
        splits=splits,
        shuffle_seed=shuffle_seed,
    )


def benchmark_data_identity(
    benchmark: str,
    data_dir: str | Path,
    *,
    splits: Iterable[str],
    shuffle_seed: int = 888,
) -> dict[str, Any]:
    return dataset_identity(
        benchmark,
        data_dir,
        splits=splits,
        shuffle_seed=shuffle_seed,
    )


def dataset_identities_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> bool:
    """Compare a stored identity with a newly computed compatible superset.

    Dataset identity schemas may gain descriptive fields over time. A formal
    run created by an older schema remains valid when every field it recorded
    still matches exactly and the new identity only adds fields. Lists retain
    exact length/order semantics; scalar values, paths, counts, hashes, and
    shuffle seeds must remain equal.
    """

    def matches(stored: Any, computed: Any) -> bool:
        if isinstance(stored, Mapping):
            return isinstance(computed, Mapping) and all(
                key in computed and matches(value, computed[key])
                for key, value in stored.items()
            )
        if isinstance(stored, (list, tuple)):
            return (
                isinstance(computed, (list, tuple))
                and len(stored) == len(computed)
                and all(matches(left, right) for left, right in zip(stored, computed))
            )
        return stored == computed

    return matches(expected, actual)


def default_data_dir(benchmark: str) -> Path:
    try:
        return DEFAULT_DATA_DIRS[str(benchmark).lower()]
    except KeyError as error:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}; choose from {', '.join(BENCHMARKS)}"
        ) from error


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
    benchmark: str = "mmlu",
) -> Path:
    root = Path(results_root or os.environ.get("RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    if run_name:
        return root / run_name
    name = benchmark_definition(benchmark).name
    return root / name / model_slug(model_name) / f"seed_{seed}"


def worker_metadata(
    num_workers: int,
    *,
    benchmark: str = "mmlu",
) -> list[dict[str, Any]]:
    roles = benchmark_definition(benchmark).worker_roles
    if not 1 <= num_workers <= len(roles):
        raise ValueError(f"num_workers must be between 1 and {len(roles)}")
    return [
        {
            "agent_id": agent_id,
            "role": role,
            "agent_type": "AnalyzeAgent",
            "node_index": index,
        }
        for index, (agent_id, role) in enumerate(roles[:num_workers])
    ]


def prompt_contract(benchmark: str) -> dict[str, str] | None:
    """Return the versioned prompt identity that must match formal runs."""

    prompt_set = benchmark_definition(benchmark).prompt_set
    version = getattr(prompt_set, "VERSION", None)
    if version is None:
        return None
    return {
        "prompt_set": prompt_set.__name__,
        "version": str(version),
    }


def validate_prompt_contract(
    config: Mapping[str, Any],
    *,
    benchmark: str | None = None,
) -> None:
    """Reject generation with a run created under another prompt version."""

    benchmark_name = str(benchmark or config.get("benchmark", "mmlu")).lower()
    expected = prompt_contract(benchmark_name)
    if expected is None:
        return
    actual = config.get("prompt")
    if not isinstance(actual, Mapping):
        raise ValueError(
            "Run prompt contract does not match the current implementation: "
            f"stored={actual!r}, current={expected!r}. Choose a new RUN_NAME "
            "and rerun calibration before evaluation."
        )
    workers = list(config.get("workers") or [])
    prepare_reviewer_ids = getattr(
        benchmark_definition(benchmark_name).prompt_set,
        "prepare_reviewer_agent_ids",
        None,
    )
    expected_full: dict[str, Any] = dict(expected)
    if prepare_reviewer_ids is not None:
        raw_aliases = actual.get("reviewer_agent_ids")
        if not isinstance(raw_aliases, Mapping):
            raise ValueError(
                "Run prompt contract has no saved reviewer_agent_ids mapping. "
                "Choose a new RUN_NAME and rerun calibration before evaluation."
            )
        worker_ids = [str(worker["agent_id"]) for worker in workers]
        try:
            aliases = prepare_reviewer_ids(worker_ids, raw_aliases)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Run prompt contract has invalid reviewer_agent_ids: {error}"
            ) from error
        expected_full["reviewer_agent_ids"] = aliases
    if dict(actual) != expected_full:
        raise ValueError(
            "Run prompt contract does not match the current implementation: "
            f"stored={dict(actual)!r}, current={expected_full!r}. Choose a new "
            "RUN_NAME and rerun calibration before evaluation."
        )


def reviewer_agent_ids_from_config(config: Mapping[str, Any]) -> dict[str, str]:
    """Load the validated runtime IDs shown to a versioned final reviewer."""

    validate_prompt_contract(config)
    prompt = config.get("prompt")
    if not isinstance(prompt, Mapping):
        return {}
    aliases = prompt.get("reviewer_agent_ids")
    if not isinstance(aliases, Mapping):
        return {}
    return {str(agent_id): str(alias) for agent_id, alias in aliases.items()}


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
    benchmark: str = "mmlu",
    reviewer_agent_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    definition = benchmark_definition(benchmark)
    benchmark = definition.name
    workers = worker_metadata(num_workers, benchmark=benchmark)
    config: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": benchmark,
        "model_name": canonical_model_name(model_name),
        "workers": workers,
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
    configured_prompt = prompt_contract(benchmark)
    if configured_prompt is not None:
        prepare_reviewer_ids = getattr(
            definition.prompt_set, "prepare_reviewer_agent_ids", None
        )
        if prepare_reviewer_ids is not None:
            configured_prompt["reviewer_agent_ids"] = prepare_reviewer_ids(
                [str(worker["agent_id"]) for worker in workers],
                reviewer_agent_ids,
            )
        elif reviewer_agent_ids is not None:
            raise ValueError(
                f"{definition.prompt_set.__name__} does not support reviewer_agent_ids"
            )
        config["prompt"] = configured_prompt
    elif reviewer_agent_ids is not None:
        raise ValueError(
            f"{definition.prompt_set.__name__} does not support reviewer_agent_ids"
        )
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
    if "correct" in final:
        return bool(final["correct"])
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
