"""Exact original AgentAttrPrune seed0 train40/prompt contract for GSM8K baselines.

This is the cohort whose original global 70% evaluation achieved 1241/1319.
Baseline training budgets and graph algorithms remain the responsibility of each
baseline. No model is loaded and no dataset is downloaded by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from AgentAttrPrune.datasets import GSM8KDataset
from AgentAttrPrune.prompt import GSM8KPromptSet
from AgentAttrPrune.prompt.gsm8k import WORKER_ROLES
from experiments_attrprune.common import (
    DEFAULT_RESULTS_ROOT, benchmark_data_identity, config_fingerprint,
    dataset_identities_match, load_json, reviewer_agent_ids_from_config,
    validate_prompt_contract,
)

DEFAULT_REFERENCE = DEFAULT_RESULTS_ROOT / "gsm8k/qwen3_8b/train40_correctness_aware"
CALIBRATION_SIZE = 40
EVALUATION_SIZE = 1319


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify_sources(alignment: dict) -> None:
    for name, expected in alignment["source_hashes"].items():
        if file_sha(name) != expected:
            raise ValueError(f"Frozen GSM8K alignment source changed: {name}")


def _validate_trace(trace: dict, config: dict, index: int, aliases: dict) -> None:
    if (trace.get("generation_complete") is not True or trace.get("token_ids_complete") is not True
            or trace.get("truncated_event_ids")):
        raise ValueError(f"Incomplete reference calibration trace: {index}")
    for key, value in config.items():
        if trace.get("config", {}).get(key) != value:
            raise ValueError(f"Reference trace configuration differs: sample {index}, {key}")
    if trace["config"].get("sample_index") != index:
        raise ValueError("Reference trace sample identity changed")
    workers = config["workers"]
    roles = {w["agent_id"]: w["role"] for w in workers}
    expected = [(r, w["agent_id"]) for r in range(config["num_rounds"]) for w in workers]
    expected.append((config["num_rounds"], "dense_reviewer"))
    events = trace.get("events", [])
    if [(e.get("round"), e.get("agent_id")) for e in events] != expected:
        raise ValueError("Reference trace does not have the complete fixed-order nodes")
    for event in events:
        if event.get("hit_max_new_tokens"):
            raise ValueError("Truncated reference output cannot define the aligned experiment")
        incoming = event["incoming"]
        if event["agent_id"] == "dense_reviewer":
            sources = [{**item, "reviewer_agent_id": aliases[item["agent_id"]]} for item in incoming]
            messages = GSM8KPromptSet.reviewer_messages(task=trace["sample"]["task"], worker_outputs=sources)
        else:
            messages = GSM8KPromptSet.worker_messages(
                role=roles[event["agent_id"]], task=trace["sample"]["task"],
                spatial_incoming=[i for i in incoming if i["edge_type"] == "spatial"],
                temporal_incoming=[i for i in incoming if i["edge_type"] == "temporal"],
            )
        if messages != event["input_messages"]:
            raise ValueError(f"Current GSM8K prompt differs from saved reference: sample {index}, node {event['agent_id']}")


def load_alignment(reference_run: str | Path = DEFAULT_REFERENCE) -> dict:
    """Recover all 40 original questions, retaining final-incorrect examples too."""
    reference = Path(reference_run).expanduser().resolve()
    config_path = reference / "run_config.json"
    selection_path = reference / "calibration/selection_manifest.json"
    config, selection = load_json(config_path), load_json(selection_path)
    validate_prompt_contract(config, benchmark="gsm8k")
    if (config["fingerprint"] != config_fingerprint(config) or config["benchmark"] != "gsm8k"
            or config["seed"] != 0 or config["num_rounds"] != 2
            or config["splits"] != {"calibration": "train", "evaluation": "test"}
            or config["decoding"] != {"temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False}
            or [(w["agent_id"], w["role"]) for w in config["workers"]] != list(WORKER_ROLES)
            or [w["node_index"] for w in config["workers"]] != list(range(4))):
        raise ValueError("Expected the original seed0/four-worker/two-round GSM8K reference")
    selected = selection["samples"]
    if (selection.get("expected_sample_count") != CALIBRATION_SIZE
            or selection.get("expected_start_index") != 0 or len(selected) != CALIBRATION_SIZE
            or [s["sample_index"] for s in selected] != list(range(CALIBRATION_SIZE))
            or [s["sample_id"] for s in selected] != [f"sample_{i:05d}" for i in range(CALIBRATION_SIZE)]):
        raise ValueError("Reference selection must be the complete original train40")
    identity = config["dataset"]
    data_dir = Path(identity.get("requested_data_dir") or identity["data_root"])
    actual = benchmark_data_identity("gsm8k", data_dir, splits=("train", "test"), shuffle_seed=identity["shuffle_seed"])
    if not dataset_identities_match(identity, actual):
        raise ValueError("GSM8K dataset differs from the original AgentAttrPrune experiment")
    datasets = {split: GSM8KDataset(split=split, data_dir=data_dir, shuffle_seed=identity["shuffle_seed"], download=False)
                for split in ("train", "test")}
    if len(datasets["test"]) != EVALUATION_SIZE:
        raise ValueError("Expected the complete 1,319-question GSM8K test evaluation")
    aliases = reviewer_agent_ids_from_config(config)
    paths = [config_path, selection_path]
    calibration = []
    for record in selected:
        index = record["sample_index"]
        path = Path(record["trace_json"]).resolve()
        if path != (reference / "calibration/traces" / record["sample_id"] / "trace.json").resolve():
            raise ValueError("Reference selection points outside its expected trace directory")
        if file_sha(path) != record["sha256"]:
            raise ValueError(f"Reference calibration trace SHA changed: {index}")
        trace = load_json(path)
        _validate_trace(trace, config, index, aliases)
        expected_sample = datasets["train"].record_to_sample(datasets["train"][index])
        if trace["sample"] != expected_sample:
            raise ValueError("Reference trace question/answer differs from the frozen dataset ordering")
        calibration.append(trace["sample"])
        paths.append(path)
    if len({s["task"] for s in calibration}) != CALIBRATION_SIZE:
        raise ValueError("Calibration question pool contains duplicate tasks")
    evaluation = [datasets["test"].record_to_sample(row) for row in datasets["test"]]
    raw_indices = np.arange(len(datasets["train"])) if identity["shuffle_seed"] is None else (
        np.random.default_rng(identity["shuffle_seed"]).permutation(len(datasets["train"])))
    paths.extend(Path(actual["data_root"]) / f"{split}.jsonl" for split in ("train", "test"))
    repository = Path(__file__).resolve().parents[1]
    paths.extend(repository / name for name in (
        "AgentAttrPrune/prompt/gsm8k.py", "AgentAttrPrune/prompt/reviewer_ids.py", "AgentAttrPrune/datasets/gsm8k.py"))
    result = {
        "schema_version": 1, "method_reference": "AgentAttrPrune", "benchmark": "gsm8k",
        "reference_run": str(reference), "reference_run_config": config,
        "workers": config["workers"], "reviewer_agent_ids": aliases,
        "prompt_version": GSM8KPromptSet.VERSION,
        "calibration_selection": "exact original AgentAttrPrune seed0 train40, including final-incorrect examples",
        "calibration_indices": list(range(CALIBRATION_SIZE)),
        "calibration_source_indices": [int(i) for i in raw_indices[:CALIBRATION_SIZE]],
        "calibration_samples": calibration,
        "evaluation_indices": list(range(EVALUATION_SIZE)), "evaluation_samples": evaluation,
        "calibration_samples_sha256": digest(calibration), "evaluation_samples_sha256": digest(evaluation),
        "source_hashes": {str(p.resolve()): file_sha(p) for p in paths},
    }
    result["fingerprint"] = digest(result)
    return result


def freeze_alignment(run_dir: str | Path, alignment: dict) -> Path:
    from experiments_attrprune.random_order_mmlu import freeze_json
    verify_sources(alignment)
    path = Path(run_dir) / "prompt_data_alignment.json"
    freeze_json(path, alignment)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    alignment = load_alignment(parser.parse_args().reference_run)
    print(json.dumps({"calibration_questions": len(alignment["calibration_samples"]),
        "evaluation_questions": len(alignment["evaluation_samples"]),
        **{k: v for k, v in alignment.items() if k not in {
            "calibration_samples", "evaluation_samples", "source_hashes", "reference_run_config",
            "calibration_indices", "evaluation_indices"}}}, indent=2))


if __name__ == "__main__":
    main()
