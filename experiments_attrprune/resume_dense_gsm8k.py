"""Resume the original seed0 Dense test with an audited, OOM-only memory fix."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import threading

from experiments_attrprune import evaluate_mmlu as original
from experiments_attrprune import qwen3_chunked_prefill as recovery
from experiments_attrprune.common import ExperimentPaths

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware"
NAME = "dense_seed0_aligned_batch4"
LAUNCH = ROOT / "logs/gsm8k_seed0_dense/launch_20261105T071030Z.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def evaluation_args():
    return original.parse_args([
        "--benchmark", "gsm8k", "--model-name", str(ROOT.parent.parent / "Models/Qwen3-8B"),
        "--data-dir", str(ROOT / "data/GSM8K"), "--run-dir", str(RUN),
        "--mode", "dense", "--evaluation-name", NAME, "--split", "test",
        "--start-index", "0", "--limit", "1319", "--eval-batch-size", "4", "--seed", "0",
        "--resume", "--save-traces", "--no-download",
    ])


def validate():
    args = evaluation_args()
    paths = ExperimentPaths(RUN)
    config = original._load_run_config(paths, args)
    if config["fingerprint"] != "dd1386dc95e08ff0":
        raise ValueError("Expected the original GSM8K seed0 configuration")
    launch = json.loads(LAUNCH.read_text())
    for collection in ("source_sha256", "protected_prior_files_sha256"):
        for name, expected in launch[collection].items():
            if sha(name) != expected:
                raise ValueError(f"Original source or prior result changed: {name}")
    directory = paths.evaluation / f"{NAME}_test"
    policy, _ = original._resolve_policy("dense", paths=paths, config=config)
    expected = original._evaluation_state(
        config=config, policy=policy, mode="dense", evaluation_name=NAME,
        evaluation_directory=directory.name, benchmark="gsm8k", split="test",
        start=0, stop=1319, dataset_size=1319, seed=0, save_traces=True, eval_batch_size=4,
    )
    if json.loads((directory / "evaluation_state.json").read_text()) != expected:
        raise ValueError("Frozen Dense evaluation state changed")
    if json.loads((directory / "resolved_policy.json").read_text()) != policy.to_dict():
        raise ValueError("Dense policy changed")
    records = original._load_prediction_checkpoints(directory / "predictions.jsonl", start=0, stop=1319)
    if not set(range(704)).issubset(records):
        raise ValueError("The original 704 completed predictions must all be present")
    for record in records.values():
        if record["edge_counts"]["worker_active"] != 28 or record["edge_counts"]["reviewer_dense_edges"] != 4:
            raise ValueError("Saved prediction is not the complete Dense graph")
        path = Path(record["trace_json"])
        if not path.is_file():
            raise ValueError(f"Missing completed trace: {path}")
    return args, directory, records, launch


def snapshot(directory, records, launch):
    from transformers.models.qwen3 import modeling_qwen3
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    audit = directory / "recovery" / stamp
    audit.mkdir(parents=True, exist_ok=False)
    for name in ("predictions.jsonl", "evaluation_state.json", "resolved_policy.json"):
        shutil.copy2(directory / name, audit / name)
    shutil.copy2(RUN / "run_config.json", audit / "run_config.json")
    protected = {**launch["source_sha256"], **launch["protected_prior_files_sha256"]}
    protected.update({str(directory / name): sha(directory / name)
                      for name in ("evaluation_state.json", "resolved_policy.json")})
    protected.update({record["trace_json"]: sha(record["trace_json"]) for record in records.values()})
    runtime = (Path(__file__).resolve(), Path(recovery.__file__).resolve(),
               Path(modeling_qwen3.__file__).resolve())
    for path in runtime:
        shutil.copy2(path, audit / path.name)
    prediction = directory / "predictions.jsonl"
    manifest = {
        "created_utc": utc(), "host": os.uname().nodename,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "evaluation_directory": str(directory), "completed_before": len(records),
        "completed_sample_indices": sorted(records), "total": 1319,
        "protected_files_sha256": protected,
        "predictions_prefix_bytes": prediction.stat().st_size,
        "predictions_prefix_sha256": sha(prediction),
        "runtime_source_hashes": {str(path): sha(path) for path in runtime},
        "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
        "change": "After greedy CUDA OOM, retry same model.generate using Qwen3 eager query-row blocks",
        "query_chunk_size": 256, "min_query_length": 1024,
        "numerics": "Same full-context attention equations; changed GEMM shapes may affect floating-point rounding",
        "unchanged": {"temperature": 0, "max_new_tokens": 2048, "eval_batch_size": 4,
                      "seed": 0, "worker_edges": 28, "reviewer_edges": 4, "num_rounds": 2},
    }
    save_json(audit / "manifest.json", manifest)
    return audit, manifest


def verify_preserved(directory, manifest):
    for name, expected in {**manifest["protected_files_sha256"], **manifest["runtime_source_hashes"]}.items():
        if sha(name) != expected:
            raise ValueError(f"Protected source, configuration, or completed trace changed: {name}")
    with (directory / "predictions.jsonl").open("rb") as handle:
        prefix = handle.read(manifest["predictions_prefix_bytes"])
    if hashlib.sha256(prefix).hexdigest() != manifest["predictions_prefix_sha256"]:
        raise ValueError("Previously completed predictions changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    args, directory, records, launch = validate()
    if options.dry_run:
        print(json.dumps({"valid": True, "completed": len(records), "total": 1319,
                          "evaluation_directory": str(directory), "eval_batch_size": 4,
                          "recovery": "OOM-only query-row chunking; full key/value context preserved"}, indent=2))
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("Select the original GPU pair explicitly")
    with (directory / ".recovery.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args, directory, records, launch = validate()
        if len(records) == 1319:
            print(json.dumps(asyncio.run(original.evaluate(args)), indent=2))
            return
        audit, manifest = snapshot(directory, records, launch)
        lock = threading.Lock()
        counts = {}

        def emit(event):
            with lock:
                kind = event["event"]
                counts[kind] = counts.get(kind, 0) + 1
                with (audit / "runtime_events.jsonl").open("a") as stream:
                    stream.write(json.dumps({"time_utc": utc(), **event}) + "\n")
                if kind != "chunked_attention":
                    print("Dense OOM recovery: " + json.dumps(event), flush=True)

        save_json(audit / "status.json", {"status": "running", "updated_utc": utc(),
                                          "completed_before": len(records), "pid": os.getpid()})
        print(f"Preserved {len(records)}/1319 completed Dense samples. Recovery audit: {audit}", flush=True)
        error_text = None
        try:
            recovery.install_recovery(emit)
            result = asyncio.run(original.evaluate(args))
        except BaseException as error:
            error_text = repr(error)
            raise
        finally:
            try:
                verify_preserved(directory, manifest)
                completed = len(original._load_prediction_checkpoints(directory / "predictions.jsonl", start=0, stop=1319))
            except BaseException as error:
                save_json(audit / "status.json", {"status": "failed", "updated_utc": utc(),
                                                  "error": repr(error), "execution_error": error_text})
                raise
            save_json(audit / "status.json", {"status": "failed" if error_text else "completed",
                "updated_utc": utc(), "error": error_text, "completed": completed,
                "total": 1319, "runtime_events": counts,
                "original_predictions_traces_and_configuration_preserved": True})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
