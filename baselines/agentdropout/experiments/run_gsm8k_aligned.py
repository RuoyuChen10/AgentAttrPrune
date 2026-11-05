"""GSM8K training, parser and checkpoint helpers retained from the aligned runner.

The unrelated standalone CLI and dense experiments have been omitted.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict

def _flatten_answer(answer):
    return str(answer[0]) if isinstance(answer, (list, tuple)) and answer else str(answer or "")


class GSM8KDataset:
    """Adapter preserving gsm_get_predict and the released float comparison."""

    def __init__(self, path: Path, split: str):
        self.path, self.split = Path(path), split
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or not all(
                        isinstance(record.get(key), str) for key in ("question", "answer")
                    ) or not record["question"].strip() or "\n####" not in record["answer"]:
                        raise ValueError("expected a question and an answer with the #### marker")
                    records.append(record)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"invalid GSM8K record at {self.path}:{line_number}: {error}") from error
        if not records:
            raise ValueError(f"empty dataset: {self.path}")
        self.records = gsm_data_process(records)
        for record in self.records:
            if not math.isfinite(float(record["answer"])):
                raise ValueError(f"non-finite gold answer in {self.path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @staticmethod
    def record_to_input(record):
        return {"task": record["task"]}

    @staticmethod
    def record_to_target_answer(record):
        value = float(record["answer"])
        return str(0.0 if value == 0 else value)

    @staticmethod
    def postprocess_answer(answer):
        try:
            value = float(gsm_get_predict(_flatten_answer(answer)))
            return str(0.0 if value == 0 else value) if math.isfinite(value) else ""
        except (ValueError, TypeError, IndexError, ZeroDivisionError):
            return ""


async def train(graph, dataset, args, metrics_path):
    from experiments.aligned_train_mmlu import _edge_dropout_stage, _node_dropout_stage

    graph.optimized_spatial = graph.optimized_temporal = False
    await _node_dropout_stage(
        graph, dataset, num_iters=args.node_iters, batch_size=args.node_batch_size,
        num_rounds=args.num_rounds, lr=args.learning_rate, loader=iter(enumerate(dataset)),
        metrics_path=metrics_path, seed=args.seed,
    )
    graph.optimized_spatial = graph.optimized_temporal = True
    await _edge_dropout_stage(
        graph, dataset, num_iters=args.edge_iters, batch_size=args.edge_batch_size,
        num_rounds=args.num_rounds, lr=args.learning_rate, delta=args.delta,
        prune_every=args.prune_every, pruning_rate=args.pruning_rate,
        loader=iter(enumerate(dataset)), metrics_path=metrics_path, seed=args.seed,
    )


def _atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_existing(path, dataset, seed, *, repair_tail=False, tolerate_tail=False):
    records = {}
    if not path.exists():
        return records
    lines = path.read_bytes().splitlines(keepends=True)
    offset = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            offset += len(line)
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            if (repair_tail or tolerate_tail) and line_number == len(lines) and not line.endswith(b"\n"):
                if repair_tail:
                    with path.open("r+b") as handle:
                        handle.truncate(offset)
                        handle.flush()
                        os.fsync(handle.fileno())
                    print(f"Discarded incomplete final JSONL record at {path}:{line_number}", flush=True)
                else:
                    print(f"Incomplete final JSONL record will be retried on resume: {path}:{line_number}", flush=True)
                break
            raise ValueError(f"invalid prediction JSONL at {path}:{line_number}") from error
        index = record.get("sample_index") if isinstance(record, dict) else None
        if type(index) is not int or not 0 <= index < len(dataset) or index in records:
            raise ValueError(f"invalid or duplicate sample_index at {path}:{line_number}")
        if not all(isinstance(record.get(key), str) for key in ("question", "gold_answer", "raw_answer", "parsed_answer")):
            raise ValueError(f"prediction fields missing or invalid at {path}:{line_number}")
        gold = dataset.record_to_target_answer(dataset[index])
        parsed = dataset.postprocess_answer(record.get("raw_answer", ""))
        if (record.get("question") != dataset[index]["task"] or record.get("gold_answer") != gold
                or record.get("policy_seed") != seed + index or record.get("parsed_answer") != parsed
                or type(record.get("correct")) is not bool or record["correct"] != (parsed == gold)):
            raise ValueError(f"prediction does not match this run at {path}:{line_number}")
        records[index] = record
        offset += len(line)
    return records
