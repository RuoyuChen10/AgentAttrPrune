"""Original evaluation I/O helpers used by the sampled-policy evaluators."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def _read_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            records[int(item["sample_index"])] = item
    return records


def _append_batch(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _flatten_answer(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)
