from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from AgentAttrPrune.llm import generation_observer


TRACE_SCHEMA_VERSION = 1


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    return str(value)


def _first_output(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _first_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _sequence_length(value: Any) -> int:
    value = to_jsonable(value)
    if not isinstance(value, list):
        return 0
    if value and isinstance(value[0], list):
        return len(value[0])
    return len(value)


class GenerationTraceRecorder:
    """Collect one sample's exact generation inputs, outputs, and provenance."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        sample: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        self.config = to_jsonable(dict(config or {}))
        self.sample = to_jsonable(dict(sample or {}))
        self._events: list[dict[str, Any]] = []
        self._invocation_to_event: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def new_invocation_id(self) -> str:
        return uuid.uuid4().hex

    @contextmanager
    def observe(self, metadata: Mapping[str, Any]) -> Iterator[None]:
        invocation_id = str(metadata["invocation_id"])
        stable_metadata = dict(metadata)
        pending_lock = threading.Lock()
        pending_generation: dict[str, Any] | None = None
        active = True

        def callback(generation: dict[str, Any]) -> None:
            # Buffer until the awaited LLM call has completed successfully.
            # ``asyncio.wait_for`` cannot stop an already-running to_thread
            # generation, so a timed-out attempt may invoke this callback after
            # a retry has begun. Closing this attempt before committing keeps
            # that stale generation out of the formal trace.
            nonlocal pending_generation
            with pending_lock:
                if active and pending_generation is None:
                    pending_generation = dict(generation)

        token = generation_observer.set(callback)
        try:
            yield
        except BaseException:
            with pending_lock:
                active = False
                pending_generation = None
            raise
        else:
            with pending_lock:
                active = False
                captured = pending_generation
            if captured is not None:
                self.record_generation(invocation_id, stable_metadata, captured)
        finally:
            generation_observer.reset(token)

    def record_generation(
        self,
        invocation_id: str,
        metadata: Mapping[str, Any],
        generation: Mapping[str, Any],
    ) -> int:
        with self._lock:
            existing = self._invocation_to_event.get(invocation_id)
            if existing is not None:
                return existing
            event_id = len(self._events)
            incoming = list(metadata.get("incoming", []))
            event = {
                "event_id": event_id,
                "phase": metadata.get("phase", "unknown"),
                "iteration": metadata.get("iteration"),
                "batch_index": metadata.get("batch_index"),
                "sample_index": metadata.get("sample_index"),
                "round": int(metadata["round"]),
                "agent_id": str(metadata["agent_id"]),
                "agent_type": str(metadata["agent_type"]),
                "role": str(metadata["role"]),
                "incoming_from": [str(item["agent_id"]) for item in incoming],
                "spatial_incoming_from": [
                    str(item["agent_id"])
                    for item in incoming
                    if item.get("edge_type") == "spatial"
                ],
                "temporal_incoming_from": [
                    str(item["agent_id"])
                    for item in incoming
                    if item.get("edge_type") == "temporal"
                ],
                "incoming": incoming,
                "input_messages": generation.get("messages", []),
                "rendered_prompt": generation.get("rendered_prompt"),
                "input_ids": generation.get("input_ids"),
                "attention_mask": generation.get("attention_mask"),
                "generated_ids": generation.get("generated_ids"),
                "output": _first_output(generation.get("output_text", "")),
                "finish_reason": str(
                    _first_value(generation.get("finish_reasons"), "unknown")
                ),
                "hit_max_new_tokens": bool(
                    _first_value(
                        generation.get("hit_max_new_tokens"),
                        False,
                    )
                ),
                "llm_call": generation.get("generation_kwargs", {}),
                "model_name": generation.get("model_name"),
                "model_path": generation.get("model_path"),
            }
            json_event = to_jsonable(event)
            self._events.append(json_event)
            self._invocation_to_event[invocation_id] = event_id
            return event_id

    def record_fallback(
        self,
        invocation_id: str,
        metadata: Mapping[str, Any],
        *,
        messages: list[dict[str, str]],
        output: str,
        model_name: str | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> int:
        """Record non-HF test/custom backends that do not emit an observer event."""
        return self.record_generation(
            invocation_id,
            metadata,
            {
                "messages": messages,
                "rendered_prompt": None,
                "input_ids": None,
                "attention_mask": None,
                "generated_ids": None,
                "output_text": output,
                "generation_kwargs": dict(generation_kwargs or {}),
                "model_name": model_name,
            },
        )

    def event_id_for(self, invocation_id: str) -> int | None:
        with self._lock:
            return self._invocation_to_event.get(invocation_id)

    def event(self, event_id: int) -> dict[str, Any]:
        with self._lock:
            return dict(self._events[event_id])

    @property
    def summary(self) -> dict[str, int]:
        with self._lock:
            events = list(self._events)
        return {
            "llm_calls": len(events),
            "prompt_tokens": sum(
                _sequence_length(event.get("input_ids")) for event in events
            ),
            "generated_tokens": sum(
                _sequence_length(event.get("generated_ids")) for event in events
            ),
            "truncated_events": sum(
                bool(event.get("hit_max_new_tokens")) for event in events
            ),
        }

    def payload(self, final: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        truncated_event_ids = [
            int(event["event_id"])
            for event in events
            if bool(event.get("hit_max_new_tokens"))
        ]
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_kind": "agentattrprune_generation_trace",
            "token_ids_complete": bool(events)
            and all(
                event.get("input_ids") is not None
                and event.get("generated_ids") is not None
                for event in events
            ),
            "generation_complete": bool(events) and not truncated_event_ids,
            "truncated_event_ids": truncated_event_ids,
            "config": self.config,
            "sample": self.sample,
            "events": events,
            "usage": self.summary,
            "final": to_jsonable(dict(final or {})),
        }

    def write(
        self,
        final: Mapping[str, Any] | None = None,
        *,
        output_path: str | Path | None = None,
    ) -> Path:
        path = Path(output_path) if output_path is not None else self.output_path
        if path is None:
            raise ValueError("output_path must be set in the constructor or write()")
        payload = self.payload(final)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        return path


def load_trace(
    path: str | Path,
    *,
    allow_legacy: bool = True,
    for_aggregation: bool = False,
) -> dict[str, Any]:
    """Read schema v1 or a legacy trace for debugging.

    Legacy traces without ``schema_version`` are marked as schema 0. They are
    deliberately rejected for aggregation so a formal run cannot silently mix
    incompatible trace writers.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version is None:
        if not allow_legacy:
            raise ValueError("Legacy trace has no schema_version")
        payload["schema_version"] = 0
        payload["legacy_schema"] = True
        schema_version = 0
    if schema_version not in {0, TRACE_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported trace schema_version: {schema_version}")
    if for_aggregation and schema_version != TRACE_SCHEMA_VERSION:
        raise ValueError(
            "Legacy traces are debug-only and cannot be used for aggregation"
        )
    return payload
