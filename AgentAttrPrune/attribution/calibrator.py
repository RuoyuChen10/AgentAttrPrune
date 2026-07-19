"""High-level orchestration for attribution-based edge calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import traceback
from typing import Any, Mapping, Optional, Sequence

import torch

from AgentAttrPrune.attribution.edge import (
    GreedyFullScoreAttributor,
    aggregate_sample_matrices,
    apply_early_stop_sentence_saliency,
    build_incoming_sentence_masks,
    build_pruned_masks,
    build_sample_matrices,
    compute_event_edge_scores,
    infer_num_rounds,
    infer_worker_agents,
    load_json,
    select_target_token_indices,
    target_ids_from_event,
    trace_is_correct,
    write_json,
)
from AgentAttrPrune.attribution.scoring import TeacherForcedTextScorer
from AgentAttrPrune.attribution.visualization import (
    write_aggregate_matrix_png,
    write_event_html,
    write_event_insertion_curve_svg,
    write_event_svg,
    write_sample_matrix_png,
)
from AgentAttrPrune.pruning import RoundPolicy


ALGORITHM_VERSION = 3


def _normalize_model_name(value: Optional[str]) -> Optional[str]:
    if value and value.startswith(("hf:", "local:")):
        return value.split(":", 1)[1]
    return value


def _canonical_model_name(value: Optional[str]) -> Optional[str]:
    value = _normalize_model_name(value)
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value.rstrip("/")


@dataclass(frozen=True)
class AttributionConfig:
    """Configuration shared by every event in one calibration aggregate."""

    sample_glob: str = "sample_*"
    sample_limit: Optional[int] = None
    sample_scope: str = "all"
    node_scope: str = "all"
    event_limit: Optional[int] = None
    event_ids: Optional[tuple[int, ...]] = None
    min_incoming: int = 2
    include_reviewer: bool = True
    enable_thinking: bool = False

    target_mode: str = "all"
    target_token_limit: Optional[int] = None
    score_type: str = "logit"
    score_reduction: str = "sum"
    target_recovery: float = 0.95
    absolute_atol: float = 1e-4
    relative_rtol: float = 1e-3
    patience: int = 1
    batch_size: int = 4
    max_selected_sentences: Optional[int] = None
    accept_incomplete: bool = False
    min_selected_sentences: int = 2
    min_sentence_words: int = 4
    saliency_floor: float = 0.4

    torch_dtype: str = "auto"
    device_map: str = "auto"
    policy_pruning_rate: float = 0.25
    overwrite: bool = False
    keep_going: bool = False
    allow_legacy_traces: bool = False
    visualizations: bool = True

    def __post_init__(self) -> None:
        if not self.sample_glob:
            raise ValueError("sample_glob must be non-empty")
        if self.sample_scope not in {"all", "correct"}:
            raise ValueError("sample_scope must be all or correct")
        if self.node_scope not in {"all", "workers", "multi-input"}:
            raise ValueError(
                "node_scope must be all, workers, or multi-input"
            )
        optional_positive = {
            "sample_limit": self.sample_limit,
            "event_limit": self.event_limit,
            "target_token_limit": self.target_token_limit,
            "max_selected_sentences": self.max_selected_sentences,
        }
        for name, value in optional_positive.items():
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided")
        for name in (
            "min_incoming",
            "patience",
            "batch_size",
            "min_selected_sentences",
            "min_sentence_words",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.target_mode not in {"decision", "all"}:
            raise ValueError("target_mode must be decision or all")
        if self.score_type not in {"logit", "logprob", "probability"}:
            raise ValueError("score_type must be logit, logprob, or probability")
        if self.score_reduction not in {"sum", "mean"}:
            raise ValueError("score_reduction must be sum or mean")
        if not 0 < self.target_recovery <= 1:
            raise ValueError("target_recovery must be in (0, 1]")
        if not math.isfinite(self.absolute_atol) or not math.isfinite(self.relative_rtol):
            raise ValueError("absolute_atol and relative_rtol must be finite")
        if self.absolute_atol < 0 or self.relative_rtol < 0:
            raise ValueError("absolute_atol and relative_rtol must be non-negative")
        if not math.isfinite(self.policy_pruning_rate) or not 0 <= self.policy_pruning_rate <= 1:
            raise ValueError("policy_pruning_rate must be in [0, 1]")
        if (
            not math.isfinite(self.saliency_floor)
            or not 0 <= self.saliency_floor < 1
        ):
            raise ValueError("saliency_floor must be finite and in [0, 1)")
        if self.event_ids is not None:
            values = tuple(sorted({int(value) for value in self.event_ids}))
            if any(value < 0 for value in values):
                raise ValueError("event_ids must be non-negative")
            object.__setattr__(self, "event_ids", values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["event_ids"] is not None:
            payload["event_ids"] = list(payload["event_ids"])
        return payload


class EdgeAttributionCalibrator:
    """Attribute correct traces, aggregate matrices, and emit edge policies.

    The scoring model is loaded lazily. Tests or alternative backends may inject
    any ``scorer`` exposing ``tokenizer`` and
    ``score_teacher_forced_targets(...)``.
    """

    def __init__(
        self,
        run_dir: str | Path,
        output_dir: str | Path,
        model_name: Optional[str] = None,
        config: Optional[AttributionConfig | Mapping[str, Any]] = None,
        scorer: Optional[Any] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir)
        self.model_name = _normalize_model_name(model_name)
        if config is None:
            self.config = AttributionConfig()
        elif isinstance(config, AttributionConfig):
            self.config = config
        else:
            self.config = AttributionConfig(**dict(config))
        self._scorer = scorer

    @property
    def scorer(self):
        if self._scorer is None:
            if not self.model_name:
                paths = self._trace_paths()
                if not paths:
                    raise FileNotFoundError(f"No calibration traces under {self.run_dir}")
                self.model_name = self._infer_model_name(load_json(paths[0]))
            self._scorer = TeacherForcedTextScorer.from_pretrained(
                self.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
            )
        return self._scorer

    def _trace_root(self) -> Path:
        candidates = (
            self.run_dir,
            self.run_dir / "traces",
            self.run_dir / "calibration" / "traces",
        )
        for candidate in candidates:
            if any(candidate.glob(f"{self.config.sample_glob}/trace.json")):
                return candidate
        return self.run_dir

    def _trace_paths(self) -> list[Path]:
        paths = sorted(
            self._trace_root().glob(f"{self.config.sample_glob}/trace.json")
        )
        if self.config.sample_limit is not None:
            paths = paths[: self.config.sample_limit]
        return paths

    @staticmethod
    def _infer_model_name(trace: Mapping[str, Any]) -> str:
        config = trace.get("config") or {}
        value = _normalize_model_name(config.get("model_name") or config.get("llm_name"))
        if not value:
            for event in trace.get("events") or []:
                value = _normalize_model_name(event.get("model_name"))
                if value:
                    break
        if not value:
            raise ValueError("No model name in trace; pass model_name explicitly")
        return value

    def _validate_trace(self, trace: Mapping[str, Any], trace_path: Path) -> None:
        schema = trace.get("schema_version")
        if schema is None and not self.config.allow_legacy_traces:
            raise ValueError(
                f"Legacy trace without schema_version is debug-only: {trace_path}. "
                "Set allow_legacy_traces=True only for isolated compatibility checks."
            )
        if schema is not None and int(schema) != 1:
            raise ValueError(f"Unsupported trace schema_version {schema!r}: {trace_path}")
        if schema is not None and int(schema) == 1:
            if not str((trace.get("config") or {}).get("fingerprint", "")):
                raise ValueError(
                    f"Schema-v1 trace is missing its run fingerprint: {trace_path}"
                )
            incomplete_events = [
                event.get("event_id")
                for event in trace.get("events") or []
                if event.get("input_ids") is None
                or event.get("generated_ids") is None
            ]
            if incomplete_events:
                raise ValueError(
                    "Schema-v1 formal attribution requires exact input_ids and "
                    f"generated_ids; missing for event(s) {incomplete_events}: "
                    f"{trace_path}"
                )
            truncated_events = set(
                int(event_id)
                for event_id in trace.get("truncated_event_ids") or []
            )
            for event in trace.get("events") or []:
                event_id = int(event.get("event_id", -1))
                if (
                    event.get("finish_reason") == "length"
                    or bool(event.get("hit_max_new_tokens"))
                ):
                    truncated_events.add(event_id)
                    continue
                if event.get("finish_reason") not in (None, "unknown"):
                    continue
                max_tokens = (event.get("llm_call") or {}).get("max_tokens")
                generated = event.get("generated_ids")
                if max_tokens is None or generated is None:
                    continue
                generated_tensor = torch.as_tensor(generated)
                while (
                    generated_tensor.ndim > 1
                    and generated_tensor.shape[0] == 1
                ):
                    generated_tensor = generated_tensor.squeeze(0)
                if generated_tensor.numel() >= int(max_tokens):
                    truncated_events.add(event_id)
            if truncated_events:
                raise ValueError(
                    "Formal attribution requires complete node outputs; event(s) "
                    f"{sorted(truncated_events)} hit max_new_tokens in {trace_path}. "
                    "Regenerate the run with a larger MAX_NEW_TOKENS value."
                )
        trace_model = None
        try:
            trace_model = self._infer_model_name(trace)
        except ValueError:
            pass
        scoring_model = self.model_name or getattr(self.scorer, "model_name", None)
        if trace_model and scoring_model:
            if _canonical_model_name(trace_model) != _canonical_model_name(scoring_model):
                raise ValueError(
                    "Trace generation model and attribution model differ: "
                    f"{trace_model!r} vs {scoring_model!r}. Exact target ids are not portable."
                )

    @staticmethod
    def _sample_name(trace_path: Path, trace: Mapping[str, Any]) -> str:
        if trace_path.parent.name.startswith("sample_"):
            return trace_path.parent.name
        sample = trace.get("sample") or {}
        index = (trace.get("config") or {}).get(
            "sample_index", sample.get("sample_index", 0)
        )
        return f"sample_{int(index):05d}"

    @staticmethod
    def _event_slug(event: Mapping[str, Any]) -> str:
        agent = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(event.get("agent_id", "agent"))
        ).strip("_")
        return (
            f"event_{int(event.get('event_id', 0)):03d}_"
            f"r{int(event.get('round', 0)):02d}_{agent}"
        )

    def _eligible_events(self, trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        workers = set(infer_worker_agents(trace))
        events = []
        for event in trace.get("events") or []:
            if (
                self.config.event_ids is not None
                and int(event.get("event_id", -1)) not in self.config.event_ids
            ):
                continue
            agent_id = str(event.get("agent_id", ""))
            if self.config.node_scope == "workers" and agent_id not in workers:
                continue
            if self.config.node_scope == "multi-input":
                if len(event.get("incoming") or []) < self.config.min_incoming:
                    continue
                if not self.config.include_reviewer and agent_id not in workers:
                    continue
            events.append(event)
        if self.config.event_limit is not None:
            events = events[: self.config.event_limit]
        return events

    def _validate_all_node_coverage(
        self,
        trace: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        if (
            self.config.node_scope != "all"
            or self.config.event_limit is not None
            or self.config.event_ids is not None
        ):
            return
        workers = infer_worker_agents(trace)
        num_rounds = infer_num_rounds(trace)
        expected = {
            (round_index, agent_id)
            for round_index in range(num_rounds)
            for agent_id in workers
        }
        worker_set = set(workers)
        actual_worker = [
            (int(event.get("round", -1)), str(event.get("agent_id", "")))
            for event in events
            if str(event.get("agent_id", "")) in worker_set
        ]
        reviewer = [
            event
            for event in events
            if str(event.get("agent_id", "")) not in worker_set
            and int(event.get("round", -1)) == num_rounds
        ]
        actual_set = set(actual_worker)
        duplicates = len(actual_worker) != len(actual_set)
        missing = sorted(expected - actual_set)
        unexpected = sorted(actual_set - expected)
        expected_count = len(workers) * num_rounds + 1
        if (
            duplicates
            or missing
            or unexpected
            or len(reviewer) != 1
            or len(events) != expected_count
        ):
            raise ValueError(
                "Incomplete all-node trace coverage: "
                f"expected={expected_count}, actual={len(events)}, "
                f"missing={missing}, unexpected={unexpected}, "
                f"reviewers={len(reviewer)}, duplicates={duplicates}"
            )

    def _selection_configuration(self) -> dict[str, Any]:
        return {
            "sample_scope": self.config.sample_scope,
            "node_scope": self.config.node_scope,
            "min_incoming": self.config.min_incoming,
            "include_reviewer": self.config.include_reviewer,
            "event_ids": (
                list(self.config.event_ids) if self.config.event_ids is not None else None
            ),
            "event_limit": self.config.event_limit,
        }

    def _event_configuration(self, event: Mapping[str, Any]) -> dict[str, Any]:
        trace_payload = {
            key: event.get(key)
            for key in (
                "round",
                "agent_id",
                "incoming",
                "input_messages",
                "rendered_prompt",
                "input_ids",
                "attention_mask",
                "output",
                "generated_ids",
                "model_name",
                "llm_call",
            )
        }
        encoded = json.dumps(
            trace_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        scorer = self.scorer
        model = getattr(scorer, "model", None)
        tokenizer = scorer.tokenizer
        try:
            dtype = str(next(model.parameters()).dtype)
        except (AttributeError, StopIteration):
            dtype = "unknown"
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            vocab_size = None
        config = self.config.to_dict()
        for key in (
            "sample_glob",
            "sample_limit",
            "sample_scope",
            "node_scope",
            "event_limit",
            "event_ids",
            "min_incoming",
            "include_reviewer",
            "overwrite",
            "keep_going",
            "allow_legacy_traces",
            "policy_pruning_rate",
            "saliency_floor",
            "visualizations",
        ):
            config.pop(key, None)
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "model_name": str(getattr(scorer, "model_name", self.model_name)),
            "model_config_name": str(
                getattr(getattr(model, "config", None), "_name_or_path", self.model_name)
            ),
            "model_dtype": dtype,
            "tokenizer_name": str(
                getattr(tokenizer, "name_or_path", getattr(scorer, "model_name", "unknown"))
            ),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": vocab_size,
            "event_sha256": hashlib.sha256(encoded).hexdigest(),
            **config,
        }

    def _process_event(
        self,
        trace_path: Path,
        sample_id: str,
        trace: Mapping[str, Any],
        event: Mapping[str, Any],
        output_path: Path,
    ) -> dict[str, Any]:
        event_model = _normalize_model_name(event.get("model_name"))
        scorer_model = self.model_name or getattr(self.scorer, "model_name", None)
        if event_model and scorer_model:
            if _canonical_model_name(event_model) != _canonical_model_name(scorer_model):
                raise ValueError(
                    "Event generation model and attribution model differ: "
                    f"{event_model!r} vs {scorer_model!r}."
                )
        configuration = self._event_configuration(event)
        html_path = output_path.with_suffix(".html")
        svg_path = output_path.with_suffix(".svg")
        curve_path = output_path.with_suffix(".insertion_curve.svg")
        if output_path.exists() and not self.config.overwrite:
            cached = load_json(output_path)
            if cached.get("configuration") == configuration:
                apply_early_stop_sentence_saliency(
                    cached,
                    saliency_floor=self.config.saliency_floor,
                )
                cached["artifacts"] = {
                    "json": str(output_path.resolve()),
                    "svg": (
                        str(svg_path.resolve())
                        if self.config.visualizations
                        else None
                    ),
                    "visualization": (
                        str(svg_path.resolve())
                        if self.config.visualizations
                        else None
                    ),
                    "insertion_curve_svg": (
                        str(curve_path.resolve())
                        if self.config.visualizations
                        else None
                    ),
                    "html": (
                        str(html_path.resolve())
                        if self.config.visualizations
                        else None
                    ),
                }
                if self.config.visualizations:
                    write_event_svg(cached, svg_path)
                    write_event_insertion_curve_svg(cached, curve_path)
                    write_event_html(cached, html_path)
                write_json(output_path, cached)
                return cached

        tokenizer = self.scorer.tokenizer
        if not event.get("incoming"):
            rendered = str(event.get("rendered_prompt") or "")
            if not rendered:
                try:
                    rendered = tokenizer.apply_chat_template(
                        list(event.get("input_messages") or []),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.config.enable_thinking,
                    )
                except TypeError:
                    rendered = tokenizer.apply_chat_template(
                        list(event.get("input_messages") or []),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            generated_ids, target_id_source = target_ids_from_event(event, tokenizer)
            target_indices = select_target_token_indices(
                tokenizer,
                generated_ids,
                mode=self.config.target_mode,
                target_token_limit=self.config.target_token_limit,
            )
            result = self._base_event_result(
                trace_path=trace_path,
                sample_id=sample_id,
                trace=trace,
                event=event,
                configuration=configuration,
                rendered=rendered,
                input_ids=event.get("input_ids") or [],
                generated_ids=generated_ids,
                target_id_source=target_id_source,
                target_indices=target_indices,
                status="no_attributable_input",
                incoming_edges=[],
                sentences=[],
                attribution={
                    "performed": False,
                    "score_type": self.config.score_type,
                    "reduction": self.config.score_reduction,
                    "selected_region_indices": [],
                    "curve": [],
                    "stop_reason": "no_attributable_input",
                    "stop_step": 0,
                    "selected_recovery": None,
                },
                edge_scores=[],
            )
            result["attributable_sentence_count"] = 0
            result["attributable_token_count"] = 0
            apply_early_stop_sentence_saliency(
                result,
                saliency_floor=self.config.saliency_floor,
            )
            result["artifacts"] = {
                "json": str(output_path.resolve()),
                "svg": (
                    str(svg_path.resolve())
                    if self.config.visualizations
                    else None
                ),
                "visualization": (
                    str(svg_path.resolve())
                    if self.config.visualizations
                    else None
                ),
                "insertion_curve_svg": (
                    str(curve_path.resolve())
                    if self.config.visualizations
                    else None
                ),
                "html": str(html_path.resolve()) if self.config.visualizations else None,
            }
            write_json(output_path, result)
            if self.config.visualizations:
                write_event_svg(result, svg_path)
                write_event_insertion_curve_svg(result, curve_path)
                write_event_html(result, html_path)
            return result

        rendered, input_ids, masks, spans, edges = build_incoming_sentence_masks(
            tokenizer,
            event,
            enable_thinking=self.config.enable_thinking,
        )
        recorded_ids = event.get("input_ids")
        if recorded_ids is not None:
            recorded = torch.as_tensor(recorded_ids, dtype=torch.long)
            while recorded.ndim > 1 and recorded.shape[0] == 1:
                recorded = recorded.squeeze(0)
            if recorded.ndim != 1 or recorded.tolist() != input_ids:
                raise ValueError(
                    "Recorded input_ids do not match the attribution tokenizer/rendered prompt."
                )
        generated_ids, target_id_source = target_ids_from_event(event, tokenizer)
        target_indices = select_target_token_indices(
            tokenizer,
            generated_ids,
            mode=self.config.target_mode,
            target_token_limit=self.config.target_token_limit,
        )
        attributor = GreedyFullScoreAttributor(
            self.scorer,
            score_type=self.config.score_type,
            reduction=self.config.score_reduction,
            target_recovery=self.config.target_recovery,
            absolute_atol=self.config.absolute_atol,
            relative_rtol=self.config.relative_rtol,
            patience=self.config.patience,
            batch_size=self.config.batch_size,
            max_selected_sentences=self.config.max_selected_sentences,
        )
        attribution = attributor(
            input_ids,
            masks,
            generated_ids=generated_ids,
            selected_target_indices=target_indices,
        )
        if (
            attribution["stop_reason"] == "max_selected_sentences"
            and not self.config.accept_incomplete
        ):
            raise RuntimeError(
                "Attribution hit max_selected_sentences before target recovery; "
                "the incomplete prefix is excluded from calibration."
            )
        edge_scores = compute_event_edge_scores(
            spans,
            edges,
            attribution["selected_region_indices"],
            min_selected_sentences=self.config.min_selected_sentences,
            min_sentence_words=self.config.min_sentence_words,
        )
        selected = set(attribution["selected_region_indices"])
        result = self._base_event_result(
            trace_path=trace_path,
            sample_id=sample_id,
            trace=trace,
            event=event,
            configuration=configuration,
            rendered=rendered,
            input_ids=input_ids,
            generated_ids=generated_ids,
            target_id_source=target_id_source,
            target_indices=target_indices,
            status="attributed",
            incoming_edges=[edge.to_dict() for edge in edges],
            sentences=[
                {**span.to_dict(), "selected": span.index in selected}
                for span in spans
            ],
            attribution={"performed": True, **attribution},
            edge_scores=edge_scores,
        )
        result["attributable_sentence_count"] = len(spans)
        result["attributable_token_count"] = int(masks.max(axis=0).sum())
        apply_early_stop_sentence_saliency(
            result,
            saliency_floor=self.config.saliency_floor,
        )
        result["artifacts"] = {
            "json": str(output_path.resolve()),
            "svg": (
                str(svg_path.resolve())
                if self.config.visualizations
                else None
            ),
            "visualization": (
                str(svg_path.resolve())
                if self.config.visualizations
                else None
            ),
            "insertion_curve_svg": (
                str(curve_path.resolve())
                if self.config.visualizations
                else None
            ),
            "html": str(html_path.resolve()) if self.config.visualizations else None,
        }
        write_json(output_path, result)
        if self.config.visualizations:
            write_event_svg(result, svg_path)
            write_event_insertion_curve_svg(result, curve_path)
            write_event_html(result, html_path)
        return result

    def _base_event_result(
        self,
        *,
        trace_path: Path,
        sample_id: str,
        trace: Mapping[str, Any],
        event: Mapping[str, Any],
        configuration: Mapping[str, Any],
        rendered: str,
        input_ids: Sequence[int] | Any,
        generated_ids: Sequence[int],
        target_id_source: str,
        target_indices: Sequence[int],
        status: str,
        incoming_edges: Sequence[Mapping[str, Any]],
        sentences: Sequence[Mapping[str, Any]],
        attribution: Mapping[str, Any],
        edge_scores: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        tokenizer = self.scorer.tokenizer
        prompt_ids = torch.as_tensor(input_ids, dtype=torch.long)
        while prompt_ids.ndim > 1 and prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids.squeeze(0)
        prompt_count = int(prompt_ids.numel()) if prompt_ids.ndim == 1 else 0
        return {
            "schema_version": 1,
            "trace_json": str(trace_path.resolve()),
            "sample_id": sample_id,
            "sample": dict(trace.get("sample") or {}),
            "correct_verified": trace_is_correct(trace),
            "configuration": dict(configuration),
            "status": status,
            "event": {
                "event_id": event.get("event_id"),
                "round": event.get("round"),
                "agent_id": event.get("agent_id"),
                "agent_type": event.get("agent_type"),
                "role": event.get("role"),
                "incoming_count": len(event.get("incoming") or []),
            },
            "scope": "incoming_agent_output_sentences_only",
            "input": {
                "messages": list(event.get("input_messages") or []),
                "rendered_prompt": rendered,
                "input_ids": event.get("input_ids"),
                "attention_mask": event.get("attention_mask"),
            },
            # Retained as a compatibility convenience for earlier readers.
            "rendered_prompt": rendered,
            "prompt_token_count": prompt_count,
            "output": {
                "text": str(event.get("output") or ""),
                "generated_ids": list(generated_ids),
            },
            "target": {
                "mode": self.config.target_mode,
                "id_source": target_id_source,
                "generated_ids": list(generated_ids),
                "selected_indices": list(target_indices),
                "selected_ids": [
                    generated_ids[index] for index in target_indices
                ],
                "selected_tokens": [
                    tokenizer.decode(
                        [generated_ids[index]],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for index in target_indices
                ],
            },
            "incoming_edges": list(incoming_edges),
            "sentences": list(sentences),
            "attribution": dict(attribution),
            "edge_scores": list(edge_scores),
        }

    def attribute(self) -> dict[str, Any]:
        """Attribute all eligible events and write per-sample matrices/manifest."""

        trace_paths = self._trace_paths()
        if not trace_paths:
            raise FileNotFoundError(
                f"No {self.config.sample_glob}/trace.json files under {self.run_dir}"
            )
        if not self.model_name:
            self.model_name = self._infer_model_name(load_json(trace_paths[0]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_dir": str(self.run_dir.resolve()),
            "output_dir": str(self.output_dir.resolve()),
            "model_name": self.model_name,
            "configuration": self.config.to_dict(),
            "samples": [],
            "skipped_incorrect": [],
            "failures": [],
        }
        for trace_path in trace_paths:
            trace = load_json(trace_path)
            sample_id = self._sample_name(trace_path, trace)
            correct_verified = trace_is_correct(trace)
            if not correct_verified and self.config.sample_scope == "correct":
                manifest["skipped_incorrect"].append(str(trace_path.resolve()))
                continue
            sample_dir = self.output_dir / sample_id
            try:
                self._validate_trace(trace, trace_path)
                eligible_events = self._eligible_events(trace)
                self._validate_all_node_coverage(trace, eligible_events)
                event_results = []
                event_paths = []
                for event in eligible_events:
                    event_path = sample_dir / f"{self._event_slug(event)}.json"
                    event_results.append(
                        self._process_event(
                            trace_path,
                            sample_id,
                            trace,
                            event,
                            event_path,
                        )
                    )
                    event_paths.append(str(event_path.resolve()))
                if not event_results:
                    raise ValueError("No eligible generation-node events were found.")

                workers = infer_worker_agents(trace)
                expected_count = len(workers) * infer_num_rounds(trace) + 1
                index_path = sample_dir / "node_attribution_index.json"
                index_payload = {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "sample_index": (trace.get("config") or {}).get("sample_index"),
                    "correct_verified": correct_verified,
                    "node_scope": self.config.node_scope,
                    "expected_event_count": expected_count,
                    "actual_event_count": len(event_results),
                    "complete": (
                        self.config.node_scope == "all"
                        and self.config.event_limit is None
                        and self.config.event_ids is None
                        and len(event_results) == expected_count
                    ),
                    "events": [
                        {
                            "event_id": result["event"]["event_id"],
                            "round": result["event"]["round"],
                            "agent_id": result["event"]["agent_id"],
                            "role": result["event"]["role"],
                            "status": result["status"],
                            "json": result["artifacts"]["json"],
                            "svg": result["artifacts"]["svg"],
                            "visualization": result["artifacts"]["visualization"],
                            "insertion_curve_svg": result["artifacts"][
                                "insertion_curve_svg"
                            ],
                            "html": result["artifacts"]["html"],
                        }
                        for result in event_results
                    ],
                }
                if (
                    self.config.node_scope == "all"
                    and self.config.event_limit is None
                    and self.config.event_ids is None
                    and not index_payload["complete"]
                ):
                    raise RuntimeError(
                        f"{sample_id} is missing one or more node attribution results"
                    )
                write_json(index_path, index_payload)

                sample_matrix = build_sample_matrices(trace, event_results)
                sample_matrix["selection_configuration"] = self._selection_configuration()
                sample_matrix["event_attribution_files"] = event_paths
                sample_matrix["node_attribution_index"] = str(index_path.resolve())
                matrix_path = sample_dir / "sample_edge_matrices.json"
                matrix_png = sample_dir / "sample_token_ratio_matrices.png"
                sample_matrix["visualizations"] = {
                    "token_ratio_png": (
                        str(matrix_png.resolve())
                        if self.config.visualizations
                        else None
                    )
                }
                write_json(matrix_path, sample_matrix)
                if self.config.visualizations:
                    write_sample_matrix_png(sample_matrix, matrix_png)
                manifest["samples"].append(
                    {
                        "sample_id": sample_id,
                        "trace_json": str(trace_path.resolve()),
                        "correct_verified": correct_verified,
                        "event_count": len(event_results),
                        "node_attribution_index": str(index_path.resolve()),
                        "sample_matrix": str(matrix_path.resolve()),
                        "sample_matrix_png": (
                            str(matrix_png.resolve())
                            if self.config.visualizations
                            else None
                        ),
                    }
                )
            except Exception as error:
                manifest["failures"].append(
                    {
                        "sample_id": sample_id,
                        "trace_json": str(trace_path.resolve()),
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                if not self.config.keep_going:
                    write_json(self.output_dir / "manifest.json", manifest)
                    raise
        write_json(self.output_dir / "manifest.json", manifest)
        if manifest["failures"]:
            raise RuntimeError(
                f"Attribution failed for {len(manifest['failures'])} sample(s); "
                "no formal aggregate or policy was published. See manifest.json."
            )
        if not manifest["samples"]:
            raise RuntimeError("No samples were successfully attributed.")
        return manifest

    def render_existing_visualizations(self) -> dict[str, Any]:
        """Rebuild per-node SVG/HTML reports from cached event JSON only."""

        sample_dirs = sorted(
            path
            for path in self.output_dir.glob(self.config.sample_glob)
            if path.is_dir()
        )
        rendered: dict[str, Any] = {
            "schema_version": 1,
            "output_dir": str(self.output_dir.resolve()),
            "mode": "render_existing_visualizations",
            "samples": [],
        }
        for sample_dir in sample_dirs:
            event_paths = sorted(sample_dir.glob("event_*.json"))
            if not event_paths:
                continue
            event_results: dict[int, dict[str, Any]] = {}
            for event_path in event_paths:
                result = load_json(event_path)
                apply_early_stop_sentence_saliency(
                    result,
                    saliency_floor=self.config.saliency_floor,
                )
                svg_path = event_path.with_suffix(".svg")
                curve_path = event_path.with_suffix(".insertion_curve.svg")
                html_path = event_path.with_suffix(".html")
                result["artifacts"] = {
                    "json": str(event_path.resolve()),
                    "svg": str(svg_path.resolve()),
                    "visualization": str(svg_path.resolve()),
                    "insertion_curve_svg": str(curve_path.resolve()),
                    "html": str(html_path.resolve()),
                }
                write_event_svg(result, svg_path)
                write_event_insertion_curve_svg(result, curve_path)
                write_event_html(result, html_path)
                write_json(event_path, result)
                event_id = int((result.get("event") or {}).get("event_id", -1))
                event_results[event_id] = result

            index_path = sample_dir / "node_attribution_index.json"
            if index_path.exists():
                index = load_json(index_path)
                for item in index.get("events") or []:
                    event_id = int(item.get("event_id", -1))
                    result = event_results.get(event_id)
                    if result is None:
                        continue
                    item.update(
                        {
                            "json": result["artifacts"]["json"],
                            "svg": result["artifacts"]["svg"],
                            "visualization": result["artifacts"]["visualization"],
                            "insertion_curve_svg": result["artifacts"][
                                "insertion_curve_svg"
                            ],
                            "html": result["artifacts"]["html"],
                        }
                    )
                write_json(index_path, index)

            rendered["samples"].append(
                {
                    "sample_id": sample_dir.name,
                    "event_count": len(event_paths),
                    "node_attribution_index": (
                        str(index_path.resolve()) if index_path.exists() else None
                    ),
                    "svg_files": [
                        str(path.with_suffix(".svg").resolve())
                        for path in event_paths
                    ],
                    "insertion_curve_svg_files": [
                        str(path.with_suffix(".insertion_curve.svg").resolve())
                        for path in event_paths
                    ],
                }
            )
        if not rendered["samples"]:
            raise FileNotFoundError(
                f"No cached event_*.json files under {self.output_dir}"
            )
        write_json(self.output_dir / "visualization_manifest.json", rendered)
        return rendered

    def aggregate(
        self,
        sample_matrix_paths: Optional[Sequence[str | Path]] = None,
    ) -> dict[str, Any]:
        """Aggregate correct matrices and write all supported metric policies."""

        if sample_matrix_paths is None:
            paths = sorted(
                self.output_dir.glob("sample_*/sample_edge_matrices.json")
            )
        else:
            paths = sorted(Path(path) for path in sample_matrix_paths)
        if not paths:
            raise ValueError("No sample_edge_matrices.json files were provided")
        sample_matrices = [load_json(path) for path in paths]
        legacy_paths = [
            path
            for path, matrix in zip(paths, sample_matrices)
            if matrix.get("trace_schema_version", "legacy") != 1
        ]
        if legacy_paths:
            raise ValueError(
                "Legacy traces are debug-only and cannot be used to publish a "
                "formal calibration aggregate: "
                + ", ".join(str(path) for path in legacy_paths)
            )
        aggregate = aggregate_sample_matrices(sample_matrices)
        if not str(aggregate.get("run_fingerprint") or ""):
            raise ValueError(
                "Schema-v1 sample matrices are missing a run fingerprint; "
                "refusing to publish an unbound policy."
            )
        aggregate["calibration"] = {
            "source": "verified-correct traces only",
            "run_dir": str(self.run_dir.resolve()),
            "sample_matrix_files": [str(path.resolve()) for path in paths],
        }
        aggregate_png = self.output_dir / "aggregate_token_ratio_matrices.png"
        aggregate["visualizations"] = {
            "token_ratio_conditional_mean_png": (
                str(aggregate_png.resolve())
                if self.config.visualizations
                else None
            )
        }
        write_json(self.output_dir / "aggregate_edge_matrices.json", aggregate)
        if self.config.visualizations:
            write_aggregate_matrix_png(aggregate, aggregate_png)
        policies = {}
        for metric in ("token_ratio", "binary", "word_ratio"):
            payload = build_pruned_masks(
                aggregate,
                metric=metric,
                statistic="conditional_mean",
                threshold=None,
                pruning_rate=self.config.policy_pruning_rate,
                unobserved="keep",
            )
            # Validate the on-disk public policy contract before publishing it.
            policies[metric] = RoundPolicy.from_dict(payload)
            write_json(self.output_dir / f"pruned_policy_{metric}.json", payload)
        return {"aggregate": aggregate, "policies": policies}

    def run(
        self,
        *,
        aggregate_only: bool = False,
        aggregate_all_existing: bool = False,
        attribute_only: bool = False,
        render_only: bool = False,
    ) -> dict[str, Any]:
        """Run attribution/aggregation or rebuild cached visual reports only."""

        if aggregate_all_existing and not aggregate_only:
            raise ValueError("aggregate_all_existing requires aggregate_only=True")
        selected_modes = sum(
            bool(value) for value in (aggregate_only, attribute_only, render_only)
        )
        if selected_modes > 1:
            raise ValueError(
                "aggregate_only, attribute_only, and render_only are mutually exclusive"
            )
        if render_only:
            return {
                "manifest": self.render_existing_visualizations(),
                "aggregate": None,
                "policies": {},
            }
        if aggregate_only:
            paths = None
            manifest = None
            manifest_path = self.output_dir / "manifest.json"
            if manifest_path.exists():
                manifest = load_json(manifest_path)
                if manifest.get("failures"):
                    raise ValueError(
                        "The latest attribution manifest contains failures; "
                        "refusing to publish a partial formal aggregate."
                    )
                if not aggregate_all_existing:
                    paths = [
                        Path(item["sample_matrix"])
                        for item in manifest.get("samples", [])
                    ]
                    if not paths:
                        raise ValueError(
                            "The latest manifest has no successful sample matrices"
                        )
            aggregation = self.aggregate(paths)
            return {"manifest": manifest, **aggregation}

        manifest = self.attribute()
        if attribute_only:
            return {"manifest": manifest, "aggregate": None, "policies": {}}
        paths = [Path(item["sample_matrix"]) for item in manifest["samples"]]
        return {"manifest": manifest, **self.aggregate(paths)}
