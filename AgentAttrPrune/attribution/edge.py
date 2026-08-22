"""Sentence-level attribution utilities for round-specific edge pruning.

The calibration unit is one recorded receiver event.  Only sentences inside
``event["incoming"]`` source outputs are removable; the system prompt, task,
options, chat-template tokens, and source headers remain fixed.  Matrices use
``[round][source][target]`` orientation throughout this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch

from AgentAttrPrune.attribution.sentences import split_sentence_spans


EDGE_TYPES = ("spatial", "temporal")
METRICS = ("binary", "word_ratio", "token_ratio")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*|[\u3400-\u9fff]")


def word_count(text: str) -> int:
    """Count English-like words and individual CJK characters."""

    return len(_WORD_RE.findall(str(text)))


def _incoming_output(incoming: Mapping[str, Any]) -> str:
    return str(incoming.get("source_output") or incoming.get("output") or "")


def _render_messages(tokenizer, messages: Sequence[Mapping[str, Any]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )


@dataclass(frozen=True)
class IncomingSentenceSpan:
    index: int
    incoming_index: int
    source_agent: str
    target_agent: str
    edge_type: str
    source_round: int
    target_round: int
    source_event_id: Optional[int]
    text: str
    char_start: int
    char_end: int
    token_indices: Tuple[int, ...]
    word_count: int
    source_word_count: int
    source_char_start: int = 0
    source_char_end: int = 0
    source_token_indices: Tuple[int, ...] = ()
    source_token_count: int = 0

    @property
    def edge_key(self) -> Tuple[str, str, str, int]:
        return self.edge_type, self.source_agent, self.target_agent, self.target_round

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["token_indices"] = list(self.token_indices)
        return value


@dataclass(frozen=True)
class IncomingEdge:
    incoming_index: int
    source_agent: str
    target_agent: str
    edge_type: str
    source_round: int
    target_round: int
    source_event_id: Optional[int]
    source_output: str
    source_word_count: int
    source_token_count: int = 0
    source_generated_ids: Tuple[int, ...] = ()
    source_generated_token_count: int = 0

    @property
    def key(self) -> Tuple[str, str, str, int]:
        return self.edge_type, self.source_agent, self.target_agent, self.target_round

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _content_ranges(
    rendered: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> List[Tuple[int, int]]:
    """Locate exact message content in rendered-chat order for one role."""

    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for message in messages:
        content = str(message.get("content") or "")
        if not content:
            continue
        start = rendered.find(content, cursor)
        if start < 0:
            preview = re.sub(r"\s+", " ", content)[:120]
            raise ValueError(f"Could not locate exact message content in rendered prompt: {preview!r}")
        end = start + len(content)
        if str(message.get("role", "")) == role:
            ranges.append((start, end))
        cursor = end
    return ranges


def _all_occurrences_within(
    rendered: str,
    needle: str,
    ranges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    if not needle:
        return []
    matches: List[Tuple[int, int]] = []
    for range_start, range_end in ranges:
        cursor = range_start
        while True:
            start = rendered.find(needle, cursor, range_end)
            if start < 0:
                break
            end = start + len(needle)
            if end <= range_end:
                matches.append((start, end))
            cursor = start + 1
    return matches


def _locate_incoming_outputs(
    rendered: str,
    messages: Sequence[Mapping[str, Any]],
    source_outputs: Sequence[str],
) -> List[Tuple[int, int]]:
    """Find the latest feasible ordered matches inside user messages only.

    Agent outputs are appended after the task in AgentPrune prompts. Choosing
    the latest complete non-overlapping chain avoids collisions with identical
    text in the question while preserving distinct repeated spatial/temporal
    occurrences. Exact matching is intentional: normalized trace text cannot
    be mapped to tokenizer offsets safely.
    """

    user_ranges = _content_ranges(rendered, messages, role="user")
    if not user_ranges:
        raise ValueError("Rendered prompt contains no locatable user message content.")
    candidates = [
        _all_occurrences_within(rendered, source_output, user_ranges)
        for source_output in source_outputs
    ]
    for source_output, matches in zip(source_outputs, candidates):
        if not matches:
            preview = re.sub(r"\s+", " ", source_output)[:120]
            raise ValueError(
                f"Could not locate exact incoming source output inside user messages: {preview!r}"
            )

    @lru_cache(maxsize=None)
    def solve(incoming_index: int, minimum_start: int):
        if incoming_index == len(candidates):
            return ()
        # Descending order selects the latest chain that still leaves room for
        # every later incoming output.
        for start, end in reversed(candidates[incoming_index]):
            if start < minimum_start:
                continue
            tail = solve(incoming_index + 1, end)
            if tail is not None:
                return ((start, end),) + tail
        return None

    resolved = solve(0, 0)
    if resolved is None:
        raise ValueError("Incoming source outputs cannot be aligned in recorded prompt order.")
    return list(resolved)


def _token_offsets(tokenizer, rendered_text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]
    return [int(value) for value in input_ids], [tuple(map(int, pair)) for pair in offsets]


def _flatten_token_ids(value: Any) -> Tuple[int, ...]:
    if value is None:
        return ()
    tensor = torch.as_tensor(value, dtype=torch.long)
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        return ()
    return tuple(int(item) for item in tensor.tolist())


def _source_sentence_tokens(
    tokenizer,
    source_output: str,
    sentence_specs: Sequence[Tuple[str, int, int]],
) -> Tuple[int, List[Tuple[int, ...]]]:
    """Assign source-output tokenizer tokens to exactly one sentence.

    The ratio in the method PDF is defined over the source agent's output,
    rather than over prompt/header tokens. Retokenizing the exact visible
    source output with offsets gives a source-local denominator and lets
    sentence tokens be deduplicated even when a tokenizer token crosses a
    sentence boundary.
    """

    source_ids, offsets = _token_offsets(tokenizer, source_output)
    assigned: List[List[int]] = [[] for _ in sentence_specs]
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        best_index = None
        best_overlap = 0
        for sentence_index, (_, sentence_start, sentence_end) in enumerate(
            sentence_specs
        ):
            overlap = min(token_end, sentence_end) - max(token_start, sentence_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = sentence_index
        if best_index is not None and best_overlap > 0:
            assigned[best_index].append(token_index)
    return len(source_ids), [tuple(indices) for indices in assigned]


def build_incoming_sentence_masks(
    tokenizer,
    event: Mapping[str, Any],
    *,
    enable_thinking: bool = False,
    prefer_recorded_prompt: bool = True,
) -> Tuple[str, List[int], np.ndarray, List[IncomingSentenceSpan], List[IncomingEdge]]:
    """Build non-overlapping masks only for incoming agent-output sentences.

    Tokens crossing a sentence boundary are assigned to the sentence with the
    largest character overlap.  This guarantees that candidate masks are
    disjoint and prevents a token from being credited to two edges/sentences.
    """

    messages = event.get("input_messages") or []
    incoming_items = event.get("incoming") or []
    if not messages:
        raise ValueError("Event has no input_messages.")
    if not incoming_items:
        raise ValueError("Event has no incoming agent outputs.")

    rendered = str(event.get("rendered_prompt") or "") if prefer_recorded_prompt else ""
    if not rendered:
        rendered = _render_messages(tokenizer, messages, enable_thinking)
    input_ids, offsets = _token_offsets(tokenizer, rendered)
    target_agent = str(event.get("agent_id", ""))
    target_round = int(event.get("round", 0))

    edges: List[IncomingEdge] = []
    sentence_specs: List[Dict[str, Any]] = []
    nonempty_incoming = [
        (incoming_index, incoming, _incoming_output(incoming))
        for incoming_index, incoming in enumerate(incoming_items)
        if _incoming_output(incoming)
    ]
    source_locations = _locate_incoming_outputs(
        rendered,
        messages,
        [source_output for _, _, source_output in nonempty_incoming],
    )
    for (incoming_index, incoming, source_output), (source_start, source_end) in zip(
        nonempty_incoming, source_locations
    ):
        edge_type = str(incoming.get("edge_type", "spatial"))
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unsupported edge_type {edge_type!r}.")
        source_round = int(incoming.get("round", target_round if edge_type == "spatial" else target_round - 1))
        local_sentences = [
            span
            for span in split_sentence_spans(source_output)
            if word_count(span[0]) > 0
        ]
        source_token_count, local_sentence_tokens = _source_sentence_tokens(
            tokenizer,
            source_output,
            local_sentences,
        )
        source_generated_ids = _flatten_token_ids(
            incoming.get("source_generated_ids")
        )
        edge = IncomingEdge(
            incoming_index=incoming_index,
            source_agent=str(incoming.get("agent_id", "")),
            target_agent=target_agent,
            edge_type=edge_type,
            source_round=source_round,
            target_round=target_round,
            source_event_id=(
                int(incoming["event_id"]) if incoming.get("event_id") is not None else None
            ),
            source_output=source_output,
            source_word_count=word_count(source_output),
            source_token_count=source_token_count,
            source_generated_ids=source_generated_ids,
            source_generated_token_count=len(source_generated_ids),
        )
        edges.append(edge)

        for (
            sentence_text,
            local_start,
            local_end,
        ), source_token_indices in zip(local_sentences, local_sentence_tokens):
            count = word_count(sentence_text)
            sentence_specs.append(
                {
                    "edge": edge,
                    "text": sentence_text,
                    "source_char_start": local_start,
                    "source_char_end": local_end,
                    "source_token_indices": source_token_indices,
                    "char_start": source_start + local_start,
                    "char_end": source_start + local_end,
                    "word_count": count,
                }
            )

    if not sentence_specs:
        raise ValueError("Incoming outputs contain no attributable sentence text.")

    # Assign each tokenizer token to at most one sentence by maximum overlap.
    sentence_token_indices: List[List[int]] = [[] for _ in sentence_specs]
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        best_index = None
        best_overlap = 0
        for sentence_index, spec in enumerate(sentence_specs):
            overlap = min(token_end, spec["char_end"]) - max(token_start, spec["char_start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = sentence_index
        if best_index is not None and best_overlap > 0:
            sentence_token_indices[best_index].append(token_index)

    spans: List[IncomingSentenceSpan] = []
    masks: List[np.ndarray] = []
    for spec, token_indices in zip(sentence_specs, sentence_token_indices):
        if not token_indices:
            continue
        edge: IncomingEdge = spec["edge"]
        mask = np.zeros(len(input_ids), dtype=np.float32)
        mask[token_indices] = 1.0
        span = IncomingSentenceSpan(
            index=len(spans),
            incoming_index=edge.incoming_index,
            source_agent=edge.source_agent,
            target_agent=edge.target_agent,
            edge_type=edge.edge_type,
            source_round=edge.source_round,
            target_round=edge.target_round,
            source_event_id=edge.source_event_id,
            text=spec["text"],
            char_start=spec["char_start"],
            char_end=spec["char_end"],
            token_indices=tuple(token_indices),
            word_count=spec["word_count"],
            source_word_count=edge.source_word_count,
            source_char_start=spec["source_char_start"],
            source_char_end=spec["source_char_end"],
            source_token_indices=tuple(spec["source_token_indices"]),
            source_token_count=edge.source_token_count,
        )
        spans.append(span)
        masks.append(mask)

    if not masks:
        raise ValueError("Incoming sentence spans did not align to tokenizer offsets.")
    return rendered, input_ids, np.stack(masks), spans, edges


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class GreedyFullScoreAttributor:
    """Greedily add sentences until the target score matches the full prompt.

    ``scorer`` must expose ``score_teacher_forced_targets`` with the API of
    :class:`TeacherForcedTextScorer`. Candidate prompt variants are evaluated in
    batches by passing an attention mask; zeroed sentence tokens are compacted
    (actually deleted) by that scoring API before teacher forcing.
    """

    def __init__(
        self,
        scorer,
        *,
        score_type: str = "logit",
        reduction: str = "sum",
        target_recovery: float = 0.95,
        absolute_atol: float = 1e-4,
        relative_rtol: float = 1e-3,
        patience: int = 1,
        batch_size: int = 4,
        max_selected_sentences: Optional[int] = None,
    ) -> None:
        if score_type not in {"logit", "logprob", "probability"}:
            raise ValueError("score_type must be logit, logprob, or probability")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be sum or mean")
        if not 0.0 < target_recovery <= 1.0:
            raise ValueError("target_recovery must be in (0, 1]")
        if not math.isfinite(absolute_atol) or not math.isfinite(relative_rtol):
            raise ValueError("absolute_atol and relative_rtol must be finite")
        if absolute_atol < 0 or relative_rtol < 0:
            raise ValueError("absolute_atol and relative_rtol must be non-negative")
        if patience < 1 or batch_size < 1:
            raise ValueError("patience and batch_size must be positive")
        if max_selected_sentences is not None and max_selected_sentences < 1:
            raise ValueError("max_selected_sentences must be positive when provided")
        self.scorer = scorer
        self.score_type = score_type
        self.reduction = reduction
        self.target_recovery = float(target_recovery)
        self.absolute_atol = float(absolute_atol)
        self.relative_rtol = float(relative_rtol)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.max_selected_sentences = max_selected_sentences

    def _score_masks(
        self,
        input_ids: torch.Tensor,
        visible_masks: torch.Tensor,
        generated_ids: torch.Tensor,
        selected_target_indices: Sequence[int],
        *,
        diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        chunks: Dict[str, List[torch.Tensor]] = {}
        for start in range(0, visible_masks.shape[0], self.batch_size):
            end = min(start + self.batch_size, visible_masks.shape[0])
            rows = input_ids.unsqueeze(0).expand(end - start, -1)
            result = self.scorer.score_teacher_forced_targets(
                rows,
                generated_ids=generated_ids,
                selected_output_token_indices=selected_target_indices,
                attention_mask=visible_masks[start:end].to(torch.long),
                score_type=self.score_type,
                reduction=self.reduction,
            )
            keys = ["score"]
            if diagnostics:
                keys.extend(["logit_sum", "logit_mean", "logprob_sum", "logprob_mean"])
            for key in keys:
                chunks.setdefault(key, []).append(result[key].detach().to(torch.float32).cpu())
        payload = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
        for key, values in payload.items():
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"Model produced non-finite attribution values for {key}.")
        return payload

    def __call__(
        self,
        source_input_ids: Sequence[int],
        sentence_masks: np.ndarray,
        *,
        generated_ids: Sequence[int],
        selected_target_indices: Sequence[int],
    ) -> Dict[str, Any]:
        input_ids = torch.as_tensor(source_input_ids, dtype=torch.long)
        masks = torch.as_tensor(sentence_masks, dtype=torch.bool)
        target_ids = torch.as_tensor(generated_ids, dtype=torch.long)
        if masks.ndim != 2 or masks.shape[1] != input_ids.numel():
            raise ValueError("sentence_masks must have shape [sentence, prompt_token]")
        if masks.shape[0] == 0:
            raise ValueError("At least one sentence mask is required")
        if not selected_target_indices:
            raise ValueError("At least one target token must be selected")

        explainable = masks.any(dim=0)
        fixed = ~explainable
        baseline_and_full = torch.stack([fixed, torch.ones_like(fixed)])
        endpoint_scores = self._score_masks(
            input_ids,
            baseline_and_full,
            target_ids,
            selected_target_indices,
            diagnostics=True,
        )
        baseline_score = float(endpoint_scores["score"][0])
        full_score = float(endpoint_scores["score"][1])
        scale = abs(full_score - baseline_score)
        absolute_tolerance = self.absolute_atol + self.relative_rtol * abs(full_score)

        result: Dict[str, Any] = {
            "score_type": self.score_type,
            "reduction": self.reduction,
            "target_recovery": self.target_recovery,
            "absolute_atol": self.absolute_atol,
            "relative_rtol": self.relative_rtol,
            "patience": self.patience,
            "baseline_score": baseline_score,
            "full_score": full_score,
            "score_scale": scale,
            "baseline_diagnostics": {
                key: float(endpoint_scores[key][0])
                for key in ("logit_sum", "logit_mean", "logprob_sum", "logprob_mean")
            },
            "full_diagnostics": {
                key: float(endpoint_scores[key][1])
                for key in ("logit_sum", "logit_mean", "logprob_sum", "logprob_mean")
            },
            "selected_region_indices": [],
            "curve": [],
            "stop_reason": None,
            "verification_steps": 0,
        }

        if scale <= absolute_tolerance:
            result["stop_reason"] = "fixed_prompt_already_matches_full"
            result["stop_step"] = 0
            result["selected_score"] = baseline_score
            result["selected_recovery"] = 1.0
            return result

        selected: List[int] = []
        selected_visible = torch.zeros_like(explainable)
        remaining = list(range(masks.shape[0]))
        close_streak = 0
        first_close_prefix: Optional[List[int]] = None
        first_close_record: Optional[Dict[str, Any]] = None
        max_steps = len(remaining)
        if self.max_selected_sentences is not None:
            max_steps = min(max_steps, int(self.max_selected_sentences))

        for step in range(max_steps):
            candidate_visible = torch.stack(
                [fixed | selected_visible | masks[index] for index in remaining]
            )
            candidate_payload = self._score_masks(
                input_ids,
                candidate_visible,
                target_ids,
                selected_target_indices,
                diagnostics=True,
            )
            candidate_scores = candidate_payload["score"]
            gaps = torch.abs(candidate_scores - full_score)
            local_index = int(torch.argmin(gaps).item())
            region_index = remaining.pop(local_index)
            chosen_score = float(candidate_scores[local_index])
            selected.append(region_index)
            selected_visible |= masks[region_index]

            gap = abs(chosen_score - full_score)
            recovery = float(np.clip(1.0 - gap / max(scale, 1e-12), 0.0, 1.0))
            normalized_close = recovery >= self.target_recovery
            absolute_close = gap <= absolute_tolerance
            close = normalized_close or absolute_close
            record = {
                "step": step + 1,
                "selected_region_index": region_index,
                "score": chosen_score,
                "absolute_gap_to_full": gap,
                "recovery": recovery,
                "normalized_close": normalized_close,
                "absolute_close": absolute_close,
                "selected_sentence_fraction": len(selected) / masks.shape[0],
                "selected_token_fraction": float(
                    (selected_visible & explainable).sum().item() / max(1, explainable.sum().item())
                ),
                "diagnostics": {
                    key: float(candidate_payload[key][local_index])
                    for key in ("logit_sum", "logit_mean", "logprob_sum", "logprob_mean")
                },
            }
            result["curve"].append(record)

            if close:
                close_streak += 1
                if first_close_prefix is None:
                    first_close_prefix = list(selected)
                    first_close_record = dict(record)
                if close_streak >= self.patience:
                    result["stop_reason"] = "near_full_score"
                    result["verification_steps"] = self.patience - 1
                    break
            else:
                close_streak = 0
                first_close_prefix = None
                first_close_record = None

        if result["stop_reason"] == "near_full_score" and first_close_prefix is not None:
            selected = first_close_prefix
            selected_record = first_close_record
        else:
            selected_record = result["curve"][-1] if result["curve"] else None
            if len(selected) == len(masks):
                result["stop_reason"] = "all_sentences_selected"
            else:
                result["stop_reason"] = "max_selected_sentences"

        result["selected_region_indices"] = selected
        result["stop_step"] = len(selected)
        result["selected_score"] = (
            float(selected_record["score"]) if selected_record is not None else baseline_score
        )
        result["selected_recovery"] = (
            float(selected_record["recovery"]) if selected_record is not None else 0.0
        )
        return result


def apply_early_stop_sentence_saliency(
    result: MutableMapping[str, Any],
    *,
    saliency_floor: float = 0.4,
) -> Dict[int, float]:
    """Backfill reference-style cumulative-difference sentence saliency.

    Only the final selected prefix participates.  Later curve records produced
    solely for patience verification retain explicit ``None`` saliency fields.
    The first selected sentence is normalized to 1.0 and the final one to
    ``saliency_floor``.
    """

    floor = float(saliency_floor)
    if not math.isfinite(floor) or not 0.0 <= floor < 1.0:
        raise ValueError("saliency_floor must be finite and in [0, 1)")

    attribution = result.setdefault("attribution", {})
    if not isinstance(attribution, MutableMapping):
        raise TypeError("result.attribution must be a mutable mapping")
    curve = list(attribution.get("curve") or [])
    selected_ids = [
        int(index) for index in attribution.get("selected_region_indices") or []
    ]

    for record in curve:
        record["absolute_score_delta"] = None
        record["cumulative_absolute_score_delta"] = None
        record["saliency"] = None
    for sentence in result.get("sentences") or []:
        sentence["selection_step"] = None
        sentence["saliency"] = None

    configuration: Dict[str, Any] = {
        "method": "cumulative_absolute_consecutive_score_delta",
        "floor": floor,
        "ceiling": 1.0,
        "selected_count": len(selected_ids),
        "uses_final_selected_prefix_only": True,
        "normalization": "empty",
        "degenerate_fallback_used": False,
    }
    if not selected_ids:
        attribution["saliency_configuration"] = configuration
        return {}

    baseline = attribution.get("baseline_score")
    if baseline is None or not math.isfinite(float(baseline)):
        raise ValueError(
            "A finite baseline_score is required for selected-sentence saliency"
        )
    record_by_region = {
        int(record["selected_region_index"]): record
        for record in curve
        if record.get("selected_region_index") is not None
    }
    missing = [index for index in selected_ids if index not in record_by_region]
    if missing:
        raise ValueError(
            "Selected sentence indices are missing from the attribution curve: "
            f"{missing}"
        )

    previous_score = float(baseline)
    cumulative = 0.0
    selected_records: List[MutableMapping[str, Any]] = []
    cumulative_values: List[float] = []
    delta_values: List[float] = []
    for region_index in selected_ids:
        record = record_by_region[region_index]
        score = float(record["score"])
        if not math.isfinite(score):
            raise ValueError("Attribution curve contains a non-finite score")
        delta = abs(score - previous_score)
        cumulative += delta
        selected_records.append(record)
        delta_values.append(delta)
        cumulative_values.append(cumulative)
        previous_score = score

    selected_count = len(selected_ids)
    if selected_count == 1:
        saliencies = [1.0]
        normalization = "single_selected"
    else:
        denominator = cumulative_values[-1] - cumulative_values[0]
        tolerance = 1e-12 * max(
            1.0,
            abs(cumulative_values[-1]),
            abs(cumulative_values[0]),
        )
        if denominator > tolerance:
            saliencies = [
                floor
                + (1.0 - floor)
                * (cumulative_values[-1] - value)
                / denominator
                for value in cumulative_values
            ]
            normalization = "selected_cumulative_minmax"
        else:
            saliencies = [
                floor
                + (1.0 - floor)
                * (selected_count - step)
                / (selected_count - 1)
                for step in range(1, selected_count + 1)
            ]
            normalization = "rank_linear_fallback"
            configuration["degenerate_fallback_used"] = True

    saliency_by_region: Dict[int, float] = {}
    for step, (
        region_index,
        record,
        delta,
        cumulative_delta,
        saliency,
    ) in enumerate(
        zip(
            selected_ids,
            selected_records,
            delta_values,
            cumulative_values,
            saliencies,
        ),
        start=1,
    ):
        saliency = float(np.clip(saliency, floor, 1.0))
        record["absolute_score_delta"] = float(delta)
        record["cumulative_absolute_score_delta"] = float(cumulative_delta)
        record["saliency"] = saliency
        saliency_by_region[region_index] = saliency

    selected_steps = {
        region_index: step
        for step, region_index in enumerate(selected_ids, start=1)
    }
    for sentence in result.get("sentences") or []:
        region_index = int(sentence.get("index", -1))
        is_selected = region_index in saliency_by_region
        sentence["selected"] = is_selected
        sentence["selection_step"] = selected_steps.get(region_index)
        sentence["saliency"] = saliency_by_region.get(region_index)

    configuration["normalization"] = normalization
    attribution["saliency_configuration"] = configuration
    return saliency_by_region


def compute_event_edge_scores(
    spans: Sequence[IncomingSentenceSpan],
    edges: Sequence[IncomingEdge],
    selected_region_indices: Sequence[int],
    *,
    min_selected_sentences: int = 2,
    min_sentence_words: int = 4,
) -> List[Dict[str, Any]]:
    """Compute the two user-requested importance scores for every observed edge."""

    if min_selected_sentences < 1 or min_sentence_words < 1:
        raise ValueError("min_selected_sentences and min_sentence_words must be positive")

    selected_ids = {int(index) for index in selected_region_indices}
    by_incoming: Dict[int, List[IncomingSentenceSpan]] = {}
    for span in spans:
        by_incoming.setdefault(span.incoming_index, []).append(span)

    scores = []
    for edge in edges:
        source_spans = by_incoming.get(edge.incoming_index, [])
        selected_spans = [span for span in source_spans if span.index in selected_ids]
        substantive = [span for span in selected_spans if span.word_count >= min_sentence_words]
        selected_words = sum(span.word_count for span in selected_spans)
        word_denominator = max(1, edge.source_word_count)
        selected_source_tokens = {
            token_index
            for span in selected_spans
            for token_index in span.source_token_indices
        }
        token_denominator = max(1, int(edge.source_token_count))
        scores.append(
            {
                **edge.to_dict(),
                "opportunity": 1,
                "source_sentence_count": len(source_spans),
                "selected_sentence_count": len(selected_spans),
                "selected_substantive_sentence_count": len(substantive),
                "selected_region_indices": [span.index for span in selected_spans],
                "selected_words": selected_words,
                "selected_token_count": len(selected_source_tokens),
                "source_token_count": int(edge.source_token_count),
                "source_generated_token_count": int(
                    edge.source_generated_token_count
                ),
                "binary": int(len(substantive) >= min_selected_sentences),
                "word_ratio": min(1.0, selected_words / word_denominator),
                "token_ratio": min(
                    1.0, len(selected_source_tokens) / token_denominator
                ),
                "min_selected_sentences": min_selected_sentences,
                "min_sentence_words": min_sentence_words,
            }
        )
    return scores


def trace_is_correct(trace: Mapping[str, Any]) -> bool:
    final = trace.get("final") or {}
    if "correct" in final:
        return bool(final["correct"])
    parsed = final.get("parsed_answer")
    gold = final.get("gold_answer")
    return str(parsed).strip().upper() == str(gold).strip().upper() and gold is not None


def target_ids_from_event(event: Mapping[str, Any], tokenizer) -> Tuple[List[int], str]:
    """Prefer exact trace token ids and fall back to re-tokenizing output text."""

    generated = event.get("generated_ids")
    if generated is not None:
        tensor = torch.as_tensor(generated, dtype=torch.long)
        while tensor.ndim > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 1 and tensor.numel() > 0:
            return [int(value) for value in tensor.tolist()], "trace_generated_ids"

    output = str(event.get("output", "") or "")
    encoded = tokenizer(output, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    ids = [int(value) for value in encoded]
    if not ids:
        raise ValueError("Event output produced no target token ids.")
    return ids, "retokenized_output"


def select_target_token_indices(
    tokenizer,
    generated_ids: Sequence[int],
    *,
    mode: str = "decision",
    target_token_limit: Optional[int] = None,
) -> List[int]:
    """Select all output tokens or the final standalone MMLU A/B/C/D token."""

    if mode not in {"decision", "all"}:
        raise ValueError("target mode must be decision or all")
    if target_token_limit is not None and target_token_limit < 1:
        raise ValueError("target_token_limit must be positive when provided")
    if mode == "all":
        indices = list(range(len(generated_ids)))
        if target_token_limit is not None:
            indices = indices[: int(target_token_limit)]
        if not indices:
            raise ValueError("No output target tokens were selected.")
        return indices

    decoded = [
        tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for token_id in generated_ids
    ]
    for index in range(len(decoded) - 1, -1, -1):
        if re.fullmatch(r"[ABCD]", decoded[index].strip(), flags=re.IGNORECASE):
            return [index]
    raise ValueError("Could not find a standalone A/B/C/D decision token in event output.")


def infer_worker_metadata(trace: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    workers = (trace.get("topology") or {}).get("workers") or []
    if not workers:
        workers = (trace.get("config") or {}).get("workers") or []
    workers = list(workers)
    if workers and all(item.get("node_index") is not None for item in workers):
        node_indices = [int(item["node_index"]) for item in workers]
        if len(set(node_indices)) != len(node_indices):
            raise ValueError("Trace worker node_index values must be unique.")
        workers = [
            item
            for _, item in sorted(
                zip(node_indices, workers), key=lambda pair: pair[0]
            )
        ]
    return workers


def infer_worker_agents(trace: Mapping[str, Any]) -> List[str]:
    workers = infer_worker_metadata(trace)
    if workers:
        return [str(item["agent_id"]) for item in workers]

    num_rounds = infer_num_rounds(trace)
    agents = []
    for event in trace.get("events") or []:
        if int(event.get("round", 0)) >= num_rounds:
            continue
        agent = str(event.get("agent_id", ""))
        if agent and agent not in agents:
            agents.append(agent)
    return agents


def infer_num_rounds(trace: Mapping[str, Any]) -> int:
    config = trace.get("config") or {}
    rounds = config.get("rounds")
    num_rounds = config.get("num_rounds")
    if rounds is not None and num_rounds is not None and int(rounds) != int(num_rounds):
        raise ValueError("Trace config rounds and num_rounds disagree.")
    value = rounds if rounds is not None else num_rounds
    value = 1 if value is None else int(value)
    if value < 1:
        raise ValueError("Trace must configure at least one round.")
    return value


def empty_sample_matrices(num_rounds: int, num_agents: int) -> Dict[str, Dict[str, List]]:
    def zeros():
        return np.zeros((num_rounds, num_agents, num_agents), dtype=np.float64)

    return {
        metric: {edge_type: zeros().tolist() for edge_type in EDGE_TYPES}
        for metric in (*METRICS, "opportunity", "candidate")
    }


def build_sample_matrices(
    trace: Mapping[str, Any],
    event_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    trace_schema = trace.get("schema_version")
    if trace_schema is not None and int(trace_schema) != 1:
        raise ValueError(f"Unsupported trace schema_version {trace_schema!r}.")
    agents = infer_worker_agents(trace)
    if not agents:
        raise ValueError("Could not infer worker agent order from trace.")
    agent_index = {agent: index for index, agent in enumerate(agents)}
    topology_workers = infer_worker_metadata(trace)
    roles_by_agent = {
        str(item.get("agent_id", "")): str(item.get("role", ""))
        for item in topology_workers
    }
    types_by_agent = {
        str(item.get("agent_id", "")): str(item.get("agent_type", ""))
        for item in topology_workers
    }
    for event in trace.get("events") or []:
        agent_id = str(event.get("agent_id", ""))
        if (
            agent_id in agent_index
            and event.get("agent_type")
            and not types_by_agent.get(agent_id)
        ):
            types_by_agent[agent_id] = str(event["agent_type"])
    agent_roles = [roles_by_agent.get(agent, "") for agent in agents]
    agent_types = [types_by_agent.get(agent, "") for agent in agents]
    num_rounds = infer_num_rounds(trace)
    matrices = empty_sample_matrices(num_rounds, len(agents))
    shared_event_configurations = []
    for event_result in event_results:
        configuration = dict(event_result.get("configuration") or {})
        configuration.pop("event_sha256", None)
        if configuration and configuration not in shared_event_configurations:
            shared_event_configurations.append(configuration)
    if len(shared_event_configurations) > 1:
        raise ValueError("All event attributions in one sample must use the same configuration.")
    attribution_configuration = (
        shared_event_configurations[0] if shared_event_configurations else {}
    )

    # Candidate topology is inferred from every recorded worker input, including
    # receivers skipped by --min-incoming. This prevents an "unobserved=keep"
    # policy from inventing edges that were never part of the dense baseline.
    for event in trace.get("events") or []:
        target = str(event.get("agent_id", ""))
        round_index = int(event.get("round", 0))
        if target not in agent_index or not 0 <= round_index < num_rounds:
            continue
        for incoming in event.get("incoming") or []:
            edge_type = str(incoming.get("edge_type", "spatial"))
            source = str(incoming.get("agent_id", ""))
            if edge_type not in EDGE_TYPES or source not in agent_index:
                continue
            if edge_type == "temporal" and round_index == 0:
                continue
            matrices["candidate"][edge_type][round_index][agent_index[source]][
                agent_index[target]
            ] = 1.0

    for event_result in event_results:
        for edge in event_result.get("edge_scores", []):
            edge_type = edge["edge_type"]
            source = edge.get("source_agent")
            target = edge.get("target_agent")
            round_index = int(edge.get("target_round", 0))
            if edge_type not in EDGE_TYPES or source not in agent_index or target not in agent_index:
                continue
            if not 0 <= round_index < num_rounds:
                continue
            if edge_type == "temporal" and round_index == 0:
                continue
            i, j = agent_index[source], agent_index[target]
            matrices["opportunity"][edge_type][round_index][i][j] = 1.0
            for metric in METRICS:
                value = edge.get(metric)
                if value is None and metric == "token_ratio":
                    # Compatibility for older cached/debug event fixtures.
                    value = edge.get("word_ratio", 0.0)
                matrices[metric][edge_type][round_index][i][j] = max(
                    matrices[metric][edge_type][round_index][i][j], float(value)
                )

    sample = trace.get("sample") or {}
    sample_index = (trace.get("config") or {}).get("sample_index", sample.get("sample_index"))
    return {
        "schema_version": 1,
        "trace_schema_version": "legacy" if trace_schema is None else int(trace_schema),
        "sample_index": sample_index,
        "run_fingerprint": (trace.get("config") or {}).get("fingerprint"),
        "correct_verified": trace_is_correct(trace),
        "agents": agents,
        "agent_roles": agent_roles,
        "agent_types": agent_types,
        "attribution_configuration": attribution_configuration,
        "matrix_orientation": "[round][source][target]",
        "num_rounds": num_rounds,
        "matrices": matrices,
    }


def aggregate_sample_matrices(sample_matrices: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not sample_matrices:
        raise ValueError("No sample matrices were provided for aggregation.")
    verified = [item for item in sample_matrices if item.get("correct_verified")]
    if not verified:
        raise ValueError("No verified-correct sample matrices were provided.")
    agents = list(verified[0]["agents"])
    agent_roles = list(verified[0].get("agent_roles", [""] * len(agents)))
    agent_types = list(verified[0].get("agent_types", [""] * len(agents)))
    attribution_configuration = dict(
        verified[0].get("attribution_configuration") or {}
    )
    selection_configuration = dict(
        verified[0].get("selection_configuration") or {}
    )
    num_rounds = int(verified[0]["num_rounds"])
    trace_schema_version = verified[0].get("trace_schema_version", "legacy")
    run_fingerprint = verified[0].get("run_fingerprint")
    for item in verified[1:]:
        if list(item["agents"]) != agents or int(item["num_rounds"]) != num_rounds:
            raise ValueError("All sample matrices must use the same agent order and rounds.")
        item_roles = list(item.get("agent_roles", [""] * len(agents)))
        if item_roles != agent_roles:
            raise ValueError("All sample matrices must use the same worker roles.")
        item_types = list(item.get("agent_types", [""] * len(agents)))
        if item_types != agent_types:
            raise ValueError("All sample matrices must use the same worker agent types.")
        if dict(item.get("attribution_configuration") or {}) != attribution_configuration:
            raise ValueError(
                "All sample matrices must use the same attribution configuration."
            )
        if dict(item.get("selection_configuration") or {}) != selection_configuration:
            raise ValueError(
                "All sample matrices must use the same event selection configuration."
            )
        if item.get("trace_schema_version", "legacy") != trace_schema_version:
            raise ValueError(
                "Legacy and schema-v1 traces cannot be mixed in one calibration aggregate."
            )
        if item.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                "All sample matrices must come from the same run fingerprint."
            )

    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "trace_schema_version": trace_schema_version,
        "run_fingerprint": run_fingerprint,
        "num_correct_samples": len(verified),
        "agents": agents,
        "agent_roles": agent_roles,
        "agent_types": agent_types,
        "attribution_configuration": attribution_configuration,
        "selection_configuration": selection_configuration,
        "num_rounds": num_rounds,
        "matrix_orientation": "[round][source][target]",
        "metrics": {},
    }
    for metric in METRICS:
        aggregate["metrics"][metric] = {}
        for edge_type in EDGE_TYPES:
            values = np.asarray(
                [item["matrices"][metric][edge_type] for item in verified], dtype=np.float64
            )
            opportunities = np.asarray(
                [item["matrices"]["opportunity"][edge_type] for item in verified],
                dtype=np.float64,
            )
            candidates = np.asarray(
                [
                    item["matrices"].get("candidate", item["matrices"]["opportunity"])[
                        edge_type
                    ]
                    for item in verified
                ],
                dtype=np.float64,
            )
            value_sum = values.sum(axis=0)
            opportunity_count = opportunities.sum(axis=0)
            conditional = np.divide(
                value_sum,
                opportunity_count,
                out=np.zeros_like(value_sum),
                where=opportunity_count > 0,
            )
            aggregate["metrics"][metric][edge_type] = {
                "sample_mean": values.mean(axis=0).tolist(),
                "conditional_mean": conditional.tolist(),
                "sum": value_sum.tolist(),
                "opportunity_count": opportunity_count.tolist(),
                "candidate_count": candidates.sum(axis=0).tolist(),
            }
    return aggregate


def build_pruned_masks(
    aggregate: Mapping[str, Any],
    *,
    metric: str = "binary",
    statistic: str = "conditional_mean",
    threshold: Optional[float] = None,
    pruning_rate: Optional[float] = 0.25,
    unobserved: str = "keep",
    pruning_scope: str = "per_round_edge_type",
) -> Dict[str, Any]:
    """Convert aggregate importance into deterministic round masks.

    ``per_round_edge_type`` preserves the original allocation rule: the
    requested rate is rounded and applied independently to every round of
    every edge type. ``global`` flattens all observed spatial and temporal
    matrices and allocates the pruning budget once across the whole graph.
    Threshold-based pruning is inherently global and is unchanged by this
    option.
    """

    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if statistic not in {"sample_mean", "conditional_mean"}:
        raise ValueError("statistic must be sample_mean or conditional_mean")
    if (threshold is None) == (pruning_rate is None):
        raise ValueError("Provide exactly one of threshold or pruning_rate")
    if unobserved not in {"keep", "drop"}:
        raise ValueError("unobserved must be keep or drop")
    if pruning_scope not in {"per_round_edge_type", "global"}:
        raise ValueError(
            "pruning_scope must be per_round_edge_type or global"
        )
    if threshold is not None and not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if pruning_rate is not None and (
        not math.isfinite(float(pruning_rate)) or not 0 <= pruning_rate <= 1
    ):
        raise ValueError("pruning_rate must be in [0, 1]")

    agents = list(aggregate["agents"])
    expected_shape = (int(aggregate["num_rounds"]), len(agents), len(agents))
    arrays: Dict[str, Dict[str, np.ndarray]] = {}

    def project_spatial_dag(
        requested: np.ndarray,
        importance: np.ndarray,
        observed: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Greedily retain high-evidence edges while making each round acyclic."""

        effective = np.zeros_like(requested, dtype=bool)
        dropped = 0
        for round_index in range(requested.shape[0]):
            adjacency = effective[round_index]
            positions = np.argwhere(requested[round_index])
            ranked = sorted(
                positions.tolist(),
                key=lambda pos: (
                    -int(observed[round_index, pos[0], pos[1]]),
                    -float(importance[round_index, pos[0], pos[1]]),
                    pos[0],
                    pos[1],
                ),
            )
            for source, target in ranked:
                if source == target:
                    dropped += 1
                    continue
                # Adding source -> target is safe iff target cannot already
                # reach source in the partially constructed graph.
                stack = [target]
                seen = set()
                closes_cycle = False
                while stack:
                    node = stack.pop()
                    if node == source:
                        closes_cycle = True
                        break
                    if node in seen:
                        continue
                    seen.add(node)
                    stack.extend(np.flatnonzero(adjacency[node]).tolist())
                if closes_cycle:
                    dropped += 1
                else:
                    adjacency[source, target] = True
        return effective, dropped

    for edge_type in EDGE_TYPES:
        payload = aggregate["metrics"][metric][edge_type]
        importance = np.asarray(payload[statistic], dtype=np.float64)
        opportunities = np.asarray(payload["opportunity_count"], dtype=np.float64)
        candidate_counts = np.asarray(
            payload.get("candidate_count", opportunities), dtype=np.float64
        )
        if (
            importance.shape != expected_shape
            or opportunities.shape != expected_shape
            or candidate_counts.shape != expected_shape
        ):
            raise ValueError(
                f"Aggregate {edge_type} tensors must have shape {expected_shape}."
            )
        if not (
            np.isfinite(importance).all()
            and np.isfinite(opportunities).all()
            and np.isfinite(candidate_counts).all()
        ):
            raise ValueError(f"Aggregate {edge_type} tensors must be finite.")
        if (opportunities < 0).any() or (candidate_counts < 0).any():
            raise ValueError(f"Aggregate {edge_type} counts must be non-negative.")
        candidates = candidate_counts > 0
        mask = candidates.copy() if unobserved == "keep" else np.zeros_like(importance, dtype=bool)
        observed = opportunities > 0
        if bool((observed & ~candidates).any()):
            raise ValueError("Every observed edge must belong to candidate topology.")
        arrays[edge_type] = {
            "importance": importance,
            "candidates": candidates,
            "observed": observed,
            "mask": mask,
        }

    if threshold is not None:
        for edge_type in EDGE_TYPES:
            edge_arrays = arrays[edge_type]
            edge_arrays["mask"][edge_arrays["observed"]] = (
                edge_arrays["importance"][edge_arrays["observed"]]
                >= float(threshold)
            )
    elif pruning_scope == "global":
        ranked_edges = []
        for edge_type_index, edge_type in enumerate(EDGE_TYPES):
            edge_arrays = arrays[edge_type]
            edge_arrays["mask"][edge_arrays["observed"]] = True
            for round_index, source, target in np.argwhere(
                edge_arrays["observed"]
            ).tolist():
                ranked_edges.append(
                    (
                        float(
                            edge_arrays["importance"][
                                round_index, source, target
                            ]
                        ),
                        edge_type_index,
                        round_index,
                        source,
                        target,
                        edge_type,
                    )
                )
        prune_count = int(round(len(ranked_edges) * float(pruning_rate)))
        if ranked_edges and pruning_rate > 0:
            prune_count = max(1, prune_count)
        prune_count = min(len(ranked_edges), prune_count)
        ranked_edges.sort(key=lambda item: item[:5])
        for _, _, round_index, source, target, edge_type in ranked_edges[
            :prune_count
        ]:
            arrays[edge_type]["mask"][round_index, source, target] = False
    else:
        for edge_type in EDGE_TYPES:
            edge_arrays = arrays[edge_type]
            importance = edge_arrays["importance"]
            observed = edge_arrays["observed"]
            mask = edge_arrays["mask"]
            for round_index in range(importance.shape[0]):
                positions = np.argwhere(observed[round_index])
                mask[round_index][observed[round_index]] = True
                prune_count = int(round(len(positions) * float(pruning_rate)))
                if positions.size and pruning_rate > 0:
                    prune_count = max(1, prune_count)
                prune_count = min(len(positions), prune_count)
                ranked = sorted(
                    positions.tolist(),
                    key=lambda pos: (
                        importance[round_index, pos[0], pos[1]],
                        pos[0],
                        pos[1],
                    ),
                )
                for source, target in ranked[:prune_count]:
                    mask[round_index, source, target] = False

    masks: Dict[str, Any] = {}
    retained = {}
    for edge_type in EDGE_TYPES:
        edge_arrays = arrays[edge_type]
        importance = edge_arrays["importance"]
        candidates = edge_arrays["candidates"]
        observed = edge_arrays["observed"]
        mask = edge_arrays["mask"]
        if edge_type == "spatial":
            for round_index in range(mask.shape[0]):
                np.fill_diagonal(mask[round_index], False)
            mask, dag_projection_dropped = project_spatial_dag(
                mask, importance, observed
            )
        else:
            dag_projection_dropped = 0
        if edge_type == "temporal":
            mask[0] = False
        masks[edge_type] = mask.astype(int).tolist()
        candidate_count = int(candidates.sum())
        retained_candidate_count = int((mask & candidates).sum())
        retained[edge_type] = {
            "observed_edges": int(observed.sum()),
            "retained_observed_edges": int((mask & observed).sum()),
            "candidate_edges": candidate_count,
            "retained_candidate_edges": retained_candidate_count,
            "effective_candidate_pruning_rate": (
                1.0 - retained_candidate_count / candidate_count
                if candidate_count
                else 0.0
            ),
            "dag_projection_dropped_edges": dag_projection_dropped,
        }
    return {
        "schema_version": 1,
        "run_fingerprint": aggregate.get("run_fingerprint"),
        "calibration_num_correct_samples": int(
            aggregate.get("num_correct_samples", 0)
        ),
        "agents": agents,
        "agent_roles": list(aggregate.get("agent_roles", [""] * len(agents))),
        "agent_types": list(aggregate.get("agent_types", [""] * len(agents))),
        "num_rounds": int(aggregate["num_rounds"]),
        "matrix_orientation": "[round][source][target]",
        "metric": metric,
        "statistic": statistic,
        "threshold": threshold,
        "pruning_rate": pruning_rate,
        "pruning_scope": (
            "observed_attributed_edges_global"
            if pruning_scope == "global"
            else "observed_attributed_edges_per_round_edge_type"
        ),
        "pruning_allocation": pruning_scope,
        "unobserved": unobserved,
        "masks": masks,
        "retention": retained,
    }


def load_json(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, value: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path
