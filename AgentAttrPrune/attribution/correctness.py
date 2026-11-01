"""Local gold-prefix interventions on a frozen MMLU or GSM8K worker event.

Only the receiving worker is replayed. Its upstream outputs and the source
trace are never changed, and the gold prefix is used only for scoring.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import re
from typing import Any, Callable, Mapping, Sequence

from AgentAttrPrune.attribution.edge import (
    _locate_incoming_outputs,
    build_incoming_sentence_masks,
)
from AgentAttrPrune.attribution.gold_prefix import GreedyPrefixLogitAttributor


DECISION_PARSER_VERSION = "explicit_decision_v1"
RESCUE_PROMPT_VERSION = "mmlu_answer_first_v1"
GSM8K_ANSWER_PARSER_VERSION = "explicit_gsm8k_answer_v1"
GSM8K_RESCUE_PROMPT_VERSION = "gsm8k_answer_first_v1"

# Commas must be thousands separators; arbitrary reasoning and nonfinite values
# are deliberately not accepted as numeric answers.
_DECIMAL_PATTERN = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
_NUMBER_PATTERN = rf"{_DECIMAL_PATTERN}(?:/{_DECIMAL_PATTERN})?"
_MARKDOWN_MARKERS = ("**", "__", "`", "*", "_")
_GSM8K_FIELD_PATTERN = re.compile(
    r"(?P<label_mark>\*\*|__|`|\*|_)?The\s+answer\s+is"
    r"(?(label_mark)(?P=label_mark))\s+"
    rf"(?P<number_mark>\*\*|__|`|\*|_)?(?P<number>{_NUMBER_PATTERN})"
    r"(?(number_mark)(?P=number_mark))\s*[.!]?",
    flags=re.IGNORECASE,
)


def normalize_numeric_answer(value: Any) -> str:
    """Canonicalize a finite numeric literal exactly, without float rounding."""

    text = str(value).strip()
    if re.fullmatch(_NUMBER_PATTERN, text) is None:
        raise ValueError("GSM8K answer must be a finite numeric literal")
    numerator, separator, denominator = text.replace(",", "").partition("/")
    try:
        number = Fraction(numerator)
        if separator:
            number /= Fraction(denominator)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("GSM8K answer must be a finite numeric literal") from error
    if number.denominator == 1:
        return str(number.numerator)
    # Finite decimals get their shortest exact spelling; recurring fractions
    # retain a reduced rational spelling rather than an arbitrary precision.
    divisor = number.denominator
    powers = []
    for prime in (2, 5):
        power = 0
        while divisor % prime == 0:
            divisor //= prime
            power += 1
        powers.append(power)
    if divisor != 1:
        return f"{number.numerator}/{number.denominator}"
    scale = max(powers)
    magnitude = abs(number.numerator) * (10 ** scale // number.denominator)
    digits = str(magnitude).zfill(scale + 1)
    decimal = (digits[:-scale] + "." + digits[-scale:]).rstrip("0").rstrip(".")
    return ("-" if number < 0 else "") + decimal


def normalize_gold_answer(value: Any, *, benchmark: str = "mmlu") -> str:
    if benchmark == "gsm8k":
        return normalize_numeric_answer(value)
    if benchmark != "mmlu":
        raise ValueError(f"Unsupported correctness-aware benchmark: {benchmark!r}")
    gold = str(value).strip().upper()
    if gold not in {"A", "B", "C", "D"}:
        raise ValueError("MMLU gold answer must be one of A, B, C, D")
    return gold


def parse_worker_answer(text: str, *, benchmark: str = "mmlu") -> str | None:
    """Parse only the explicit answer field required by the worker prompt."""

    if benchmark == "mmlu":
        return parse_worker_decision(text)
    if benchmark != "gsm8k":
        raise ValueError(f"Unsupported correctness-aware benchmark: {benchmark!r}")
    answers: list[str] = []
    for line in str(text).splitlines():
        normalized = line.strip()
        if not re.match(r"^[*_`]*\s*The\s+answer\s+is(?=$|[^A-Za-z0-9])", normalized, flags=re.IGNORECASE):
            continue
        # Accept Markdown around the whole field, label, or numeric value;
        # never erase operators inside a value (e.g. 2*3 must not become 23).
        variants = [normalized]
        checked: set[str] = set()
        match = None
        while variants:
            candidate = variants.pop()
            if candidate in checked:
                continue
            checked.add(candidate)
            match = _GSM8K_FIELD_PATTERN.fullmatch(candidate)
            if match is not None:
                break
            for marker in _MARKDOWN_MARKERS:
                if (candidate.startswith(marker) and candidate.endswith(marker)
                        and len(candidate) > 2 * len(marker)):
                    variants.append(candidate[len(marker):-len(marker)].strip())
        if match is None:
            return None
        try:
            answers.append(normalize_numeric_answer(match.group("number")))
        except ValueError:
            return None
    return answers[0] if answers and len(set(answers)) == 1 else None


def parse_worker_decision(text: str) -> str | None:
    """Read explicit Decision fields, never infer an answer from reasoning.

    Whitespace, case, and Markdown emphasis are accepted. Repeated identical
    fields are unambiguous; conflicting or malformed fields are invalid.
    """

    answers: list[str] = []
    for line in str(text).splitlines():
        normalized = re.sub(r"[*_`]", "", line).strip()
        if not re.match(r"^Decision\s*:", normalized, flags=re.IGNORECASE):
            continue
        match = re.fullmatch(r"Decision\s*:\s*([A-D])\s*[.!]?", normalized, flags=re.IGNORECASE)
        if match is None:
            return None
        answers.append(match.group(1).upper())
    return answers[0] if answers and len(set(answers)) == 1 else None


def answer_first_system_prompt(original: str, *, benchmark: str = "mmlu") -> str:
    if benchmark == "gsm8k":
        old_format = (
            "Keep the reasoning concise. End with the result in exactly this form:\n"
            "The answer is <number>\n"
            "Do not put units or explanatory text after the final number."
        )
        new_format = (
            "First write the result in exactly this form:\n"
            "The answer is <number>\n"
            "Then write concise reasoning on the following lines.\n"
            "Do not put units or explanatory text on the answer line."
        )
        if old_format not in original:
            raise ValueError("Source worker system prompt does not match the expected GSM8K contract")
        return original.replace(old_format, new_format)
    if benchmark != "mmlu":
        raise ValueError(f"Unsupported correctness-aware benchmark: {benchmark!r}")
    old_order = (
        "First write a concise role-specific reasoning analysis. "
        "Then write your final choice."
    )
    new_order = (
        "First write your final choice. Then write a concise role-specific "
        "reasoning analysis."
    )
    old_format = (
        "Use exactly this format:\n"
        "Reasoning: <your brief analysis>\n"
        "Decision: <one letter: A, B, C, or D>"
    )
    new_format = (
        "Use exactly this format:\n"
        "Decision: <one letter: A, B, C, or D>\n"
        "Reasoning: <your brief analysis>"
    )
    if old_order not in original or old_format not in original:
        raise ValueError("Source worker system prompt does not match the expected MMLU contract")
    return original.replace(old_order, new_order).replace(old_format, new_format)


def _render_messages(tokenizer, messages: Sequence[Mapping[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )


def _flatten_ids(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def gold_prefix_target(tokenizer, gold: str, *, benchmark: str = "mmlu") -> dict[str, Any]:
    """Locate answer-bearing tokens without assuming the answer is one token."""

    gold = normalize_gold_answer(gold, benchmark=benchmark)
    prefix_format = "The answer is <gold>" if benchmark == "gsm8k" else "Decision: <gold>"
    prefix = prefix_format.replace("<gold>", gold)
    encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
    ids = _flatten_ids(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    answer_start = len(prefix) - len(gold)
    answer_indices = [
        index for index, (start, end) in enumerate(offsets)
        if int(end) > answer_start and int(start) < len(prefix)
    ]
    if not ids or len(offsets) != len(ids) or not answer_indices:
        raise ValueError("Gold answer prefix did not align to tokenizer offsets")
    if answer_indices != list(range(answer_indices[0], len(ids))):
        raise ValueError("Gold answer must occupy a contiguous target-token suffix")
    return {
        "text": prefix,
        "generated_ids": ids,
        "selected_indices": list(range(len(ids))),
        "answer_token_indices": answer_indices,
        "selected_ids": ids,
        "selected_tokens": [
            tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for token_id in ids
        ],
        "probability_semantics": {
            "prefix_joint_probability": f"P(all {prefix_format} tokens | visible input)",
            "gold_answer_conditional_probability": (
                "P(answer-bearing token suffix | visible input, preceding forced format tokens); "
                "answer-bearing tokens may include adjacent formatting whitespace"
            ),
        },
    }


def selected_source_texts(spans, edges, selected_ids: set[int]) -> dict[int, str]:
    retained: dict[int, str] = {}
    for edge in edges:
        selected = sorted(
            (span for span in spans if span.incoming_index == edge.incoming_index and span.index in selected_ids),
            key=lambda span: span.source_char_start,
        )
        retained[edge.incoming_index] = " ".join(
            edge.source_output[span.source_char_start:span.source_char_end].strip()
            for span in selected
        ).strip()
    return retained


def prune_input_messages(original_messages, edges, retained_by_incoming) -> list[dict[str, Any]]:
    """Replace only source-output occurrences, preserving task and headers."""

    messages = copy.deepcopy(list(original_messages))
    users = [index for index, item in enumerate(messages) if item.get("role") == "user"]
    if len(users) != 1:
        raise ValueError("Expected exactly one user message in a worker event")
    user_index = users[0]
    content = str(messages[user_index]["content"])
    ordered_edges = sorted(edges, key=lambda edge: edge.incoming_index)
    # Use the same latest-feasible matching as mask construction. A question
    # can itself quote an upstream output, and repeated history is common.
    locations = _locate_incoming_outputs(
        content, [{"role": "user", "content": content}],
        [edge.source_output for edge in ordered_edges],
    )
    pieces: list[str] = []
    cursor = 0
    for edge, (start, end) in zip(ordered_edges, locations):
        pieces.extend([content[cursor:start], retained_by_incoming.get(edge.incoming_index, "")])
        cursor = end
    pieces.append(content[cursor:])
    messages[user_index]["content"] = "".join(pieces)
    return messages


class RescueGenerationError(RuntimeError):
    """Incomplete generation is an execution failure, not a zero attribution."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _generation(scorer, messages, max_new_tokens: int, *, benchmark: str = "mmlu") -> dict[str, Any]:
    rendered = _render_messages(scorer.tokenizer, messages)
    generation = scorer.generate_from_rendered_chat(
        rendered, max_new_tokens=max_new_tokens, do_sample=False
    )
    ids = _flatten_ids(generation["generated_answer_ids"])
    input_ids = _flatten_ids(generation["input_ids"])
    eos_value = getattr(getattr(getattr(scorer, "model", None), "generation_config", None), "eos_token_id", None)
    if eos_value is None:
        eos_value = getattr(scorer.tokenizer, "eos_token_id", None)
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, (list, tuple, set))
        else ({int(eos_value)} if eos_value is not None else set())
    )
    ended_eos = bool(ids and ids[-1] in eos_ids)
    truncated = len(ids) >= max_new_tokens and not ended_eos
    text = str(generation["output_text"])
    return {
        "messages": copy.deepcopy(list(messages)),
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "text": text,
        "parsed_answer": parse_worker_answer(text, benchmark=benchmark),
        "generated_ids": ids,
        "generated_token_count": len(ids),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "finish_reason": "eos" if ended_eos else ("length" if truncated else "stopped"),
        "hit_max_new_tokens": truncated,
    }


def run_gold_prefix_rescue(
    scorer,
    event: Mapping[str, Any],
    gold: str,
    *,
    batch_size: int,
    max_new_tokens: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    benchmark: str = "mmlu",
) -> dict[str, Any]:
    """Score the local intervention and independently verify its peak subset.

    The returned spans, edges and masks are in-memory objects for integration
    into the normal event report. Callback records are JSON-serializable and
    contain each completed stage, including every candidate ranking score.
    """

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    target = gold_prefix_target(scorer.tokenizer, gold, benchmark=benchmark)
    gold = normalize_gold_answer(gold, benchmark=benchmark)
    answer_first_event = copy.deepcopy(dict(event))
    messages = answer_first_event.get("input_messages") or []
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Expected the worker system message first")
    messages[0]["content"] = answer_first_system_prompt(str(messages[0]["content"]), benchmark=benchmark)
    for key in ("rendered_prompt", "input_ids", "attention_mask"):
        answer_first_event.pop(key, None)
    rendered, input_ids, masks, spans, edges = build_incoming_sentence_masks(
        scorer.tokenizer, answer_first_event, enable_thinking=False, prefer_recorded_prompt=False
    )
    answer_first_event.update(
        rendered_prompt=rendered, input_ids=input_ids, attention_mask=[1] * len(input_ids)
    )

    def emit(payload: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(payload)

    emit({
        "stage": "rescue_input", "target": target,
        "messages": messages, "rendered_prompt": rendered, "input_ids": input_ids,
        "sentences": [span.to_dict() for span in spans],
        "incoming_edges": [edge.to_dict() for edge in edges],
    })
    full_generation = _generation(scorer, messages, max_new_tokens, benchmark=benchmark)
    emit({"stage": "full_answer_first_generation", "generation": full_generation})
    if full_generation["hit_max_new_tokens"]:
        raise RescueGenerationError("Full answer-first diagnostic generation was truncated", full_generation)

    attribution = GreedyPrefixLogitAttributor(
        scorer, score_type="logit", reduction="sum", batch_size=batch_size
    )(
        input_ids, masks, generated_ids=target["generated_ids"],
        selected_target_indices=target["selected_indices"],
        answer_target_indices=target["answer_token_indices"],
        progress_callback=emit,
    )
    attribution.update(
        forced_prefix=target["text"],
        target_token_ids=target["generated_ids"],
        answer_token_indices=target["answer_token_indices"],
        probability_semantics=target["probability_semantics"],
    )
    selected_ids = set(attribution["best_region_indices"])
    retained = selected_source_texts(spans, edges, selected_ids)
    verification_messages = prune_input_messages(event["input_messages"], edges, retained)
    verification = _generation(scorer, verification_messages, max_new_tokens, benchmark=benchmark)
    verification["correct"] = verification["parsed_answer"] == gold
    verification["prompt_mode"] = "original_reasoning_first"
    emit({"stage": "peak_verification", "generation": verification})
    if verification["hit_max_new_tokens"]:
        raise RescueGenerationError("Peak-subset verification generation was truncated", verification)
    return {
        "event": answer_first_event,
        "rendered": rendered,
        "input_ids": input_ids,
        "masks": masks,
        "spans": spans,
        "edges": edges,
        "target": target,
        "attribution": attribution,
        "full_generation": full_generation,
        "verification": verification,
        "success": verification["correct"],
    }
