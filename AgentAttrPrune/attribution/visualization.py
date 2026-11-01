"""Self-contained attribution reports and round-specific matrix heatmaps."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _edge_svg(result: Mapping[str, Any]) -> str:
    incoming = list(result.get("incoming_edges") or [])
    if not incoming:
        return (
            '<svg viewBox="0 0 900 130" role="img" '
            'aria-label="No attributable incoming edges">'
            '<rect x="335" y="35" width="230" height="58" rx="12" '
            'class="target-node"/><text x="450" y="69" text-anchor="middle">'
            "No attributable incoming edge</text></svg>"
        )
    scores = {
        int(item.get("incoming_index", -1)): item
        for item in result.get("edge_scores") or []
    }
    height = max(180, 90 * len(incoming))
    target = result.get("event") or {}
    target_label = (
        f"{target.get('agent_id', 'target')} · round {target.get('round', '?')}"
    )
    pieces = [
        f'<svg viewBox="0 0 900 {height}" role="img" '
        'aria-label="Incoming attribution graph">',
        (
            f'<rect x="625" y="{height / 2 - 38:.1f}" width="235" height="76" '
            'rx="13" class="target-node"/>'
        ),
        (
            f'<text x="742" y="{height / 2 - 5:.1f}" text-anchor="middle">'
            f"{html.escape(target_label)}</text>"
        ),
        (
            f'<text x="742" y="{height / 2 + 19:.1f}" text-anchor="middle" '
            'class="small">receiving node</text>'
        ),
    ]
    for index, edge in enumerate(incoming):
        y = 45 + index * ((height - 90) / max(1, len(incoming) - 1))
        incoming_index = int(edge.get("incoming_index", index))
        score = scores.get(incoming_index, {})
        ratio = float(score.get("token_ratio", 0.0))
        selected = int(score.get("selected_token_count", 0))
        total = int(score.get("source_token_count", 0))
        source = str(edge.get("source_agent", "source"))
        edge_type = str(edge.get("edge_type", ""))
        source_round = edge.get("source_round", "?")
        width = 1.5 + ratio * 8.0
        color = "#2563eb" if edge_type == "spatial" else "#9333ea"
        pieces.extend(
            [
                (
                    f'<rect x="28" y="{y - 31:.1f}" width="235" height="62" '
                    'rx="11" class="source-node"/>'
                ),
                (
                    f'<text x="145" y="{y - 4:.1f}" text-anchor="middle">'
                    f"{html.escape(source)}</text>"
                ),
                (
                    f'<text x="145" y="{y + 18:.1f}" text-anchor="middle" '
                    f'class="small">{html.escape(edge_type)} · round '
                    f"{html.escape(str(source_round))}</text>"
                ),
                (
                    f'<line x1="263" y1="{y:.1f}" x2="625" '
                    f'y2="{height / 2:.1f}" stroke="{color}" '
                    f'stroke-width="{width:.2f}" marker-end="url(#arrow)"/>'
                ),
                (
                    f'<text x="438" y="{y - 8:.1f}" text-anchor="middle" '
                    f'class="edge-label">{ratio:.3f} ({selected}/{total} tokens)'
                    "</text>"
                ),
            ]
        )
    pieces.insert(
        1,
        (
            '<defs><marker id="arrow" markerWidth="10" markerHeight="7" '
            'refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" '
            'fill="#475569"/></marker></defs>'
        ),
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _gold_prefix_curve_svg(result: Mapping[str, Any], width: int, height: int) -> str:
    """Show the complete ranking, including post-peak decline and deletion."""
    attr = result["attribution"]
    curve = attr["curve"]
    n = len(curve)
    insertion = [float(attr["baseline_score"])] + [float(row["insertion_score"]) for row in curve]
    deletion = [float(attr["full_score"])] + [float(row["deletion_score"]) for row in curve]
    low, high = min(insertion + deletion), max(insertion + deletion)
    pad = max((high - low) * .12, 1e-4)
    low, high = low - pad, high + pad
    left, right, top, bottom = 115, width - 45, 115, height - 175

    def point(step, value):
        return left + step / max(1, n) * (right - left), bottom - (value - low) / (high - low) * (bottom - top)

    peak = int(attr["best_step"])
    px, py = point(peak, insertion[peak])
    success = (result.get("rescue") or {}).get("success")
    label = "verified correct" if success else "verification failed; incoming edge importance = 0"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<title>Complete gold-prefix insertion and deletion curves</title>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827;font-size:19px}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="35" y="42" style="font-size:28px;font-weight:bold">Gold-prefix Insertion / Deletion</text>',
        f'<text x="35" y="77">{html.escape(str(result.get("sample_id", "")))} · {html.escape(str(result.get("event", {}).get("agent_id", "")))}</text>',
    ]
    for tick in range(0, 101, 20):
        x, _ = point(tick * n / 100, low)
        parts += [f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#e5e7eb"/>',
                  f'<text x="{x}" y="{bottom + 30}" text-anchor="middle">{tick}%</text>']
    for value in np.linspace(low, high, 5):
        _, y = point(0, value)
        parts += [f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#e5e7eb"/>',
                  f'<text x="{left - 12}" y="{y + 6}" text-anchor="end">{value:.2f}</text>']
    for values, color in ((insertion, "#ea580c"), (deletion, "#2563eb")):
        points = " ".join(f"{point(k, value)[0]:.2f},{point(k, value)[1]:.2f}" for k, value in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    parts += [
        f'<circle cx="{px}" cy="{py}" r="7" fill="#15803d"/>',
        f'<text x="{(left + right) / 2}" y="{bottom + 63}" text-anchor="middle">Ranked prefix retained (%) · deletion removes that same prefix</text>',
        '<text x="22" y="300" transform="rotate(-90 22 300)">Raw-logit sum</text>',
        f'<text x="{left}" y="{height - 77}">Orange: insertion · Blue: deletion · Green: earliest global peak {peak}/{n}</text>',
        f'<text x="{left}" y="{height - 43}">{html.escape(label)}</text>',
        '</svg>',
    ]
    return "".join(parts)


def _insertion_curve_svg(
    result: Mapping[str, Any],
    *,
    width: int = 1400,
    height: int = 900,
) -> str:
    attribution = result.get("attribution") or {}
    if attribution.get("method") == "exact_greedy_gold_prefix_insertion_maximization":
        return _gold_prefix_curve_svg(result, width, height)
    curve = list(attribution.get("curve") or [])
    baseline = attribution.get("baseline_score")
    full = attribution.get("full_score")
    event = result.get("event") or {}
    left, right, top, bottom = 155, 70, 125, 145
    chart_width = width - left - right
    chart_height = height - top - bottom
    style = (
        "text{font-family:Inter,Arial,sans-serif;fill:#111827}"
        ".title{font-size:38px;font-weight:760}"
        ".subtitle{font-size:24px;fill:#475569}"
        ".axis-label{font-size:27px;font-weight:700}"
        ".tick{font-size:21px;fill:#374151}"
        ".annotation{font-size:22px;font-weight:700;fill:#9a3412}"
        ".legend{font-size:20px;fill:#374151}"
    )
    pieces = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="curve-title curve-description">'
        ),
        "<style>" + style + "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            '<title id="curve-title">Early-stop insertion score curve</title>'
            '<desc id="curve-description">Insertion score against retained '
            "significant sentence percentage. The observed curve ends at the "
            "early-stop prefix; the full-input endpoint is shown separately.</desc>"
        ),
        (
            f'<text x="{left}" y="52" class="title">Early-stop Insertion '
            "Score Curve</text>"
        ),
        (
            f'<text x="{left}" y="88" class="subtitle">sample '
            f"{html.escape(str(result.get('sample_id', '')))} · round "
            f"{html.escape(str(event.get('round', '?')))} · "
            f"{html.escape(str(event.get('agent_id', 'node')))}</text>"
        ),
    ]
    if baseline is None or full is None:
        pieces.extend(
            [
                (
                    f'<rect x="{left}" y="{top}" width="{chart_width}" '
                    f'height="{chart_height}" fill="#f8fafc" stroke="#cbd5e1"/>'
                ),
                (
                    f'<text x="{left + chart_width / 2:.1f}" '
                    f'y="{top + chart_height / 2:.1f}" text-anchor="middle" '
                    'class="subtitle">No attributable incoming input; no '
                    "insertion search was required.</text>"
                ),
                "</svg>",
            ]
        )
        return "".join(pieces)

    selected_ids = {
        int(index)
        for index in attribution.get("selected_region_indices") or []
    }
    selected_curve = [
        item
        for item in curve
        if int(item.get("selected_region_index", -1)) in selected_ids
    ]
    points = [(0.0, float(baseline))]
    points.extend(
        (
            100.0 * float(item.get("selected_sentence_fraction", 0.0)),
            float(item.get("score", baseline)),
        )
        for item in selected_curve
    )
    target_recovery = float(attribution.get("target_recovery", 0.95))
    score_scale = abs(float(full) - float(baseline))
    direction = 1.0 if float(full) >= float(baseline) else -1.0
    target_score = float(full) - direction * (1.0 - target_recovery) * score_scale
    all_values = [value for _, value in points] + [
        float(full),
        target_score,
    ]
    low, high = min(all_values), max(all_values)
    span = high - low
    padding = max(span * 0.1, abs(high) * 0.02, 1e-6)
    low -= padding
    high += padding

    def xy(percent: float, value: float) -> tuple[float, float]:
        x = left + float(percent) / 100.0 * chart_width
        y = top + (high - float(value)) / (high - low) * chart_height
        return x, y

    polyline = " ".join(
        f"{xy(percent, value)[0]:.1f},{xy(percent, value)[1]:.1f}"
        for percent, value in points
    )
    if len(points) > 1:
        first_x, _ = xy(*points[0])
        last_x, _ = xy(*points[-1])
        area_points = (
            f"{first_x:.1f},{top + chart_height:.1f} "
            + polyline
            + f" {last_x:.1f},{top + chart_height:.1f}"
        )
        pieces.append(
            f'<polygon points="{area_points}" fill="#e5e7eb" opacity="0.85"/>'
        )
    for tick in range(0, 101, 20):
        x, _ = xy(tick, low)
        pieces.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                    f'y2="{top + chart_height}" stroke="#e5e7eb" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x:.1f}" y="{top + chart_height + 36}" '
                    f'text-anchor="middle" class="tick">{tick}</text>'
                ),
            ]
        )
    for value in np.linspace(low, high, 6):
        _, y = xy(0, float(value))
        pieces.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.1f}" '
                    f'x2="{left + chart_width}" y2="{y:.1f}" '
                    'stroke="#e5e7eb" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 18}" y="{y + 7:.1f}" text-anchor="end" '
                    f'class="tick">{float(value):.5g}</text>'
                ),
            ]
        )
    target_y = xy(0, target_score)[1]
    pieces.extend(
        [
            (
                f'<line x1="{left}" y1="{target_y:.1f}" '
                f'x2="{left + chart_width}" y2="{target_y:.1f}" '
                'stroke="#64748b" stroke-width="3" stroke-dasharray="10 8"/>'
            ),
            (
                f'<text x="{left + 12}" y="{target_y - 12:.1f}" '
                f'text-anchor="start" class="legend">{target_recovery:.0%} '
                f"recovery target · {target_score:.5g}</text>"
            ),
            (
                f'<polyline points="{polyline}" fill="none" stroke="#111827" '
                'stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>'
            ),
        ]
    )
    for percent, value in points:
        x, y = xy(percent, value)
        pieces.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#111827"/>'
        )
    full_x, full_y = xy(100.0, float(full))
    diamond = (
        f"{full_x:.1f},{full_y - 10:.1f} "
        f"{full_x + 10:.1f},{full_y:.1f} "
        f"{full_x:.1f},{full_y + 10:.1f} "
        f"{full_x - 10:.1f},{full_y:.1f}"
    )
    pieces.extend(
        [
            f'<polygon points="{diamond}" fill="#2563eb"/>',
            (
                f'<text x="{full_x - 12:.1f}" y="{full_y + 32:.1f}" '
                f'text-anchor="end" class="legend">full input · '
                f"{float(full):.5g}</text>"
            ),
        ]
    )

    selected_count = len(selected_ids)
    total_sentences = int(
        result.get("attributable_sentence_count")
        or len(result.get("sentences") or [])
    )
    if selected_curve:
        stop_record = selected_curve[-1]
        stop_sentence_percent = 100.0 * float(
            stop_record.get("selected_sentence_fraction", 0.0)
        )
        stop_token_percent = 100.0 * float(
            stop_record.get("selected_token_fraction", 0.0)
        )
        stop_score = float(stop_record.get("score", baseline))
        stop_x, stop_y = xy(stop_sentence_percent, stop_score)
        label_to_right = stop_sentence_percent <= 58.0
        label_x = stop_x + 24 if label_to_right else stop_x - 24
        anchor = "start" if label_to_right else "end"
        label_y = max(top + 42, stop_y - 92)
        recovery = float(
            attribution.get(
                "selected_recovery",
                stop_record.get("recovery", 0.0),
            )
        )
        pieces.extend(
            [
                (
                    f'<line x1="{stop_x:.1f}" y1="{top}" x2="{stop_x:.1f}" '
                    f'y2="{top + chart_height}" stroke="#f97316" '
                    'stroke-width="3" stroke-dasharray="8 7"/>'
                ),
                (
                    f'<circle cx="{stop_x:.1f}" cy="{stop_y:.1f}" r="11" '
                    'fill="#f97316" stroke="#9a3412" stroke-width="3"/>'
                ),
                (
                    f'<text x="{label_x:.1f}" y="{label_y:.1f}" '
                    f'text-anchor="{anchor}" class="annotation">'
                    f'<tspan x="{label_x:.1f}" dy="0">stop: '
                    f"{stop_sentence_percent:.1f}% sentences "
                    f"({selected_count}/{total_sentences})</tspan>"
                    f'<tspan x="{label_x:.1f}" dy="30">'
                    f"{stop_token_percent:.1f}% tokens · recovery "
                    f"{recovery:.1%}</tspan></text>"
                ),
            ]
        )

    pieces.extend(
        [
            (
                f'<line x1="{left}" y1="{top}" x2="{left}" '
                f'y2="{top + chart_height}" stroke="#111827" stroke-width="3"/>'
            ),
            (
                f'<line x1="{left}" y1="{top + chart_height}" '
                f'x2="{left + chart_width}" y2="{top + chart_height}" '
                'stroke="#111827" stroke-width="3"/>'
            ),
            (
                f'<text x="{left + chart_width / 2:.1f}" y="{height - 42}" '
                'text-anchor="middle" class="axis-label">Retained Significant '
                "Sentences (%)</text>"
            ),
            (
                f'<text x="40" y="{top + chart_height / 2:.1f}" '
                f'transform="rotate(-90 40 {top + chart_height / 2:.1f})" '
                'text-anchor="middle" class="axis-label">Insertion Score '
                f"({html.escape(str(attribution.get('score_type', 'score')))} · "
                f"{html.escape(str(attribution.get('reduction', '')))}"
                ")</text>"
            ),
            "</svg>",
        ]
    )
    return "".join(pieces)


def _curve_svg(result: Mapping[str, Any]) -> str:
    return _insertion_curve_svg(result, width=1200, height=720)


def write_event_insertion_curve_svg(
    result: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write the standalone per-node early-stop insertion score curve."""

    return _write_text_atomic(
        Path(path),
        _insertion_curve_svg(result),
    )


def _highlight_source(
    source_output: str,
    spans: Sequence[Mapping[str, Any]],
    *,
    saliency_floor: float = 0.4,
) -> str:
    cursor = 0
    pieces: list[str] = []
    ordered = sorted(
        spans,
        key=lambda item: (
            int(item.get("source_char_start", 0)),
            int(item.get("source_char_end", 0)),
        ),
    )
    for span in ordered:
        start = max(cursor, int(span.get("source_char_start", 0)))
        end = max(start, int(span.get("source_char_end", start)))
        pieces.append(html.escape(source_output[cursor:start]))
        css = "selected-sentence" if span.get("selected") else "candidate-sentence"
        saliency = span.get("saliency")
        style = ""
        if span.get("selected") and saliency is not None:
            style = (
                f' style="background:{_saliency_color(float(saliency), saliency_floor)};'
                f'outline-color:{_saliency_color(float(saliency), saliency_floor, stroke=True)}"'
            )
        title = (
            f"sentence {span.get('index')}; "
            f"{len(span.get('source_token_indices') or [])} source tokens; "
            f"saliency={saliency}"
        )
        pieces.append(
            f'<mark class="{css}" title="{html.escape(title)}"{style}>'
            f"{html.escape(source_output[start:end])}</mark>"
        )
        cursor = end
    pieces.append(html.escape(source_output[cursor:]))
    return "".join(pieces)


def _sentence_chunks(text: str) -> list[str]:
    """Split fixed/output prose for display without changing attribution spans."""

    chunks: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?。！？])\s+(?=\S)", line)
        chunks.extend(part.strip() for part in parts if part.strip())
    return chunks


def _svg_lines(text: str, *, width: int = 112) -> list[str]:
    lines = textwrap.wrap(
        " ".join(str(text).split()),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    )
    return lines or [""]


def _mix_hex(low: str, high: str, amount: float) -> str:
    amount = float(np.clip(amount, 0.0, 1.0))
    low_rgb = tuple(int(low[index : index + 2], 16) for index in (1, 3, 5))
    high_rgb = tuple(int(high[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(start + (end - start) * amount)
        for start, end in zip(low_rgb, high_rgb)
    )
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _saliency_color(
    saliency: float,
    floor: float,
    *,
    stroke: bool = False,
) -> str:
    denominator = max(1e-12, 1.0 - float(floor))
    amount = (float(saliency) - float(floor)) / denominator
    if stroke:
        return _mix_hex("#fb923c", "#c2410c", amount)
    return _mix_hex("#fed7aa", "#f97316", amount)


def _sentence_selection_steps(result: Mapping[str, Any]) -> dict[int, int]:
    steps: dict[int, int] = {}
    for default_step, point in enumerate(
        (result.get("attribution") or {}).get("curve") or [],
        start=1,
    ):
        index = point.get("selected_region_index")
        if index is None:
            index = point.get("region_index")
        if index is not None:
            steps[int(index)] = int(point.get("step", default_step))
    return steps


def write_event_svg(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write a single-column, sentence-level node-attribution figure.

    The layout intentionally mirrors the sentence panel used by the MMLU trace
    examples: selected incoming sentences are orange, unselected candidates
    are gray, and fixed prompt/task text is visibly neutral. It contains no
    token-granularity panel.
    """

    path = Path(path)
    event = result.get("event") or {}
    sample = result.get("sample") or {}
    attribution = result.get("attribution") or {}
    visualization = result.get("visualization") or {}
    display_mode = str(visualization.get("display_mode", "attribution"))
    pruning_mask_mode = display_mode == "pruning_mask"
    output_tone = str(visualization.get("output_tone", "success"))
    scores = {
        int(item.get("incoming_index", -1)): item
        for item in result.get("edge_scores") or []
    }
    spans_by_edge: dict[int, list[Mapping[str, Any]]] = {}
    for sentence in result.get("sentences") or []:
        spans_by_edge.setdefault(
            int(sentence.get("incoming_index", -1)), []
        ).append(sentence)
    for spans in spans_by_edge.values():
        spans.sort(key=lambda item: int(item.get("index", 0)))

    selected_tokens = sum(
        int(item.get("selected_token_count", 0))
        for item in scores.values()
    )
    source_tokens = sum(
        int(item.get("source_token_count", 0))
        for item in scores.values()
    )
    selected_ratio = selected_tokens / source_tokens if source_tokens else 0.0
    selection_steps = _sentence_selection_steps(result)
    saliency_configuration = attribution.get("saliency_configuration") or {}
    saliency_floor = float(saliency_configuration.get("floor", 0.4))
    low_saliency_color = _saliency_color(saliency_floor, saliency_floor)
    high_saliency_color = _saliency_color(1.0, saliency_floor)

    blocks: list[dict[str, Any]] = []

    def add_section(label: str, detail: str = "") -> None:
        blocks.append({"kind": "section", "label": label, "detail": detail})

    def add_sentence(
        text: str,
        *,
        selected: bool = False,
        tone: str = "neutral",
        title: str = "",
        badge: str = "",
        saliency: float | None = None,
    ) -> None:
        blocks.append(
            {
                "kind": "sentence",
                "text": str(text),
                "lines": _svg_lines(str(text)),
                "selected": bool(selected),
                "tone": tone,
                "title": title,
                "badge": badge,
                "saliency": saliency,
            }
        )

    system_messages = [
        str(message.get("content", ""))
        for message in (result.get("input") or {}).get("messages") or []
        if str(message.get("role", "")).lower() == "system"
    ]
    if system_messages:
        add_section("Fixed system instruction", "shown for context · never attributed")
        for chunk in _sentence_chunks("\n".join(system_messages)):
            add_sentence(chunk, tone="fixed", title="Fixed system-prompt text")

    task = str(sample.get("task") or "")
    if task:
        add_section("Fixed task / user question", "shown for context · never attributed")
        for chunk in _sentence_chunks(task):
            add_sentence(chunk, tone="fixed", title="Fixed task text")

    incoming = list(result.get("incoming_edges") or [])
    if not incoming:
        add_section(
            "Incoming agent/history messages",
            "no attributable input for this node",
        )
        add_sentence(
            "This node has no incoming agent output or history message to attribute.",
            tone="empty",
            title="No attributable input",
        )
    for edge in incoming:
        incoming_index = int(edge.get("incoming_index", -1))
        score = scores.get(incoming_index, {})
        ratio = float(score.get("token_ratio", 0.0))
        selected = int(score.get("selected_token_count", 0))
        total = int(score.get("source_token_count", 0))
        source_label = str(
            edge.get("source_agent")
            or edge.get("source_event_id")
            or "source"
        )
        edge_type = str(edge.get("edge_type") or "incoming")
        detail = (
            f"{edge_type} · source round {edge.get('source_round', '?')} · "
            f"selected source tokens {selected}/{total} ({ratio:.1%})"
        )
        add_section(f"Agent {source_label}", detail)
        spans = spans_by_edge.get(incoming_index, [])
        if not spans:
            add_sentence(
                str(edge.get("source_output") or ""),
                tone="candidate",
                title=f"Incoming edge {incoming_index}; no sentence spans recorded",
            )
        for sentence in spans:
            sentence_index = int(sentence.get("index", -1))
            is_selected = bool(sentence.get("selected"))
            token_count = len(sentence.get("source_token_indices") or [])
            step = sentence.get("selection_step") or selection_steps.get(sentence_index)
            saliency = sentence.get("saliency")
            display_badge = str(sentence.get("display_badge") or "")
            add_sentence(
                str(sentence.get("text") or ""),
                selected=is_selected,
                tone="candidate",
                title=(
                    f"sentence {sentence_index}; edge {incoming_index}; "
                    f"{token_count} source tokens; selected={is_selected}; "
                    f"saliency={saliency}"
                ),
                badge=display_badge or (
                    (
                        f"saliency {float(saliency):.2f} · step {step}"
                        if saliency is not None
                        else (f"selected · step {step}" if step else "selected")
                    )
                    if is_selected
                    else ""
                ),
                saliency=float(saliency) if saliency is not None else None,
            )

    add_section(
        str(visualization.get("output_section_label", "Receiving node output")),
        str(
            visualization.get(
                "output_section_detail",
                f"Agent {event.get('agent_id', 'node')} · target text being explained",
            )
        ),
    )
    output_chunks = _sentence_chunks((result.get("output") or {}).get("text", ""))
    if not output_chunks:
        output_chunks = ["(empty output)"]
    for chunk in output_chunks:
        add_sentence(chunk, tone="output", title="Receiving node output")

    width = 3040
    left = 110
    content_width = width - 2 * left
    top = 255
    gap = 18
    section_height = 105
    line_height = 54
    sentence_padding = 28
    sentence_min_height = 92
    total_height = top + 135
    for block in blocks:
        if block["kind"] == "section":
            total_height += section_height
        else:
            total_height += max(
                sentence_min_height,
                len(block["lines"]) * line_height + 2 * sentence_padding,
            )
        total_height += gap

    node_label = (
        f"sample {result.get('sample_id', '')} · "
        f"round {event.get('round', '?')} · "
        f"{event.get('agent_id', 'node')} ({event.get('role', '')})"
    )
    status_line = str(
        visualization.get(
            "status_line",
            f"status: {result.get('status', 'unknown')} · "
            f"stop: {attribution.get('stop_reason', '')} · "
            f"recovery: {attribution.get('selected_recovery', 'n/a')}",
        )
    )
    figure_title = str(
        visualization.get("title", "Selected Incoming-Sentence Attribution")
    )
    description = str(
        visualization.get(
            "description",
            "Single-column sentence attribution. Orange intensity shows "
            "normalized early-stop saliency for selected incoming sentences; "
            "gray incoming sentences were not selected. Fixed prompt text is "
            "never attributed.",
        )
    )
    panel_label = str(
        visualization.get(
            "panel_label",
            f"Sentence-level ({selected_ratio:.1%} selected source tokens)",
        )
    )
    selected_low_label = str(
        visualization.get("selected_low_label", f"{saliency_floor:.1f} later selected")
    )
    selected_high_label = str(
        visualization.get("selected_high_label", "1.0 top1")
    )
    unselected_label = str(
        visualization.get("unselected_label", "unselected / fixed text")
    )
    pieces = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{total_height}" viewBox="0 0 {width} {total_height}" '
            'role="img" aria-labelledby="title description">'
        ),
        f'<title id="title">{html.escape(figure_title)}</title>',
        f'<desc id="description">{html.escape(description)}</desc>',
        (
            '<defs><linearGradient id="saliency-gradient" x1="0%" y1="0%" '
            'x2="100%" y2="0%">'
            f'<stop offset="0%" stop-color="{low_saliency_color}"/>'
            f'<stop offset="100%" stop-color="{high_saliency_color}"/>'
            "</linearGradient></defs>"
        ),
        "<style>"
        "text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".main{font-size:66px;font-weight:760}"
        ".sub{font-size:34px;fill:#526071}"
        ".panel{font-size:58px;font-weight:730}"
        ".section{font-size:42px;font-weight:740}"
        ".detail{font-size:29px;fill:#5e6b7c}"
        ".body{font-size:38px}"
        ".badge{font-size:27px;font-weight:720;fill:#7c2d12}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text id="title-text" x="{left}" y="78" class="main">'
            f"{html.escape(figure_title)}</text>"
        ),
        (
            f'<text x="{left}" y="128" class="sub">'
            f"{html.escape(node_label)}</text>"
        ),
        (
            f'<text x="{left}" y="173" class="sub">'
            f"{html.escape(status_line)}</text>"
        ),
        (
            f'<text x="{left}" y="238" class="panel">'
            f"{html.escape(panel_label)}</text>"
        ),
    ]

    if pruning_mask_mode:
        pieces.extend(
            [
                (
                    f'<rect x="{left}" y="265" width="54" height="38" rx="8" '
                    'fill="#ffffff" stroke="#a7afb9" stroke-width="2"/>'
                    f'<text x="{left + 72}" y="296" class="detail">'
                    f"{html.escape(selected_high_label)}</text>"
                ),
                (
                    f'<rect x="{left + 650}" y="265" width="54" height="38" rx="8" '
                    'fill="#374151" stroke="#111827" stroke-width="2"/>'
                    f'<text x="{left + 722}" y="296" class="detail">'
                    f"{html.escape(unselected_label)}</text>"
                ),
            ]
        )
    else:
        pieces.extend(
            [
                (
                    f'<rect x="{left}" y="265" width="430" height="34" rx="8" '
                    'fill="url(#saliency-gradient)" stroke="#c2410c" stroke-width="2"/>'
                    f'<text x="{left}" y="328" class="detail">'
                    f"{html.escape(selected_low_label)}</text>"
                    f'<text x="{left + 430}" y="328" text-anchor="end" class="detail">'
                    f"{html.escape(selected_high_label)}</text>"
                ),
                (
                    f'<rect x="{left + 650}" y="265" width="54" height="38" rx="8" '
                    'fill="#e5e7eb" stroke="#a7afb9" stroke-width="2"/>'
                    f'<text x="{left + 722}" y="296" class="detail">'
                    f"{html.escape(unselected_label)}</text>"
                ),
            ]
        )

    y = 335
    fill_by_tone = {
        "fixed": "#e5e7eb",
        "candidate": "#e5e7eb",
        "empty": "#f1f5f9",
        "output": "#fee2e2" if output_tone == "error" else "#dcfce7",
        "neutral": "#f1f5f9",
    }
    stroke_by_tone = {
        "fixed": "#a7afb9",
        "candidate": "#a7afb9",
        "empty": "#cbd5e1",
        "output": "#fca5a5" if output_tone == "error" else "#86c99a",
        "neutral": "#cbd5e1",
    }
    for block in blocks:
        if block["kind"] == "section":
            pieces.append(
                f'<text x="{left + 18}" y="{y + 48}" class="section">'
                f"{html.escape(block['label'])}</text>"
            )
            if block["detail"]:
                pieces.append(
                    f'<text x="{left + 18}" y="{y + 85}" class="detail">'
                    f"{html.escape(block['detail'])}</text>"
                )
            y += section_height + gap
            continue

        box_height = max(
            sentence_min_height,
            len(block["lines"]) * line_height + 2 * sentence_padding,
        )
        selected = bool(block["selected"])
        saliency = block.get("saliency")
        is_masked = (
            pruning_mask_mode
            and block["tone"] == "candidate"
            and not selected
        )
        if pruning_mask_mode:
            if is_masked:
                fill, stroke = "#374151", "#111827"
            elif block["tone"] == "output":
                fill = "#fee2e2" if output_tone == "error" else "#dcfce7"
                stroke = "#fca5a5" if output_tone == "error" else "#86c99a"
            else:
                fill, stroke = "#ffffff", "#a7afb9"
        else:
            fill = (
                _saliency_color(
                    float(saliency) if saliency is not None else 1.0,
                    saliency_floor,
                )
                if selected
                else fill_by_tone.get(block["tone"], "#e5e7eb")
            )
            stroke = (
                _saliency_color(
                    float(saliency) if saliency is not None else 1.0,
                    saliency_floor,
                    stroke=True,
                )
                if selected
                else stroke_by_tone.get(block["tone"], "#a7afb9")
            )
        pieces.append("<g>")
        pieces.append(f"<title>{html.escape(block['title'])}</title>")
        pieces.append(
            f'<rect x="{left}" y="{y}" width="{content_width}" '
            f'height="{box_height}" rx="18" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"/>'
        )
        text_x = left + 28
        text_y = y + sentence_padding + 38
        masked_fill_attribute = ' style="fill:#f8fafc"' if is_masked else ""
        for line_index, line in enumerate(block["lines"]):
            pieces.append(
                f'<text x="{text_x}" y="{text_y + line_index * line_height}" '
                f'class="body"{masked_fill_attribute}>'
                f"{html.escape(line)}</text>"
            )
        if block["badge"]:
            pieces.append(
                f'<text x="{left + content_width - 28}" y="{y + box_height - 18}" '
                f'text-anchor="end" class="badge"'
                f"{masked_fill_attribute}>"
                f"{html.escape(block['badge'])}</text>"
            )
        pieces.append("</g>")
        y += box_height + gap

    pieces.append("</svg>")
    return _write_text_atomic(path, "".join(pieces))


def write_event_html(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write one standalone, directly readable report for a generation node."""

    path = Path(path)
    event = result.get("event") or {}
    input_payload = result.get("input") or {}
    output_payload = result.get("output") or {}
    attribution = result.get("attribution") or {}
    saliency_floor = float(
        (attribution.get("saliency_configuration") or {}).get("floor", 0.4)
    )
    edge_spans: dict[int, list[Mapping[str, Any]]] = {}
    for sentence in result.get("sentences") or []:
        edge_spans.setdefault(int(sentence.get("incoming_index", -1)), []).append(
            sentence
        )
    incoming_sections = []
    for edge in result.get("incoming_edges") or []:
        incoming_index = int(edge.get("incoming_index", -1))
        source_output = str(edge.get("source_output", ""))
        highlighted = _highlight_source(
            source_output,
            edge_spans.get(incoming_index, []),
            saliency_floor=saliency_floor,
        )
        incoming_sections.append(
            (
                '<section class="message"><h3>'
                f"{html.escape(str(edge.get('source_agent', 'source')))} "
                f"→ {html.escape(str(edge.get('target_agent', 'target')))} "
                f"({html.escape(str(edge.get('edge_type', '')))})"
                "</h3><pre>"
                f"{highlighted}"
                "</pre></section>"
            )
        )
    if not incoming_sections:
        incoming_sections.append(
            '<p class="muted">This node has no agent/history input to attribute.</p>'
        )
    messages_json = json.dumps(
        input_payload.get("messages", []), ensure_ascii=False, indent=2
    )
    status = str(result.get("status", "unknown"))
    rescue_sections = ""
    if result.get("rescue"):
        rescue = result["rescue"]
        verification = rescue["verification"]
        full_generation = rescue["full_generation"]
        peak = int(attribution["best_step"])
        peak_diagnostics = (
            attribution["curve"][peak - 1]["insertion_diagnostics"]
            if peak else attribution["baseline_diagnostics"]
        )
        rescue_sections = (
            '<section class="card"><h2>Gold-prefix rescue verification</h2>'
            '<p>Highlighted sentences show the attempted peak subset. A failed verification contributes '
            'observed zero edge importance. The original dense trace and downstream inputs are unchanged.</p>'
            f'<p>Target: {html.escape(str(result["target"]["text"]))}; '
            f'verified correct: {rescue["success"]}; retained: {peak}/{len(attribution["curve"])}.</p>'
            '<h3>Full-input answer-first generation (diagnostic)</h3>'
            f'<pre>{html.escape(full_generation["text"])}</pre>'
            '<h3>Peak-subset generation using the original prompt, without gold supplied</h3>'
            f'<pre>{html.escape(verification["text"])}</pre>'
            '<details><summary>Actual verification input</summary>'
            f'<pre>{html.escape(verification["rendered_prompt"])}</pre></details>'
            '<details><summary>Answer-first scoring input</summary>'
            f'<pre>{html.escape(result["scoring_input"]["rendered_prompt"])}</pre></details>'
            '<details><summary>Peak probability diagnostics (not optimization or stopping criteria)</summary>'
            f'<pre>{html.escape(json.dumps({"semantics": result["target"].get("probability_semantics"), "peak": peak_diagnostics}, indent=2, ensure_ascii=False))}</pre>'
            '</details></section>'
        )
    css = """
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;
background:#f8fafc;color:#0f172a}.wrap{max-width:1180px;margin:auto;padding:28px}
.card{background:white;border:1px solid #dbe4ee;border-radius:16px;padding:22px;
margin:18px 0;box-shadow:0 4px 20px #0f172a0a}h1,h2,h3{margin-top:0}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.pill{background:#e2e8f0;border-radius:999px;padding:7px 12px;display:inline-block}
pre{white-space:pre-wrap;word-break:break-word;background:#f8fafc;border-radius:10px;
padding:16px;line-height:1.55;overflow:auto}.selected-sentence{background:#fde68a;
outline:2px solid #f59e0b}.candidate-sentence{background:#e0f2fe;color:#334155}
.target-node{fill:#dbeafe;stroke:#2563eb;stroke-width:2}.source-node{fill:#f1f5f9;
stroke:#64748b;stroke-width:1.5}.small{font-size:13px;fill:#475569}
.edge-label{font-size:12px;fill:#334155}.axis{stroke:#64748b;stroke-width:1.5}
.full-score{stroke:#dc2626;stroke-width:2;stroke-dasharray:8 6}.muted{color:#64748b}
details summary{cursor:pointer;font-weight:650;margin-bottom:12px}
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Node attribution ·
{html.escape(str(event.get("agent_id", "node")))}</title><style>{css}</style></head>
<body><main class="wrap">
<h1>Node attribution</h1>
<div class="meta">
<span class="pill">sample: {html.escape(str(result.get("sample_id", "")))}</span>
<span class="pill">round: {html.escape(str(event.get("round", "")))}</span>
<span class="pill">node: {html.escape(str(event.get("agent_id", "")))}</span>
<span class="pill">role: {html.escape(str(event.get("role", "")))}</span>
<span class="pill">status: {html.escape(status)}</span>
<span class="pill">sample correct: {html.escape(str(result.get("correct_verified")))}</span>
</div>
<section class="card"><h2>Incoming attribution graph</h2>{_edge_svg(result)}</section>
<section class="card"><h2>Attributed incoming messages</h2>
<p><mark class="selected-sentence">selected sentence (saliency gradient)</mark> &nbsp;
<mark class="candidate-sentence">unselected candidate</mark></p>
{''.join(incoming_sections)}</section>
<section class="card"><h2>Search result</h2>
<p>Stop reason: <strong>{html.escape(str(attribution.get("stop_reason", "")))}</strong>;
selected recovery: <strong>{html.escape(str(attribution.get("selected_recovery", "")))}</strong></p>
{_curve_svg(result)}</section>
<section class="card"><h2>Receiving node output</h2>
<pre>{html.escape(str(output_payload.get("text", "")))}</pre></section>
{rescue_sections}
<section class="card"><details><summary>Complete rendered prompt</summary>
<pre>{html.escape(str(input_payload.get("rendered_prompt", "")))}</pre></details>
<details><summary>Complete input messages</summary>
<pre>{html.escape(messages_json)}</pre></details></section>
</main></body></html>"""
    return _write_text_atomic(path, page)


def _plot_matrix_grid(
    *,
    matrices: Mapping[str, Any],
    candidates: Mapping[str, Any],
    agents: Sequence[str],
    title: str,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spatial = np.asarray(matrices["spatial"], dtype=np.float64)
    temporal = np.asarray(matrices["temporal"], dtype=np.float64)
    candidate_spatial = np.asarray(candidates["spatial"], dtype=np.float64)
    candidate_temporal = np.asarray(candidates["temporal"], dtype=np.float64)
    num_rounds = int(spatial.shape[0])
    fig, axes = plt.subplots(
        num_rounds,
        2,
        figsize=(max(10, len(agents) * 1.45 * 2), max(4, num_rounds * 4.4)),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e2e8f0")
    image = None
    for round_index in range(num_rounds):
        for column, (edge_type, values, support) in enumerate(
            (
                ("spatial", spatial, candidate_spatial),
                ("temporal", temporal, candidate_temporal),
            )
        ):
            axis = axes[round_index][column]
            masked = np.ma.masked_where(
                support[round_index] <= 0,
                values[round_index],
            )
            image = axis.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap)
            axis.set_title(f"round {round_index} · {edge_type}")
            axis.set_xticks(range(len(agents)), labels=agents, rotation=45, ha="right")
            axis.set_yticks(range(len(agents)), labels=agents)
            axis.set_xlabel("target")
            axis.set_ylabel("source")
            for source in range(len(agents)):
                for target in range(len(agents)):
                    if support[round_index, source, target] > 0:
                        value = float(values[round_index, source, target])
                        axis.text(
                            target,
                            source,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value >= 0.5 else "black",
                            fontsize=8,
                        )
    fig.suptitle(title, fontsize=15)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.78, label="token ratio")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    temporary.replace(path)
    return path


def write_sample_matrix_png(
    sample_matrix: Mapping[str, Any],
    path: str | Path,
    *,
    metric: str = "token_ratio",
) -> Path:
    matrices = sample_matrix["matrices"]
    return _plot_matrix_grid(
        matrices=matrices[metric],
        candidates=matrices["candidate"],
        agents=list(sample_matrix["agents"]),
        title=(
            f"Sample {sample_matrix.get('sample_index')} · {metric} "
            f"· correct={sample_matrix.get('correct_verified')}"
        ),
        path=Path(path),
    )


def write_aggregate_matrix_png(
    aggregate: Mapping[str, Any],
    path: str | Path,
    *,
    metric: str = "token_ratio",
    statistic: str = "conditional_mean",
) -> Path:
    payload = aggregate["metrics"][metric]
    matrices = {
        edge_type: payload[edge_type][statistic]
        for edge_type in ("spatial", "temporal")
    }
    candidates = {
        edge_type: payload[edge_type]["candidate_count"]
        for edge_type in ("spatial", "temporal")
    }
    return _plot_matrix_grid(
        matrices=matrices,
        candidates=candidates,
        agents=list(aggregate["agents"]),
        title=(
            f"{'All-sample' if aggregate.get('aggregation_scope') == 'all' else 'Correct-only'} aggregate · {metric} {statistic} "
            f"· n={aggregate.get('num_aggregated_samples', aggregate.get('num_correct_samples'))}"
        ),
        path=Path(path),
    )
