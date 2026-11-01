"""Sentence-level attribution and edge-calibration public API."""

from AgentAttrPrune.attribution.calibrator import (
    ALGORITHM_VERSION,
    AttributionConfig,
    EdgeAttributionCalibrator,
)
from AgentAttrPrune.attribution.edge import (
    EDGE_TYPES,
    METRICS,
    GreedyFullScoreAttributor,
    IncomingEdge,
    IncomingSentenceSpan,
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
    word_count,
    write_json,
)
from AgentAttrPrune.attribution.gold_prefix import GreedyPrefixLogitAttributor
from AgentAttrPrune.attribution.scoring import TeacherForcedTextScorer
from AgentAttrPrune.attribution.visualization import (
    write_aggregate_matrix_png,
    write_event_html,
    write_event_insertion_curve_svg,
    write_event_svg,
    write_sample_matrix_png,
)

__all__ = [
    "ALGORITHM_VERSION",
    "AttributionConfig",
    "EDGE_TYPES",
    "EdgeAttributionCalibrator",
    "GreedyFullScoreAttributor",
    "GreedyPrefixLogitAttributor",
    "IncomingEdge",
    "IncomingSentenceSpan",
    "METRICS",
    "TeacherForcedTextScorer",
    "aggregate_sample_matrices",
    "apply_early_stop_sentence_saliency",
    "build_incoming_sentence_masks",
    "build_pruned_masks",
    "build_sample_matrices",
    "compute_event_edge_scores",
    "infer_num_rounds",
    "infer_worker_agents",
    "load_json",
    "select_target_token_indices",
    "target_ids_from_event",
    "trace_is_correct",
    "word_count",
    "write_aggregate_matrix_png",
    "write_event_html",
    "write_event_insertion_curve_svg",
    "write_event_svg",
    "write_json",
    "write_sample_matrix_png",
]
