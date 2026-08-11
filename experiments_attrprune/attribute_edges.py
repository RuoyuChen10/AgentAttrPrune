"""Calibrate round-specific edge policies from verified-correct traces."""

from __future__ import annotations

import argparse
from pathlib import Path

from AgentAttrPrune.attribution import AttributionConfig, EdgeAttributionCalibrator
from AgentAttrPrune.datasets import BENCHMARKS
from experiments_attrprune.common import (
    DEFAULT_MODEL,
    ExperimentPaths,
    canonical_model_name,
    config_fingerprint,
    load_json,
    resolve_run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute incoming agent-output sentences, aggregate correct-sample "
            "edge matrices, and publish token/binary/word-ratio policies."
        )
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="mmlu")
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Accepted for a uniform shell interface; attribution reads saved traces.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sample-glob", default="sample_*")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--sample-scope",
        choices=["all", "correct"],
        default="all",
        help="Attribute every saved dev trace, or only final-correct traces.",
    )
    parser.add_argument(
        "--node-scope",
        choices=["all", "workers", "multi-input"],
        default="all",
        help=(
            "all: every worker-round plus reviewer; workers: every worker-round; "
            "multi-input: only events passing --min-incoming."
        ),
    )
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--event-id", type=int, action="append", default=None)
    parser.add_argument("--min-incoming", type=int, default=2)
    parser.add_argument(
        "--include-reviewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compatibility control for --node-scope multi-input.",
    )
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--target-mode", choices=["decision", "all"], default="all")
    parser.add_argument("--target-token-limit", type=int, default=None)
    parser.add_argument(
        "--score-type",
        choices=["logit", "logprob", "probability"],
        default="logit",
    )
    parser.add_argument("--score-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--target-recovery", type=float, default=0.95)
    parser.add_argument("--absolute-atol", type=float, default=1e-4)
    parser.add_argument("--relative-rtol", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-selected-sentences", type=int, default=None)
    parser.add_argument("--accept-incomplete", action="store_true")
    parser.add_argument("--min-selected-sentences", type=int, default=2)
    parser.add_argument("--min-sentence-words", type=int, default=4)
    parser.add_argument(
        "--saliency-floor",
        type=float,
        default=0.4,
        help=(
            "Minimum normalized saliency for the final selected sentence; "
            "post-processing only and does not change attribution search."
        ),
    )

    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--policy-pruning-rate", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--allow-legacy-traces", action="store_true")
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip per-node SVG/HTML reports and matrix PNG heatmaps.",
    )
    parser.add_argument(
        "--attribute-only",
        action="store_true",
        help="Write per-node/sample outputs without publishing an aggregate policy.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--aggregate-all-existing", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "Rebuild per-node sentence-level SVG/HTML files from cached event "
            "JSON without loading the attribution model."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    del args.data_dir
    run_dir = args.run_dir or resolve_run_dir(
        model_name=args.model_name,
        seed=args.seed,
        results_root=args.results_root,
        run_name=args.run_name,
        benchmark=args.benchmark,
    )
    paths = ExperimentPaths(run_dir)
    if not paths.run_config.exists():
        raise FileNotFoundError(
            f"Missing {paths.run_config}; run the traces stage first."
        )
    run_config = load_json(paths.run_config)
    if run_config.get("fingerprint") != config_fingerprint(run_config):
        raise ValueError(
            "run_config.json has a missing or invalid fingerprint; refusing attribution."
        )
    if str(run_config.get("benchmark", "mmlu")) != args.benchmark:
        raise ValueError(
            "Attribution benchmark must match run_config.json exactly: "
            f"configured={run_config.get('benchmark', 'mmlu')!r}, "
            f"requested={args.benchmark!r}"
        )
    configured_model = canonical_model_name(run_config["model_name"])
    requested_model = canonical_model_name(args.model_name)
    if configured_model != requested_model:
        raise ValueError(
            "Attribution model must match run_config.json exactly: "
            f"configured={configured_model!r}, requested={requested_model!r}"
        )

    config = AttributionConfig(
        sample_glob=args.sample_glob,
        sample_limit=args.sample_limit,
        sample_scope=args.sample_scope,
        node_scope=args.node_scope,
        event_limit=args.event_limit,
        event_ids=tuple(args.event_id) if args.event_id else None,
        min_incoming=args.min_incoming,
        include_reviewer=args.include_reviewer,
        enable_thinking=args.enable_thinking,
        target_mode=args.target_mode,
        target_token_limit=args.target_token_limit,
        score_type=args.score_type,
        score_reduction=args.score_reduction,
        target_recovery=args.target_recovery,
        absolute_atol=args.absolute_atol,
        relative_rtol=args.relative_rtol,
        patience=args.patience,
        batch_size=args.batch_size,
        max_selected_sentences=args.max_selected_sentences,
        accept_incomplete=args.accept_incomplete,
        min_selected_sentences=args.min_selected_sentences,
        min_sentence_words=args.min_sentence_words,
        saliency_floor=args.saliency_floor,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        policy_pruning_rate=args.policy_pruning_rate,
        overwrite=args.overwrite,
        keep_going=args.keep_going,
        allow_legacy_traces=args.allow_legacy_traces,
        visualizations=not args.no_visualizations,
    )
    calibrator = EdgeAttributionCalibrator(
        run_dir=paths.traces,
        output_dir=paths.attribution,
        model_name=requested_model,
        config=config,
    )
    result = calibrator.run(
        aggregate_only=args.aggregate_only,
        aggregate_all_existing=args.aggregate_all_existing,
        attribute_only=args.attribute_only,
        render_only=args.render_only,
    )
    aggregate = result["aggregate"]
    if aggregate is None:
        if args.render_only:
            print(
                f"Rebuilt sentence-level and insertion-curve SVG reports for "
                f"{len(result['manifest']['samples'])} sample(s) under "
                f"{paths.attribution}"
            )
            return
        print(
            f"Saved {len(result['manifest']['samples'])} sample attribution "
            f"bundle(s) under {paths.attribution}"
        )
        return
    print(
        f"Aggregated {aggregate['num_correct_samples']} correct samples into "
        f"{paths.attribution / 'aggregate_edge_matrices.json'}"
    )
    print(f"Binary policy: {paths.attribution / 'pruned_policy_binary.json'}")
    print(
        "Token-ratio policy: "
        f"{paths.attribution / 'pruned_policy_token_ratio.json'}"
    )
    print(
        "Word-ratio policy: "
        f"{paths.attribution / 'pruned_policy_word_ratio.json'}"
    )


if __name__ == "__main__":
    main()
