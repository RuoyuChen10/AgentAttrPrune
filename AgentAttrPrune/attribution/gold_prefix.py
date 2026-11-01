"""Exact sentence ranking for a teacher-forced answer prefix.

This module is intentionally separate from the calibration attributor.  The
formal calibration objective reconstructs a recorded node output and may stop
early.  Here the target can instead be a counterfactual, gold answer prefix,
and every incoming sentence is ranked.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch


class GreedyPrefixLogitAttributor:
    """Rank every sentence by exact greedy insertion maximization.

    At each step all remaining sentences are evaluated.  The selected sentence
    is the one whose addition gives the largest teacher-forced target score.
    The paired deletion score is evaluated for the complement of the same
    selected prefix.  Search never stops early; after the full ordering is
    available, ``best_step`` is the global maximum of the insertion curve.
    """

    def __init__(
        self,
        scorer,
        *,
        score_type: str = "logit",
        reduction: str = "sum",
        batch_size: int = 4,
    ) -> None:
        if score_type not in {"logit", "logprob", "probability"}:
            raise ValueError("score_type must be logit, logprob, or probability")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be sum or mean")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.scorer = scorer
        self.score_type = score_type
        self.reduction = reduction
        self.batch_size = int(batch_size)

    def _score_masks(
        self,
        input_ids: torch.Tensor,
        visible_masks: torch.Tensor,
        generated_ids: torch.Tensor,
        selected_target_indices: Sequence[int],
        answer_target_indices: Sequence[int] = (),
    ) -> Dict[str, torch.Tensor]:
        chunks: Dict[str, List[torch.Tensor]] = {}
        for start in range(0, visible_masks.shape[0], self.batch_size):
            end = min(start + self.batch_size, visible_masks.shape[0])
            rows = input_ids.unsqueeze(0).expand(end - start, -1)
            payload = self.scorer.score_teacher_forced_targets(
                rows,
                generated_ids=generated_ids,
                selected_output_token_indices=selected_target_indices,
                attention_mask=visible_masks[start:end].to(torch.long),
                score_type=self.score_type,
                reduction=self.reduction,
            )
            for key in (
                "score",
                "logit_sum",
                "logit_mean",
                "logprob_sum",
                "logprob_mean",
            ):
                chunks.setdefault(key, []).append(
                    payload[key].detach().to(torch.float32).cpu()
                )
            if answer_target_indices:
                if "target_logprobs" not in payload:
                    raise ValueError("Gold probability diagnostics require target_logprobs")
                target_logprobs = payload["target_logprobs"].detach().to(torch.float32).cpu()
                answer_columns = [
                    list(selected_target_indices).index(index)
                    for index in answer_target_indices
                ]
                chunks.setdefault("target_logprobs", []).append(target_logprobs)
                chunks.setdefault("answer_logprob", []).append(
                    target_logprobs[:, answer_columns].sum(dim=-1)
                )
        result = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
        for key, values in result.items():
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"Model produced non-finite values for {key}.")
        return result

    @staticmethod
    def _diagnostics(payload: Dict[str, torch.Tensor], index: int) -> Dict[str, Any]:
        result = {
            key: float(payload[key][index])
            for key in (
                "logit_sum",
                "logit_mean",
                "logprob_sum",
                "logprob_mean",
            )
        }
        if "answer_logprob" in payload:
            result.update(
                target_token_logprobs=payload["target_logprobs"][index].tolist(),
                target_token_probabilities=payload["target_logprobs"][index].exp().tolist(),
                prefix_joint_probability=math.exp(result["logprob_sum"]),
                gold_answer_logprob=float(payload["answer_logprob"][index]),
                gold_answer_conditional_probability=math.exp(
                    float(payload["answer_logprob"][index])
                ),
            )
        return result

    def __call__(
        self,
        source_input_ids: Sequence[int],
        sentence_masks: np.ndarray,
        *,
        generated_ids: Sequence[int],
        selected_target_indices: Sequence[int],
        answer_target_indices: Sequence[int] = (),
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
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
        if any(index not in selected_target_indices for index in answer_target_indices):
            raise ValueError("Answer target indices must be included in selected target indices")

        explainable = masks.any(dim=0)
        fixed = ~explainable
        endpoints = self._score_masks(
            input_ids,
            torch.stack([fixed, torch.ones_like(fixed)]),
            target_ids,
            selected_target_indices,
            answer_target_indices,
        )
        baseline_score = float(endpoints["score"][0])
        full_score = float(endpoints["score"][1])
        if progress_callback is not None:
            progress_callback({
                "stage": "ranking_endpoints",
                "baseline_score": baseline_score,
                "full_score": full_score,
                "baseline_diagnostics": self._diagnostics(endpoints, 0),
                "full_diagnostics": self._diagnostics(endpoints, 1),
            })

        selected: List[int] = []
        selected_visible = torch.zeros_like(explainable)
        remaining = list(range(masks.shape[0]))
        previous_insertion = baseline_score
        previous_deletion = full_score
        curve: List[Dict[str, Any]] = []

        for step in range(1, masks.shape[0] + 1):
            candidate_selected = torch.stack(
                [selected_visible | masks[index] for index in remaining]
            )
            insertion_visible = fixed.unsqueeze(0) | candidate_selected
            deletion_visible = fixed.unsqueeze(0) | (
                explainable.unsqueeze(0) & ~candidate_selected
            )
            paired = self._score_masks(
                input_ids,
                torch.cat([insertion_visible, deletion_visible], dim=0),
                target_ids,
                selected_target_indices,
                answer_target_indices,
            )
            candidate_count = len(remaining)
            insertion_scores = paired["score"][:candidate_count]
            deletion_scores = paired["score"][candidate_count:]
            local_index = int(torch.argmax(insertion_scores).item())
            region_index = remaining.pop(local_index)
            chosen_insertion = float(insertion_scores[local_index])
            chosen_deletion = float(deletion_scores[local_index])

            candidate_evaluations = []
            candidate_regions = list(remaining)
            candidate_regions.insert(local_index, region_index)
            for candidate_index, candidate_region in enumerate(candidate_regions):
                candidate_evaluations.append(
                    {
                        "region_index": int(candidate_region),
                        "insertion_score": float(insertion_scores[candidate_index]),
                        "deletion_score": float(deletion_scores[candidate_index]),
                        "insertion_diagnostics": self._diagnostics(
                            paired, candidate_index
                        ),
                        "deletion_diagnostics": self._diagnostics(
                            paired, candidate_count + candidate_index
                        ),
                    }
                )

            selected.append(region_index)
            selected_visible |= masks[region_index]
            curve.append(
                {
                    "step": step,
                    "selected_region_index": int(region_index),
                    "insertion_score": chosen_insertion,
                    "deletion_score": chosen_deletion,
                    "score": chosen_insertion,
                    "marginal_insertion_delta": chosen_insertion
                    - previous_insertion,
                    "marginal_deletion_drop": previous_deletion - chosen_deletion,
                    "selected_sentence_fraction": step / masks.shape[0],
                    "selected_token_fraction": float(
                        (selected_visible & explainable).sum().item()
                        / max(1, explainable.sum().item())
                    ),
                    "insertion_diagnostics": self._diagnostics(
                        paired, local_index
                    ),
                    "deletion_diagnostics": self._diagnostics(
                        paired, candidate_count + local_index
                    ),
                    "candidate_evaluations": candidate_evaluations,
                }
            )
            previous_insertion = chosen_insertion
            previous_deletion = chosen_deletion
            if progress_callback is not None:
                progress_callback({"stage": "ranking_step", **curve[-1]})

        insertion_scores = [baseline_score] + [
            float(record["insertion_score"]) for record in curve
        ]
        best_step = int(np.argmax(np.asarray(insertion_scores, dtype=np.float64)))
        best_region_indices = selected[:best_step]
        best_record = curve[best_step - 1] if best_step else None
        return {
            "performed": True,
            "method": "exact_greedy_gold_prefix_insertion_maximization",
            "ranking_objective": f"maximize_{self.score_type}_{self.reduction}",
            "score_type": self.score_type,
            "reduction": self.reduction,
            "early_stop": False,
            "baseline_score": baseline_score,
            "full_score": full_score,
            "baseline_diagnostics": self._diagnostics(endpoints, 0),
            "full_diagnostics": self._diagnostics(endpoints, 1),
            "ordered_region_indices": selected,
            "selected_region_indices": best_region_indices,
            "curve": curve,
            "stop_reason": "all_sentences_ranked",
            "stop_step": len(selected),
            "best_step": best_step,
            "best_region_indices": best_region_indices,
            "best_score": float(insertion_scores[best_step]),
            "best_sentence_fraction": best_step / masks.shape[0],
            "best_token_fraction": (
                float(best_record["selected_token_fraction"])
                if best_record is not None
                else 0.0
            ),
            "deletion_start_score": full_score,
            "deletion_end_score": baseline_score,
        }
