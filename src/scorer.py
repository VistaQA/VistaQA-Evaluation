from __future__ import annotations

import math
from typing import Any

from data_loader import SAESample
from mask_evaluator import MaskEvaluator


def compute_joint_score(text_score: float, mask_score: float, epsilon: float) -> float:
    t_prime = max(float(text_score), float(epsilon))
    s_prime = max(float(mask_score), float(epsilon))
    return math.sqrt(t_prime * s_prime)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _instance_group(num_gt_masks: int) -> str:
    if num_gt_masks == 0:
        return "zero_instance"
    if num_gt_masks == 1:
        return "single_instance"
    return "multi_instance"


class SAEScorer:
    def __init__(self, epsilon: float) -> None:
        if float(epsilon) <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.mask_evaluator = MaskEvaluator()

    def score_samples(
        self,
        samples: list[SAESample],
        text_evaluator: Any,
        judge_model_name: str,
    ) -> list[dict[str, Any]]:
        if text_evaluator is None:
            raise ValueError("text_evaluator must not be None")

        results: list[dict[str, Any]] = []
        for sample in samples:
            judge_result = text_evaluator.evaluate(
                image_id=sample.image_id,
                question=sample.question,
                gt_answer=sample.gt_answer,
                pred_answer=sample.pred_text,
                task_type=sample.task_type,
                domain=sample.domain,
            )
            text_score = int(judge_result.get("correct", 0))
            mask_score = self.mask_evaluator.score(
                gt_masks=sample.gt_masks,
                pred_masks=sample.pred_masks,
            )
            joint = compute_joint_score(
                text_score=text_score,
                mask_score=mask_score,
                epsilon=self.epsilon,
            )
            results.append(
                {
                    "image_id": sample.image_id,
                    "task_type": sample.task_type,
                    "domain": sample.domain,
                    "is_hallucination": bool(sample.is_hallucination),
                    "instance_group": _instance_group(len(sample.gt_masks)),
                    "standard_average": float(joint),
                    "judge_model": str(judge_result.get("judge_model", judge_model_name)),
                }
            )
        return results

    @staticmethod
    def _aggregate_by_group(
        per_sample_results: list[dict[str, Any]],
        key: str,
        group_order: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[float]] = {}
        for item in per_sample_results:
            group = str(item.get(key, ""))
            grouped.setdefault(group, []).append(float(item.get("standard_average", 0.0)))

        groups = list(grouped.keys())
        if group_order is None:
            groups.sort()
        else:
            groups = [g for g in group_order if g in grouped]

        rows: list[dict[str, Any]] = []
        for group in groups:
            rows.append(
                {
                    "group": group,
                    "standard_average": _mean(grouped[group]),
                }
            )
        return rows

    def build_standard_average_report(self, per_sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        overall_scores = [float(item.get("standard_average", 0.0)) for item in per_sample_results]

        hallucination_view: list[dict[str, Any]] = []
        for item in per_sample_results:
            copied = dict(item)
            copied["hallucination_group"] = "true" if bool(item.get("is_hallucination", False)) else "false"
            hallucination_view.append(copied)

        return {
            "overall": {
                "standard_average": _mean(overall_scores),
            },
            "breakdown": {
                "task_type": self._aggregate_by_group(per_sample_results, key="task_type"),
                "domain": self._aggregate_by_group(per_sample_results, key="domain"),
                "is_hallucination": self._aggregate_by_group(
                    hallucination_view,
                    key="hallucination_group",
                    group_order=["false", "true"],
                ),
                "instance_group": self._aggregate_by_group(
                    per_sample_results,
                    key="instance_group",
                    group_order=["zero_instance", "single_instance", "multi_instance"],
                ),
            },
        }
