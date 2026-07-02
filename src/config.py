from __future__ import annotations

from pathlib import Path
from typing import Final


BASE_DIR: Final[Path] = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR: Final[Path] = BASE_DIR / "outputs"

RESULTS_FILENAME: Final[str] = "evaluation_results.json"

DEFAULT_EPSILON: Final[float] = 0.01

DEFAULT_JUDGE_MODEL: Final[str] = "Qwen2.5-72B-Instruct"

JUDGE_PROMPT_TEMPLATE: Final[str] = """You are an evaluator for a visual question answering task. Your task is to determine whether the predicted answer is correct with respect to the ground-truth answer for the given question.

Evaluation rules:
1. Accept minor wording differences, paraphrases, singular/plural variation, and equivalent expressions.
2. Accept answers that are semantically equivalent to the ground-truth answer.
3. Reject answers that are incomplete, overly vague, or refer to the wrong object, attribute, count number, text, or relation.
4. For counting questions, the numeric value must match exactly unless the reference explicitly allows a range.
5. For OCR questions, ignore capitalization and minor punctuation differences, but do not ignore incorrect characters or different words.
6. For hallucination questions, accept only answers that clearly indicate the target does not exist or is not present or the answer cannot be determined from the image.
7. Be strict: the predicted answer should mean the same thing as the ground-truth answer in the context of the question.

Return your decision in JSON with the following fields:
{{
  "correct": 0 or 1,
  "reason": "short explanation"
}}

Question: {question}
Ground-truth answer: {gt_answer}
Predicted answer: {pred_answer}
Task type: {task_type}
Domain: {domain}
"""

