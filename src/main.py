from __future__ import annotations

import argparse
import os
from pathlib import Path

from config import DEFAULT_EPSILON, DEFAULT_JUDGE_MODEL, DEFAULT_OUTPUT_DIR, RESULTS_FILENAME
from data_loader import load_and_merge_samples
from utils import ensure_dir, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VistaQA evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt", required=True, type=str, help="Path to GT JSON/JSONL file or directory")
    parser.add_argument("--pred", required=True, type=str, help="Path to prediction JSON/JSONL file or directory")
    parser.add_argument(
        "--llm_model",
        required=False,
        default=DEFAULT_JUDGE_MODEL,
        type=str,
        help="Judge model name for OpenAI-compatible API",
    )
    parser.add_argument(
        "--epsilon",
        required=False,
        default=DEFAULT_EPSILON,
        type=float,
        help="Single epsilon value used in J = sqrt(max(T,epsilon) * max(S,epsilon))",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default=str(DEFAULT_OUTPUT_DIR),
        type=str,
        help="Directory for evaluation_results.json",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from scorer import SAEScorer
    from text_evaluator import TextEvaluator

    epsilon = float(args.epsilon)
    if epsilon <= 0:
        raise ValueError("--epsilon must be positive")

    samples = load_and_merge_samples(gt_path=args.gt, pred_path=args.pred)
    if not samples:
        raise ValueError("No evaluable samples were loaded")

    text_evaluator = TextEvaluator(
        model_name=str(args.llm_model),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    scorer = SAEScorer(epsilon=epsilon)
    per_sample_results = scorer.score_samples(
        samples=samples,
        text_evaluator=text_evaluator,
        judge_model_name=str(args.llm_model),
    )
    report = scorer.build_standard_average_report(per_sample_results)

    output = {
        "metadata": {
            "epsilon": epsilon,
            "judge_model": str(args.llm_model),
            "total_samples": len(samples),
        },
        "overall": report["overall"],
        "breakdown": report["breakdown"],
    }

    output_dir = ensure_dir(Path(args.output_dir))
    output_path = output_dir / RESULTS_FILENAME
    write_json(output_path, output)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
