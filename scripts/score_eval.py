#!/usr/bin/env python3
"""Compute pass@k and avg@k from a processed parquet dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from verl.utils.reward_score import default_compute_score


def _get_ground_truth(row: Dict[str, Any]) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model["ground_truth"]
    return row.get("response")


def _get_responses(row: Dict[str, Any]) -> List[str]:
    if "responses" in row and isinstance(row["responses"], list):
        return row["responses"]
    if "response" in row:
        return [row["response"]]
    return []


def score_dataset(path: Path, k: int, max_samples: int) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    if max_samples > 0:
        df = df.head(max_samples)

    pass_scores: List[float] = []
    avg_scores: List[float] = []

    for row in df.to_dict(orient="records"):
        data_source = row.get("data_source", "")
        ground_truth = _get_ground_truth(row)
        extra_info = row.get("extra_info")
        responses = _get_responses(row)

        scores = [
            default_compute_score(data_source, resp, ground_truth, extra_info)
            for resp in responses
        ]
        if not scores:
            pass_scores.append(0.0)
            avg_scores.append(0.0)
            continue

        pass_scores.append(1.0 if any(score > 0 for score in scores) else 0.0)
        avg_scores.append(sum(scores) / len(scores))

    total = len(pass_scores)
    pass_at_k = sum(pass_scores) / total if total else 0.0
    avg_at_k = sum(avg_scores) / total if total else 0.0

    return {
        "pass_at_k": {"k": k, "value": pass_at_k},
        "avg_at_k": {"k": k, "value": avg_at_k},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed parquet")
    parser.add_argument("--k", type=int, required=True, help="K for pass@k/avg@k")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--output", help="Write metrics JSON to file")
    args = parser.parse_args()

    metrics = score_dataset(Path(args.input), args.k, args.max_samples)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
