#!/usr/bin/env python3
"""Run model-based eval on a processed parquet dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score import default_compute_score


def _get_ground_truth(row: Dict[str, Any]) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model["ground_truth"]
    return row.get("response")


def _build_prompt(context: str) -> str:
    context = context.strip()
    if context.endswith("\n"):
        context = context[:-1]
    return context


def _load_model(model_path: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def run_eval(
    model_path: str,
    dataset_path: Path,
    k: int,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_parquet(dataset_path)
    if max_samples > 0:
        df = df.head(max_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = _load_model(model_path, dtype)

    pass_scores: List[float] = []
    avg_scores: List[float] = []

    for row in df.to_dict(orient="records"):
        context = row.get("context", "")
        data_source = row.get("data_source", "")
        ground_truth = _get_ground_truth(row)
        extra_info = row.get("extra_info")

        prompt = _build_prompt(context)
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        scores: List[float] = []
        for _ in range(k):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            score = default_compute_score(data_source, response, ground_truth, extra_info)
            scores.append(score)

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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    metrics = run_eval(
        model_path=args.model_path,
        dataset_path=Path(args.dataset),
        k=args.k,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
