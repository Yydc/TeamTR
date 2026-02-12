#!/usr/bin/env python3
"""Team evaluation for multiple models (per-model, best-of, router, or chained team flow)."""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score import default_compute_score


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _messages_to_text(messages: Any) -> str:
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return str(messages)
    if len(messages) == 1 and isinstance(messages[0], dict):
        return _normalize_message_content(messages[0].get("content", ""))

    lines: List[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", "")).strip()
            content = _normalize_message_content(msg.get("content", ""))
            if not content:
                continue
            lines.append(f"{role}: {content}" if role else content)
        else:
            lines.append(str(msg))
    return "\n".join(lines)


def _normalize_context(context: Any) -> str:
    if context is None:
        return ""
    if isinstance(context, (list, dict)):
        return _messages_to_text(context)
    if not isinstance(context, str):
        return str(context)

    text = context.strip()
    if text and text[0] in "[{" and text[-1] in "]}":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, (list, dict)):
                return _messages_to_text(parsed)
    return context


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _get_ground_truth(row: Dict[str, Any]) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model["ground_truth"]
    return row.get("response")


def _build_prompt(context: str) -> str:
    context = _normalize_context(context).strip()
    if context.endswith("\n"):
        context = context[:-1]
    return context


def _load_records(dataset_path: Path, max_samples: int) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(dataset_path)
    if max_samples > 0:
        df = df.head(max_samples)
    return df.to_dict(orient="records")


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


def _score_to_float(score: Any) -> float:
    """Normalize reward outputs (float/dict/etc.) into a scalar score."""
    if isinstance(score, dict):
        primary = score.get("score")
        if isinstance(primary, (int, float, bool)):
            return float(primary)
        acc = score.get("acc")
        if isinstance(acc, (int, float, bool)):
            return float(acc)
        for value in score.values():
            if isinstance(value, (int, float, bool)):
                return float(value)
        return 0.0
    if isinstance(score, (int, float, bool)):
        return float(score)
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _generate_text(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    response_ids = outputs[0, prompt_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def _eval_model(
    *,
    model_path: str,
    records: List[Dict[str, Any]],
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = _load_model(model_path, dtype)

    per_pass: List[int] = []
    per_avg: List[float] = []

    for row in records:
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
                do_sample = temperature > 0
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "num_return_sequences": 1,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p
                outputs = model.generate(**inputs, **gen_kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            response_ids = outputs[0, prompt_len:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            score = default_compute_score(data_source, response, ground_truth, extra_info)
            scores.append(_score_to_float(score))

        per_pass.append(1 if any(score > 0 for score in scores) else 0)
        per_avg.append(sum(scores) / len(scores) if scores else 0.0)

    total = len(per_pass)
    metrics = {
        "pass_at_k": {"k": k, "value": sum(per_pass) / total if total else 0.0},
        "avg_at_k": {"k": k, "value": sum(per_avg) / total if total else 0.0},
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return per_pass, per_avg, metrics


def _build_chain_prompt(question: str, prior_turns: List[Dict[str, Any]]) -> str:
    """Build next-agent input: question + previous agent answers."""
    if not prior_turns:
        return question

    lines = [question.strip(), "", "Previous agents' answers:"]
    for t in prior_turns:
        agent_id = int(t["agent_id"]) + 1
        lines.append(f"Agent {agent_id}:")
        lines.append(str(t["output"]).strip())
        lines.append("")
    lines.append("Provide your best final answer.")
    return "\n".join(lines).strip()


def _eval_chain_team(
    *,
    model_paths: List[str],
    records: List[Dict[str, Any]],
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    router: str,
    seed: int | None,
    trace_samples: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    unique_loaded: Dict[str, Dict[str, Any]] = {}
    for model_path in sorted(set(model_paths)):
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        unique_loaded[model_path] = {"tokenizer": tok, "model": _load_model(model_path, dtype)}

    tokenizers = [unique_loaded[p]["tokenizer"] for p in model_paths]
    models = [unique_loaded[p]["model"] for p in model_paths]

    rng = random.Random(seed)
    per_pass: List[int] = []
    per_avg: List[float] = []
    traces: List[Dict[str, Any]] = []
    n_models = len(model_paths)

    try:
        for row_idx, row in enumerate(records):
            question = _build_prompt(row.get("context", ""))
            data_source = row.get("data_source", "")
            ground_truth = _get_ground_truth(row)
            extra_info = row.get("extra_info")

            rollout_scores: List[float] = []
            trace_turns: List[Dict[str, Any]] | None = None

            for sample_idx in range(k):
                if router == "random":
                    order = list(range(n_models))
                    rng.shuffle(order)
                else:
                    start = (row_idx + sample_idx) % n_models
                    order = [(start + j) % n_models for j in range(n_models)]

                turn_records: List[Dict[str, Any]] = []
                for turn_idx, model_idx in enumerate(order):
                    prompt = _build_chain_prompt(question, turn_records)
                    response = _generate_text(
                        model=models[model_idx],
                        tokenizer=tokenizers[model_idx],
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    turn_records.append(
                        {
                            "turn": turn_idx,
                            "agent_id": model_idx,
                            "prompt": prompt,
                            "output": response,
                        }
                    )

                final_response = turn_records[-1]["output"] if turn_records else ""
                score = default_compute_score(data_source, final_response, ground_truth, extra_info)
                rollout_scores.append(_score_to_float(score))

                # Keep one representative trajectory for this sample.
                if sample_idx == 0 and len(traces) < trace_samples:
                    trace_turns = turn_records

            per_pass.append(1 if any(s > 0 for s in rollout_scores) else 0)
            per_avg.append(sum(rollout_scores) / len(rollout_scores) if rollout_scores else 0.0)

            if trace_turns is not None:
                traces.append(
                    {
                        "sample_index": row_idx,
                        "question": question,
                        "turns": trace_turns,
                    }
                )
    finally:
        for entry in unique_loaded.values():
            model = entry["model"]
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = len(per_pass)
    metrics = {
        "pass_at_k": {"k": k, "value": sum(per_pass) / total if total else 0.0},
        "avg_at_k": {"k": k, "value": sum(per_avg) / total if total else 0.0},
    }
    return metrics, traces


def _aggregate_best_of(per_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_models:
        return {"pass_at_k": {"k": 0, "value": 0.0}, "avg_at_k": {"k": 0, "value": 0.0}}

    num_samples = len(per_models[0]["per_pass"])
    best_pass = []
    best_avg = []
    for idx in range(num_samples):
        best_pass.append(max(m["per_pass"][idx] for m in per_models))
        best_avg.append(max(m["per_avg"][idx] for m in per_models))

    total = len(best_pass)
    return {
        "pass_at_k": {"k": per_models[0]["metrics"]["pass_at_k"]["k"], "value": sum(best_pass) / total if total else 0.0},
        "avg_at_k": {"k": per_models[0]["metrics"]["avg_at_k"]["k"], "value": sum(best_avg) / total if total else 0.0},
    }


def _aggregate_router(per_models: List[Dict[str, Any]], router: str, seed: int | None) -> Dict[str, Any]:
    if not per_models:
        return {"pass_at_k": {"k": 0, "value": 0.0}, "avg_at_k": {"k": 0, "value": 0.0}}

    num_samples = len(per_models[0]["per_pass"])
    n_models = len(per_models)
    if router == "random":
        rng = random.Random(seed)
        model_indices = [rng.randrange(n_models) for _ in range(num_samples)]
    else:
        model_indices = [idx % n_models for idx in range(num_samples)]

    routed_pass = []
    routed_avg = []
    for sample_idx, model_idx in enumerate(model_indices):
        routed_pass.append(per_models[model_idx]["per_pass"][sample_idx])
        routed_avg.append(per_models[model_idx]["per_avg"][sample_idx])

    total = len(routed_pass)
    return {
        "pass_at_k": {"k": per_models[0]["metrics"]["pass_at_k"]["k"], "value": sum(routed_pass) / total if total else 0.0},
        "avg_at_k": {"k": per_models[0]["metrics"]["avg_at_k"]["k"], "value": sum(routed_avg) / total if total else 0.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-paths", required=True, help="Comma-separated list of model paths (HF format).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["per_model", "best_of", "router", "chain"], default="per_model")
    parser.add_argument("--router", choices=["round_robin", "random"], default="round_robin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--chain-trace-samples", type=int, default=5)
    parser.add_argument("--chain-trace-output", default=None, help="Optional JSONL path to dump chain turn traces.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip()]
    if not model_paths:
        raise SystemExit("No model paths provided.")

    records = _load_records(Path(args.dataset), args.max_samples)

    per_models: List[Dict[str, Any]] = []
    per_model_cache: Dict[str, Tuple[List[int], List[float], Dict[str, Any]]] = {}
    for model_path in model_paths:
        if model_path not in per_model_cache:
            per_model_cache[model_path] = _eval_model(
                model_path=model_path,
                records=records,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        per_pass, per_avg, metrics = per_model_cache[model_path]
        per_models.append(
            {
                "model_path": model_path,
                "per_pass": list(per_pass),
                "per_avg": list(per_avg),
                "metrics": dict(metrics),
            }
        )

    output: Dict[str, Any] = {
        "schema": "teamtr.eval.team.v1",
        "mode": args.mode,
        "router": args.router if args.mode in {"router", "chain"} else None,
        "k": args.k,
        "max_samples": args.max_samples,
        "models": [{"model_path": m["model_path"], "metrics": m["metrics"]} for m in per_models],
        "generated_at": _now_iso(),
    }

    if args.mode == "best_of":
        output["team"] = _aggregate_best_of(per_models)
    elif args.mode == "router":
        output["team"] = _aggregate_router(per_models, args.router, args.seed)
    elif args.mode == "chain":
        chain_metrics, traces = _eval_chain_team(
            model_paths=model_paths,
            records=records,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            router=args.router,
            seed=args.seed,
            trace_samples=max(0, args.chain_trace_samples),
        )
        output["team"] = chain_metrics
        output["team_trace_samples"] = traces
        if args.chain_trace_output:
            trace_path = Path(args.chain_trace_output)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w") as f:
                for rec in traces:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
