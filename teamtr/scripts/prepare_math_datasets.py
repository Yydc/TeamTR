#!/usr/bin/env python3
"""
Download HF datasets and export to parquet for TeamTR multi-turn format.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import datasets
import pandas as pd


def _get_split_name(ds_dict, preferred: list[str]) -> Optional[str]:
    for name in preferred:
        if name in ds_dict:
            return name
    return None


def _get_nested_value(obj, dotted_key: str):
    value = obj.to_dict() if hasattr(obj, "to_dict") else obj
    for part in dotted_key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _normalize_prompt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                parts.append(f"{role}: {content}".strip())
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def _make_records(df: pd.DataFrame, prompt_key: str, response_key: str, data_source: str | None) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        prompt_val = row.get(prompt_key, None)
        response_val = row.get(response_key, None)
        if prompt_val is None and "." in prompt_key:
            prompt_val = _get_nested_value(row, prompt_key)
        if response_val is None and "." in response_key:
            response_val = _get_nested_value(row, response_key)

        prompt = _normalize_prompt(prompt_val)
        response = "" if response_val is None else str(response_val)
        reward_model_val = row.get("reward_model", None)
        if reward_model_val is None:
            reward_model_val = {"ground_truth": response}
        elif isinstance(reward_model_val, dict):
            reward_model_val = dict(reward_model_val)
            reward_model_val.setdefault("ground_truth", response)
        else:
            reward_model_val = {"ground_truth": str(reward_model_val)}

        data_source_val = data_source or row.get("data_source", None)
        extra_info_val = row.get("extra_info", None)
        if isinstance(extra_info_val, dict) and len(extra_info_val) == 0:
            extra_info_val = None
        rows.append(
            {
                "context": prompt,
                "response": response,
                "agent_id": -1,
                "turn_id": 0,
                "conversation_id": idx,
                "data_source": data_source_val,
                "reward_model": reward_model_val,
                "extra_info": extra_info_val,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare math datasets for TeamTR.")
    parser.add_argument("--dataset-name", required=True, help="HF dataset name.")
    parser.add_argument("--config-name", default=None, help="HF dataset config name.")
    parser.add_argument("--prompt-key", required=True, help="Column for prompts.")
    parser.add_argument("--response-key", required=True, help="Column for responses.")
    parser.add_argument("--output-dir", required=True, help="Output directory for parquet files.")
    parser.add_argument("--train-split", default=None, help="Train split name override.")
    parser.add_argument("--val-split", default=None, help="Validation split name override.")
    parser.add_argument("--val-size", type=float, default=0.02, help="Validation split ratio if missing.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for splitting.")
    parser.add_argument("--max-train", type=int, default=None, help="Limit number of train samples.")
    parser.add_argument("--max-val", type=int, default=None, help="Limit number of val samples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before truncating.")
    parser.add_argument("--data-source", default=None, help="Override data_source value.")
    args = parser.parse_args()

    ds = datasets.load_dataset(args.dataset_name, args.config_name) if args.config_name else datasets.load_dataset(
        args.dataset_name
    )

    train_split = args.train_split or _get_split_name(ds, ["train"])
    val_split = args.val_split or _get_split_name(ds, ["validation", "val", "test"])

    if train_split is None:
        raise ValueError("No train split found. Use --train-split to specify.")

    train_ds = ds[train_split]
    if val_split is None:
        split = train_ds.train_test_split(test_size=args.val_size, seed=args.seed)
        train_ds = split["train"]
        val_ds = split["test"]
    else:
        val_ds = ds[val_split]

    if args.shuffle:
        train_ds = train_ds.shuffle(seed=args.seed)
        val_ds = val_ds.shuffle(seed=args.seed) if val_ds is not None else val_ds

    if args.max_train is not None:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    if args.max_val is not None and val_ds is not None:
        val_ds = val_ds.select(range(min(args.max_val, len(val_ds))))

    train_df = train_ds.to_pandas()
    val_df = val_ds.to_pandas()

    train_out = _make_records(train_df, args.prompt_key, args.response_key, args.data_source)
    val_out = _make_records(val_df, args.prompt_key, args.response_key, args.data_source)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)

    print(f"Wrote train to {train_path} ({len(train_out)} rows)")
    print(f"Wrote val to {val_path} ({len(val_out)} rows)")


if __name__ == "__main__":
    main()
