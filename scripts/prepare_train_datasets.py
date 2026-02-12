#!/usr/bin/env python3
"""Prepare TeamTR train/val datasets in processed parquet format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


RAW_DEEPSCALER_ARROW = Path(
    "${TEAMTR_DATASETS_ROOT:-./data}/deepscaler_preview/hf_arrow/train/data-00000-of-00001.arrow"
)
DEEPSCALER_OUT_DIR = Path("${TEAMTR_DATASETS_ROOT:-./data}/deepscaler_preview/processed")

RAW_DAPO_ARROW_DIR = Path("${TEAMTR_DATASETS_ROOT:-./data}/dapo_math_17k/hf_arrow/train")
DAPO_OUT_DIR = Path("${TEAMTR_DATASETS_ROOT:-./data}/dapo_math_17k/processed")


def _wrap_problem(problem: str) -> str:
    return (
        "Solve the following math problem step by step. The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        f"{problem.strip()}"
    )


def _load_raw_table(path: Path) -> pd.DataFrame:
    with pa.memory_map(str(path)) as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()
    return table.to_pandas()


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
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
        return _message_content_to_text(messages[0].get("content", ""))

    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", "")).strip()
            content = _message_content_to_text(msg.get("content", ""))
            if not content:
                continue
            lines.append(f"{role}: {content}" if role else content)
        else:
            lines.append(str(msg))
    return "\n".join(lines)


def _extract_prompt_text(prompt: Any) -> str:
    text = _messages_to_text(prompt)
    return text.strip()


def _build_deepscaler_df(raw: pd.DataFrame) -> pd.DataFrame:
    problems = raw["problem"].astype(str).tolist()
    answers = raw["answer"].astype(str).tolist()
    solutions = raw["solution"].astype(str).tolist()

    rows = {
        "context": [_wrap_problem(p) for p in problems],
        "response": ["" for _ in problems],
        "agent_id": [-1 for _ in problems],
        "turn_id": [0 for _ in problems],
        "conversation_id": list(range(len(problems))),
        "data_source": ["deepscaler" for _ in problems],
        "reward_model": [{"ground_truth": ans, "style": "deepscaler_preview"} for ans in answers],
        "extra_info": [{"solution": sol} for sol in solutions],
    }
    return pd.DataFrame(rows)


def _build_dapo_df(dataset: datasets.Dataset) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(dataset):
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict):
            reward_model = {}
        ground_truth = str(reward_model.get("ground_truth", ""))
        style = reward_model.get("style", "rule-lighteval/MATH_v2")

        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}

        rows.append(
            {
                "context": _extract_prompt_text(row.get("prompt", "")),
                "response": "",
                "agent_id": -1,
                "turn_id": 0,
                "conversation_id": idx,
                "data_source": row.get("data_source", "math_dapo"),
                "reward_model": {"ground_truth": ground_truth, "style": style},
                "extra_info": extra_info,
            }
        )
    return pd.DataFrame(rows)


def _split_train_val(df: pd.DataFrame, val_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise SystemExit("Need at least 2 rows to split train/val.")
    val_size = min(max(1, val_size), len(df) - 1)
    val_df = df.sample(n=val_size, random_state=seed)
    train_df = df.drop(val_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _existing_val_size(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(pd.read_parquet(path, engine="pyarrow"))
    except Exception:
        return None


def prepare_deepscaler(*, force: bool, seed: int) -> None:
    if not RAW_DEEPSCALER_ARROW.exists():
        raise SystemExit(f"Missing raw DeepScaleR arrow: {RAW_DEEPSCALER_ARROW}")

    DEEPSCALER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out = DEEPSCALER_OUT_DIR / "train.parquet"
    val_out = DEEPSCALER_OUT_DIR / "val.parquet"
    if not force and train_out.exists() and val_out.exists():
        print("Processed DeepScaleR train/val already exist; use --force to overwrite.")
        return

    raw_df = _load_raw_table(RAW_DEEPSCALER_ARROW)
    processed_df = _build_deepscaler_df(raw_df)

    val_size = _existing_val_size(val_out) or int(round(len(processed_df) * 0.02))
    train_df, val_df = _split_train_val(processed_df, val_size, seed)

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)
    print(f"Wrote {train_out} ({len(train_df)} rows)")
    print(f"Wrote {val_out} ({len(val_df)} rows)")


def _load_dapo_dataset() -> datasets.Dataset:
    if not RAW_DAPO_ARROW_DIR.exists():
        raise SystemExit(f"Missing DAPO arrow directory: {RAW_DAPO_ARROW_DIR}")
    arrow_files = sorted(str(p) for p in RAW_DAPO_ARROW_DIR.glob("*.arrow"))
    if not arrow_files:
        raise SystemExit(f"No .arrow files found under {RAW_DAPO_ARROW_DIR}")
    return datasets.load_dataset("arrow", data_files=arrow_files, split="train")


def _take_subset(ds: datasets.Dataset, max_rows: int, seed: int) -> datasets.Dataset:
    if max_rows <= 0 or len(ds) <= max_rows:
        return ds
    return ds.shuffle(seed=seed).select(range(max_rows))


def prepare_dapo(*, force: bool, seed: int, train_size: int, val_size: int) -> None:
    DAPO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out = DAPO_OUT_DIR / "train.parquet"
    val_out = DAPO_OUT_DIR / "val.parquet"
    if not force and train_out.exists() and val_out.exists():
        print("Processed DAPO train/val already exist; use --force to overwrite.")
        return

    raw_ds = _load_dapo_dataset()
    target_total = train_size + val_size
    subset_ds = _take_subset(raw_ds, target_total, seed)
    processed_df = _build_dapo_df(subset_ds)
    train_df, val_df = _split_train_val(processed_df, val_size, seed)

    if len(train_df) < train_size:
        print(
            f"Warning: requested dapo_train_size={train_size}, got {len(train_df)} "
            f"(raw subset size was {len(processed_df)})."
        )

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)
    print(f"Wrote {train_out} ({len(train_df)} rows)")
    print(f"Wrote {val_out} ({len(val_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits/subsampling")
    parser.add_argument("--dapo-train-size", type=int, default=17000, help="Target DAPO train rows")
    parser.add_argument("--dapo-val-size", type=int, default=500, help="Target DAPO val rows")
    args = parser.parse_args()

    if args.dapo_train_size < 1:
        raise SystemExit("--dapo-train-size must be >= 1")
    if args.dapo_val_size < 1:
        raise SystemExit("--dapo-val-size must be >= 1")

    prepare_deepscaler(force=args.force, seed=args.seed)
    prepare_dapo(force=args.force, seed=args.seed, train_size=args.dapo_train_size, val_size=args.dapo_val_size)


if __name__ == "__main__":
    main()
