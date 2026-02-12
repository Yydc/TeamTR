#!/usr/bin/env python3
"""Prepare eval datasets into TeamTR processed parquet format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from datasets import load_from_disk


DATASETS_ROOT = Path("${TEAMTR_DATASETS_ROOT:-./data}")


def _to_json_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)


def _build_rows(
    records: Iterable[Dict[str, Any]],
    *,
    prompt_key: str,
    answer_key: str,
    data_source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(records):
        prompt = ex[prompt_key]
        answer = ex[answer_key]
        answer_str = _to_json_str(answer)
        rows.append(
            {
                "context": prompt,
                "response": answer_str,
                "agent_id": -1,
                "turn_id": 0,
                "conversation_id": idx,
                "data_source": data_source,
                "reward_model": {"ground_truth": answer_str},
                "extra_info": None,
            }
        )
    return rows


def _write_parquet(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)


def prepare_dataset(name: str, split: str, prompt_key: str, answer_key: str, data_source: str) -> Path:
    dataset_dir = DATASETS_ROOT / name
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset not found: {dataset_dir}")
    ds = load_from_disk(str(dataset_dir))
    if split not in ds:
        raise SystemExit(f"Split {split} not found in {name}: {list(ds.keys())}")
    records = ds[split]
    rows = _build_rows(records, prompt_key=prompt_key, answer_key=answer_key, data_source=data_source)
    out_path = dataset_dir / "processed" / "val.parquet"
    _write_parquet(rows, out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed files")
    args = parser.parse_args()

    jobs = [
        ("aime24", "test", "problem", "solution", "aime24"),
        ("aime_2025", "train", "problem", "answer", "aime25"),
        ("ZebraLogic_grid_mode", "test", "puzzle", "solution", "zebralogic"),
    ]

    for name, split, prompt_key, answer_key, data_source in jobs:
        out_path = DATASETS_ROOT / name / "processed" / "val.parquet"
        if out_path.exists() and not args.force:
            continue
        prepare_dataset(name, split, prompt_key, answer_key, data_source)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
