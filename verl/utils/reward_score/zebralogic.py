"""Exact-match reward for ZebraLogic outputs."""
from __future__ import annotations

import json
import re
from typing import Any


def _normalize(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, sort_keys=True, ensure_ascii=True)
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def compute_score(solution_str: str, ground_truth: Any) -> float:
    pred = _normalize(solution_str)
    gold = _normalize(ground_truth)
    return 1.0 if pred == gold else 0.0
