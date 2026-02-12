# Helpers to turn multi-agent conversations into flat samples for training.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from verl.teamtr.context_rollout import Turn


@dataclass
class FlatSample:
    prompt: str
    response: str
    agent_id: int


def flatten_turns(turns_per_prompt: Sequence[Sequence[Turn]]) -> List[FlatSample]:
    """Flatten multi-turn conversations into per-turn samples with agent_id."""
    flat: List[FlatSample] = []
    for convo in turns_per_prompt:
        for t in convo:
            flat.append(FlatSample(prompt=t.prompt, response=t.output, agent_id=t.agent_id))
    return flat


def to_dataframe(turns_per_prompt: Sequence[Sequence[Turn]]) -> pd.DataFrame:
    """Convert flattened samples to a DataFrame (prompt, response, agent_id)."""
    rows = []
    for convo in turns_per_prompt:
        for t in convo:
            rows.append({"prompt": t.prompt, "response": t.output, "agent_id": t.agent_id})
    return pd.DataFrame(rows)


@dataclass
class TurnSample:
    context: str
    response: str
    agent_id: int
    turn_id: int
    conversation_id: int


def flatten_turns_with_context(
    turns_per_prompt: Sequence[Sequence[Turn]], sep: str = "\n"
) -> List[TurnSample]:
    """Flatten multi-turn conversations into per-turn samples with history context."""
    flat: List[TurnSample] = []
    for conv_id, convo in enumerate(turns_per_prompt):
        history: List[str] = []
        for turn_id, t in enumerate(convo):
            context = getattr(t, "context", None)
            if context is None:
                prompt = getattr(t, "prompt", "")
                if prompt:
                    context = sep.join(history + [prompt])
                else:
                    context = sep.join(history)
            flat.append(
                TurnSample(
                    context=context,
                    response=t.output,
                    agent_id=t.agent_id,
                    turn_id=turn_id,
                    conversation_id=conv_id,
                )
            )
            history.extend([t.prompt, t.output])
    return flat


def to_dataframe_multi_turn(turns_per_prompt: Sequence[Sequence[Turn]]) -> pd.DataFrame:
    """Convert multi-turn samples to a DataFrame (context, response, agent_id, turn_id, conversation_id)."""
    rows = []
    for conv_id, convo in enumerate(turns_per_prompt):
        history: List[str] = []
        for turn_id, t in enumerate(convo):
            context = getattr(t, "context", None)
            if context is None:
                prompt = getattr(t, "prompt", "")
                if prompt:
                    context = "\n".join(history + [prompt])
                else:
                    context = "\n".join(history)
            rows.append(
                {
                    "context": context,
                    "response": t.output,
                    "agent_id": t.agent_id,
                    "turn_id": turn_id,
                    "conversation_id": conv_id,
                }
            )
            history.extend([t.prompt, t.output])
    return pd.DataFrame(rows)
