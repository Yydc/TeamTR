#!/usr/bin/env python3
"""
Block-coordinate (stage-wise) driver for TeamTR.

Workflow:
1) Provide a list of prompts and per-agent HF model paths.
2) Generate multi-agent conversations under the current team using a router (default random).
3) Flatten to per-agent parquet files (prompt, response, agent_id).
4) Print suggested PPO commands to update each agent with block-coordinate filtering (`teamtr_target_agent_id`).

This is an offline approximation of the stage-wise loop: rollouts are collected under the
current team, then each agent is trained independently using its own samples.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from verl.teamtr.context_rollout import MultiAgentContextRollout
from verl.teamtr.data_utils import flatten_turns
from verl.teamtr.router import RandomRouter


def parse_args():
    p = argparse.ArgumentParser(description="TeamTR block-coordinate stage driver")
    p.add_argument("--prompts", type=str, required=True, help="Text file, one prompt per line")
    p.add_argument(
        "--model-paths",
        type=str,
        required=True,
        help="Comma-separated HF model paths for each agent (order defines agent_id)",
    )
    p.add_argument("--output-dir", type=str, default="teamtr_outputs", help="Where to write parquet/conversations")
    p.add_argument("--max-turns", type=int, default=4, help="Max conversation turns")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per agent reply")
    p.add_argument("--device", type=str, default="cuda", help="Device for HF generation")
    p.add_argument("--dtype", type=str, default="bfloat16", help="Torch dtype for generation")
    return p.parse_args()


def load_prompts(path: str):
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return [ln for ln in lines if ln]


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)
    model_paths = args.model_paths.split(",")
    num_agents = len(model_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    router = RandomRouter(num_agents=num_agents)
    rollout = MultiAgentContextRollout(
        model_paths=model_paths,
        router=router,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"[TeamTR] Generating conversations for {len(prompts)} prompts with {num_agents} agents...")
    all_turns = []
    for p in prompts:
        turns = rollout.run(init_prompt=p, max_turns=args.max_turns)
        all_turns.append(turns)
    # save raw conversations
    conv_path = Path(args.output_dir) / "conversations.jsonl"
    with conv_path.open("w") as f:
        for convo in all_turns:
            rec = [
                {"agent_id": t.agent_id, "prompt": t.prompt, "output": t.output, "context": t.context}
                for t in convo
            ]
            f.write(str(rec) + "\n")
    # flatten and split per agent
    flat = flatten_turns(all_turns)
    df = pd.DataFrame([{"prompt": s.prompt, "response": s.response, "agent_id": s.agent_id} for s in flat])
    parquet_all = Path(args.output_dir) / "stage_data_all.parquet"
    df.to_parquet(parquet_all, index=False)
    print(f"[TeamTR] Wrote all samples to {parquet_all}")

    for aid in range(num_agents):
        df_agent = df[df.agent_id == aid]
        out_path = Path(args.output_dir) / f"stage_data_agent{aid}.parquet"
        df_agent.to_parquet(out_path, index=False)
        print(f"[TeamTR] agent {aid} samples: {len(df_agent)}, saved to {out_path}")

    print("\n[TeamTR] Suggested PPO commands per agent (block coordinate update):")
    for aid, mpath in enumerate(model_paths):
        out_path = Path(args.output_dir) / f"stage_data_agent{aid}.parquet"
        cmd = (
            "python -m verl.trainer.main_ppo "
            f"data.train_files={out_path} "
            f"actor_rollout_ref.actor.model.path={mpath} "
            f"actor_rollout_ref.ref.model.path={mpath} "
            "algorithm.adv_estimator=grpo "
            "actor_rollout_ref.actor.use_kl_loss=True "
            f"teamtr_target_agent_id={aid} "
            f"teamtr_agent_id={aid} "
            "trainer.total_epochs=1 "
            "trainer.logger='[\"console\"]'"
        )
        print(f"  # agent {aid}\n  {cmd}\n")


if __name__ == "__main__":
    sys.exit(main())
