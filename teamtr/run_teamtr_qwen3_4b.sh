#!/usr/bin/env bash
# Minimal TeamTR example with 3 agents sharing Qwen3-4B as the base model.
# Requires: data parquet paths set via DATA_TRAIN/ DATA_VAL envs (or edit below),
# Ray cluster available (single-node is fine), and verl installed.

set -euo pipefail
set -x

DATA_TRAIN=${DATA_TRAIN:-"$HOME/data/gsm8k/train.parquet"}
DATA_VAL=${DATA_VAL:-"$HOME/data/gsm8k/test.parquet"}
TEAMTR_MODEL_PATHS=${TEAMTR_MODEL_PATHS:-"${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B"}

python3 - <<'PY'
import os

from omegaconf import OmegaConf

from verl.teamtr.runner import TeamTRRunner, AgentSpec
from verl.teamtr.router import RandomRouter

# 1) Base config: load PPO template and apply common overrides
base_cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
overrides = {
    "data.train_files": os.environ.get("DATA_TRAIN", ""),
    "data.val_files": os.environ.get("DATA_VAL", ""),
    "data.train_batch_size": 512,
    "data.max_prompt_length": 512,
    "data.max_response_length": 768,
    "data.truncation": "error",
    "actor_rollout_ref.actor.ppo_mini_batch_size": 128,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 16,
    "actor_rollout_ref.actor.use_kl_loss": True,
    "actor_rollout_ref.actor.kl_loss_coef": 0.001,
    "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff": 0.0,
    "actor_rollout_ref.model.enable_gradient_checkpointing": True,
    "actor_rollout_ref.rollout.name": "hf",
    "actor_rollout_ref.rollout.n": 4,
    "actor_rollout_ref.rollout.max_model_len": 2048,
    "actor_rollout_ref.ref.fsdp_config.param_offload": True,
    "critic.enabled": False,  # GRPO/DAPO 可选地移除 value head
    "trainer.logger": '["console"]',
    "trainer.project_name": "teamtr_qwen3_4b",
    "trainer.experiment_name": "teamtr_grpo_stagewise_multiagent",
    "trainer.n_gpus_per_node": 4,
    "trainer.nnodes": 1,
    "trainer.save_freq": 50,
    "trainer.test_freq": 10,
    "trainer.total_epochs": 3,
}
for k, v in overrides.items():
    OmegaConf.update(base_cfg, k, v, merge=True)

# 2) Define agents (K=3) – default to local Qwen3-8B, override via TEAMTR_MODEL_PATHS
model_paths = [p.strip() for p in os.environ.get("TEAMTR_MODEL_PATHS", "").split(",") if p.strip()]
if len(model_paths) != 3:
    raise ValueError("TEAMTR_MODEL_PATHS must provide exactly 3 comma-separated model paths.")
agents = [AgentSpec(model_path=mpath) for mpath in model_paths]

# 3) Trust radii (token-KL) per agent
trust = [0.05, 0.05, 0.05]

# 4) Build and run a stage with default order (random permutation)
runner = TeamTRRunner(
    base_cfg=base_cfg,
    agents=agents,
    trust_radii=trust,
    algo="teamtr-grpo",  # also supports teamtr-dapo / teamtr-gmpo
    router=RandomRouter(num_agents=len(agents)),
)
# NOTE: This script still trains one agent at a time; to use block coordinate
# updates, set TEAMTR_TARGET_AGENT_ID in env or override in cfg.
runner.run_stage()
PY
