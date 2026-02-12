# TeamTR: Trust Region Fine-tuning for Multi-Agent LLM Teams

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

TeamTR is a framework for fine-tuning multi-agent large language model teams using trust region methods with token-decomposed KL divergence constraints and intermediate-occupancy evaluation.

## Key Features

- 🎯 **Token-Decomposed Trust Regions**: Per-agent KL constraints computed at token level
- 📊 **Intermediate-Occupancy Evaluation**: Reduces compounding occupancy shift from O(n²√δ) to O(n√δ)
- 🔄 **Stage-Wise Sequential Updates**: Block-coordinate optimization with certified improvement bounds
- 🤝 **Multi-Agent Infrastructure**: Heterogeneous teams with per-agent models, routers, and checkpointing
- ⚡ **Distributed Training**: Built on Ray and FSDP for scalable multi-GPU training

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Details](#training-details)
  - [Offline Stage-Wise Training](#offline-stage-wise-training)
  - [Online Multi-Agent Training](#online-multi-agent-training)
- [Testing & Evaluation](#testing--evaluation)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Architecture](#architecture)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- 16GB+ GPU memory per agent (H100/H200 recommended)

### Quick Install

```bash
# 1. Clone repository
git clone https://github.com/Yydc/TeamTR.git
cd TeamTR

# 2. Create environment (using micromamba - recommended)
micromamba create -f teamtr-env.yml -y
micromamba activate TeamTR

# 3. Configure paths
cp .env.example .env
# Edit .env to set your TEAMTR_DATASETS_ROOT and TEAMTR_MODELS_ROOT

# 4. Verify installation
python -c "from verl.teamtr import RandomRouter; print('✓ Installation successful')"
./run_tests.sh unit
```

For detailed installation instructions, see [SETUP.md](SETUP.md).

---

## Quick Start

### 1. Prepare Data

TeamTR expects parquet files with multi-turn conversations:

```python
import pandas as pd

data = pd.DataFrame([
    {"conversation_id": 0, "turn_id": 0, "agent_id": 0,
     "prompt": "What is 2+2?", "response": "4"},
    {"conversation_id": 0, "turn_id": 1, "agent_id": 1,
     "prompt": "4", "response": "That's correct!"},
])
data.to_parquet("data/train.parquet")
```

See [Data Format](#data-format) for details.

### 2. Run Quick Test

```bash
# Smoke test with minimal setup (requires models & data)
export TEAMTR_CONFIG_NAME="teamtr_quick_test"
export TEAMTR_TOTAL_TRAINING_STEPS=10
export TEAMTR_MODEL_PATHS="path/to/model1,path/to/model2,path/to/model3"

python -m verl.trainer.main_ppo \
    --config-name $TEAMTR_CONFIG_NAME \
    trainer.total_epochs=1
```

---

## Training Details

TeamTR supports two training modes:

### Offline Stage-Wise Training

**Block-coordinate approach**: Generate conversations offline, then train each agent independently.

#### Step 1: Generate Multi-Agent Conversations

```bash
python teamtr/run_bc_stage.py \
  --prompts data/prompts.txt \
  --model-paths "model1,model2,model3" \
  --output-dir outputs/stage_data \
  --max-turns 4 \
  --max-new-tokens 256 \
  --device cuda
```

**Output:**
- `outputs/stage_data/stage_data_all.parquet` - All samples
- `outputs/stage_data/stage_data_agent{0,1,2}.parquet` - Per-agent samples
- `outputs/stage_data/conversations.jsonl` - Raw conversations

#### Step 2: Train Each Agent with Block-Coordinate Filtering

The script above prints commands like:

```bash
# Agent 0
python -m verl.trainer.main_ppo \
    --config-name teamtr_base_full \
    data.train_files=outputs/stage_data/stage_data_agent0.parquet \
    actor_rollout_ref.actor.model.path=model1 \
    +teamtr_target_agent_id=0 \
    +experiment_name=stage1_agent0

# Agent 1
python -m verl.trainer.main_ppo \
    --config-name teamtr_base_full \
    data.train_files=outputs/stage_data/stage_data_agent1.parquet \
    actor_rollout_ref.actor.model.path=model2 \
    +teamtr_target_agent_id=1 \
    +experiment_name=stage1_agent1

# Agent 2 (similarly)
```

**Key Parameter:** `teamtr_target_agent_id` filters training data to specific agent.

### Online Multi-Agent Training

**End-to-end approach**: Rollouts collected under current team, agents updated sequentially.

#### Single-Node Training

```bash
# Example: 3x Qwen3-8B models, 4 GPUs
export TEAMTR_CONFIG_NAME="teamtr_homo_full"
export TEAMTR_EXPERIMENT_NAME="my_experiment"
export TEAMTR_MODEL_PATHS="Qwen/Qwen3-8B,Qwen/Qwen3-8B,Qwen/Qwen3-8B"
export TEAMTR_TRAIN_FILES="data/train.parquet"
export TEAMTR_VAL_FILES="data/val.parquet"
export TEAMTR_N_GPUS_PER_NODE=4

python -m verl.trainer.main_ppo \
    --config-name $TEAMTR_CONFIG_NAME \
    +experiment_name=$TEAMTR_EXPERIMENT_NAME
```

#### Slurm Cluster

```bash
# Edit Slurm parameters in run_teamtr_slurm.sh
export TEAMTR_CONFIG_NAME="teamtr_homo_full"
export TEAMTR_EXPERIMENT_NAME="cluster_run"
export TEAMTR_MODEL_PATHS="model1,model2,model3"
export TEAMTR_TRAIN_FILES="data/train.parquet"
export TEAMTR_TOTAL_TRAINING_STEPS=1000
export TEAMTR_N_GPUS_PER_NODE=8

sbatch run_teamtr_slurm.sh
```

#### Heterogeneous Teams

```bash
# Different model sizes
export TEAMTR_CONFIG_NAME="teamtr_hetero_full"
export TEAMTR_MODEL_PATHS="Qwen/Qwen3-1.7B,Qwen/Qwen3-8B,Qwen/Qwen3-14B"

python -m verl.trainer.main_ppo \
    --config-name $TEAMTR_CONFIG_NAME
```

#### LoRA Training (Low Memory)

```bash
export TEAMTR_CONFIG_NAME="teamtr_base_lora"
export TEAMTR_MODEL_PATHS="model1,model2,model3"

python -m verl.trainer.main_ppo \
    --config-name $TEAMTR_CONFIG_NAME \
    actor_rollout_ref.model.lora_rank=32
```

### Training Configuration via Environment Variables

```bash
# Core settings
export TEAMTR_CONFIG_NAME="teamtr_homo_full"        # Config file
export TEAMTR_EXPERIMENT_NAME="my_exp"              # Experiment name
export TEAMTR_MODEL_PATHS="m1,m2,m3"                # 3 model paths (comma-sep)

# Data
export TEAMTR_TRAIN_FILES="train.parquet"           # Training data
export TEAMTR_VAL_FILES="val.parquet"               # Validation data

# Training parameters
export TEAMTR_TOTAL_TRAINING_STEPS=1000             # Total steps
export TEAMTR_TRAIN_BATCH_SIZE=64                   # Training batch size
export TEAMTR_GEN_BATCH_SIZE=128                    # Generation batch size
export TEAMTR_ROLLOUT_N=4                           # Rollouts per prompt

# Multi-turn chaining (intermediate occupancy)
export TEAMTR_CHAIN_TURNS=3                         # Number of turns per chain
export TEAMTR_CHAIN_RANDOM_ORDER=1                  # Random agent order
export TEAMTR_CHAIN_SEED=42                         # Random seed

# Hardware
export TEAMTR_N_GPUS_PER_NODE=8                     # GPUs per node

# Output
export TEAMTR_OUTPUT_DIR="./outputs/my_exp"         # Output directory
export TEAMTR_SAVE_FREQ=100                         # Checkpoint frequency
export TEAMTR_TEST_FREQ=50                          # Validation frequency
```

---

## Testing & Evaluation

### Unit Tests

```bash
# Run all tests
./run_tests.sh

# Run specific test suite
./run_tests.sh unit

# Run with coverage
pytest tests/ --cov=verl --cov=teamtr --cov-report=html
```

### Model Evaluation

#### Single Model Evaluation

```bash
python scripts/run_model_eval.py \
    --model-path outputs/my_exp/checkpoints/final/agent0 \
    --dataset data/eval.parquet \
    --output-dir outputs/eval_results \
    --k 4 \
    --max-samples 500
```

#### Team Evaluation (All Agents)

```bash
python scripts/run_team_eval.py \
    --model-paths "model0,model1,model2" \
    --dataset data/eval.parquet \
    --output-dir outputs/team_eval \
    --k 8
```

#### Full Evaluation Pipeline

```bash
# Includes training + evaluation on multiple benchmarks
python scripts/run_pipeline.py \
    --mode train_eval_full_parallel \
    --config teamtr_homo_full \
    --profile 3x8b
```

**Supported Modes:**
- `server_smoke` - Quick smoke test
- `train_eval_full_parallel` - Full train + eval
- `eval_only` - Evaluation without training

### Evaluation Datasets

Configure via environment variables:

```bash
export TEAMTR_MATH500_PATH="data/math500/val.parquet"
export TEAMTR_AIME24_PATH="data/aime24/val.parquet"
export TEAMTR_AIME25_PATH="data/aime_2025/val.parquet"
```

---

## Configuration

TeamTR uses Hydra for configuration. Available configs in `teamtr/configs/`:

| Config | Description |
|--------|-------------|
| `teamtr_homo_full.yaml` | Homogeneous team (3x same model), full fine-tuning |
| `teamtr_hetero_full.yaml` | Heterogeneous team (1.7B, 8B, 14B), full fine-tuning |
| `teamtr_base_lora.yaml` | LoRA-based training (low memory) |
| `teamtr_quick_test.yaml` | Quick smoke test configuration |

### Key Configuration Parameters

```yaml
# teamtr/configs/teamtr_base_full.yaml

# Data
data:
  format: multi_agent_turn
  train_files: ["train.parquet"]
  agent_id_key: "agent_id"

# Algorithm
algorithm:
  adv_estimator: grpo              # Group-normalized advantages
  use_kl_in_reward: false

# Actor
actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_coef: 0.001             # Adaptive KL penalty coefficient
    kl_loss_type: low_var_kl        # Low-variance KL estimator
  rollout:
    name: multi_agent_hf            # Multi-agent HuggingFace rollout
    n: 4                            # Rollouts per prompt

# Multi-agent
teamtr_router: random               # Router type (random, round_robin)
teamtr_router_seed: 42
teamtr_target_agent_id: null        # null=train all, int=specific agent
```

### Overriding Config from Command Line

```bash
python -m verl.trainer.main_ppo \
    --config-name teamtr_base_full \
    trainer.total_epochs=5 \
    data.train_files=["new_data.parquet"] \
    algorithm.adv_estimator=reinforce++ \
    actor_rollout_ref.rollout.n=8 \
    +experiment_name=custom_run
```

---

## Data Format

TeamTR expects **multi-turn conversation data** in Parquet format:

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | int | Unique conversation ID |
| `turn_id` | int | Turn index (0, 1, 2, ...) |
| `agent_id` | int | Agent ID (0-indexed) |
| `prompt` | str | Input prompt for this turn |
| `response` | str | Agent's response |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `context` | str | Full conversation history (auto-generated if missing) |

### Example Data Creation

```python
import pandas as pd

conversations = [
    # Conversation 1: Agent 0 → Agent 1
    {"conversation_id": 0, "turn_id": 0, "agent_id": 0,
     "prompt": "Solve x^2 = 4", "response": "x = ±2"},
    {"conversation_id": 0, "turn_id": 1, "agent_id": 1,
     "prompt": "x = ±2", "response": "Correct! Both 2 and -2 satisfy the equation."},

    # Conversation 2: Agent 1 → Agent 0 → Agent 2
    {"conversation_id": 1, "turn_id": 0, "agent_id": 1,
     "prompt": "What is 5! ?", "response": "5! = 120"},
    {"conversation_id": 1, "turn_id": 1, "agent_id": 0,
     "prompt": "5! = 120", "response": "Yes, 5×4×3×2×1 = 120"},
    {"conversation_id": 1, "turn_id": 2, "agent_id": 2,
     "prompt": "Yes, 5×4×3×2×1 = 120", "response": "Great explanation!"},
]

df = pd.DataFrame(conversations)
df.to_parquet("data/train.parquet", index=False)

print(f"✓ Created dataset with {len(df)} turns across {df['conversation_id'].nunique()} conversations")
```

### Data Preparation Scripts

```bash
# Convert existing datasets
python scripts/prepare_train_datasets.py \
    --input data/raw/*.arrow \
    --output data/processed/train.parquet

python scripts/prepare_eval_datasets.py \
    --input data/eval/*.jsonl \
    --output data/processed/val.parquet
```

---

## Architecture

TeamTR implements stage-wise block-coordinate fine-tuning:

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
└─────────────────────────────────────────────────────────┘

Data Preparation
  └─> MultiAgentTurnDataset
       ├─ context_input_ids
       ├─ response_ids
       └─ agent_id
           ↓
Multi-Agent Rollout (bucketing by agent_id)
  └─> MultiAgentHFRollout
       ├─ Groups samples by agent_id
       └─ Generates with per-agent models
           ↓
Log Probability Computation
  └─> Per-agent actor/ref log probs
       └─ Token-level KL computation
           ↓
Advantages & Rewards
  └─> GRPO (Group-normalized)
       ├─ Group by prompt (uid)
       ├─ Normalize: (R - μ) / σ
       └─ Token-level masking (response_mask)
           ↓
Per-Agent Updates (Trust Region)
  └─> ActorRolloutRefWorker
       ├─ Filter by agent_id
       ├─ PPO with clipped objective
       ├─ Adaptive KL penalty (β)
       └─ Independent optimizer per agent
           ↓
Checkpointing
  └─> ./checkpoints/agent{0,1,2}/
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Training Loop** | [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) | Stage-wise PPO with agent routing |
| **Trust Region** | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) | Token-decomposed KL, adaptive controllers |
| **Multi-Agent Worker** | [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py) | Per-agent model management |
| **Multi-Agent Rollout** | [verl/workers/rollout/multi_agent_hf_rollout.py](verl/workers/rollout/multi_agent_hf_rollout.py) | Agent-bucketed generation |
| **Dataset** | [verl/utils/dataset/multi_agent_dataset.py](verl/utils/dataset/multi_agent_dataset.py) | Multi-turn conversation dataset |
| **Routers** | [verl/teamtr/router.py](verl/teamtr/router.py) | RandomRouter, RoundRobinRouter |
| **Context Rollout** | [verl/teamtr/context_rollout.py](verl/teamtr/context_rollout.py) | Offline conversation generation |

---

## Theoretical Framework

TeamTR is based on the following key results:

1. **Token-Decomposed KL** (Eq. 5): KL divergence computed per-token via chain rule
   ```
   KL(π||π') = E[Σ_u KL(π_u || π'_u)]
   ```

2. **Intermediate-Occupancy Surrogates** (Eq. 9): Evaluated under current team occupancy
   ```
   L_i^seq = (1-γ)^(-1) E_{s~d^{π̂^{i-1}}} [Â_{i-1}(s,a)]
   ```

3. **Occupancy-Shift Reduction** (Proposition 2): Quadratic→Linear scaling
   ```
   |E_{d^{π̂^{i-1}}} - E_{d^{π̂^0}}| ≤ (√2γ/(1-γ)) Σ_{k<i} √δ_k
   ```

4. **Stage-Wise Improvement Certificate** (Theorem 4):
   ```
   J(π̄) - J(π_cur) ≥ Σ_i L_i^seq - O(n√δ̄)
   ```

---

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'ray'`**
```bash
pip install -e .[gpu]  # Reinstall with all dependencies
```

**Issue: Out of Memory**
```bash
# Use LoRA
export TEAMTR_CONFIG_NAME="teamtr_base_lora"

# Or reduce batch sizes
export TEAMTR_TRAIN_BATCH_SIZE=16
export TEAMTR_GEN_BATCH_SIZE=32
```

**Issue: Flash-Attention build fails**
```bash
export TORCH_CUDA_ARCH_LIST="9.0"  # H100/H200
pip install flash-attn --no-build-isolation
```

See [SETUP.md](SETUP.md) for comprehensive troubleshooting.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

TeamTR is built on top of [VERL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning) by ByteDance.

---

## Contact & Support

- **GitHub**: [https://github.com/Yydc/TeamTR](https://github.com/Yydc/TeamTR)
- **Issues**: [GitHub Issues](https://github.com/Yydc/TeamTR/issues)
- **Author**: Yi Xie

For questions or collaboration, please open an issue on GitHub.
