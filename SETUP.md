# TeamTR Setup Guide

This guide provides detailed instructions for setting up TeamTR on your system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Environment Configuration](#environment-configuration)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Setup](#model-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

- **CPU**: 16+ cores recommended for data processing
- **GPU**:
  - Minimum: 1x A100 (40GB) or H100 (80GB)
  - Recommended: 4-8x H100/H200 for multi-agent training
  - Compute capability: 8.0+ (Ampere or newer)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ for models, datasets, and checkpoints

### Software

- **OS**: Linux (Ubuntu 20.04+, CentOS 7+)
- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: 12.1 or higher
- **Driver**: NVIDIA driver 525+ for CUDA 12.1

---

## Installation Methods

### Method 1: Micromamba (Recommended)

Micromamba is a lightweight, fast conda alternative.

```bash
# 1. Install micromamba (if not already installed)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# 2. Clone TeamTR
git clone https://github.com/your-org/TeamTR.git
cd TeamTR

# 3. Create environment
micromamba create -f teamtr-env.yml -y

# 4. Activate environment
micromamba activate TeamTR

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Method 2: Conda/Mamba

```bash
# 1. Clone repository
git clone https://github.com/your-org/TeamTR.git
cd TeamTR

# 2. Create environment
conda env create -f teamtr-env.yml
# OR: mamba env create -f teamtr-env.yml

# 3. Activate
conda activate TeamTR
```

### Method 3: pip (Manual Dependency Installation)

```bash
# 1. Clone repository
git clone https://github.com/your-org/TeamTR.git
cd TeamTR

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install PyTorch (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install flash-attention (requires CUDA)
# Set CUDA architecture list for your GPU (9.0 for H100/H200)
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" pip install flash-attn --no-build-isolation

# 5. Install TeamTR with dependencies
pip install -e .[gpu]

# If the above fails, install from requirements.txt
pip install -r requirements.txt
```

### Method 4: Docker (Coming Soon)

```bash
# Pull pre-built image
docker pull your-org/teamtr:latest

# Run container
docker run --gpus all -it your-org/teamtr:latest
```

---

## Environment Configuration

### Step 1: Copy Environment Template

```bash
cp .env.example .env
```

### Step 2: Edit Configuration

Open `.env` and set the following paths:

```bash
# Required: Dataset and model locations
TEAMTR_DATASETS_ROOT=/path/to/your/datasets
TEAMTR_MODELS_ROOT=/path/to/your/models

# Optional: Training files
TEAMTR_TRAIN_FILES=${TEAMTR_DATASETS_ROOT}/train.parquet
TEAMTR_VAL_FILES=${TEAMTR_DATASETS_ROOT}/val.parquet

# Optional: Specific model paths
TEAMTR_MODEL_1P7B=${TEAMTR_MODELS_ROOT}/Qwen3-1.7B
TEAMTR_MODEL_8B=${TEAMTR_MODELS_ROOT}/Qwen3-8B
TEAMTR_MODEL_14B=${TEAMTR_MODELS_ROOT}/Qwen3-14B

# Optional: Output directories
TEAMTR_OUTPUTS_DIR=./outputs
TEAMTR_LOGS_DIR=./logs
```

### Step 3: Load Environment

```bash
# For bash/zsh
source .env

# Or use direnv (auto-loads .env when entering directory)
sudo apt install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
direnv allow .
```

---

## Dataset Preparation

### Dataset Format

TeamTR requires parquet files with the following schema:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `conversation_id` | int | Yes | Unique ID for each conversation |
| `turn_id` | int | Yes | Turn index (0, 1, 2, ...) |
| `agent_id` | int | Yes | Agent ID (0-indexed) |
| `prompt` | str | Yes | Input prompt for this turn |
| `response` | str | Yes | Agent's response |
| `context` | str | No | Full conversation history (auto-generated if missing) |

### Example Dataset Creation

```python
import pandas as pd
from pathlib import Path

# Create sample data
data = [
    {
        "conversation_id": 0,
        "turn_id": 0,
        "agent_id": 0,
        "prompt": "What is 2+2?",
        "response": "4"
    },
    {
        "conversation_id": 0,
        "turn_id": 1,
        "agent_id": 1,
        "prompt": "4",
        "response": "That's correct!"
    },
    # Add more samples...
]

# Save to parquet
df = pd.DataFrame(data)
output_path = Path("data/train.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path, index=False)

print(f"✓ Saved {len(df)} samples to {output_path}")
```

### Data Preparation Scripts

TeamTR includes scripts to convert common formats:

```bash
# Convert arrow files to parquet
python scripts/prepare_train_datasets.py \
  --input data/raw/*.arrow \
  --output data/processed/train.parquet

# Prepare evaluation datasets
python scripts/prepare_eval_datasets.py \
  --input data/eval/*.jsonl \
  --output data/processed/val.parquet
```

---

## Model Setup

### Option 1: Download Pre-trained Models

```bash
# Create models directory
mkdir -p models

# Download from Hugging Face
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B

# Or use git lfs
cd models
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B
```

### Option 2: Use Existing Models

If you already have models, create symlinks:

```bash
ln -s /path/to/existing/models ./models
```

### Option 3: Use Model Identifiers

You can also use Hugging Face model IDs directly in configs:

```yaml
actor_rollout_ref:
  actor:
    model:
      path: "Qwen/Qwen3-8B"  # Will download automatically
```

---

## Verification

### Test 1: Import Check

```bash
python -c "
from verl.teamtr import RandomRouter, MultiAgentContextRollout, Turn
from teamtr.config import print_config
print('✓ All imports successful')
print_config()
"
```

### Test 2: Configuration Check

```bash
python teamtr/config.py
```

Expected output:
```
============================================================
TeamTR Configuration
============================================================
Project Root:    /path/to/TeamTR
Datasets Root:   /path/to/datasets
Models Root:     /path/to/models
Outputs Dir:     ./outputs
Logs Dir:        ./logs
============================================================
```

### Test 3: Unit Tests

```bash
./run_tests.sh unit
```

### Test 4: Smoke Test (Requires Models & Data)

```bash
export TEAMTR_CONFIG_NAME="teamtr_quick_test"
export TEAMTR_TOTAL_TRAINING_STEPS=5

python scripts/run_pipeline.py --mode server_smoke
```

---

## Troubleshooting

### Issue 1: Flash-Attention Build Fails

**Error**: `RuntimeError: No GPUs found during flash-attn build`

**Solution**:
```bash
# Set CUDA architecture for your GPU
# H100/H200: 9.0, A100: 8.0, RTX 4090: 8.9
export TORCH_CUDA_ARCH_LIST="9.0"
pip install flash-attn --no-build-isolation
```

### Issue 2: CUDA Version Mismatch

**Error**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**:
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: Ray Import Errors

**Error**: `ModuleNotFoundError: No module named 'ray'`

**Solution**:
```bash
pip install ray[default]

# Or reinstall with all dependencies
pip install -e .[gpu]
```

### Issue 4: Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. **Use LoRA**: Reduces memory by ~50%
   ```bash
   export TEAMTR_CONFIG_NAME="teamtr_base_lora"
   ```

2. **Reduce Batch Size**:
   ```bash
   export TEAMTR_TRAIN_BATCH_SIZE=16
   export TEAMTR_GEN_BATCH_SIZE=32
   ```

3. **Use Smaller Models**:
   ```bash
   export TEAMTR_MODEL_PATHS="model1.7b,model1.7b,model1.7b"
   ```

4. **Enable Gradient Checkpointing**: Already enabled in default configs

### Issue 5: Import Fails Due to Missing Dependencies

**Error**: Various `ModuleNotFoundError` messages

**Solution**:
```bash
# Install all optional dependencies
pip install -e .[gpu,math,test]

# Or install missing package individually
pip install <missing-package>
```

### Issue 6: Tokenizer Warnings

**Warning**: `Unsupported processor type: Qwen2TokenizerFast`

**Note**: This warning is harmless for text-only training. It only affects multimodal processors.

### Issue 7: Path Not Found

**Error**: `FileNotFoundError: Datasets root not found`

**Solution**:
1. Check `.env` file has correct paths
2. Create directories if they don't exist:
   ```bash
   mkdir -p data models outputs logs
   ```
3. Set environment variables:
   ```bash
   export TEAMTR_DATASETS_ROOT=/path/to/datasets
   ```

---

## Advanced Configuration

### Using Slurm

For cluster environments with Slurm:

```bash
# Copy and edit submission script
cp run_teamtr_slurm.sh my_experiment.sh

# Submit job
sbatch my_experiment.sh
```

### Custom Configurations

Create your own config by extending base configs:

```yaml
# configs/my_experiment.yaml
defaults:
  - teamtr_base_full

experiment_name: my_custom_experiment

trainer:
  total_epochs: 10

algorithm:
  adv_estimator: reinforce++
```

Run with:
```bash
python -m verl.trainer.main_ppo --config-name my_experiment
```

### Distributed Training

TeamTR uses Ray for distributed training. Configure in `.env`:

```bash
# Number of GPUs per node
export TEAMTR_N_GPUS_PER_NODE=8

# Ray head address (for multi-node)
export RAY_ADDRESS="auto"
```

---



