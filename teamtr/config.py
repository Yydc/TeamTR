"""TeamTR configuration management.

This module centralizes all path and environment configurations,
making the codebase portable across different environments.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ============================================================================
# Dataset Paths
# ============================================================================

DATASETS_ROOT = Path(os.environ.get(
    "TEAMTR_DATASETS_ROOT",
    PROJECT_ROOT / "data"  # Default: ./data in project root
))

# Training datasets
TRAIN_FILES_DEFAULT = os.environ.get(
    "TEAMTR_TRAIN_FILES",
    str(DATASETS_ROOT / "train.parquet")
)

# Validation datasets
VAL_FILES_DEFAULT = os.environ.get(
    "TEAMTR_VAL_FILES",
    str(DATASETS_ROOT / "val.parquet")
)

# Specific evaluation datasets
MATH500_DATASET = Path(os.environ.get(
    "TEAMTR_MATH500_PATH",
    DATASETS_ROOT / "math500/processed/val.parquet"
))

AIME24_DATASET = Path(os.environ.get(
    "TEAMTR_AIME24_PATH",
    DATASETS_ROOT / "aime24/processed/val.parquet"
))

AIME25_DATASET = Path(os.environ.get(
    "TEAMTR_AIME25_PATH",
    DATASETS_ROOT / "aime_2025/processed/val.parquet"
))

ZEBRALOGIC_DATASET = Path(os.environ.get(
    "TEAMTR_ZEBRALOGIC_PATH",
    DATASETS_ROOT / "ZebraLogic_grid_mode/processed/val.parquet"
))

# ============================================================================
# Model Paths
# ============================================================================

MODELS_ROOT = Path(os.environ.get(
    "TEAMTR_MODELS_ROOT",
    PROJECT_ROOT / "models"  # Default: ./models in project root
))

# Default model paths for different sizes
MODEL_1P7B_DEFAULT = os.environ.get(
    "TEAMTR_MODEL_1P7B",
    str(MODELS_ROOT / "Qwen__Qwen3-1.7B")
)

MODEL_8B_DEFAULT = os.environ.get(
    "TEAMTR_MODEL_8B",
    str(MODELS_ROOT / "Qwen__Qwen3-8B")
)

MODEL_14B_DEFAULT = os.environ.get(
    "TEAMTR_MODEL_14B",
    str(MODELS_ROOT / "Qwen__Qwen3-14B")
)

# ============================================================================
# Output Directories
# ============================================================================

OUTPUTS_DIR = Path(os.environ.get(
    "TEAMTR_OUTPUTS_DIR",
    PROJECT_ROOT / "outputs"
))

LOGS_DIR = Path(os.environ.get(
    "TEAMTR_LOGS_DIR",
    PROJECT_ROOT / "logs"
))

# ============================================================================
# Environment Configuration
# ============================================================================

MAMBA_ROOT_PREFIX = os.environ.get(
    "MAMBA_ROOT_PREFIX",
    os.path.expanduser("~/.micromamba")
)

MAMBA_ENV_NAME = os.environ.get(
    "TEAMTR_ENV_NAME",
    "TeamTR"
)

# ============================================================================
# Cluster/Slurm Configuration
# ============================================================================

SERVER_GPU_LIMIT = int(os.environ.get("TEAMTR_GPU_LIMIT", "16"))
ALLOWED_PARALLEL_SHAPES = {"2x8", "4x4"}

# ============================================================================
# Helper Functions
# ============================================================================

def validate_paths(require_datasets: bool = False, require_models: bool = False):
    """Validate that required paths exist.

    Args:
        require_datasets: If True, raise error if dataset paths don't exist
        require_models: If True, raise error if model paths don't exist

    Raises:
        FileNotFoundError: If required paths don't exist
    """
    errors = []

    if require_datasets:
        if not DATASETS_ROOT.exists():
            errors.append(
                f"Datasets root not found: {DATASETS_ROOT}\n"
                f"Set TEAMTR_DATASETS_ROOT environment variable or create ./data directory"
            )

    if require_models:
        if not MODELS_ROOT.exists():
            errors.append(
                f"Models root not found: {MODELS_ROOT}\n"
                f"Set TEAMTR_MODELS_ROOT environment variable or create ./models directory"
            )

    if errors:
        raise FileNotFoundError("\n\n".join(errors))


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("TeamTR Configuration")
    print("=" * 60)
    print(f"Project Root:    {PROJECT_ROOT}")
    print(f"Datasets Root:   {DATASETS_ROOT}")
    print(f"Models Root:     {MODELS_ROOT}")
    print(f"Outputs Dir:     {OUTPUTS_DIR}")
    print(f"Logs Dir:        {LOGS_DIR}")
    print("=" * 60)
    print("\nTo customize, set environment variables:")
    print("  TEAMTR_DATASETS_ROOT  - Root directory for datasets")
    print("  TEAMTR_MODELS_ROOT    - Root directory for models")
    print("  TEAMTR_OUTPUTS_DIR    - Output directory")
    print("  TEAMTR_LOGS_DIR       - Logs directory")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
