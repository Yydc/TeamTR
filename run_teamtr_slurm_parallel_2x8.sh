#!/bin/bash
set -euo pipefail

CODE_ROOT="${TEAMTR_CODE_ROOT:-${CODE_ROOT}}"
FULL_SCRIPT="${TEAMTR_FULL_SCRIPT:-${TEAMTR_PROJECT_ROOT:-./TeamTR}/run_teamtr_slurm_full_8gpu_8h.sh}"

if [ ! -f "${FULL_SCRIPT}" ]; then
  echo "Missing full-run script: ${FULL_SCRIPT}"
  exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH."
  exit 1
fi

TRAIN_FILES="${TEAMTR_TRAIN_FILES:-${TEAMTR_DATASETS_ROOT:-./data}/dapo_math_17k/processed/train.parquet,${TEAMTR_DATASETS_ROOT:-./data}/deepscaler_preview/processed/train.parquet}"
VAL_FILES="${TEAMTR_VAL_FILES:-${TEAMTR_DATASETS_ROOT:-./data}/math500/processed/val.parquet}"

# 3x8B job
JOB1=$(TEAMTR_CODE_ROOT="${CODE_ROOT}" \
  TEAMTR_CONFIG_NAME="teamtr_homo_full" \
  TEAMTR_EXPERIMENT_NAME="teamtr_full_3x8b" \
  TEAMTR_MODEL_PATHS="${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B" \
  TEAMTR_TRAIN_FILES="${TRAIN_FILES}" \
  TEAMTR_VAL_FILES="${VAL_FILES}" \
  TEAMTR_OUTPUT_DIR="${CODE_ROOT}/outputs/full/3x8b" \
  sbatch --export=ALL "${FULL_SCRIPT}" | awk '{print $4}')

# 1.7B + 8B + 14B job
JOB2=$(TEAMTR_CODE_ROOT="${CODE_ROOT}" \
  TEAMTR_CONFIG_NAME="teamtr_hetero_full" \
  TEAMTR_EXPERIMENT_NAME="teamtr_full_1p7b_8b_14b" \
  TEAMTR_MODEL_PATHS="${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-1.7B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-14B" \
  TEAMTR_TRAIN_FILES="${TRAIN_FILES}" \
  TEAMTR_VAL_FILES="${VAL_FILES}" \
  TEAMTR_OUTPUT_DIR="${CODE_ROOT}/outputs/full/1p7b_8b_14b" \
  sbatch --export=ALL "${FULL_SCRIPT}" | awk '{print $4}')

echo "Submitted parallel 2x8 jobs: ${JOB1} (3x8B) and ${JOB2} (1p7b_8b_14b)"
echo "Logs: ${TEAMTR_LOGS_DIR:-./logs}/teamtr_${JOB1}.log and ${TEAMTR_LOGS_DIR:-./logs}/teamtr_${JOB2}.log"
