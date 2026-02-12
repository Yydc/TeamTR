#!/bin/bash
#SBATCH --job-name=teamtr_eval
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu&zone-sof1
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=${TEAMTR_LOGS_DIR:-./logs}/teamtr_eval_%j.log
#SBATCH --error=${TEAMTR_LOGS_DIR:-./logs}/teamtr_eval_%j.error
#SBATCH --mail-type=ALL

set -euo pipefail

CODE_ROOT="${TEAMTR_CODE_ROOT:-${CODE_ROOT}}"
cd "${CODE_ROOT}"

export TEAMTR_ENV_NAME="${TEAMTR_ENV_NAME:-TeamTR}"
export TEAMTR_MAMBA_ROOT_PREFIX="${TEAMTR_MAMBA_ROOT_PREFIX:-${TEAMTR_MAMBA_ROOT_PREFIX}}"
export TEAMTR_MODEL_PATH="${TEAMTR_MODEL_PATH:?TEAMTR_MODEL_PATH is required}"
export TEAMTR_EVAL_DATASET="${TEAMTR_EVAL_DATASET:?TEAMTR_EVAL_DATASET is required}"
export TEAMTR_EVAL_OUTPUT="${TEAMTR_EVAL_OUTPUT:?TEAMTR_EVAL_OUTPUT is required}"
export TEAMTR_EVAL_K="${TEAMTR_EVAL_K:-4}"
export TEAMTR_EVAL_MAX_SAMPLES="${TEAMTR_EVAL_MAX_SAMPLES:-16}"
export TEAMTR_EVAL_MAX_NEW_TOKENS="${TEAMTR_EVAL_MAX_NEW_TOKENS:-256}"
export TEAMTR_EVAL_TEMPERATURE="${TEAMTR_EVAL_TEMPERATURE:-0.8}"
export TEAMTR_EVAL_TOP_P="${TEAMTR_EVAL_TOP_P:-1.0}"

RUNNER=(micromamba run -r "${TEAMTR_MAMBA_ROOT_PREFIX}" -n "${TEAMTR_ENV_NAME}")

srun -- "${RUNNER[@]}" bash -c "set -euo pipefail; \
  export PYTHONPATH=${CODE_ROOT}; \
  export TOKENIZERS_PARALLELISM=false; \
  python -u ${CODE_ROOT}/scripts/run_model_eval.py \
    --model-path ${TEAMTR_MODEL_PATH} \
    --dataset ${TEAMTR_EVAL_DATASET} \
    --k ${TEAMTR_EVAL_K} \
    --max-samples ${TEAMTR_EVAL_MAX_SAMPLES} \
    --max-new-tokens ${TEAMTR_EVAL_MAX_NEW_TOKENS} \
    --temperature ${TEAMTR_EVAL_TEMPERATURE} \
    --top-p ${TEAMTR_EVAL_TOP_P} \
    --output ${TEAMTR_EVAL_OUTPUT}"
