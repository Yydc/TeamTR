#!/bin/bash
#SBATCH --job-name=teamtr_quick_6h
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=${TEAMTR_PROJECT_ROOT:-./TeamTR}/teamtr_%j.log
#SBATCH --error=${TEAMTR_PROJECT_ROOT:-./TeamTR}/teamtr_%j.error
#SBATCH --mail-type=ALL

cd ${TEAMTR_PROJECT_ROOT:-./TeamTR}
echo "Working directory after cd: $(pwd)"

# Env + experiment settings (quick test defaults).
export TEAMTR_ENV_FILE="${TEAMTR_ENV_FILE:-${TEAMTR_PROJECT_ROOT:-./TeamTR}/teamtr-env-test.yml}"
export TEAMTR_ENV_NAME="${TEAMTR_ENV_NAME:-TeamTR}"
export TEAMTR_REFRESH_ENV="${TEAMTR_REFRESH_ENV:-0}"
export TEAMTR_MAMBA_ROOT_PREFIX="${TEAMTR_MAMBA_ROOT_PREFIX:-${TEAMTR_MAMBA_ROOT_PREFIX}}"
export TEAMTR_CONFIG_NAME="${TEAMTR_CONFIG_NAME:-teamtr_quick_test}"
export TEAMTR_EXPERIMENT_NAME="${TEAMTR_EXPERIMENT_NAME:-teamtr_quick_test_1p7b}"
export TEAMTR_N_GPUS_PER_NODE="${TEAMTR_N_GPUS_PER_NODE:-2}"

# Paths to datasets (DAPO train, MATH-500 eval).
export TEAMTR_TRAIN_FILES="${TEAMTR_TRAIN_FILES:-${TEAMTR_DATASETS_ROOT:-./data}/dapo_math_17k/processed/train.parquet}"
export TEAMTR_VAL_FILES="${TEAMTR_VAL_FILES:-${TEAMTR_DATASETS_ROOT:-./data}/math500/processed/val.parquet}"

# Quick-test overrides.
export TEAMTR_TOTAL_TRAINING_STEPS="${TEAMTR_TOTAL_TRAINING_STEPS:-10}"
export TEAMTR_SAVE_FREQ="${TEAMTR_SAVE_FREQ:-5}"
export TEAMTR_TEST_FREQ="${TEAMTR_TEST_FREQ:-10}"
export TEAMTR_TRAIN_BATCH_SIZE="${TEAMTR_TRAIN_BATCH_SIZE:-10}"
export TEAMTR_GEN_BATCH_SIZE="${TEAMTR_GEN_BATCH_SIZE:-10}"
export TEAMTR_ROLLOUT_N="${TEAMTR_ROLLOUT_N:-1}"
export TEAMTR_PPO_MINI_BATCH_SIZE="${TEAMTR_PPO_MINI_BATCH_SIZE:-10}"
export TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU="${TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export TEAMTR_VAL_BEFORE_TRAIN="${TEAMTR_VAL_BEFORE_TRAIN:-false}"

# Model paths (default to 1.7B homo for quick test).
export TEAMTR_MODEL_PATHS="${TEAMTR_MODEL_PATHS:-${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-1.7B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-1.7B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-1.7B}"
MODEL_OVERRIDE_ARGS=""
if [ -n "${TEAMTR_MODEL_PATHS:-}" ]; then
  IFS=',' read -r -a MODEL_PATH_ARR <<< "${TEAMTR_MODEL_PATHS}"
  if [ "${#MODEL_PATH_ARR[@]}" -ne 3 ]; then
    echo "TEAMTR_MODEL_PATHS must contain exactly 3 comma-separated model paths."
    exit 1
  fi
  MODEL_OVERRIDE_ARGS="actor_rollout_ref.rollout.agent_model_paths=['${MODEL_PATH_ARR[0]}','${MODEL_PATH_ARR[1]}','${MODEL_PATH_ARR[2]}'] \
actor_rollout_ref.model.path=${MODEL_PATH_ARR[1]}"
fi

# Logs
export TEAMTR_LOG_DIR="${TEAMTR_LOG_DIR:-${TEAMTR_PROJECT_ROOT:-./TeamTR}/logs}"
mkdir -p "${TEAMTR_LOG_DIR}"
LOG_FILE="${TEAMTR_LOG_DIR}/${TEAMTR_EXPERIMENT_NAME}_slurm_${SLURM_JOB_ID}.log"

# Use existing env if present, otherwise create it with mkenv.
ENV_PREFIX="${TEAMTR_MAMBA_ROOT_PREFIX}/envs/${TEAMTR_ENV_NAME}"
ALT_ENV_PREFIX="${MAMBA_ROOT_PREFIX:-~/.local/share/mamba}/envs/${TEAMTR_ENV_NAME}"
REFRESH_FLAG=""
if [ "${TEAMTR_REFRESH_ENV}" -eq 1 ]; then
  REFRESH_FLAG="--refresh"
fi

if [ -d "${ENV_PREFIX}" ] && [ "${TEAMTR_REFRESH_ENV}" -eq 0 ]; then
  RUNNER=(micromamba run -r "${TEAMTR_MAMBA_ROOT_PREFIX}" -n "${TEAMTR_ENV_NAME}")
elif [ -d "${ALT_ENV_PREFIX}" ] && [ "${TEAMTR_REFRESH_ENV}" -eq 0 ]; then
  RUNNER=(micromamba run -n "${TEAMTR_ENV_NAME}")
else
  RUNNER=(mkenv -r "${TEAMTR_MAMBA_ROOT_PREFIX}" -n "${TEAMTR_ENV_NAME}" -f "${TEAMTR_ENV_FILE}")
  if [ -n "${REFRESH_FLAG}" ]; then
    RUNNER+=("${REFRESH_FLAG}")
  fi
fi

srun -- "${RUNNER[@]}" bash -c "export PYTHONPATH=${TEAMTR_PROJECT_ROOT:-./TeamTR}; \
  export PYTHONUNBUFFERED=1; \
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0; \
  export RAY_DEDUP_LOGS=0; \
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1; \
  export TORCH_NCCL_ENABLE_MONITORING=0; \
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200; \
  command -v python; \
  echo \"Launching TeamTR quick test...\"; \
  python -u -m verl.trainer.main_ppo \
    --config-path ${TEAMTR_PROJECT_ROOT:-./TeamTR}/teamtr/configs \
    --config-name ${TEAMTR_CONFIG_NAME} \
    hydra.searchpath=[file://${TEAMTR_PROJECT_ROOT:-./TeamTR}/verl/trainer/config] \
    trainer.experiment_name=${TEAMTR_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${TEAMTR_N_GPUS_PER_NODE} \
    trainer.val_before_train=${TEAMTR_VAL_BEFORE_TRAIN} \
    trainer.test_freq=${TEAMTR_TEST_FREQ} \
    trainer.save_freq=${TEAMTR_SAVE_FREQ} \
    trainer.total_training_steps=${TEAMTR_TOTAL_TRAINING_STEPS} \
    data.train_files=\"['${TEAMTR_TRAIN_FILES}']\" \
    data.val_files=\"['${TEAMTR_VAL_FILES}']\" \
    data.train_batch_size=${TEAMTR_TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${TEAMTR_GEN_BATCH_SIZE} \
    actor_rollout_ref.rollout.n=${TEAMTR_ROLLOUT_N} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TEAMTR_PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU} \
    ${MODEL_OVERRIDE_ARGS} \
    trainer.logger=[console] |& tee \"${LOG_FILE}\""
