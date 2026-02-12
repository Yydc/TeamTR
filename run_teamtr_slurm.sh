#!/bin/bash
#SBATCH --job-name=teamtr_test_2gpu
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu&zone-sof1
#SBATCH --gpus=h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=${TEAMTR_LOGS_DIR:-./logs}/teamtr_%j.log
#SBATCH --error=${TEAMTR_LOGS_DIR:-./logs}/teamtr_%j.error
#SBATCH --mail-type=ALL

CODE_ROOT="${TEAMTR_CODE_ROOT:-${CODE_ROOT}}"
cd "${CODE_ROOT}"
echo "Working directory after cd: $(pwd)"

# Env + experiment settings (test run defaults).
export TEAMTR_ENV_FILE="${TEAMTR_ENV_FILE:-${CODE_ROOT}/env/teamtr-env.yml}"
export TEAMTR_ENV_NAME="${TEAMTR_ENV_NAME:-TeamTR}"
export TEAMTR_REFRESH_ENV="${TEAMTR_REFRESH_ENV:-0}"
export TEAMTR_MAMBA_ROOT_PREFIX="${TEAMTR_MAMBA_ROOT_PREFIX:-${TEAMTR_MAMBA_ROOT_PREFIX}}"
export TEAMTR_CONFIG_NAME="${TEAMTR_CONFIG_NAME:-teamtr_homo_full}"
export TEAMTR_EXPERIMENT_NAME="${TEAMTR_EXPERIMENT_NAME:-teamtr_homo_full_1p7b_test}"
export TEAMTR_N_GPUS_PER_NODE="${TEAMTR_N_GPUS_PER_NODE:-2}"
export RAY_DISABLE_DASHBOARD="${RAY_DISABLE_DASHBOARD:-1}"
export RAY_DISABLE_MEMORY_MONITOR="${RAY_DISABLE_MEMORY_MONITOR:-1}"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="${RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE:-1}"
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS="${RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS:-120}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Paths to datasets (DAPO train, MATH-500 eval).
export TEAMTR_TRAIN_FILES="${TEAMTR_TRAIN_FILES:-${TEAMTR_DATASETS_ROOT:-./data}/deepscaler_preview/processed/train.parquet,${TEAMTR_DATASETS_ROOT:-./data}/dapo_math_17k/processed/train.parquet}"
export TEAMTR_VAL_FILES="${TEAMTR_VAL_FILES:-${TEAMTR_DATASETS_ROOT:-./data}/math500/processed/val.parquet}"

# Test-mode overrides.
export TEAMTR_TOTAL_TRAINING_STEPS="${TEAMTR_TOTAL_TRAINING_STEPS:-100}"
export TEAMTR_SAVE_FREQ="${TEAMTR_SAVE_FREQ:-50}"
export TEAMTR_TEST_FREQ="${TEAMTR_TEST_FREQ:-100}"
export TEAMTR_TRAIN_BATCH_SIZE="${TEAMTR_TRAIN_BATCH_SIZE:-10}"
export TEAMTR_GEN_BATCH_SIZE="${TEAMTR_GEN_BATCH_SIZE:-10}"
export TEAMTR_ROLLOUT_N="${TEAMTR_ROLLOUT_N:-1}"
export TEAMTR_PPO_MINI_BATCH_SIZE="${TEAMTR_PPO_MINI_BATCH_SIZE:-10}"
export TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU="${TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export TEAMTR_VAL_BEFORE_TRAIN="${TEAMTR_VAL_BEFORE_TRAIN:-false}"
export TEAMTR_OUTPUT_DIR="${TEAMTR_OUTPUT_DIR:-${CODE_ROOT}/outputs/${TEAMTR_EXPERIMENT_NAME}}"
export TEAMTR_MODEL_USE_SHM="${TEAMTR_MODEL_USE_SHM:-false}"
export TEAMTR_STAGE_MODELS="${TEAMTR_STAGE_MODELS:-1}"
export TEAMTR_MODEL_CACHE_DIR="${TEAMTR_MODEL_CACHE_DIR:-/tmp/teamtr_models}"

# Optional override for model paths (comma-separated, 3 paths).
export TEAMTR_MODEL_PATHS="${TEAMTR_MODEL_PATHS:-${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-1.7B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-1.7B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-1.7B}"
MODEL_OVERRIDE_ARGS=""
if [ -n "${TEAMTR_MODEL_PATHS:-}" ]; then
  model_raw="${TEAMTR_MODEL_PATHS//;/,}"
  IFS=',' read -r -a MODEL_PATH_ARR <<< "${model_raw}"
  if [ "${#MODEL_PATH_ARR[@]}" -ne 3 ]; then
    echo "TEAMTR_MODEL_PATHS must contain exactly 3 comma-separated model paths."
    exit 1
  fi
  if [ "${TEAMTR_STAGE_MODELS}" = "1" ]; then
    mkdir -p "${TEAMTR_MODEL_CACHE_DIR}"
    for idx in "${!MODEL_PATH_ARR[@]}"; do
      src="${MODEL_PATH_ARR[$idx]}"
      if [ ! -d "${src}" ]; then
        echo "Model path does not exist: ${src}"
        exit 1
      fi
      hash=$(printf "%s" "${src}" | md5sum | awk '{print $1}')
      dest_dir="${TEAMTR_MODEL_CACHE_DIR}/${hash}"
      dest="${dest_dir}/$(basename "${src}")"
      if [ ! -d "${dest}" ]; then
        mkdir -p "${dest_dir}"
        if command -v rsync >/dev/null 2>&1; then
          rsync -a --delete "${src}/" "${dest}/"
        else
          rm -rf "${dest}"
          cp -a "${src}" "${dest}"
        fi
      fi
      MODEL_PATH_ARR[$idx]="${dest}"
    done
    TEAMTR_MODEL_PATHS="$(IFS=','; echo "${MODEL_PATH_ARR[*]}")"
  fi
  MODEL_OVERRIDE_ARGS="actor_rollout_ref.rollout.agent_model_paths=['${MODEL_PATH_ARR[0]}','${MODEL_PATH_ARR[1]}','${MODEL_PATH_ARR[2]}'] \
actor_rollout_ref.model.path=${MODEL_PATH_ARR[1]}"
fi

# Logs
export TEAMTR_LOG_DIR="${TEAMTR_LOG_DIR:-${TEAMTR_LOGS_DIR:-./logs}}"
mkdir -p "${TEAMTR_LOG_DIR}"
if ! mkdir -p "${RAY_TMPDIR}" 2>/dev/null; then
  export RAY_TMPDIR="/tmp/ray"
  mkdir -p "${RAY_TMPDIR}"
fi
LOG_FILE="${TEAMTR_LOG_DIR}/${TEAMTR_EXPERIMENT_NAME}_slurm_${SLURM_JOB_ID}.log"

build_py_list() {
  local raw="$1"
  local IFS=','
  read -r -a arr <<< "${raw}"
  local out="["
  local first=1
  for item in "${arr[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [ -z "${item}" ] && continue
    if [ "${first}" -eq 0 ]; then
      out+=","
    else
      first=0
    fi
    out+="'${item}'"
  done
  out+="]"
  echo "${out}"
}

TRAIN_LIST="$(build_py_list "${TEAMTR_TRAIN_FILES}")"
VAL_LIST="$(build_py_list "${TEAMTR_VAL_FILES}")"

# Use existing env if present; avoid auto-create unless explicitly requested.
ENV_PREFIX="${TEAMTR_MAMBA_ROOT_PREFIX}/envs/${TEAMTR_ENV_NAME}"
ALT_ENV_PREFIX="${MAMBA_ROOT_PREFIX:-~/.local/share/mamba}/envs/${TEAMTR_ENV_NAME}"
if [ -d "${ENV_PREFIX}" ] && [ "${TEAMTR_REFRESH_ENV}" -eq 0 ]; then
  RUNNER=(micromamba run -r "${TEAMTR_MAMBA_ROOT_PREFIX}" -n "${TEAMTR_ENV_NAME}")
elif [ -d "${ALT_ENV_PREFIX}" ] && [ "${TEAMTR_REFRESH_ENV}" -eq 0 ]; then
  RUNNER=(micromamba run -n "${TEAMTR_ENV_NAME}")
else
  if [ "${TEAMTR_REFRESH_ENV}" -eq 1 ]; then
    RUNNER=(mkenv -r "${TEAMTR_MAMBA_ROOT_PREFIX}" -n "${TEAMTR_ENV_NAME}" -f "${TEAMTR_ENV_FILE}" --refresh)
  else
    echo "Env ${TEAMTR_ENV_NAME} not found under ${TEAMTR_MAMBA_ROOT_PREFIX} (or ${ALT_ENV_PREFIX}). Create it before running."
    exit 1
  fi
fi

srun -- "${RUNNER[@]}" bash -c "set -euo pipefail; \
  export PYTHONPATH=${CODE_ROOT}; \
  export PYTHONUNBUFFERED=1; \
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0; \
  export RAY_DEDUP_LOGS=0; \
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1; \
  export TORCH_NCCL_ENABLE_MONITORING=0; \
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200; \
  export TORCHDYNAMO_DISABLE=1; \
  command -v python; \
  echo \"Launching TeamTR training...\"; \
  python -u -m verl.trainer.main_ppo \
    --config-path ${CODE_ROOT}/teamtr/configs \
    --config-name ${TEAMTR_CONFIG_NAME} \
    hydra.searchpath=[file://${CODE_ROOT}/verl/trainer/config] \
    trainer.experiment_name=${TEAMTR_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${TEAMTR_N_GPUS_PER_NODE} \
    trainer.val_before_train=${TEAMTR_VAL_BEFORE_TRAIN} \
    trainer.test_freq=${TEAMTR_TEST_FREQ} \
    trainer.save_freq=${TEAMTR_SAVE_FREQ} \
    trainer.total_training_steps=${TEAMTR_TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir=${TEAMTR_OUTPUT_DIR} \
    data.train_files=\"${TRAIN_LIST}\" \
    data.val_files=\"${VAL_LIST}\" \
    data.train_batch_size=${TEAMTR_TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${TEAMTR_GEN_BATCH_SIZE} \
    actor_rollout_ref.rollout.n=${TEAMTR_ROLLOUT_N} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TEAMTR_PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.model.use_shm=${TEAMTR_MODEL_USE_SHM} \
    ${MODEL_OVERRIDE_ARGS} \
    trainer.logger=[console] 2>&1 | tee \"${LOG_FILE}\"; \
  status=\\${PIPESTATUS[0]}; \
  exit \\$status"
