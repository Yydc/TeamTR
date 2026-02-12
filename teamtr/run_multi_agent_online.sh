#!/usr/bin/env bash
# Online multi-agent TeamTR example using multi_agent_hf rollout.

set -euo pipefail
set -x

TEAMTR_REPO_ROOT=${TEAMTR_REPO_ROOT:-"${TEAMTR_PROJECT_ROOT:-./TeamTR}"}
cd "${TEAMTR_REPO_ROOT}"

TEAMTR_CONFIG_PATH=${TEAMTR_CONFIG_PATH:-"${TEAMTR_PROJECT_ROOT:-./TeamTR}/teamtr/configs"}
TEAMTR_CONFIG_NAME=${TEAMTR_CONFIG_NAME:-"teamtr_homo_lora"}
TEAMTR_TRAIN_FILES=${TEAMTR_TRAIN_FILES:-""}
TEAMTR_VAL_FILES=${TEAMTR_VAL_FILES:-""}
TEAMTR_MODEL_PATHS=${TEAMTR_MODEL_PATHS:-""}
TEAMTR_LOG_DIR=${TEAMTR_LOG_DIR:-"${TEAMTR_PROJECT_ROOT:-./TeamTR}/logs"}

OVERRIDES=()

if [ -n "${TEAMTR_TRAIN_FILES}" ]; then
  IFS=',' read -r -a TRAIN_ARR <<< "${TEAMTR_TRAIN_FILES}"
  if [ "${#TRAIN_ARR[@]}" -lt 1 ]; then
    echo "TEAMTR_TRAIN_FILES must contain at least 1 path."
    exit 1
  fi
  TRAIN_LIST="['${TRAIN_ARR[0]}'"
  for ((i=1; i<${#TRAIN_ARR[@]}; i++)); do
    TRAIN_LIST+=", '${TRAIN_ARR[$i]}'"
  done
  TRAIN_LIST+="]"
  OVERRIDES+=("data.train_files=${TRAIN_LIST}")
fi

if [ -n "${TEAMTR_VAL_FILES}" ]; then
  IFS=',' read -r -a VAL_ARR <<< "${TEAMTR_VAL_FILES}"
  if [ "${#VAL_ARR[@]}" -lt 1 ]; then
    echo "TEAMTR_VAL_FILES must contain at least 1 path."
    exit 1
  fi
  VAL_LIST="['${VAL_ARR[0]}'"
  for ((i=1; i<${#VAL_ARR[@]}; i++)); do
    VAL_LIST+=", '${VAL_ARR[$i]}'"
  done
  VAL_LIST+="]"
  OVERRIDES+=("data.val_files=${VAL_LIST}")
fi

if [ -n "${TEAMTR_MODEL_PATHS}" ]; then
  IFS=',' read -r -a MODEL_PATH_ARR <<< "${TEAMTR_MODEL_PATHS}"
  if [ "${#MODEL_PATH_ARR[@]}" -ne 3 ]; then
    echo "TEAMTR_MODEL_PATHS must contain exactly 3 comma-separated model paths."
    exit 1
  fi
  AGENT_MODEL_PATHS="['${MODEL_PATH_ARR[0]}','${MODEL_PATH_ARR[1]}','${MODEL_PATH_ARR[2]}']"
  OVERRIDES+=("actor_rollout_ref.rollout.agent_model_paths=${AGENT_MODEL_PATHS}")
  OVERRIDES+=("actor_rollout_ref.model.path=${MODEL_PATH_ARR[1]}")
fi

mkdir -p "${TEAMTR_LOG_DIR}"
LOG_FILE="${TEAMTR_LOG_DIR}/${TEAMTR_CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

python -m verl.trainer.main_ppo \
  --config-path "${TEAMTR_CONFIG_PATH}" \
  --config-name "${TEAMTR_CONFIG_NAME}" \
  "${OVERRIDES[@]}" |& tee "${LOG_FILE}"
