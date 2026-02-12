#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${TEAMTR_CODE_ROOT:-${CODE_ROOT}}"
SBATCH_SCRIPT="${TEAMTR_FULL_SCRIPT:-${TEAMTR_PROJECT_ROOT:-./TeamTR}/run_teamtr_slurm_full_8gpu_8h.sh}"
STAMP="${TEAMTR_SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${TEAMTR_SWEEP_ROOT:-${TEAMTR_PROJECT_ROOT:-./TeamTR}/outputs/sweeps/3x8b_${STAMP}}"
JOB_LIST="${SWEEP_ROOT}/jobs.txt"

mkdir -p "${SWEEP_ROOT}"
: > "${JOB_LIST}"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
  echo "Missing sbatch script: ${SBATCH_SCRIPT}"
  exit 1
fi

submit_one() {
  local exp="$1"
  local extra_args="$2"
  local router_seed="$3"
  local chain_seed="$4"

  local job_id
  job_id=$(
    TEAMTR_CODE_ROOT="${CODE_ROOT}" \
    TEAMTR_CONFIG_NAME="teamtr_homo_full" \
    TEAMTR_EXPERIMENT_NAME="${exp}" \
    TEAMTR_MODEL_PATHS="${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B,${TEAMTR_MODELS_ROOT:-./models}/Qwen__Qwen3-8B" \
    TEAMTR_ROUTER="random" \
    TEAMTR_ROUTER_SEED="${router_seed}" \
    TEAMTR_CHAIN_TURNS="3" \
    TEAMTR_CHAIN_RANDOM_ORDER="1" \
    TEAMTR_CHAIN_SEED="${chain_seed}" \
    TEAMTR_CHAIN_MAX_MODEL_LEN="1536" \
    TEAMTR_TOTAL_TRAINING_STEPS="360" \
    TEAMTR_SAVE_FREQ="60" \
    TEAMTR_TEST_FREQ="0" \
    TEAMTR_VAL_BEFORE_TRAIN="false" \
    TEAMTR_TRAIN_BATCH_SIZE="8" \
    TEAMTR_GEN_BATCH_SIZE="8" \
    TEAMTR_PPO_MINI_BATCH_SIZE="8" \
    TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU="1" \
    TEAMTR_MODEL_USE_SHM="false" \
    TEAMTR_SAVE_HF_MODEL="1" \
    TEAMTR_OUTPUT_DIR="${SWEEP_ROOT}/${exp}" \
    TEAMTR_EXTRA_ARGS="${extra_args}" \
    sbatch \
      --partition=batch \
      --constraint='type-gpu&zone-sof1' \
      --nodes=1 \
      --ntasks=1 \
      --gpus=h200:8 \
      --cpus-per-gpu=8 \
      --time=08:00:00 \
      --job-name="${exp}" \
      --export=ALL \
      "${SBATCH_SCRIPT}" | awk '{print $4}'
  )

  echo "${exp} ${job_id}" | tee -a "${JOB_LIST}"
}

# Four 8-GPU runs in parallel => 32 GPUs total requested.
submit_one \
  "sweep_3x8b_v1_chain_base_${STAMP}" \
  "trainer.resume_mode=disable data.max_prompt_length=768 data.max_response_length=256 actor_rollout_ref.rollout.max_model_len=1536 actor_rollout_ref.rollout.response_length=256 actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.temperature=0.8 actor_rollout_ref.actor.kl_loss_coef=0.001" \
  "42" \
  "42"

submit_one \
  "sweep_3x8b_v2_chain_more_samples_${STAMP}" \
  "trainer.resume_mode=disable data.max_prompt_length=768 data.max_response_length=256 actor_rollout_ref.rollout.max_model_len=1536 actor_rollout_ref.rollout.response_length=256 actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.temperature=0.8 actor_rollout_ref.actor.kl_loss_coef=0.001" \
  "43" \
  "43"

submit_one \
  "sweep_3x8b_v3_chain_low_temp_high_kl_${STAMP}" \
  "trainer.resume_mode=disable data.max_prompt_length=768 data.max_response_length=192 actor_rollout_ref.rollout.max_model_len=1408 actor_rollout_ref.rollout.response_length=192 actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.temperature=0.6 actor_rollout_ref.actor.kl_loss_coef=0.002" \
  "44" \
  "44"

submit_one \
  "sweep_3x8b_v4_chain_high_temp_low_kl_${STAMP}" \
  "trainer.resume_mode=disable data.max_prompt_length=768 data.max_response_length=256 actor_rollout_ref.rollout.max_model_len=1536 actor_rollout_ref.rollout.response_length=256 actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.actor.kl_loss_coef=0.0005" \
  "45" \
  "45"

echo ""
echo "Submitted jobs (experiment job_id):"
cat "${JOB_LIST}"
echo "Job list saved to: ${JOB_LIST}"
