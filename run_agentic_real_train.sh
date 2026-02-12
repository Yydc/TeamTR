#!/bin/bash
set -euo pipefail
cd $(dirname $(dirname $0))

# Ensure codex CLI and python are on PATH for non-interactive runs.
export PATH="~/.nvm/versions/node/v20.20.0/bin:~/.local/bin:${PATH}"

# Real training pipeline via agentic. No prompts, auto-retry handled by run_pipeline.py.
/usr/bin/python3 tools/agentic.py run --project projects/TeamTR --codex-allow-shell --max-iters 50

# Submit full 8-GPU, 8h Slurm job for real training.
SLURM_SCRIPT="${TEAMTR_PROJECT_ROOT:-./TeamTR}/run_teamtr_slurm_full_8gpu_8h.sh"
if [ ! -f "${SLURM_SCRIPT}" ]; then
  echo "Missing Slurm script: ${SLURM_SCRIPT}"
  exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH."
  exit 1
fi

echo "Submitting ${SLURM_SCRIPT} via sbatch..."
sbatch "${SLURM_SCRIPT}"
