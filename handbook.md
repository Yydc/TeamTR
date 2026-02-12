# TeamTR Handbook

This is a practical guide for training and evaluating TeamTR on the INSAIT Slurm cluster.
All paths are for this server and match `target.md`.

## Quick facts
- Dataset root: `/home/yuanqi_yao/dataset`
- Model root: `/home/yuanqi_yao/yyq/models`
- Code root: `/home/yuanqi_yao/agentic/agentic/projects/TeamTR`
- Slurm logs: `/home/yuanqi_yao/slurm_logs`
- GPU cap: 16 total at once (`2x8` or `4x4` parallel shapes)

## Preflight
```bash
cd /home/yuanqi_yao/agentic/agentic/projects/TeamTR
python scripts/run_pipeline.py --mode server_smoke
```

## Fast smoke run (tiny data, quick validation)
Uses the small DAPO subset and math500 for eval. This is for testing the loop only.
```bash
cd /home/yuanqi_yao/agentic/agentic/projects/TeamTR
TEAMTR_TRAIN_FILES=/home/yuanqi_yao/dataset/dapo_math_17k/processed/train_quick.parquet \
TEAMTR_VAL_FILES=/home/yuanqi_yao/dataset/dapo_math_17k/processed/val_quick.parquet \
TEAMTR_TOTAL_TRAINING_STEPS=50 \
TEAMTR_SAVE_FREQ=50 \
TEAMTR_TEST_FREQ=0 \
TEAMTR_TRAIN_BATCH_SIZE=8 \
TEAMTR_GEN_BATCH_SIZE=8 \
TEAMTR_PPO_MINI_BATCH_SIZE=8 \
TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
TEAMTR_ROLLOUT_N=1 \
TEAMTR_VAL_BEFORE_TRAIN=false \
TEAMTR_SAVE_HF_MODEL=1 \
TEAMTR_EVAL_ONLY_MATH=1 \
TEAMTR_EVAL_MAX_SAMPLES_FULL=100 \
TEAMTR_EVAL_FROM_CHECKPOINT=1 \
TEAMTR_EVAL_AGENT_INDEX=1 \
TEAMTR_EXTRA_ARGS="data.max_prompt_length=256 data.max_response_length=256 data.truncation=right actor_rollout_ref.rollout.max_model_len=1024" \
python scripts/run_pipeline.py --mode train_eval_full_parallel --max_total_gpus 16 --parallel_shape 2x8 \
  --configs configs/teamtr_3x8b.yaml configs/teamtr_1p7b_8b_14b.yaml
```

## Full training (2x8 parallel)
Launches two 8-GPU jobs in parallel: 3x8B and 1.7B+8B+14B.
```bash
cd /home/yuanqi_yao/agentic/agentic/projects/TeamTR
TEAMTR_ROUTER=round_robin \
TEAMTR_SAVE_HF_MODEL=1 \
python scripts/run_pipeline.py --mode train_eval_full_parallel --max_total_gpus 16 --parallel_shape 2x8 \
  --configs configs/teamtr_3x8b.yaml configs/teamtr_1p7b_8b_14b.yaml
```

If you want to submit directly:
```bash
cd /home/yuanqi_yao/agentic/agentic/projects/TeamTR
bash run_teamtr_slurm_parallel_2x8.sh
```

## Single 8-GPU job (manual)
```bash
cd /home/yuanqi_yao/agentic/agentic/projects/TeamTR
TEAMTR_CONFIG_NAME=teamtr_homo_full \
TEAMTR_EXPERIMENT_NAME=teamtr_full_3x8b \
TEAMTR_MODEL_PATHS=/home/yuanqi_yao/yyq/models/Qwen__Qwen3-8B,/home/yuanqi_yao/yyq/models/Qwen__Qwen3-8B,/home/yuanqi_yao/yyq/models/Qwen__Qwen3-8B \
TEAMTR_ROUTER=round_robin \
TEAMTR_SAVE_HF_MODEL=1 \
sbatch /home/yuanqi_yao/TeamTR/run_teamtr_slurm_full_8gpu_8h.sh
```

## Evaluation (single model)
Use the eval Slurm script (1 GPU):
```bash
cd /home/yuanqi_yao/agentic/agentic/projects/TeamTR
TEAMTR_MODEL_PATH=/path/to/hf_checkpoint_or_model \
TEAMTR_EVAL_DATASET=/home/yuanqi_yao/dataset/math500/processed/val.parquet \
TEAMTR_EVAL_K=4 \
TEAMTR_EVAL_MAX_SAMPLES=100 \
TEAMTR_EVAL_MAX_NEW_TOKENS=256 \
TEAMTR_EVAL_OUTPUT=/home/yuanqi_yao/agentic/agentic/projects/TeamTR/outputs/eval_math500.json \
sbatch run_teamtr_eval.sh
```

## Evaluation (team of 3 models)
This supports three modes:
- `per_model`: evaluate each model separately
- `best_of`: per-sample best of the three
- `router`: select by round-robin or random per sample

```bash
cd /home/yuanqi_yao/agentic/agentic/projects/TeamTR
python scripts/run_team_eval.py \
  --model-paths /path/to/agent0/huggingface,/path/to/agent1/huggingface,/path/to/agent2/huggingface \
  --dataset /home/yuanqi_yao/dataset/math500/processed/val.parquet \
  --k 4 \
  --max-samples 100 \
  --mode per_model \
  --output outputs/team_eval/per_model.json
```

Best-of / router:
```bash
python scripts/run_team_eval.py \
  --model-paths /path/to/agent0/huggingface,/path/to/agent1/huggingface,/path/to/agent2/huggingface \
  --dataset /home/yuanqi_yao/dataset/math500/processed/val.parquet \
  --k 4 \
  --max-samples 100 \
  --mode best_of \
  --output outputs/team_eval/best_of.json

python scripts/run_team_eval.py \
  --model-paths /path/to/agent0/huggingface,/path/to/agent1/huggingface,/path/to/agent2/huggingface \
  --dataset /home/yuanqi_yao/dataset/math500/processed/val.parquet \
  --k 4 \
  --max-samples 100 \
  --mode router --router round_robin --seed 42 \
  --output outputs/team_eval/router_rr.json
```

## Checkpoints (HF format)
To enable easy eval from checkpoints, set:
```
TEAMTR_SAVE_HF_MODEL=1
```
HF checkpoints land at:
```
outputs/<experiment>/global_step_<N>/actor/agent<IDX>/huggingface
```

## Monitoring and stopping jobs
- Tail logs: `tail -f /home/yuanqi_yao/slurm_logs/teamtr_<JOBID>.log`
- Job state: `squeue -j <JOBID>`
- Stop job: `scancel <JOBID>`

## Keep runs under 8 hours (fast tuning knobs)
These are safe knobs for quicker iteration:
- `TEAMTR_TOTAL_TRAINING_STEPS`: e.g., 50 or 100
- `TEAMTR_TRAIN_BATCH_SIZE` / `TEAMTR_GEN_BATCH_SIZE`: lower if OOM
- `TEAMTR_PPO_MICRO_BATCH_SIZE_PER_GPU`: lower for memory
- `TEAMTR_ROLLOUT_N`: 1 for speed
- `TEAMTR_EXTRA_ARGS`: reduce sequence length and max model length

Example:
```bash
TEAMTR_EXTRA_ARGS="data.max_prompt_length=256 data.max_response_length=256 data.truncation=right actor_rollout_ref.rollout.max_model_len=1024"
```

## Notes
- If only one model is available locally, you can reuse the same path three times in `TEAMTR_MODEL_PATHS`.
- For the official requirement, training uses DeepScaleR + DAPO and evaluation includes math500/aime/zebralogic.
