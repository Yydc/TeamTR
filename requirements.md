# TeamTR (Server Contract — Real Training, Mamba, ≤16 GPUs Total)

```toml
codex_task_spec_version = "1.1"

[project]
name = "TeamTR"
slug = "teamtr"
type = "research_coding"

# ---------------------------
# Server / cluster contract (INSAIT Slurm)
# ---------------------------
[project.server]
# Data/model stores are *absolute* on the server.
datasets_store = "/home/yuanqi_yao/dataset"
models_store = "/home/yuanqi_yao/yyq/models"

# Inside this repo, we use *relative* stable mount points (recommend: symlinks).
#   ./datasets -> ${datasets_store}
#   ./models   -> ${models_store}
datasets_root = "datasets"
models_root = "models"

# All outputs should land under this folder (recommend: symlink to a persistent work dir).
outputs_root = "outputs"

# Slurm launcher wrapper (already exists on the server; copy or symlink into repo root).
slurm_wrapper = "run_teamtr_slurm.sh"

# Hard constraint: never exceed 16 GPUs simultaneously.
# Allowed parallel shapes: "2x8" (two 8-GPU jobs) or "4x4" (four 4-GPU jobs).
max_total_gpus = 16
allowed_parallel_shapes = ["2x8", "4x4"]

# ---------------------------
# Environment (mamba)
# ---------------------------
[env]
python = "3.11"
pip = []

# Use micromamba on the cluster for `run` (and env existence checks).
# We assume an env named "TeamTR" already exists on the server.
manager = "micromamba"
name = "TeamTR"
root_prefix = "/home/yuanqi_yao/.micromamba"

# Keep an env YAML in-repo for reproducibility / reference.
# IMPORTANT: agentic runner will *not* auto-create the env by default (create_policy=never).
file = "env/teamtr-env.yml"
create_policy = "never"
update_policy = "never"

[assets]
# We require local assets (no HF downloads in smoke).
policy = "required_local"
datasets_dir = "datasets"
models_dir = "models"
hf_cache_dir = "assets/hf"
offline_smoke = true

[pipeline.repo]
required_paths = [
  # Entry points / wrappers
  "run_teamtr_slurm.sh",
  "scripts/run_pipeline.py",

  # Environment reference (mamba)
  "env/teamtr-env.yml",

  # Config profiles
  "configs/teamtr_3x1p7b_smoke.yaml",
  "configs/teamtr_3x8b.yaml",
  "configs/teamtr_1p7b_8b_14b.yaml",

  # Minimal docs
  "README.md",
]

# ---------------------------
# Gates (ordered; stops at first failure)
# ---------------------------

[[pipeline.gates]]
name = "server_smoke"
command = "python scripts/run_pipeline.py --mode server_smoke"
timeout_min = 8
expected_artifacts = ["outputs/server_smoke/metrics.json"]
metrics_file = "outputs/server_smoke/metrics.json"
criteria = [
  {path = "schema", op = "==", value = "teamtr.server_smoke.v2"},
  {path = "server_ok", op = "==", value = true},

  # Paths / mounts
  {path = "paths.slurm_wrapper_ok", op = "==", value = true},
  {path = "paths.datasets_symlink_ok", op = "==", value = true},
  {path = "paths.datasets_resolved", op = "==", value = "/home/yuanqi_yao/dataset"},
  {path = "paths.models_symlink_ok", op = "==", value = true},
  {path = "paths.models_resolved", op = "==", value = "/home/yuanqi_yao/yyq/models"},

  # GPU cap must be enforced by pipeline (static contract).
  {path = "resources.max_total_gpus", op = "==", value = 16},
]

[[pipeline.gates]]
name = "train_llm_smoke"
command = "python scripts/run_pipeline.py --mode train_llm_smoke --config configs/teamtr_3x1p7b_smoke.yaml"
timeout_min = 180
expected_artifacts = [
  "outputs/train_smoke/metrics.json",
  "outputs/train_smoke/config_resolved.json",
  "outputs/train_smoke/traces/kl_trace.json",
  "outputs/train_smoke/traces/surrogates.json",
  "outputs/train_smoke/checkpoints/manifest.json"
]
metrics_file = "outputs/train_smoke/metrics.json"
criteria = [
  {path = "schema", op = "==", value = "teamtr.train.metrics.v2"},
  {path = "train_ok", op = "==", value = true},

  # Identity / profile
  {path = "profile", op = "==", value = "3x1p7b_smoke"},
  {path = "team.n_agents", op = "==", value = 3},
  {path = "team.router", op = "==", value = "round_robin"},

  # Training data must include DeepScaleR + DAPO (standardized names).
  {path = "data.train_sets[0]", op = "==", value = "deepscaler"},
  {path = "data.train_sets[1]", op = "==", value = "dapo"},

  # Must log paper defaults.
  {path = "rollout.temperature", op = "==", value = 0.8},
  {path = "rollout.top_p", op = "==", value = 1.0},
  {path = "rollout.max_new_tokens", op = "==", value = 32768},
  {path = "tr.delta", op = "==", value = 0.01},

  # Must log key knobs (bounds-based checks to enforce presence).
  {path = "rollout.group_size", op = ">=", value = 1},
  {path = "update.ratio_clip_eps", op = ">=", value = 0.0},
  {path = "update.ratio_clip_eps", op = "<=", value = 0.5},
  {path = "train.stages", op = ">=", value = 1},

  # Trust-region sanity.
  {path = "tr.kl_violation_rate", op = "<=", value = 0.20},

  # Logging contract.
  {path = "logging.complete", op = "==", value = true},
]

[[pipeline.gates]]
name = "eval_llm_smoke"
command = "python scripts/run_pipeline.py --mode eval_llm_smoke --config configs/teamtr_3x1p7b_smoke.yaml"
timeout_min = 120
expected_artifacts = [
  "outputs/eval_smoke/metrics.json",
  "outputs/eval_smoke/config_resolved.json"
]
metrics_file = "outputs/eval_smoke/metrics.json"
criteria = [
  {path = "schema", op = "==", value = "teamtr.eval.metrics.v2"},
  {path = "eval_ok", op = "==", value = true},
  {path = "profile", op = "==", value = "3x1p7b_smoke"},

  # Smoke budget: smaller K, but must cover the real target benchmarks.
  {path = "benchmarks.aime24.pass_at_k.k", op = "==", value = 8},
  {path = "benchmarks.aime25.pass_at_k.k", op = "==", value = 8},
  {path = "benchmarks.zebralogic.pass_at_k.k", op = "==", value = 8},
  {path = "benchmarks.math500.pass_at_k.k", op = "==", value = 4},

  # Require avg@K too (not just pass@K).
  {path = "benchmarks.aime24.avg_at_k.k", op = "==", value = 8},
  {path = "benchmarks.aime25.avg_at_k.k", op = "==", value = 8},
  {path = "benchmarks.zebralogic.avg_at_k.k", op = "==", value = 8},
  {path = "benchmarks.math500.avg_at_k.k", op = "==", value = 4},

  {path = "logging.complete", op = "==", value = true},
]

[[pipeline.gates]]
name = "config_check"
command = "python scripts/run_pipeline.py --mode config_check --configs configs/teamtr_3x8b.yaml configs/teamtr_1p7b_8b_14b.yaml"
timeout_min = 10
expected_artifacts = ["outputs/config_check/metrics.json"]
metrics_file = "outputs/config_check/metrics.json"
criteria = [
  {path = "schema", op = "==", value = "teamtr.config_check.v2"},
  {path = "config_ok", op = "==", value = true},
  {path = "resources.max_total_gpus", op = "==", value = 16},
  {path = "resources.parallel_shape", op = "==", value = "2x8"},
]

# Full runs (only executed after all smoke gates pass).
# Must run the *real* TeamTR algorithm; "smoke" here means fewer stages / fewer eval samples,
# NOT a different toy environment.

[[pipeline.gates]]
name = "train_eval_full_parallel"
command = "python scripts/run_pipeline.py --mode train_eval_full_parallel --max_total_gpus 16 --parallel_shape 2x8 --configs configs/teamtr_3x8b.yaml configs/teamtr_1p7b_8b_14b.yaml"
timeout_min = 4320
expected_artifacts = [
  "outputs/full/summary.json",
  "outputs/full/3x8b/train/metrics.json",
  "outputs/full/3x8b/eval/metrics.json",
  "outputs/full/1p7b_8b_14b/train/metrics.json",
  "outputs/full/1p7b_8b_14b/eval/metrics.json"
]
metrics_file = "outputs/full/summary.json"
criteria = [
  {path = "schema", op = "==", value = "teamtr.full.summary.v1"},
  {path = "ok", op = "==", value = true},
  {path = "resources.max_total_gpus", op = "==", value = 16},
  {path = "resources.parallel_shape", op = "==", value = "2x8"},

  # Expect both parallel jobs to complete successfully.
  {path = "results.3x8b.train_ok", op = "==", value = true},
  {path = "results.1p7b_8b_14b.train_ok", op = "==", value = true},
]
```
