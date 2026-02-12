# TeamTR: Stage-wise Block-Coordinate Finetuning for Multi-Agent LLM Teams

TeamTR 是多 agent 团队的阶段式块坐标训练系统：在多模型/多 agent 场景下，按 `agent_id` 分桶进行生成、logprob 计算与参数更新。

下面是**当前仓库已经打通的在线训练流程**，便于新同学快速理解与复现。

## 训练流程概览（在线多 agent）
```
数据准备 -> MultiAgentTurnDataset -> Rollout(按 agent_id) -> logprob/ref_logprob
-> advantage/KL(只算 response) -> update_actor(按 agent_id) -> ckpt(每 agent 独立)
```

### 1) 数据准备
- 数据被处理成 `multi_agent_turn` 格式。
- 每条样本包含 agent 的上下文与回复切分，保证只有回复段参与 KL/advantage。

### 2) Dataset / Collate
`MultiAgentTurnDataset` 产出字段：
- `context_input_ids`, `context_attention_mask`
- `response_ids`, `response_mask`
- `agent_id`, `raw_prompt_ids`

### 3) Rollout
当 `rollout.name=multi_agent_hf`：
- `MultiAgentHFRollout` 接收 `{aid: module}` 的字典。
- batch 按 `agent_id` 分桶，每个 agent 使用自己的模型生成。

### 4) LogProb / RefLogProb
- FSDP worker 按 `agent_id` 分桶计算 actor/ref logprob。

### 5) Advantage / KL / Update
- 只在 `response_mask` 覆盖的 token 上计算 KL/advantage。
- `update_actor` 按 `agent_id` 分桶，仅更新对应 agent 参数。

### 6) Checkpoint
- 每个 agent 独立目录保存：`.../agent{aid}`。

## 核心文件入口
- **多 agent 训练主线**
  - `verl/workers/fsdp_workers.py`：多模型加载与按 `agent_id` 分桶更新/保存。
  - `verl/workers/rollout/multi_agent_hf_rollout.py`：多 agent HF 生成。
  - `verl/trainer/ppo/ray_trainer.py`：写入/修复 `agent_id` 并路由。
- **多轮数据**
  - `verl/utils/dataset/multi_agent_dataset.py`
  - `teamtr/data_utils.py`
- **配置**
  - `teamtr/configs/teamtr_*`：homo/hetero, lora/full, test/full
  - `teamtr-env.yml`, `teamtr-env-test.yml`：micromamba 环境

## 环境与运行（micromamba）
我们使用 micromamba（不是 conda/anaconda）：
- env 文件：`teamtr-env.yml` 或 `teamtr-env-test.yml`
- env 名称：`TeamTR`
- root prefix：`/home/yuanqi_yao/.micromamba`

### Slurm 入口
- `run_teamtr_slurm.sh`：主提交脚本（内部处理 mkenv/micromamba run）
- `submit_teamtr_latest.sh`：轻量 wrapper，便于覆盖参数后再次提交

### 常用覆盖参数
```
TEAMTR_EXPERIMENT_NAME
TEAMTR_TOTAL_TRAINING_STEPS
TEAMTR_TRAIN_BATCH_SIZE
TEAMTR_GEN_BATCH_SIZE
TEAMTR_MODEL_PATHS   # 逗号分隔 3 个模型路径
```

## 日志位置
- Slurm 主日志：`/home/yuanqi_yao/TeamTR/teamtr_%j.log`
- 训练日志：`/home/yuanqi_yao/TeamTR/logs/<experiment>_slurm_<jobid>.log`
- 运行记录：`teamtr/RUN_LOG.md`
- Debug 记录：`teamtr/DEBUG_LOG.md`

## 常见 warning（可忽略）
- `Unsupported processor type: Qwen2TokenizerFast`
  - 仅影响多模态 processor，纯文本训练不受影响。
- `generation_config default values modified`
  - 提示默认采样参数变化，非错误。

## 离线阶段式（BCG）仍可用
```bash
# 1) 生成多 agent 对话并平铺
python teamtr/run_bc_stage.py \
  --prompts prompts.txt \
  --model-paths /path/Qwen3-8B-a0,/path/Qwen3-8B-a1,/path/Qwen3-8B-a2

# 2) 按脚本打印的命令，逐 agent 运行 PPO（带 teamtr_target_agent_id）
```

## 默认假设
- 同构 Qwen3-8B/1.7B 可共享 tokenizer；异构接口保留。
- 路由默认随机；可在 `ray_trainer` 中替换。
- 设备按 H200 进行配置。
