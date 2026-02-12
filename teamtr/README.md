# TeamTR Logs (detailed)

## 已完成
- 离线阶段式 BCG：`teamtr/run_bc_stage.py` 生成多 agent 对话、平铺 parquet、打印 per-agent PPO 命令；`teamtr/context_rollout.py`（多 agent 采样），`teamtr/data_utils.py`（平铺）。
- 多模型生成：`rollout.name=multi_agent_hf` 支持按 `agent_id` 分桶的 HF 生成（FSDP），组件 `MultiAgentHFRollout` 已导出。
- 训练侧支持：`ray_trainer` 写入 `agent_id`，支持 `teamtr_target_agent_id` 过滤做块坐标式更新；示例 `teamtr/run_teamtr_qwen3_4b.sh`。

## 缺口
1) 在线多模型训练：ActorRolloutRefWorker 仍单模型/单优化器，未按 agent_id 切换模型并冻结其他 agent；需为每 agent 构建独立 module/optimizer/ckpt，并改造 FSDP/Megatron 路径。
2) 上下文串联：主数据流仍单轮 prompt-response，未消费多轮对话，也未将上一 agent 输出拼成下一 agent prompt；需定义多轮 DataProto/schema 与 mask。
3) 示例未更新为交替对话，待 1/2 打通。

## 文件索引
- 核心：`teamtr/router.py`、`teamtr/runner.py`、`teamtr/algorithms.py`
- 采样/平铺：`teamtr/context_rollout.py`、`teamtr/data_utils.py`、`teamtr/run_bc_stage.py`
- 生成/训练支撑：`verl/workers/rollout/multi_agent_hf_rollout.py`、`verl/trainer/ppo/ray_trainer.py`
- 示例：`teamtr/run_teamtr_qwen3_4b.sh`

## 时间线
- 2026-01-18：加入多 agent 采样/平铺、multi_agent_hf 生成、target_agent 过滤、离线阶段脚本；更新示例。
