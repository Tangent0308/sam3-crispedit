# 四机运行指南

## 运行参数

- Arnold worker：4
- 每 worker：8 × H100
- 每 rank：2 个 TP=4 规划副本、6 个单卡编辑器、2 个单卡审核器
- 目标：50,000 条自动审核通过样本，`add/replace/remove/attribute` 各 12,500 条
- CUDA：12.9；PyTorch `2.13.0+cu129`
- 共享挂载：`/mnt/bn/strategy-mllm-train`
- 源记录：运行目录下已冻结的 `source_cache/`，无需 SG2 HDFS 挂载
- W&B：关闭

## 入口

四个 Arnold worker 使用同一段脚本；平台负责注入 `ARNOLD_ID`：

```bash
#!/usr/bin/env bash
set -euo pipefail
export SA1B_RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014"
export SA1B_RUNTIME_ROOT="/opt/tiger/sa1b-production-runtime"
export SA1B_SKIP_SETUP=0
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=LAZY
exec bash "$SA1B_RUN_ROOT/deployment/launch_4node.sh"
```

也可以直接调用冻结入口：

```bash
bash /mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014/deployment/arnold_entry_full.sh
```

脚本会拒绝非 4×8 配置。启动前应看到：

```text
Shared source cache verified: 16000 images, 32000 records; HDFS mount not required
Shared inputs and model configurations are accessible
```

## 安装和启动检查

`setup_runtime.sh` 为 planner/editor/audit 分别创建独立环境，并使用 `requirements/production/*.lock.txt`。编辑器固定 vLLM-Omni revision；所有角色使用 CUDA 12.9 wheel。`check_runtime.py` 在环境标记 ready 前执行 GPU matmul、vLLM RMSNorm、CUDA runtime 和编辑器 Omni 导入检查，避免 CUDA 13 wheel 混入 CUDA 12.9 任务。

## 查看进度

```bash
ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014
python -m scripts.run_production_4node --root "$ROOT" status
for n in 0 1 2 3; do tail -F "$ROOT/logs/node$n.log"; done
tail -F "$ROOT/logs/progress_monitor.log"
```

`progress.json` 的阶段依次为 `source_materialization`、`planning`、`generation_and_audit`、`complete`。planning 日志会记录模型 shard、JIT、每批 completed/accepted 和 batch_seconds；生成/审核阶段会记录队列、通过数和类别缺额。

## 恢复规则

1. 先确保旧 Arnold 任务的四个 rank 已退出，避免同一 rank 并行写共享检查点。
2. 使用完全相同的 `SA1B_RUN_ROOT`、代码包、模型和四机拓扑重新提交入口；共享 rank 锁会阻止重复 owner。
3. 已完成 wave 会跳过；已完成 planning checkpoint 会按输入 mask 数和 accepted 数校验后复用；生成 fingerprint 匹配的结果和已落盘审核结果直接恢复。
4. 已提交但结果丢失的 planner request 会记录为 `interrupted_unknown`，不重复调用同一 mask，也不伪造拒绝结果。
5. 配置、输入清单、代码 hash 或模型路径改变时不续跑，应创建新的运行根目录。
6. 最后一个 rank 结束后自动尝试合并；必要时在代码环境执行 `finalize`，并检查 `final/summary.json`。

## 故障处理

- 看到 `/mnt/hdfs/...index`：确认使用的是更新后的冻结 `deployment/code.tar.gz`，并确认 `/mnt/bn/strategy-mllm-train` 已挂载；当前正式配置使用 shared source cache。
- 看到 `libcudart.so.13`：不要用默认 `vllm==...` 重新安装；清理对应 role 环境，保留 CUDA 12.9 lock，重新以 `SA1B_SKIP_SETUP=0` 启动。
- 只有旧的 `nodes/node*/failure.json` 而日志时间早于本次启动：先以 `owner.json`、最新 `progress.json` 和日志 mtime 判断，不能把旧 143 记录当作当前错误。
- 磁盘或共享目录异常：停止任务后保留 root、logs 和 checkpoints；不要删除 source_cache、dispatch、results 或队列数据库。

## 完成判定

```bash
cat "$ROOT/final/summary.json"
wc -l "$ROOT/final/auto_pass.jsonl"
```

完成必须同时满足总数 50,000、四类各 12,500、case ID 和 source mask 唯一、所有图片字段可解析、每条关键审核字段为真。自动完成不等于人工质量验收；训练发布前按 [PIPELINE.md](PIPELINE.md) 的质量项抽查和记录人工结论。
