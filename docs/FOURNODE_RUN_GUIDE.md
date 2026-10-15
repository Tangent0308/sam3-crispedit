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

`setup_runtime.sh` 为 planner/editor/audit 分别创建独立环境，并使用 `requirements/production/*.lock.txt`。编辑器固定 vLLM-Omni revision；所有角色使用 CUDA 12.9 wheel。`check_runtime.py` 在环境标记 ready 前执行无界面 OpenCV 检查、spawn 子进程导入、GPU matmul、vLLM RMSNorm、CUDA runtime 和编辑器 Omni 导入检查，避免 CUDA 13 wheel 混入 CUDA 12.9 任务。

## 查看进度

```bash
ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014
python -m scripts.run_production_4node --root "$ROOT" status
for n in 0 1 2 3; do tail -F "$ROOT/logs/node$n.log"; done
tail -F "$ROOT/logs/progress_monitor_tqdm.log"
```

`progress.json` 的阶段依次为 `source_materialization`、`planning`、`model_cache`、`generation_and_audit`、`complete`。planning 日志会记录模型 shard、JIT、每批 completed/accepted 和 batch_seconds；生成/审核阶段会记录队列、通过数和类别缺额。

## 恢复规则

1. 先确保旧 Arnold 任务的四个 rank 已退出，避免同一 rank 并行写共享检查点。
2. 使用完全相同的 `SA1B_RUN_ROOT`、代码包、模型和四机拓扑重新提交入口；共享 rank 锁会阻止重复 owner。
3. 已完成 wave 会跳过；已完成 planning checkpoint 会按输入 mask 数和 accepted 数校验后复用；生成 fingerprint 匹配的结果和已落盘审核结果直接恢复。
4. 已提交但结果丢失的 planner request 会记录为 `interrupted_unknown`，不重复调用同一 mask，也不伪造拒绝结果。
5. 数据协议、输入清单、模型或生成参数改变时不混入旧运行。仅修复启动依赖、权重缓存和超时，可以在所有 rank 停止并持有共享锁时使用 `scripts/production/repair_deployment.py` 发布补丁：备份旧部署、逐项验证旧代码及 planner 快照，只更新允许的运行时文件及 `code_hashes`，保留原根目录和全部检查点。禁止手工删除 hash 检查。
6. 最后一个 rank 结束后自动尝试合并；必要时在代码环境执行 `finalize`，并检查 `final/summary.json`。

## 2026-10-14 启动修复与原地恢复

本次任务的 root 始终为 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014`，直接用上方原入口重新提交四机任务，保留 `SA1B_SKIP_SETUP=0`。无需重新 prepare，也无需换目录。

- 审核 lock 曾同时安装 `opencv-python` 与 `opencv-python-headless`，二者覆盖同一 `cv2` 目录，子进程报 `libxcb.so.1`。修复为只保留 headless；旧环境 lock 变化时自动同步，并重新安装 headless 修复共享文件。
- 编辑器从共享盘并发读权重，触发 Omni 默认 600 秒启动超时。`model_cache.py` 在启动 GPU worker 前将编辑、审核权重各复制一份到节点本地盘，按文件 SHA-256 校验、断点复用；约需 83 GiB，建议预留 100 GiB。Omni 初始化及 stage 超时均设为 1800 秒。缓存阶段日志有字节进度，复制不会占用模型初始化的超时额度。
- 数据记录和生成 fingerprint 继续使用原模型路径，只有模型加载使用本地副本；步数、seed、指令和审核门槛保持原设置。
- 已完成的 planner 副本直接复用；未完成副本只处理未提交 mask。已提交但未落盘结果的 mask 标记 `interrupted_unknown`，不会重复调用；后续候选补足最终配额。已生成图按 fingerprint 复用，已审核结果从共享 `results/` 重建计数及本地队列。
- 失败时同步写入 `progress.json.status=failed`，避免界面继续显示 running。后台监控每 300 秒写日志；聊天窗口不具备后台日志自动推送能力。

验证记录位于运行根目录 `verification/startup_repair/`：229 项自动测试通过，实际 GPU 运行 4 条（四类各一条），全部完成生成和审核，其中 3 条自动通过、1 条由质量审核拒绝；4 张生成图的恢复验证未再次调用生图，图片哈希不变。编辑器加载 27.4 秒、4 条生成与保存 84.8 秒；审核权重加载 12.7 秒（不含初始化和预热）。这些结果用于验证启动与恢复，不计入正式 50k 配额，也不能用于估计长期通过率。

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

## 分组数据的独立运行

孪生/多目标任务不能使用 50k 的 `SA1B_RUN_ROOT`、部署包或队列数据库。先创建独立目录并预留评测源图：

```bash
GROUP_INPUT_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_group_inputs_20261014
SOURCE_ROOT=/mnt/hdfs/sg_byte_ttlive_strategy_llm/user/haobo.yuan/datasets
python scripts/prepare_group_inputs.py \
  --frozen-root /mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014 \
  --benchmark-exclusion /mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014/inputs/benchmark_exclusion.jsonl \
  --source-root "$SOURCE_ROOT" --output-root "$GROUP_ROOT" \
  --eval-count 1200 --train-limit 16000 \
  --materialize-missing-train --materialize-workers 32
```

如果历史训练清单已知，应显式加入评测排除注册表（每个清单一个参数），例如：

```bash
  --skip-discovery \
  --existing-manifest /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/manifest.jsonl \
  --existing-manifest /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined/manifest.jsonl
```

缺失的训练源会以 `--materialize-workers` 路并发读取；默认 8 路。评测预留只在新根目录首次创建，恢复时复用已有 `inputs/eval_reservation.json`。

孪生第一批使用四类配额各约 1,250 组；下面命令先只做规划，确认规划质量后再启动编辑器和 27B 审核：

```bash
python scripts/run_group_pipeline.py --kind twins \
  --manifest "$GROUP_INPUT_ROOT/inputs/train_candidates.jsonl" \
  --output-root "$GROUP_INPUT_ROOT/twins_train_wave1" --target 5000 --split train \
  --reservation "$GROUP_INPUT_ROOT/inputs/eval_reservation.json" \
  --planner-model /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-Flash-Next-FP8 \
  --planner-backend vllm --planner-tp 4 --planner-batch-size 8 \
  --candidate-pool-factor 12.0 --plan-only
```

Flash-Next 仍是待验证的规划/审核候选，不代表已经达到正式发布质量。当前适配根据本地 `config.json` 的 `Qwen4ExpForConditionalGeneration` 自动启用 Triton MoE、Engram CPU offload；四卡使用 TP4，八卡使用 TEP8。它需要支持该架构的运行时；`requirements/mllm-vllm-cu129.txt` 中的旧 27B 环境不能直接视作已验证兼容。本机实验复用 `/tmp/sa1b-next-runtime/bin/python`（vLLM 0.31.1rc1.dev296、CUDA 12.9）及已下载的节点缓存。审核切换模型时同时指定 `--audit-tp 4`，默认仍为 1。每种模型/输入变体必须使用新的输出根目录，避免复用另一模型的计划缓存。

正式 wave1 选择已经固定在独立根目录
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_twins_flashnext_wave1_20261015/inputs/selected.jsonl`（5,000 组），四个分片位于同目录的 `shards/twins_train_wave1/`。这份清单已完成四类、scope、面积比和评测排除校验，不要重新选择或覆盖它：

```bash
export GROUP_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_twins_flashnext_wave1_20261015
export GROUP_INPUT_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_group_inputs_20261014
python scripts/shard_group_selection.py \
  --selection "$GROUP_ROOT/inputs/selected.jsonl" \
  --output-root "$GROUP_ROOT/shards/twins_train_wave1" --world-size 4
```

每个 Arnold worker（`ARNOLD_ID=0..3`）可以直接执行仓库内的统一入口
`scripts/group_4node_entry.sh`。它会校验 4×8 拓扑、原子复制本 rank shard、复用已有
checkpoint，并把日志写到 rank 独立目录。示例环境变量如下：

```bash
export GROUP_ROOT="$GROUP_ROOT"
export GROUP_KIND=twins GROUP_SPLIT=train
export GROUP_SHARD_ROOT="$GROUP_ROOT/shards/twins_train_wave1"
export GROUP_SELECTION="$GROUP_ROOT/inputs/selected.jsonl"
export GROUP_MANIFEST="$GROUP_INPUT_ROOT/inputs/train_candidates.jsonl"
export GROUP_RESERVATION="$GROUP_INPUT_ROOT/inputs/eval_reservation.json"
export GROUP_PLANNER_MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-Flash-Next-FP8
export GROUP_EDITOR_MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1
export GROUP_AUDIT_MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B
export GROUP_PLANNER_TP=4 GROUP_MAX_NEW_TOKENS=1200 GROUP_AUDIT_TP=1
# The entry automatically enables clean binary/outline crops, full-scene localization,
# and the grounded group-audit prompt. Leave GROUP_PUBLISH unset until the PDF ledger is complete.
bash scripts/group_4node_entry.sh
```

四个 rank 都结束后合并并做硬校验：

```bash
python scripts/merge_group_shards.py --kind twins --split train \
  --input-root "$GROUP_ROOT/runs/twins_train" --target 5000 \
  --output "$GROUP_ROOT/final/twins_train.jsonl"
```

评测 200 组使用相同流程，但输入换为 `eval_twin_pairs_selected.jsonl`，`--split eval`，并把输出写到 `twins_eval/rank${RANK}`；评测组完成后必须生成图片内嵌 PDF，逐组填写 `human_review.jsonl`，全部确认后才可发布。训练发布至少需要孪生 200 组或多目标 100 组的显式人工通过记录；评测发布必须每组都人工通过。`GROUP_PUBLISH=1` 时必须设置 `GROUP_HUMAN_REVIEW`，入口会拒绝缺失或未签字的 ledger。多目标的 1,500 训练组和 100 评测组必须在孪生第一批人工确认后执行。当前选择文件已经完成并分片：

```text
$GROUP_ROOT/inputs/train_multi_selected.jsonl       # 1,500 groups
$GROUP_ROOT/inputs/eval_multi_selected.jsonl        # 100 groups
$GROUP_ROOT/shards/multi_train_wave1/               # 4 x 375
$GROUP_ROOT/shards/multi_eval/                     # 4 x 25
```

多目标选择文件在进入规划前必须完成操作配额检查。`scripts/rebalance_multi_operations.py` 会枚举每组可行序列并只更新 `operations` 字段；训练和评测的重排证据分别记录在 `$GROUP_ROOT/verification/multi_operation_balance.json`，随后必须重新运行 `scripts/shard_group_selection.py` 更新四机分片。不要用固定的目标位置顺序代替该步骤，也不要把不可行操作通过改标签强行填入配额。

选择阶段使用快速 bbox 容量池后再运行 `scripts/recheck_multi_selection.py` 做 RLE 相邻复核；快速池本身不能进入规划或发布。四机入口将 `GROUP_KIND=multi`、`GROUP_SPLIT=train`、`GROUP_SHARD_ROOT=$GROUP_ROOT/shards/multi_train_wave1`、`GROUP_SELECTION=$GROUP_ROOT/inputs/train_multi_selected.jsonl`，目标设为 1,500；评测时改用 `multi_eval` 分片、`eval_multi_selected.jsonl` 和 `GROUP_SPLIT=eval`。多目标每组仍只做一次规划调用，随后按 `generation_order` 链式生图，组级生成/27B 审核/人工 ledger 全部通过后才可发布。

生图阶段追加 `--editor-model ... --audit-model ... --publish`，并将 `--split eval` 指向
`inputs/eval_candidates.jsonl` 生成 200 组评测；评测组必须全部通过人工 ledger 后才可发布。多目标第一批在孪生第一批训练侧确认后执行，命令把 `--kind twins` 改为 `--kind multi`，目标设为 1,500，并使用同一个 source-isolation reservation。每个组一条 planning call，生成和审核 checkpoint 可从同一独立根目录恢复；不要把失败组缩减成较少目标数。

分组规划实验可显式添加 `--outline-planning-crops`，在每个候选行追加 mask 外侧轮廓；仍只向模型传原图和一张候选拼图。该开关尚在验证，默认关闭，当前 50k 快照不受影响。
