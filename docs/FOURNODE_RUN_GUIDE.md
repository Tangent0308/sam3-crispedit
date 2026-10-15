# 四机运行指南

## 运行参数

- Arnold worker：4
- 每 worker：8 × H100
- 每 rank：2 个 TP=4 规划副本、6 个单卡编辑器、2 个单卡审核器
- 目标：50,000 条自动审核通过样本，`add/replace/remove/attribute` 各 12,500 条
- CUDA：12.9；PyTorch `2.13.0+cu129`
- 共享挂载：`/mnt/bn/strategy-mllm-train`
- 源记录：运行期间使用冻结的 `source_cache/`，无需 SG2 HDFS 挂载；本批完成后该缓存已按确认删除，恢复旧 run 前需重建
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
3. 已完成 wave 会跳过；已完成 planning checkpoint 会按输入 mask 数和 accepted 数校验后复用；生成 fingerprint 匹配的结果和已落盘审核结果直接恢复。 分组规划还会按 planner batch（默认 8）逐组原子落盘，rank 被回收时不重复已解析的 MLLM 结果。
4. 已提交但结果丢失的 planner request 会记录为 `interrupted_unknown`，不重复调用同一 mask，也不伪造拒绝结果。
5. 数据协议、输入清单、模型或生成参数改变时不混入旧运行。仅修复启动依赖、权重缓存和超时，可以在所有 rank 停止并持有共享锁时使用 `scripts/production/repair_deployment.py` 发布补丁：备份旧部署、逐项验证旧代码及 planner 快照，只更新允许的运行时文件及 `code_hashes`，保留原根目录和全部检查点。禁止手工删除 hash 检查。
6. 最后一个 rank 结束后自动尝试合并，成功后检查 `final/summary.json`。如果四个节点均已写出 `done.json` 和 `auto_pass.jsonl`，仅汇总失败（例如共享盘不支持 `flock`，报 `Errno 38`），无需重启四机或重新生成图片；在能访问共享目录的 CPU 机器上，使用更新后的仓库执行下述独立收尾入口。该入口保留冻结配置，验证原有导出及审核规则的代码哈希，使用原子建目录锁防止并发发布，并检查数量、类型配额、唯一性、审核标记及全部图片路径。图片校验包含存在性、非空及根目录内路径，不包含全量解码或人工质量复核。

```bash
python3 -u -m scripts.finalize_production_exports \
  --root /mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014 \
  --workers 16
```

收尾进度写入 `final/progress.json`；最终标注为 `final/auto_pass.jsonl`，其中图片路径相对于数据根目录；`final/summary.json` 保存统计、输入导出哈希及最终标注哈希。目录锁 `final/finalize.lock.d` 不会自动抢占；异常退出遗留锁时，先核实 `owner.json` 对应进程已退出再处理。历史 `failure.json` 保留；本次 `Errno 38` 成功恢复后更新节点进度为完成。

## 本批最终数据入口（2026-10-15）

四机配额已完成，无需重启任务。CPU 收尾及规范整理后，训练入口为 `$ROOT/release/annotations/train.jsonl`，媒体路径相对于 `$ROOT/release`；`final/auto_pass.jsonl` 作为原始合并记录保留。全量统计、字段和可视化见 [50k 数据说明](DATASET_50K_ZH.md)。源记录缓存 A 已按用户确认删除；该已完成运行不应直接重新启动，恢复/扩量前需重建并校验缓存。其余候选保留，见 [清理记录](CLEANUP_CANDIDATES_50K_ZH.md)。

## 2026-10-14 启动修复与原地恢复

以下记录 2026-10-14 的故障恢复方式：当时使用同一 root `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014` 和原入口重新提交，保留 `SA1B_SKIP_SETUP=0`，无需重新 prepare。当前本批已经完成且源缓存已清理，不再直接套用该历史重启步骤。

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
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_twins_flashnext_wave1_20261015/inputs/selected.jsonl`（5,000 组），四个分片位于同目录的 `shards/twins_train_wave1/`。这份清单已完成四类、scope、面积比和评测排除校验，不要重新选择或覆盖它：校验摘要和清单/分片哈希见 `$GROUP_ROOT/verification/selection_integrity.json`。全量训练/评测 source ID 与图像哈希隔离记录见 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_group_inputs_20261014/verification/source_isolation_full.json`。 重新生成该记录可运行 `scripts/verify_group_inputs.py`；它在模型启动前失败即停止，避免把泄漏或配额错误带入四机。

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
export GROUP_PLANNER_TP=4 GROUP_MAX_NEW_TOKENS=1600 GROUP_AUDIT_TP=1
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


### 孪生小批量的分阶段并行入口（2026-10-15）

`group_4node_entry.sh` 的默认入口仍是每 rank 一个编辑器。新增的 `--editor-devices` 是显式选择的分阶段接口，已验证单机双卡；尚未完成四机性能验证。规划/审核与编辑运行环境版本不同，应分别使用各自的 Python。以下变量沿用本节的模型和独立运行根目录；`RUN_ROOT`、`MANIFEST`、`TARGET` 应指向本 rank 的根目录、候选清单和预定组数。

```bash
COMMON=(--kind twins --manifest "$MANIFEST" --output-root "$RUN_ROOT"   --target "$TARGET" --split train --reservation "$GROUP_RESERVATION"   --outline-planning-crops --locate-planning-scene)
"$PLANNER_PYTHON" scripts/run_group_pipeline.py "${COMMON[@]}"   --plan-only --planner-model "$GROUP_PLANNER_MODEL" --planner-tp "$GROUP_PLANNER_TP"   --max-new-tokens 1600
"$EDITOR_PYTHON" scripts/run_group_pipeline.py "${COMMON[@]}"   --generate-only --editor-model "$GROUP_EDITOR_MODEL" --editor-devices 0,1
"$AUDIT_PYTHON" scripts/run_group_pipeline.py "${COMMON[@]}"   --audit-only --audit-model "$GROUP_AUDIT_MODEL"
```

GPU 列表使用本机物理可见设备标识，每个 worker 独占列表中的一个编辑器实例；不要填写重复设备。每组只调度一次，恢复使用逐成员输入哈希与输出校验和。日志包含已生成/确定性通过/总组数，`release.json` 记录各类自动接受数量与缺口；缺口不会自动变成足额数据。新 prompt 实验必须用新根目录，恢复不会重复规划已拒绝组。

`--twin-audit-mode joint` 仅用于对照实验：现有回归表明它严重漏检，不能因为调用数少而直接替代默认审核。当前小批量结论与 PDF 在 PIPELINE.md 第 7.9 节；正式分组生产尚不建议启动。当前 50k 生产快照不受这些实验入口影响。

新分组运行已切换到 instruction/mask 2.0。必须新建运行根目录，不能用 2.0 代码直接续跑 1.0 规划检查点；`config.json` 会校验协议版本。源候选选择和 reservation 可以复用。生成仍使用 Ref、40 步及原执行区域；新 add site mask 在生成后解析。最终准入额外要求 `blind_protocol.pass=true`，定位失败不会退回宿主 mask 静默发布。


### 2.0 的 JSON 约束与 site 修正

在新的小批量运行中，可给上述 COMMON 同时追加 `--structured-planning-json` 和 `--site-model /mnt/bn/strategy-mllm-train/common/models/sam3`。规划运行时设置 `SA1B_FORCE_DIRECT_VLLM=1`，使用已验证支持 StructuredOutputsParams 的直接 vLLM 后端。三个阶段必须使用相同的配置参数，否则不可变 config 校验会拒绝继续。

`--site-model` 在每个编辑 worker 首次遇到 add/site 后懒加载 SAM3，在生成后、审核前修正源 mask。SAM3 与编辑器同卡常驻的峰值显存和四机性能尚未测试；当前已实测的是独立运行时对既有目标图的修正。无需再次调用编辑器或 MLLM：

```bash
# 在仓库根目录执行；OUT 必须是尚不存在的新目录。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=4 /tmp/sa1b-next-runtime/bin/python \
  scripts/refine_group_source_regions.py \
  --manifest /path/to/run/audited.jsonl \
  --output-root /path/to/run/site_refined \
  --site-model /mnt/bn/strategy-mllm-train/common/models/sam3
```

GPU 4 仅为例子，需选择本任务可用卡。该本机运行时为 Transformers 5.17；复制到新节点时应先验证 SAM3Model/SAM3VideoModel/SAM3Processor 可导入。新 `groups.jsonl` 只复用原目标图，变更源 mask 的组会撤销旧语义通过状态并标 pending，不会发布。它是诊断修正入口，不能把旧审核的通过率当成新 mask 的通过率；新版本须完成定位/轮廓复核再进入发布流程。


### 跨类别孪生配额与盲测

新孪生默认 `--twin-same-category-fraction 0.30`，其余 70% 为跨类别。四类编辑配额、两/三成员比例、面积比和源隔离不变。旧的纯同类 5,000 组清单不满足新配比；复用源候选库与 reservation，重新选择、冻结并分片，使用新运行根目录。不可变配置包含 `twin_contract=blind-cross-category-1`，旧计划不会静默当作新协议通过。

规划阶段现在包括每组一次规划和一次独立无定位盲测；`--plan-only` 的 planned.jsonl 只包含两关都通过的组，planned/ 中保留所有规划成功记录，rejected/*_blind.json 与 blind_probes/ 保留盲测拒绝/不确定结果。可读取 blind_probes/summary.json 查看额外调用和耗时。盲测失败组不会进入编辑器。

`group_4node_entry.sh` 自动加 `--quota-from-selection`，各 rank 按冻结 shard 的操作/同类跨类计数分配最终名额；这样不会因每 rank 各自四舍五入而破坏全局配额。选择清单中的 category_relation 必须来自新选择过程。`merge_group_shards.py` 对新协议复核盲测、粒度和同类约 30% 的全局比例（容差 max(1 组, 2% 总组数)）；任何 rank 缺额都不能发布为足额数据。本轮没有启动四机生成。
