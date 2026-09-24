# CrispEdit 四机完整打标

执行顺序：质量筛选 → 难定位局部编辑筛选 → Qwen3.8 编辑单元观察/grounding → SAM3 mask → 校验。
方法及现有数据规模见[主文档](CRISPEDIT_MASK.md)。本页对应 `crispedit-labeling` 分支当前实现，不是仅 prefilter 的入口。

## 1. 运行条件与调度

四台 worker、每台 8 GPU，共 32 卡；数据、模型、仓库、基础 Python、环境包、输出目录在共享盘，依赖解压到各节点本地缓存。
沿用 SAMTokEdit 指南的 Arnold rank 与 node0 建环境方式，但本任务按 shard 独立分工，**不使用跨节点 DDP/NCCL rendezvous**。

| 阶段 | 每节点执行方式 | 输出 |
| --- | --- | --- |
| quality | Qwen3.8/vLLM，TP1，最多 8 实例，batch4 | 质量根目录 `audit/manifest` |
| scene | Qwen3.8/vLLM，TP1，最多 8 实例，batch4，仅质量 PASS | 场景根目录 `audit/manifest` |
| grounding | Qwen3.8/vLLM，TP2，最多 4 实例，batch16，仅双 PASS | `RUN_DIR/labels/grounding` |
| mask | 每卡一个 SAM3 worker | `RUN_DIR/labels/mask` |

节点内不足的任务不会强行占满实例。各阶段全部节点完成后，由 node0 校验并合并，再释放下一阶段。
quality/scene 按源行数分 shard，mask 按双 PASS 数重新平衡；只写各自节点工作目录，最终合并拒绝覆盖其他运行的结果。

实现：[bootstrap](../scripts/bootstrap_crispedit_4node.sh)、[入口](../scripts/run_crispedit_pipeline.py)、[调度与合并](../crispedit/distributed.py)、[统一环境](../scripts/setup_crispedit_env.sh)。

保护措施：冻结源 shard 大小/修改时间/行数、代码摘要和参数；校验两个 manifest 的原始 row_idx 与筛选来源；记录每节点 hostname；生产模式要求四个不同主机；失败 marker 通知其他节点停止子进程；锁住重复 rank。不同运行不要并发写同一质量/场景输出目录。

## 2. 可直接提交的完整入口

已准备不依赖远程 push 的共享代码快照：
`/mnt/bn/strategy-mllm-train/user/tanyue/workspaces/crispedit_labeling_20260924`。
已准备经过本机推理验证的环境包，启动时校验 SHA256 并缓存到节点本地。直接在共享盘导入大量 Python 依赖曾耗时十余分钟，因此推荐本地缓存；不改变推理方法。
基础 Python 位于旧共享 workspace，不要修改或删除它；每节点需要约 11GB 本地磁盘。模型权重不复制。

把下面完整内容作为 **四台 Arnold worker 的共同入口**，不是只在 node0 运行。分配 worker_num=4、worker_gpu=8。
每次新任务换一个 `CRISPEDIT_RUN_ID`，四台必须完全一致，不能各自用当前时间生成。

```bash
#!/usr/bin/env bash
set -euo pipefail
export CRISPEDIT_RUN_ID="crispedit_full_20260924_a"
export CRISPEDIT_REPO_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/workspaces/crispedit_labeling_20260924"
export CRISPEDIT_USE_EXISTING_REPO=1
export CRISPEDIT_RUNTIME_ARCHIVE="$CRISPEDIT_REPO_DIR/runtime/python312_packages.tar.gz"
export CRISPEDIT_BASE_PYTHON="/mnt/bn/strategy-mllm-train/user/tanyue/workspaces/sam3-crispedit_prefilter_4node_20260923/.uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
export CRISPEDIT_LOCAL_RUNTIME_DIR="/opt/tiger/tanyue/.cache/crispedit_runtime_20260924"
export CRISPEDIT_RUN_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_${CRISPEDIT_RUN_ID}"
export CRISPEDIT_INPUT_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M"
export CRISPEDIT_QUALITY_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-qwen38-pair-prefilter"
export CRISPEDIT_SCENE_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-difficult-local-edit"
export CRISPEDIT_LABEL_DIR="$CRISPEDIT_RUN_DIR/labels"
export CRISPEDIT_MODEL_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
export CRISPEDIT_SAM3_CHECKPOINT_PATH="/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"
export CRISPEDIT_FILTER_BATCH_SIZE=4
export CRISPEDIT_GROUNDING_BATCH_SIZE=16
bash "$CRISPEDIT_REPO_DIR/scripts/bootstrap_crispedit_4node.sh"
```

Arnold 提供 `ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8`、`ARNOLD_ID=0/1/2/3`。入口检查后 node0 准备环境，所有节点通过预检查，再进入完整 pipeline。
上面质量/场景参数是**根目录**，调度器自行追加 `manifest/`；不要传成 manifest 子目录。

默认行为：旧 1,298 shard 中有 170 个历史 background/style，保留文件但不进入调度；复用其余 1,128 shard 的两阶段结果，补做新增 780 shard，再对所需五类的全部双 PASS 打 mask。结果写入新的 labels 目录。
规划核对：有效源数据 1,908 shard / 485,964 行，每节点各补做 195 个 shard。根目录统计包含保留的历史结果，`active_source_shards` / `preserved_out_of_scope_shards` 区分本次范围；历史数据不删除。
若要两轮筛选也从头重跑，应把 QUALITY_DIR、SCENE_DIR 改为本次新目录；不要覆盖生产目录。

### 从远程 clone / 新环境启动

目前新分支尚未因本次任务提交/push；在 push 之前使用上面的共享快照。远程有分支后，可把 bootstrap 先复制到共享路径，再作为四 worker 入口：

```bash
# 提交任务前，在当前仓库执行一次
cp scripts/bootstrap_crispedit_4node.sh \
  /mnt/bn/strategy-mllm-train/user/tanyue/workspaces/bootstrap_crispedit_4node.sh

# 四个 worker 都执行（使用新的 run ID）
export CRISPEDIT_RUN_ID="crispedit_clone_run_a"
export CRISPEDIT_BRANCH="crispedit-labeling"
unset CRISPEDIT_REPO_DIR CRISPEDIT_PYTHON CRISPEDIT_SAM_PYTHON CRISPEDIT_USE_EXISTING_REPO
unset CRISPEDIT_RUN_DIR CRISPEDIT_LABEL_DIR
unset CRISPEDIT_RUNTIME_ARCHIVE CRISPEDIT_BASE_PYTHON CRISPEDIT_LOCAL_RUNTIME_DIR
bash /mnt/bn/strategy-mllm-train/user/tanyue/workspaces/bootstrap_crispedit_4node.sh
```

node0 clone 用户 GitHub 仓库，安装共享 `.uv-python` 和 `.venv-crispedit`，其他节点等待，不同时安装。
使用锁定依赖：PyTorch 2.13/cu129、vLLM 0.28、Transformers 5.15.1、NumPy 1.26.4；具体见 `scripts/crispedit_packages.txt`。
需要模型权重已存在、共享盘可执行 Python、匹配 CUDA 驱动及安装阶段网络可用；不会自动下载模型。
新环境安装后可由 node0 一次性执行 `bash scripts/pack_crispedit_runtime.sh .venv-crispedit /共享路径/runtime.tar.gz`，
得到环境包及 `.sha256`；后续用上一节的 archive/base-python/local-runtime 三个变量运行。
[缓存脚本](../scripts/cache_crispedit_runtime.sh) 使用文件锁、SHA256 和完成标记，拒绝覆盖不同版本的本地缓存。失败的临时解压目录保留用于诊断，不自动删除其他目录。

## 3. 结果、tqdm 与恢复

```text
RUN_DIR/
  logs/bootstrap.node{0..3}.log    环境、协调器输出
  logs/quality.node{0..3}.log      第一轮 tqdm
  logs/scene.node{0..3}.log        第二轮 tqdm
  logs/grounding.node{0..3}.log    Qwen 定位 tqdm
  logs/mask.node{0..3}.log         SAM3 tqdm
  logs/validate.node0.log         最终结构校验
  plan.json                      源快照、参数、代码摘要、筛选分工
  label_plan.json, selection*.json 双 PASS 行号及 mask 分工
  work/                          节点中间结果，恢复时保留
  labels/grounding/, labels/mask/  合并后的双 PASS 稀疏 parquet
  labels/validation_summary.json  QC 统计
  reports/                       两轮统计、mask 统计、运行记录
  control/initial/complete.ok     完整成功标记
```

保留原始 row_idx，不把输出中第几行当作源行号。`complete.ok` 表示全流程结构验证完成，**不是所有 mask 语义正确**。

```bash
RUN_DIR=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_crispedit_full_20260924_a
tail -f "$RUN_DIR/logs/grounding.node0.log"
tail -f "$RUN_DIR/logs/mask.node0.log"
```

失败先看 `control/本次attempt/*.failed` 和对应节点日志。确认旧四节点进程都已退出后，四台用相同原配置再增加：

```bash
export CRISPEDIT_RESUME=1
export CRISPEDIT_RESUME_TOKEN="retry_01"  # 每次恢复换新名称，四台一致
bash "$CRISPEDIT_REPO_DIR/scripts/bootstrap_crispedit_4node.sh"
```

恢复会检查代码/参数/源快照/manifest 未变化，再利用完整且签名匹配的 shard；已完成的筛选 shard 不加载模型。不手工删除失败标记或改 plan 来绕过验证。成功标记改为 `control/retry_01/complete.ok`。修改代码或数据后使用新 RUN_DIR。

Arnold worker 作业本身在后台运行，不依赖登录终端；手工登录四台机器运行时，每台分别将同一环境变量与对应 ARNOLD_ID 放入自己的 tmux，不能在一台上伪造四个 rank 当四机。

## 4. 本次验证与复现

当前可用资源只有一台 8 卡 H100。已验证四 rank 协议和真实 GPU 小批量链路；**没有声称完成四台物理机器、32 卡的实际运行**。物理四机的挂载、网络环境和性能仍需在用户分配四 worker 后验收。

189 项测试通过，覆盖完整四 rank 交接、空任务节点、重复恢复、失败传播、原始行号、历史类型隔离及拒绝覆盖，另测试环境缓存的迁移、重复使用和不同包的冲突保护。
真实 GPU 用一机 8 卡运行四个独立 rank，每 rank 2 卡，显式 `--local-test`，重新跑两轮筛选、Qwen grounding 与 SAM3。原始数据复制到隔离目录，生产结果不改动。

```bash
cd /opt/tiger/tanyue/sam3-crispedit-crispedit-labeling
tmux new-session -d -s crispedit_smoke ' \
  /opt/tiger/tanyue/sam3-crispedit/.venv-scaleedit-vllm/bin/python -u \
  scripts/smoke_crispedit_pipeline.py \
  --source-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --selection-file /mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/mask_fresh65_doublepass_20260923/selection.json \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/pipeline_smoke_new_run \
  --per-type 4 > /mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/pipeline_smoke_new_run.log 2>&1'
```

每次复现换新的 output-dir。抽样映射在 `provenance.json`，样本覆盖五种类型各 4 条，不作为无偏质量评测。

实际验证目录：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/pipeline_4rank_verified_20260924/`。

- `run/`：完整重跑两轮筛选与 mask；20 quality PASS → 19 scene PASS / 1 DROP → 19 非空 mask、35 实例，0 解析/运行错误；四 rank 均 exit 0。
- `run/control/resume_check/complete.ok`：同一运行恢复成功，复用全部结果。
- [19 条可视化](/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/pipeline_4rank_verified_20260924/review/index.html)：HTML 内嵌图片，已有问题未被当作质量成功。
- `shared_run/`：共享依赖冷启动诊断。十余分钟仍在依赖导入，主动停止；失败 marker 来自人为停止，不计作成功结果。
- `cached_run/`：共享代码 + 节点本地依赖缓存复验成功；四 rank 均完成，19 非空 mask / 35 实例，0 解析/运行错误。成功标记 `cached_run/control/initial/complete.ok`，各阶段 tqdm 在 `cached_run/logs/`。
- [缓存环境 19 条画廊](/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/pipeline_4rank_verified_20260924/cached_review/index.html)：与首轮 15 条 mask 完全一致；4 条 observation 相同但 box 有小变化，预测间 IoU 为 0.979–0.996，非真值准确率。

环境包约 5.4GB，解压约 11GB。缓存后几十秒内进入 vLLM 模型初始化；首次 FlashInfer CUDA 编译仍需额外预热，不把缓存后的导入耗时当作整个任务耗时。

全量只做了 `--plan-only`，没有启动推理：
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/repo_cleanup_20260924/production_plan/plan.json`。
