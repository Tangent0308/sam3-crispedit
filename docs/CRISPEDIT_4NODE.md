# CrispEdit 四机完整打标

当前入口：四台机器分别 git clone `crispedit-labeling`，运行仓库安装脚本，再顺序执行质量筛选 → 细粒度筛选 → Qwen3.8 grounding → SAM3 mask → 校验。
代码、基础 Python、依赖、下载缓存均在每台机器本地；共享盘只用于数据、模型、结果、日志和协调标记。
实现：[安装](../scripts/setup_crispedit_env.sh)、[预检查](../scripts/preflight_crispedit_env.py)、[四机启动](../scripts/bootstrap_crispedit_4node.sh)、[调度](../crispedit/distributed.py)。方法和生产统计见[主文档](CRISPEDIT_MASK.md)。

## 1. 四台 Arnold worker 使用相同完整入口

每台 8 GPU；Arnold 提供 `ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8` 和 `ARNOLD_ID=0/1/2/3`。
先把修复提交推送至远程分支，再提交任务。每次新运行更换唯一 RUN_ID，四台保持一致。仅在任务尚未进入数据规划、共享目录中没有数据产物时，才可清空失败状态后复用 RUN_ID。
需要 Git、Python3/pip、GNU tar、flock、可访问 GitHub/PyPI/PyTorch 下载站点的网络，以及每节点至少约 40GB 可用本地空间。

```bash
#!/usr/bin/env bash
set -euo pipefail
export CRISPEDIT_RUN_ID="crispedit_full_localenv_20260925"
export CRISPEDIT_BRANCH="crispedit-labeling"
export CRISPEDIT_REPO_URL="https://github.com/Tangent0308/sam3-crispedit.git"
: "${ARNOLD_ID:?Arnold must supply node rank}"
: "${ARNOLD_WORKER_NUM:?Arnold must supply worker count}"
: "${ARNOLD_WORKER_GPU:?Arnold must supply GPU count}"
[[ $ARNOLD_WORKER_NUM == 4 && $ARNOLD_WORKER_GPU == 8 && $ARNOLD_ID =~ ^[0-3]$ ]]
[[ $CRISPEDIT_RUN_ID =~ ^[A-Za-z0-9._-]+$ ]]

export CRISPEDIT_RUN_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_${CRISPEDIT_RUN_ID}"
export CRISPEDIT_INPUT_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M"
export CRISPEDIT_QUALITY_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-qwen38-pair-prefilter"
export CRISPEDIT_SCENE_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-difficult-local-edit"
export CRISPEDIT_LABEL_DIR="$CRISPEDIT_RUN_DIR/labels"
export CRISPEDIT_MODEL_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
export CRISPEDIT_SAM3_CHECKPOINT_PATH="/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"
export CRISPEDIT_FILTER_BATCH_SIZE=4
export CRISPEDIT_GROUNDING_BATCH_SIZE=16
export CRISPEDIT_REPO_DIR="/opt/tiger/tanyue/workspaces/sam3-crispedit-${CRISPEDIT_RUN_ID}-node${ARNOLD_ID}"
unset CRISPEDIT_RUNTIME_ARCHIVE CRISPEDIT_BASE_PYTHON CRISPEDIT_LOCAL_RUNTIME_DIR
unset CRISPEDIT_PYTHON CRISPEDIT_SAM_PYTHON CRISPEDIT_USE_EXISTING_REPO

mkdir -p "$CRISPEDIT_RUN_DIR/logs" "$(dirname "$CRISPEDIT_REPO_DIR")"
exec > >(tee -a "$CRISPEDIT_RUN_DIR/logs/entry.node${ARNOLD_ID}.log") 2>&1

# Arnold may retry the user command on the same worker. A completed clone is
# reusable, but an unrelated/partial directory or a modified checkout is not.
if [[ ! -e $CRISPEDIT_REPO_DIR ]]; then
  git clone --single-branch --branch "$CRISPEDIT_BRANCH" "$CRISPEDIT_REPO_URL" "$CRISPEDIT_REPO_DIR"
else
  [[ -d $CRISPEDIT_REPO_DIR/.git ]] || { echo "Existing path is not a Git clone: $CRISPEDIT_REPO_DIR" >&2; exit 2; }
fi
cd "$CRISPEDIT_REPO_DIR"
[[ $(git remote get-url origin) == "$CRISPEDIT_REPO_URL" ]] || { echo 'Existing clone has the wrong origin' >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Existing clone has tracked modifications' >&2; exit 2; }
git fetch --no-tags origin "$CRISPEDIT_BRANCH"
target_commit=${CRISPEDIT_COMMIT:-$(git rev-parse FETCH_HEAD)}
[[ $target_commit =~ ^[0-9a-f]{40}$ ]] || { echo 'CRISPEDIT_COMMIT must be a full 40-character SHA' >&2; exit 2; }
git merge-base --is-ancestor "$target_commit" FETCH_HEAD || { echo 'Pinned commit is not on the requested remote branch' >&2; exit 2; }
git checkout --detach "$target_commit"
bash scripts/bootstrap_crispedit_4node.sh
```

入口从 clone 开始保存 `entry.nodeN.log`。bootstrap 自动安装 uv（缺失时）、执行安装脚本并保存 `bootstrap.nodeN.log`。
各节点独立创建 `.uv-python/`、`.venv-crispedit/`，不调用其他仓库或共享目录的 Python。
固定 PyTorch 2.13/cu129、vLLM 0.28、Transformers 5.15.1、NumPy 1.26.4、headless OpenCV 4.11；完整版本见 `scripts/crispedit_packages.txt`。
安装脚本带锁和配置指纹；同一完整环境可重复检查，失败安装可继续，未知旧环境拒绝覆盖。

所有节点必须通过：OpenCV 无 GUI 构建检查、图像运算、spawn 子进程导入、8 卡 CUDA 运算，以及 GPU0 上一次真实 Qwen 双图推理。
随后比较四台的 commit、代码/安装配置摘要、依赖版本与独立 hostname。任意节点失败都不会进入数据规划。
这些预检查不生成数据集标注；可在 `preflight.nodeN.log` 查看 `VLLM_TWO_IMAGE_PROBE_OK`。

## 2. 处理范围与结果

node0 在预检查完成后动态扫描源数据、生成 `RUN_DIR/plan.json`，四节点读取同一份计划立即开始执行。
当前源数据有效类型为 add、color、motion change、remove、replace，共 1,908 shard / 485,964 行；background/style 不调度。

- 质量/细粒度：按 shard 检查 `audit` 与 `manifest` 存在且行号对齐，复用完整结果，补跑缺失结果。单侧缺文件等不完整产物会报错。
- 本次失败运行生成的计划为：复用旧 1,128 个有效 shard，两阶段各补做 780 个 shard，每节点 195 个；待质量筛选 198,775 行。
- `quality_assignments` / `scene_assignments` 是四节点待处理列表；细粒度实际只推理质量 PASS 行，零 PASS shard 生成空结果。
- 完成两轮后，从全部有效源数据汇总双 PASS，按保留行数重新均衡，全部重新 grounding 和打 mask；不扫描其他实验目录来跳过旧 mask。
- 五类全部双 PASS 的 mask 写入新 `labels/`；两阶段新增结果合并回指定质量/场景根目录。其他运行的 mask 目录拒绝覆盖。

| 阶段 | 每节点默认并行 | 最终输出 |
| --- | --- | --- |
| quality | TP1，最多 8 实例，batch4 | `CRISPEDIT_QUALITY_DIR/{audit,manifest}` |
| scene | TP1，最多 8 实例，batch4 | `CRISPEDIT_SCENE_DIR/{audit,manifest}` |
| grounding | TP2，最多 4 实例，batch16 | `RUN_DIR/labels/grounding` |
| mask | 每卡一个 SAM3 worker | `RUN_DIR/labels/mask` |

所有输出保留原始 shard / row_idx；四机 mask 为双 PASS 稀疏输出。每阶段各节点完成后，node0 校验合并并释放下一阶段。
源文件快照、manifest 来源和代码摘要被冻结；不使用跨机器 DDP/NCCL。节点之间使用共享文件协调。

## 3. 日志、调试和恢复

上述入口的具体日志目录是：
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_crispedit_full_localenv_20260925/logs/`。

```text
RUN_DIR/
  logs/entry.node{0..3}.log         clone 到任务退出的控制台输出
  logs/bootstrap.node{0..3}.log     本地安装、节点等待、调度信息
  logs/preflight.node{0..3}.log     真正加载 Qwen 并双图生成
  logs/quality.node{0..3}.log       质量筛选 tqdm
  logs/scene.node{0..3}.log         细粒度筛选 tqdm
  logs/grounding.node{0..3}.log     定位 tqdm
  logs/mask.node{0..3}.log          SAM3 tqdm
  logs/validate.node0.log          最终结构校验
  bootstrap_control/initial/      每节点预检查报告和失败标记
  plan.json, label_plan.json      shard 计划、双 PASS 打标计划
  work/                          节点中间结果，恢复需要保留
  labels/, reports/              最终打标及汇总
  control/initial/complete.ok     全链路完成标记
```

```bash
RUN_DIR=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_crispedit_full_localenv_20260925
tail -f "$RUN_DIR/logs/bootstrap.node0.log"
tail -f "$RUN_DIR/logs/quality.node0.log"
```

`complete.ok` 代表数据结构/流程校验完成；mask 语义质量需另行抽查。`Peer failed` 时查看失败标记内的上游原因和对应阶段日志。
任务由 Arnold 托管，不依赖登录终端。手动四机启动时，每台分别在 tmux 内执行完整入口。

grounding 的单行模型响应解析失败会保留在该 shard 中，记录为
`grounding_status=PARSE_ERROR`、`qc_flag=GROUND_FAIL`，并在 mask 阶段生成空 mask
行供最终统计；这类行不会再触发四机全局失败。模型引擎或批处理运行时错误仍会中止 grounding，避免把服务故障当成有效结果。

恢复仅用于已经进入数据规划、且**代码/参数/源数据完全相同**的失败任务：确认旧进程退出，保留 RUN_ID、RUN_DIR 和其他配置，四台共同增加：

```bash
export CRISPEDIT_RESUME=1
export CRISPEDIT_RESUME_TOKEN="retry_01"  # 每次换新名称，四台一致
bash scripts/bootstrap_crispedit_4node.sh
```

若调度到新机器，先按完整入口重新 clone，并设置 `CRISPEDIT_COMMIT` 为原运行的完整 commit SHA，再传上述两个变量启动。成功标记在 `control/retry_01/complete.ok`。
普通代码或安装配置修改后必须用新 RUN_ID；勿删除失败标记或修改计划摘要来绕过恢复校验。旧 prefilter 仍通过其独立结果目录复用。

续传前必须确认四台机器使用同一个 `CRISPEDIT_RESUME_TOKEN`，并保留
`RUN_DIR/plan.json`、`RUN_DIR/work/` 以及已有的 shard 结果。新的 token 只会新建
`control/<token>/` 协调目录，质量、场景和 grounding 的完整 shard 会按输入签名复用，
缺失或不完整的 shard 才会重新推理。

本次 `labeling_4node_crispedit_full_localenv_20260925` 已完成质量、细粒度、grounding 和 mask，
但旧版最终校验器把可恢复的单行解析错误当成全局失败。该 run 的 `plan.json` 固定了旧代码摘要，
应按下面的兼容续传命令启动；质量、场景、grounding 和 mask 结果会继续复用，只重新执行最终校验，
不要删除旧的 `work/` 或正式 prefilter 目录。其他代码变更仍应新建 RUN_ID。

对于本次已经完成 grounding、但在旧版汇总校验处失败的
`labeling_4node_crispedit_full_localenv_20260925`，四台机器使用最新
`crispedit-labeling` 提交重新 clone 后，可直接沿用原 RUN_ID 续传：

```bash
#!/usr/bin/env bash
set -euo pipefail

export CRISPEDIT_RUN_ID="crispedit_full_localenv_20260925"
export CRISPEDIT_BRANCH="crispedit-labeling"
export CRISPEDIT_COMMIT="af589c53e9b165091395f1dc0da3e1ffa034bba0"
export CRISPEDIT_REPO_URL="https://github.com/Tangent0308/sam3-crispedit.git"

export CRISPEDIT_RESUME=1
export CRISPEDIT_RESUME_TOKEN="retry_validate_fix_02"  # 四台相同；不能复用已使用的 token
export CRISPEDIT_ALLOW_CODE_CHANGE_ON_RESUME=1

: "${ARNOLD_ID:?Arnold must supply node rank}"
: "${ARNOLD_WORKER_NUM:?Arnold must supply worker count}"
: "${ARNOLD_WORKER_GPU:?Arnold must supply GPU count}"
[[ "$ARNOLD_WORKER_NUM" == 4 ]]
[[ "$ARNOLD_WORKER_GPU" == 8 ]]
[[ "$ARNOLD_ID" =~ ^[0-3]$ ]]
[[ "$CRISPEDIT_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]

export CRISPEDIT_RUN_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_${CRISPEDIT_RUN_ID}"
export CRISPEDIT_INPUT_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M"
export CRISPEDIT_QUALITY_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-qwen38-pair-prefilter"
export CRISPEDIT_SCENE_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-difficult-local-edit"
export CRISPEDIT_LABEL_DIR="$CRISPEDIT_RUN_DIR/labels"
export CRISPEDIT_MODEL_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
export CRISPEDIT_SAM3_CHECKPOINT_PATH="/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"
export CRISPEDIT_FILTER_BATCH_SIZE=4
export CRISPEDIT_GROUNDING_BATCH_SIZE=16
export CRISPEDIT_REPO_DIR="/opt/tiger/tanyue/workspaces/sam3-crispedit-${CRISPEDIT_RUN_ID}-node${ARNOLD_ID}"

unset CRISPEDIT_RUNTIME_ARCHIVE CRISPEDIT_BASE_PYTHON CRISPEDIT_LOCAL_RUNTIME_DIR
unset CRISPEDIT_PYTHON CRISPEDIT_SAM_PYTHON CRISPEDIT_USE_EXISTING_REPO

mkdir -p "$CRISPEDIT_RUN_DIR/logs" "$(dirname "$CRISPEDIT_REPO_DIR")"
exec > >(tee -a "$CRISPEDIT_RUN_DIR/logs/entry.node${ARNOLD_ID}.log") 2>&1

if [[ ! -e "$CRISPEDIT_REPO_DIR" ]]; then
  git clone --single-branch --branch "$CRISPEDIT_BRANCH" \
    "$CRISPEDIT_REPO_URL" "$CRISPEDIT_REPO_DIR"
else
  [[ -d "$CRISPEDIT_REPO_DIR/.git" ]]
fi

cd "$CRISPEDIT_REPO_DIR"
[[ "$(git remote get-url origin)" == "$CRISPEDIT_REPO_URL" ]]
git diff --quiet
git diff --cached --quiet
git fetch --no-tags origin "$CRISPEDIT_BRANCH"

target_commit="$CRISPEDIT_COMMIT"
[[ "$target_commit" =~ ^[0-9a-f]{40}$ ]]
git merge-base --is-ancestor "$target_commit" FETCH_HEAD
git checkout --detach "$target_commit"

bash scripts/bootstrap_crispedit_4node.sh
```

这个兼容开关只允许当前校验兼容修复继续旧 plan，仍会检查输入路径、过滤参数、源文件快照和四机配置；
旧 run 的完整质量、scene、grounding 和 mask shard 会被复用，流程会直接重新执行最终校验。
不要把该开关用于其他未审查的代码变更。若旧 `work/` 结果不完整，调度器会自动补跑缺失 shard。

## 4. 故障修复与验证

`labeling_4node_crispedit_full_20260924_a` 的四台 worker 均因 OpenCV 找不到 `libGL.so.1` 导致 vLLM 初始化失败。
先前环境混装 GUI/headless OpenCV，实际二进制启用 QT5；预检查漏查 cv2。本机有 libGL，之前的一机四 rank 验证没有暴露该问题。
该运行新增 0 行、0 parquet，未进入细粒度或 mask；失败日志保留。新实现移除 GUI 包与共享环境包方案，并在规划前运行真实引擎预检查。

本轮验证记录：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/local_env_fix_20260924/`。
`install.log` 记录全新本地安装，`preflight.log` / `preflight.json` 记录已通过的 8 卡 CUDA、spawn 和实际 Qwen 双图推理，`tests.log` 记录 198 项通过的回归测试。
`clone_install.log` 是从提交 `b045184` 克隆到干净本地目录后重新安装的记录；`install_reuse.log` 确认完整环境可重复检查复用。
安装与测试过程中基础 Python 和依赖均位于对应 clone，OpenCV 动态依赖无 libGL/Qt。
干净 clone 使用自身新装环境完成真实一机 8×H100、四 rank 全链路：20 条质量 PASS → 19 场景 PASS / 1 DROP → 19 非空 mask、35 实例，0 解析/运行错误，四 rank 均 exit 0。
成功标记：`smoke/run/control/initial/complete.ok`；逐阶段 tqdm 在 `smoke/run/logs/`，汇总在 `smoke/run/reports/`，抽样原始行号映射在 `smoke/provenance.json`。
[本轮 19 条可视化](/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/local_env_fix_20260924/review/index.html)。自动 OK 只作运行验证，不等于语义准确率。
验证期间已移除旧共享基础 Python，后续阶段仍成功；测试用 clone 完成后清理，Git 提交及日志保留。
物理四机重新提交由用户执行，不将本机四 rank 验证称为四台物理机器验收。

### 4.1 grounding 单行解析错误修复

旧版调度器在 `verify_label_shards` 中把任意 `ground_parse_ok=false` 都提升为全局异常。
`color_00003.parquet` 的一个模型响应虽然已经被 grounding runner 标为
`PARSE_ERROR/GROUND_FAIL`，仍因此触发了四个节点的 `Peer failed`。当前调度器只把带有
`runtime_error` 的推理运行时错误视为阶段失败；普通解析失败会继续合并，mask runner 会为该行
写入空 mask 和 `GROUND_FAIL`。最终校验器同步识别 `GROUND_FAIL/PARSE_ERROR`，只统计这些行，
不再因此返回全局失败；若解析错误出现在 `OK` 行，或存在运行时错误，校验仍会返回非零。

本地验证：`.venv-crispedit/bin/python -m pytest -q tests/test_crispedit_distributed.py`，9 项通过，
其中包含可恢复解析失败和运行时错误仍会阻断的回归检查。

`labeling_4node_crispedit_full_localenv_20260924_b` 首次提交没有进入环境安装或数据规划。远程
`crispedit-labeling` 当时仍为旧提交 `dd51f01`，任务 clone 到的旧 bootstrap 会再次管理仓库；它看到入口刚创建的
node-local clone 后以 `Repository exists` 退出，其他节点随即报告 `Environment setup failed`。Arnold 重试使相同文本重复写入日志。
修复后的 bootstrap 只负责当前 clone 内的安装和运行；完整入口会校验已有目录的 `.git`、origin、tracked diff 与目标 commit，
再 fetch 并以 detached commit 启动，因此同一 worker 上的命令重试不会因 clone 已存在而失败，也不会误用其他仓库或本地修改。

已删除共享启动副本 `workspaces/crispedit_labeling_20260924`（含环境包）、
`workspaces/sam3-crispedit_prefilter_4node_20260923`（含基础 Python/venv），以及本机旧的 `.cache/crispedit_runtime_20260924`。
原始数据、模型、正式筛选、历次打标与失败日志保留；环境可用 Git 中的安装脚本重建。
