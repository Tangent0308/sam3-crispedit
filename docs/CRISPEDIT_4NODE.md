# CrispEdit 四机完整打标

当前入口：四台机器分别 git clone `crispedit-labeling`，运行仓库安装脚本，再顺序执行质量筛选 → 细粒度筛选 → Qwen3.8 grounding → SAM3 mask → 校验。
代码、基础 Python、依赖、下载缓存均在每台机器本地；共享盘只用于数据、模型、结果、日志和协调标记。
实现：[安装](../scripts/setup_crispedit_env.sh)、[预检查](../scripts/preflight_crispedit_env.py)、[四机启动](../scripts/bootstrap_crispedit_4node.sh)、[调度](../crispedit/distributed.py)。方法和生产统计见[主文档](CRISPEDIT_MASK.md)。

## 1. 四台 Arnold worker 使用相同完整入口

每台 8 GPU；Arnold 提供 `ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8` 和 `ARNOLD_ID=0/1/2/3`。
先把修复提交推送至远程分支，再提交任务。每次新运行更换唯一 RUN_ID，四台保持一致；本次失败的 `crispedit_full_20260924_a` 不复用。
需要 Git、Python3/pip、GNU tar、flock、可访问 GitHub/PyPI/PyTorch 下载站点的网络，以及每节点至少约 40GB 可用本地空间。

```bash
#!/usr/bin/env bash
set -euo pipefail
export CRISPEDIT_RUN_ID="crispedit_full_localenv_20260924_b"
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
[[ ! -e $CRISPEDIT_REPO_DIR ]] || { echo 'Local clone exists; use a new run ID or the resume instructions'; exit 2; }
git clone --single-branch --branch "$CRISPEDIT_BRANCH" "$CRISPEDIT_REPO_URL" "$CRISPEDIT_REPO_DIR"
cd "$CRISPEDIT_REPO_DIR"
# Optional: set CRISPEDIT_COMMIT to a fixed commit for an exact rerun/resume.
if [[ -n ${CRISPEDIT_COMMIT:-} ]]; then
  git checkout --detach "$CRISPEDIT_COMMIT"
fi
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
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_crispedit_full_localenv_20260924_b/logs/`。

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
RUN_DIR=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/CrispEdit/labeling_4node_crispedit_full_localenv_20260924_b
tail -f "$RUN_DIR/logs/bootstrap.node0.log"
tail -f "$RUN_DIR/logs/quality.node0.log"
```

`complete.ok` 代表数据结构/流程校验完成；mask 语义质量需另行抽查。`Peer failed` 时查看失败标记内的上游原因和对应阶段日志。
任务由 Arnold 托管，不依赖登录终端。手动四机启动时，每台分别在 tmux 内执行完整入口。

恢复仅用于**同一份代码/参数/源数据**的失败任务：确认旧进程退出，保留 RUN_ID、RUN_DIR 和其他配置，四台共同增加：

```bash
export CRISPEDIT_RESUME=1
export CRISPEDIT_RESUME_TOKEN="retry_01"  # 每次换新名称，四台一致
bash scripts/bootstrap_crispedit_4node.sh
```

若调度到新机器，先按完整入口重新 clone（固定原 commit），再传上述两个变量启动。成功标记在 `control/retry_01/complete.ok`。
代码或安装配置修改后必须用新 RUN_ID；勿删除失败标记或修改计划摘要来绕过恢复校验。旧 prefilter 仍通过其独立结果目录复用。

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

已删除共享启动副本 `workspaces/crispedit_labeling_20260924`（含环境包）、
`workspaces/sam3-crispedit_prefilter_4node_20260923`（含基础 Python/venv），以及本机旧的 `.cache/crispedit_runtime_20260924`。
原始数据、模型、正式筛选、历次打标与失败日志保留；环境可用 Git 中的安装脚本重建。
