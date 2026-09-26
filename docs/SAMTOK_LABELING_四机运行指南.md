# SAMTok 派生编辑数据：四机 32 卡运行指南

本入口只扩展并行调度，不重新设计打标方法。对应远端为
`https://github.com/Tangent0308/sam3-crispedit.git`，分支为
`samtok-derived-edit-labeling`，不是 MIRAGE 的 origin/main，也不是 SAMTokEdit 的 dev。

## 1. 本次部署的流程与范围

冻结最近一轮验证的 **remove 流程**：

```text
GRES/VER 正例与每个原始 region
  → Qwen3.8-27B / vLLM，relations-v16 thinking 规划
  → 邻居保留、附件连带区域解析（无 SAM 源 mask 审核）
  → Qwen-Image-2.1 / vllm-omni，40步、seed=0
  → guard-any-v1 + visible-v1 + adaptive-remove-v5 融合
  → Qwen3.8-27B / vLLM，completion-v5 / adaptive 双图 / pixel-veto
  → 汇总全部结果与 model_pass 候选，不将模型通过冒称人工验收
```

规划的 prompt、采样设置、既有修复调用、SAM 安全约束、编辑输入、KV cache、融合和审核规则不变。
没有新增强制改写调用：最近冻结流程也是生成加审核，历史的可选改写实验不因四机部署自动启用。
其他三类编辑没有被转换为 remove；输入含其他类型会明确报错，不能把当前入口称为四类均衡生产。

本次新增的非 remove 入口是独立实验，不会覆盖上述结果：它对每个正例 source 的每个
原始 mask 分别派生 `add`、`replace`、`attribute` 三条 case（同一 source 图像复用三次），
并写入独立的 `add_replace_attribute` run/data 目录。规划改用通用的
`plan_dataset_regions.py`，编辑仍是同一套 Qwen-Image-2.1 + vLLM-Omni + 40 steps；
remove 的 relation planner、结果和续跑 checkpoint 不会被读取或修改。

四台机器独立处理数据，各用本机 8 卡。**不使用训练的 DDP/Accelerate，也不跨机切分一个模型**。
参考训练指南的 Arnold 节点编号、统一 run ID、日志和跨节点完成门禁；不复用训练 rendezvous。
`ARNOLD_WORKER_HOSTS`、`MASTER_PORT`、worker-local `PORT` 均不参与推理协调。

按 source 分组做确定性负载分配；同一张图的全部 mask 留在同一节点，保留原本的邻居保护上下文。
image ID 不重编号，每个 region 只执行一次。4个本地8卡池即32个独立模型副本并行，SAM解析阶段各机使用本机第一卡。
分片改变批次组合，MLLM 有随机采样；“方法不变”不代表跨不同拓扑逐像素、逐字符复现。

## 2. 最短正式入口：四台 worker 都执行

### 2.1 Arnold 任务配置与平台变量

仿照训练指南，在Arnold提交一个**4 workers、每worker 8张GPU**的任务，四台使用相同镜像和入口。
当前实测硬件为H100 80GB；不能把单机8卡模拟测试理解为已验证其他显存规格。
每台worker只启动一次下面的Bash入口，由pipeline创建本机GPU进程；不要额外配置成每GPU再启动一次入口。

| 配置项 | 应填写或确认的内容 |
|---|---|
| worker数量 | 4 |
| 每worker GPU数量 | 8，总计32卡 |
| worker入口 | 第3节完整Bash代码；或第2.2节的共享bootstrap调用 |
| 共享挂载 | 四机均挂载`/mnt/bn/strategy-mllm-train`；源数据/模型可读，任务experiments目录可读写 |
| 节点本地存储 | `/opt/tiger/tanyue`可写，每机至少预留150GB；repo/环境/编辑模型缓存存放这里 |
| 镜像工具与网络 | git、Bash、flock、python3/pip、可用NVIDIA驱动；每机可访问GitHub及依赖安装源 |
| 用户必填 | 新任务使用全新`SAMTOK_RUN_ID`；续跑保留run ID并填写新的`SAMTOK_ATTEMPT_ID`；无需W&B key |

Arnold平台注入的变量与本任务的使用方式：

| 变量 | 预期值/含义 | 当前打标入口如何使用 |
|---|---|---|
| `ARNOLD_WORKER_NUM` | `4` | 必填，启动前检查 |
| `ARNOLD_WORKER_GPU` | `8` | 必填，安装后还检查实际可见GPU |
| `ARNOLD_ID` | 当前机器编号`0/1/2/3` | 必填，决定分片、日志和node0协调角色 |
| `ARNOLD_WORKER_HOSTS` | 平台分配的worker地址列表 | 不解析，不作为启动前提；推理不需要rendezvous |
| `ARNOLD_WORKER_0_HOST` | 平台可能提供的node0地址 | 不使用 |
| `PORT` / `MASTER_PORT` | worker服务端口/训练遗留变量 | 不用于本流程，不需要手工统一 |

不要自行覆盖Arnold节点变量。统一run ID和共享文件系统负责节点间协调，不需要照搬训练的
`NNODES/NODE_RANK/WORLD_SIZE`或NCCL配置。也无需tmux/nohup：入口在Arnold任务前台运行，
平台负责进程生命周期，日志同时输出到任务控制台和experiments。

### 2.2 已有共享bootstrap时的最短入口

先确保分支已包含本指南和 `scripts/labeling/`，再将
`scripts/labeling/bootstrap_arnold_4node.sh` 复制到四机可访问的共享路径，或将其完整内容粘贴为 Arnold entry。
本次维护的共享副本路径为：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/launchers/bootstrap_arnold_4node.sh
```

Arnold 配置 **4 workers × 8 GPUs**，在四个 worker 共同入口填写同一个新名称：

```bash
#!/usr/bin/env bash
set -euo pipefail
export SAMTOK_RUN_ID="samtok-remove-4n-20260924-001"  # 每次提交换新名字
export SAMTOK_LIMIT_SOURCES=0                      # 0=全部正例，不是0条
bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/launchers/bootstrap_arnold_4node.sh
```

不需要 W&B key。不要仅在 worker0 执行，也不要用本机现存 repo 代替 clone。
默认会由 node0 从源 parquet 准备全部正例、每个原始 region 使用一次；不会为凑100k反复复制输入。
12,337条源记录中正例7,671条，排除4,666条No target；实际任务数等于正例所有mask之和。
历史试验是否重复不在这个全量入口额外筛选，避免擅自改变全量数据范围。

只跑小规模四机 smoke 时，设 `SAMTOK_LIMIT_SOURCES=8` 并换新 run ID。它限制源图数，不是 region 数。
若已经准备好了输入，在同一入口额外指定（仅适用于当前正式 run 的物化源目录或新建的
独立输入目录；历史 pilot 目录已经清理，不再使用）：

```bash
export SAMTOK_DATA_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID/data/source"
```

这里必须已有 `annotations.jsonl` 和 `sources/`。正式 run 默认使用
`$SAMTOK_RUN_ROOT/data/source/`，不需要额外指定。
不设置此变量时，默认数据准备位置为本次 `$SAMTOK_RUN_ROOT/data/source/`。

### 2.3 add/replace/attribute 三类型独立入口

三类型实验使用同样的 Arnold 配置（4 workers × 8 GPUs），但必须使用新的 run ID，并调用
仓库中的 `scripts/labeling/bootstrap_arnold_4node_multitype.sh`。它会把数据放到：

```text
$SAMTOK_RUN_ROOT/data/add_replace_attribute/
```

四个 worker 使用同一段入口：

```bash
#!/usr/bin/env bash
set -euo pipefail
export SAMTOK_RUN_ID="samtok-add-replace-attribute-4n-20260926"
export SAMTOK_LIMIT_SOURCES=0
bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/launchers/bootstrap_arnold_4node_multitype.sh
```

如果共享 `launchers/` 尚未同步，使用下面的完整 Arnold 入口。四个 worker 粘贴同一段；
每台机器先把当前分支 clone 到本机 bootstrap 目录，再由 bootstrap 为本机 clone 实际运行
repo、安装环境并启动 pipeline。日志统一写到共享的 experiments run root：

```bash
#!/usr/bin/env bash
set -euo pipefail

export SAMTOK_RUN_ID="samtok-add-replace-attribute-4n-20260926"
export SAMTOK_LIMIT_SOURCES=0
export SAMTOK_REPO_URL="https://github.com/Tangent0308/sam3-crispedit.git"
export SAMTOK_BRANCH="samtok-derived-edit-labeling"
export SAMTOK_PIPELINE_MODE=multitype
export SAMTOK_RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID"
export SAMTOK_DATA_ROOT="$SAMTOK_RUN_ROOT/data/add_replace_attribute"
export SAMTOK_BOOTSTRAP_DIR="/opt/tiger/tanyue/labeling_bootstrap/$SAMTOK_RUN_ID"

if [[ ! -d "$SAMTOK_BOOTSTRAP_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SAMTOK_BOOTSTRAP_DIR")"
  git clone --branch "$SAMTOK_BRANCH" --single-branch "$SAMTOK_REPO_URL" "$SAMTOK_BOOTSTRAP_DIR"
fi
git -C "$SAMTOK_BOOTSTRAP_DIR" fetch origin "$SAMTOK_BRANCH"
git -C "$SAMTOK_BOOTSTRAP_DIR" checkout --detach "origin/$SAMTOK_BRANCH"
exec bash "$SAMTOK_BOOTSTRAP_DIR/scripts/labeling/bootstrap_arnold_4node.sh"
```

正式全量运行时保持 `SAMTOK_LIMIT_SOURCES=0`。建议先用新的 run ID 设置为 `2` 做四机
联调；确认四个 node 的环境、规划和编辑日志都正常后，再以新的 run ID 运行全量。三类型
正式任务不能复用 remove 的 run ID，也不能把 `SAMTOK_DATA_ROOT` 指向 `data/source`。

如果任务中断，使用相同的 `SAMTOK_RUN_ID`，并在四个 worker 同时设置：

```bash
export SAMTOK_RESUME=1
export SAMTOK_ATTEMPT_ID="resume-001"   # 每次重启必须使用新的值
```

续跑会检查原始三类型 manifest、分片和 profile；已完成的 planning、editing、audit case
会从 checkpoint 继续，错误或不完整的 case 会重新执行。运行日志位置为：

```text
$SAMTOK_RUN_ROOT/logs/
$SAMTOK_RUN_ROOT/attempts/<attempt-id>/logs/
```

该入口的阶段顺序是：正例索引与三类型展开 → 按 source 分片 →
Qwen3.8-27B 通用 mask-grounding/scope 规划 → 8 卡 Qwen-Image-2.1 编辑 →
可选的通用二图审核 → 汇总 `results/`。三类原始输入数完全相同；规划拒绝、出图失败和审核
失败都保留在各阶段 JSONL，不会用其他类型补齐。正式生成保持 40 steps/seed 0；仅用于
冒烟时可设置 `SAMTOK_LIMIT_SOURCES=2`，但仍会为每个区域生成三种类型。

非 remove 的实验结果建议按类型单独保留（后续再合并）：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/<run-id>/
  data/add_replace_attribute/                 # 三类型输入 manifest
  nodes/node<N>/planning/                     # ground/scope/regions
  nodes/node<N>/generation/context_grounded_v4_qwen21/edited/
  nodes/node<N>/audit/
  results/all_cases.jsonl, generated.jsonl, audit.jsonl, model_pass.jsonl
  results/by_type/{add,replace,attribute}/{all_cases,generated,audit,model_pass}.jsonl
```

当前已保留的 remove 结果与新三类型实验明确分开：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/final/
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/intermediate/
```

根目录下的 `final`、`intermediate` 目前只是兼容旧路径的符号链接；后续合并四类时另建
`merged/`，不直接覆盖 `remove/`。

#### 三类型 smoke 验证记录

使用 pilot 中的真实 GRES source/mask，并补齐与正式 `regions` 相同的
`region_contract`，在 Qwen-Image-2.1 **vLLM-Omni** 后端分别跑了三类（4 steps 仅用于快速
验证类型分支，不代表正式质量配置；正式入口仍为 40 steps/seed 0）：

| 类型 | case | 结果 | 单 case 秒数 |
|---|---|---|---:|
| add | `002_gres_r94_m0_two_sheep_add_blue_collar.png` | PNG 正常生成，蓝色项圈位于指定羊的颈部 | 5.18 |
| replace | `005_gres_r376_m1_three_zebras_replace_rightmost_zebra.png` | PNG 正常生成，右侧 zebra 被替换为 antelope | 4.30 |
| attribute | `006_gres_r382_m0_two_chairs_red_left_chair.png` | PNG 正常生成，目标椅子变为红色且人物保留 | 4.20 |

输出保存在本机 smoke 目录 `/tmp/samtok_multitype_smoke.yOiM8r/`，正式运行不会读取该
临时目录。三类均通过了 PNG 解码和人工可视检查；4 steps 的 replace 结果边缘仍可能偏软，
因此不能把 smoke 图当作最终质量结论，正式审核仍由 audit 阶段决定。

## 3. 可直接提交的 Arnold 完整入口

下面整段可直接粘贴到 **四个 Arnold worker 的共同 Bash 入口**。先填写唯一的
`SAMTOK_RUN_ID`；续跑保留原run ID并设置新的attempt ID。不要四台分别生成时间戳，不要覆盖Arnold分配的`ARNOLD_ID`。
正式全量用`SAMTOK_LIMIT_SOURCES=0`，首次真实四机联调建议先改为8。

它不依赖已存在的本地仓库，也不需要先下载或复制bootstrap：
**Arnold拓扑检查 → 共享日志/失败标记 → 每机本地clone → 安装uv和三套环境 →
本地权重缓存 → node0准备源数据 → 四节点一致性检查及分片 → 生成、审核和汇总**。
以下主体与仓库的`scripts/labeling/bootstrap_arnold_4node.sh`一致，额外提供用户填写区。

```bash
#!/usr/bin/env bash

# 四个 worker 填写相同配置。新任务默认 SAMTOK_RESUME=0。
export SAMTOK_RUN_ID="samtok-derived-4n-20260925"
export SAMTOK_LIMIT_SOURCES=0
export SAMTOK_REPO_URL="https://github.com/Tangent0308/sam3-crispedit.git"
export SAMTOK_BRANCH="samtok-derived-edit-labeling"
# 恢复已停止的同名任务时，取消下面两行注释；每次重启换新的 attempt ID。
# export SAMTOK_RESUME=1
# export SAMTOK_ATTEMPT_ID="resume-001"
# 可选：四机填写相同完整SHA，固定本次运行代码。
# export SAMTOK_EXPECTED_COMMIT="实际的40位commit"

# Copy this complete file to a shared pre-clone path or paste it into Arnold entry.
# Run the SAME entry on all four workers. Each worker clones to node-local storage.
set -euo pipefail
: "${SAMTOK_RUN_ID:?Set one run ID, identical on all four workers}"
: "${ARNOLD_WORKER_NUM:?Arnold topology missing}"
: "${ARNOLD_WORKER_GPU:?Arnold topology missing}"
: "${ARNOLD_ID:?Arnold topology missing}"
[[ "$SAMTOK_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'Requires 4 nodes x 8 GPUs' >&2; exit 2; }
export SAMTOK_RUN_ROOT="${SAMTOK_RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID}"
export SAMTOK_RESUME="${SAMTOK_RESUME:-0}"
[[ "$SAMTOK_RESUME" == 0 || "$SAMTOK_RESUME" == 1 ]] || { echo 'SAMTOK_RESUME must be 0 or 1' >&2; exit 2; }
resume_args=()
attempt_suffix=""
export SAMTOK_CONTROL_ROOT="$SAMTOK_RUN_ROOT"
if [[ "$SAMTOK_RESUME" == 1 ]]; then
  : "${SAMTOK_ATTEMPT_ID:?Resume needs a NEW attempt ID, identical on all four workers}"
  [[ "$SAMTOK_ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid attempt ID' >&2; exit 2; }
  [[ -f "$SAMTOK_RUN_ROOT/reports/partition.json" ]] || { echo 'No prepared run to resume' >&2; exit 2; }
  export SAMTOK_CONTROL_ROOT="$SAMTOK_RUN_ROOT/attempts/$SAMTOK_ATTEMPT_ID"
  attempt_suffix="/attempts/$SAMTOK_ATTEMPT_ID"
  resume_args=(--resume)
elif [[ -n "${SAMTOK_ATTEMPT_ID:-}" ]]; then
  echo 'SAMTOK_ATTEMPT_ID is only valid with SAMTOK_RESUME=1' >&2; exit 2
fi
export SAMTOK_DATA_ROOT="${SAMTOK_DATA_ROOT:-$SAMTOK_RUN_ROOT/data/source}"
export SAMTOK_PARQUET="${SAMTOK_PARQUET:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet}"
export SAMTOK_REPO_DIR="${SAMTOK_REPO_DIR:-/opt/tiger/tanyue/labeling_runs/$SAMTOK_RUN_ID$attempt_suffix/node$ARNOLD_ID/repo}"
export SAMTOK_REPO_URL="${SAMTOK_REPO_URL:-https://github.com/Tangent0308/sam3-crispedit.git}"
export SAMTOK_BRANCH="${SAMTOK_BRANCH:-samtok-derived-edit-labeling}"
mkdir -p "$SAMTOK_CONTROL_ROOT/logs" "$SAMTOK_CONTROL_ROOT/control" "$SAMTOK_RUN_ROOT/control"
# Held by the shell and inherited by exec; released automatically after job exit.
exec 9>>"$SAMTOK_RUN_ROOT/control/bootstrap.node$ARNOLD_ID.lock"
flock -n 9 || { echo 'Another attempt is active on this rank' >&2; exit 1; }
# Each restart gets new barriers and claims; old markers remain as evidence.
(set -o noclobber; printf '%s\n' "$(hostname) $$" > "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.claim") || exit 1
exec > >(tee -a "$SAMTOK_CONTROL_ROOT/logs/bootstrap.node$ARNOLD_ID.log") 2>&1
on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf '{"error":"bootstrap node %s exited %s; see bootstrap log"}\n' "$ARNOLD_ID" "$rc" \
      > "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json.tmp"
    mv "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json.tmp" "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json"
  fi
}
trap on_exit EXIT
check_peers() {
  if compgen -G "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/*.failed.json" > /dev/null; then
    echo 'Peer bootstrap failed; see experiments logs' >&2; exit 1
  fi
}
export PYTHONUNBUFFERED=1
# No credentials are printed; retain standard git credential helpers.
if [[ "${SAMTOK_KEEP_PROXY:-0}" != 1 ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
fi
[[ ! -e "$SAMTOK_REPO_DIR" ]] || { echo "Clone path already exists: $SAMTOK_REPO_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$SAMTOK_REPO_DIR")"
git clone --branch "$SAMTOK_BRANCH" --single-branch "$SAMTOK_REPO_URL" "$SAMTOK_REPO_DIR"
cd "$SAMTOK_REPO_DIR"
mkdir -p "$SAMTOK_CONTROL_ROOT/reports"
git rev-parse HEAD > "$SAMTOK_CONTROL_ROOT/reports/checkout.node$ARNOLD_ID.txt"
if [[ -n "${SAMTOK_EXPECTED_COMMIT:-}" ]]; then
  [[ "$(git rev-parse HEAD)" == "$SAMTOK_EXPECTED_COMMIT" ]] || { echo 'Unexpected branch revision' >&2; exit 1; }
fi
python3 -m pip install --user --index-url "${SAMTOK_PACKAGE_INDEX:-https://bytedpypi.byted.org/simple/}" 'uv==0.11.32'
export UV_BIN="$(python3 -c 'import site; print(site.getuserbase())')/bin/uv"
export SAMTOK_RUNTIME_ROOT="$SAMTOK_REPO_DIR/.runtime"
export SAMTOK_MLLM_PYTHON="$SAMTOK_RUNTIME_ROOT/mllm/bin/python"
export SAMTOK_EDITOR_PYTHON="$SAMTOK_RUNTIME_ROOT/editor/bin/python"
export SAMTOK_SAM_PYTHON="$SAMTOK_RUNTIME_ROOT/sam/bin/python"
export SAMTOK_SAM3_SOURCE="$SAMTOK_RUNTIME_ROOT/sam3-source"
export SAMTOK_QWEN38_MODEL="${SAMTOK_QWEN38_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}"
export SAMTOK_QWEN21_MODEL="${SAMTOK_QWEN21_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1}"
export SAMTOK_SAM3_CHECKPOINT="${SAMTOK_SAM3_CHECKPOINT:-/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt}"
for file in "$SAMTOK_QWEN38_MODEL/config.json" "$SAMTOK_QWEN21_MODEL/model_index.json" "$SAMTOK_SAM3_CHECKPOINT"; do
  [[ -r "$file" ]] || { echo "Missing shared input: $file" >&2; exit 1; }
done
bash scripts/labeling/setup_env.sh
export SAMTOK_ENV_REPORT="$SAMTOK_RUNTIME_ROOT/environment.json"
# Do not stage tens of GB or prepare data while another worker failed its imports.
check_peers
printf '{"ready":true}\n' > "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$ARNOLD_ID.ok.json.tmp"
mv "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$ARNOLD_ID.ok.json.tmp" "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$ARNOLD_ID.ok.json"
start_wait=$SECONDS
while true; do
  check_peers
  all_ready=1
  for peer_rank in 0 1 2 3; do
    [[ -f "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$peer_rank.ok.json" ]] || all_ready=0
  done
  (( all_ready == 1 )) && break
  (( SECONDS - start_wait < 10800 )) || { echo 'Peer environment timeout' >&2; exit 1; }
  sleep 2
done
export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "${SAMTOK_STAGE_EDITOR_MODEL:-1}" == 1 ]]; then
  editor_cache="${SAMTOK_MODEL_CACHE_ROOT:-/opt/tiger/tanyue/labeling_model_cache/$SAMTOK_RUN_ID}/qwen21"
  "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.stage_labeling_model \
    --source "$SAMTOK_QWEN21_MODEL" --destination "$editor_cache" --run-root "$SAMTOK_CONTROL_ROOT" "${resume_args[@]}"
  export SAMTOK_QWEN21_MODEL="$editor_cache"
fi
# Node 0 materializes once if no prepared manifest was supplied. All original
# regions are kept; 0 means all positives, not an invented 100k duplication.
check_peers
if [[ "$ARNOLD_ID" == 0 ]]; then
  if [[ ! -f "$SAMTOK_DATA_ROOT/annotations.jsonl" ]]; then
    "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.prepare_removal_inputs \
      --parquet "$SAMTOK_PARQUET" --out-root "$SAMTOK_DATA_ROOT" \
      --limit-sources "${SAMTOK_LIMIT_SOURCES:-0}" --run-root "$SAMTOK_CONTROL_ROOT" "${resume_args[@]}"
  fi
  check_peers
  printf '{"ready":true}\n' > "$SAMTOK_CONTROL_ROOT/control/data.ok.json.tmp"
  mv "$SAMTOK_CONTROL_ROOT/control/data.ok.json.tmp" "$SAMTOK_CONTROL_ROOT/control/data.ok.json"
fi
start_wait=$SECONDS
until [[ -f "$SAMTOK_CONTROL_ROOT/control/data.ok.json" ]]; do
  check_peers
  (( SECONDS - start_wait < 10800 )) || { echo 'Data preparation timeout' >&2; exit 1; }
  sleep 2
done
check_peers
# No MASTER_PORT / ARNOLD_WORKER_HOSTS rendezvous: independent data shards.
if [[ "$SAMTOK_RESUME" == 1 ]]; then resume_args+=(--attempt-id "$SAMTOK_ATTEMPT_ID"); fi
exec "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.run_multinode_labeling \
  --data-root "$SAMTOK_DATA_ROOT" --run-root "$SAMTOK_RUN_ROOT" --run-id "$SAMTOK_RUN_ID" \
  --rank "$ARNOLD_ID" --gpus 0,1,2,3,4,5,6,7 "${resume_args[@]}"
```

注意：

- `ARNOLD_WORKER_NUM`、`ARNOLD_WORKER_GPU`、`ARNOLD_ID`由平台提供，上面只检查和读取；
  不要通过手工设置4/8伪装资源已分配，也不要让所有worker使用同一个rank。
- 这不是训练入口，不设置`MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`，也不启动
  `torchrun`或`accelerate launch`。`ARNOLD_WORKER_HOSTS`即使存在，也不用于共享文件系统协调。
- 各机本地clone和环境分别安装；只有源图准备由node0执行一次。与训练指南的
  “node0安装共享环境、其他节点等待”不同，不要把本地`SAMTOK_REPO_DIR`改成四机共同写的同一个目录。
- 不需要`source .venv/bin/activate`：pipeline显式使用SAM、MLLM、editor各自的Python，
  激活一套环境不能替代另外两套。
- 日志从建立run目录后开始覆盖clone、安装、缓存、数据准备及调度；最前面的必填值/拓扑错误
  只显示在Arnold任务控制台，因为这时尚未建立安全的run日志路径。
- 镜像缺少git/python3/pip或GPU驱动时，应先选择/配置正确镜像；无需照搬训练指南的
  `sudo apt-get install ffmpeg libsm6 libxext6 tmux htop`。本流程使用headless图像依赖。
- 默认清除代理直连GitHub；环境必须走代理时，在用户填写区设置`SAMTOK_KEEP_PROXY=1`。
  不要把代理凭据、访问令牌写进文档。四机都需要访问代码仓库、依赖仓库和安装源。
- 新任务使用新run ID；恢复已停止任务时按3.2节保留run ID、设置新的attempt ID。旧claim和失败日志保留，新一轮使用独立协调目录。

### 3.1 提交前确认远端已包含所需文件

在当前开发仓库运行以下不修改工作树的检查（远端名为`sam3`；新clone的默认远端名则为`origin`）：

```bash
git fetch sam3 samtok-derived-edit-labeling
git ls-tree -r --name-only sam3/samtok-derived-edit-labeling -- \
  scripts/labeling requirements synthesis_pipeline/run_multinode_labeling.py \
  synthesis_pipeline/prepare_removal_inputs.py synthesis_pipeline/check_labeling_environment.py \
  synthesis_pipeline/stage_labeling_model.py
```

应能看到两个labeling shell脚本、三份锁定依赖文件和上面的四个Python入口。
本地有文件不等于远端有文件；四机实际执行的是clone下来的分支内容。
若用`SAMTOK_EXPECTED_COMMIT`固定版本，四台必须填写相同的完整SHA。


### 3.2 从当前进度续跑（包括旧版产生的规划结果）

先在Arnold停止旧的整个四机任务，确认四个worker及其子进程已退出，再提交一个新的4 workers × 8 GPUs任务。
四个worker使用同一入口、同一run ID和同一新的attempt ID：

```bash
#!/usr/bin/env bash
set -euo pipefail
export SAMTOK_RUN_ID="samtok-derived-4n-20260925"
export SAMTOK_RESUME=1
export SAMTOK_ATTEMPT_ID="resume-001"  # 下次再重启改成resume-002，不重复使用
export SAMTOK_LIMIT_SOURCES=0
bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/launchers/bootstrap_arnold_4node.sh
```

这仍会在每台机器本地clone对应分支并安装环境。clone目录增加
`/attempts/$SAMTOK_ATTEMPT_ID/`，避免碰到上次留下的repo/venv。
已准备好的源数据不重新生成，节点分片保持原ID、原归属；输入manifest、节点数或冻结方法不一致时明确拒绝续跑。
同节点的并发attempt通过文件锁互斥。旧版任务没有新锁，所以第一次升级前必须先停止旧任务。

| 阶段 | 可复用的进度 | 会重新执行的部分 |
|---|---|---|
| 规划 | 旧版完整JSONL中与输入一致、原始回复可解析且重新通过原校验器的accept/defer；不加载模型处理已完成项 | 未落盘、截断、解析失败、字段错误、target point错误等记录 |
| 输入空mask | 保留原ID并记录`invalid_input_empty_mask`，不调用模型、不出图，汇总保留no_output原因 | 不伪造mask，也不把空mask送入crop函数 |
| 邻居/附件区域解析 | 本版本逐case完成记录及依赖一致；成功解析或明确defer均可复用 | 缺失完成记录、规划改变、源图或输出support损坏 |
| 编辑 | 本版本完成记录、输入/40步/seed/策略一致，最终PNG校验值匹配 | 只有文件但无完成记录、半写入PNG、内容损坏、上游输入改变 |
| 审核 | 已完整解析的pass或fail，原图/编辑图/指令及审核设置一致 | 回复无法解析、缺失完成记录、图片或指令变化 |
| 汇总 | 从本轮经过检查的各阶段结果重新汇总 | 不将旧done或finalize标记视作本轮完成 |

“正确进度”指完整且通过一致性检查的阶段结果，不代表所有模型判定都人工正确。
有效的质量fail或语义defer会保留，避免续跑变成反复抽样直到pass；技术失败则重算覆盖当前manifest中的错误记录。
本轮升级可以直接读取已有规划JSONL；旧版未产生逐case校验记录的后续阶段采用保守重算。
当前run尚在规划阶段，因此这一限制不会丢弃已经生成的成图。

新日志与旧日志分开：

```text
$SAMTOK_RUN_ROOT/attempts/resume-001/
├── logs/bootstrap.node<N>.log
├── logs/pipeline.node<N>.log
├── logs/audit.node<N>.log
├── reports/topology.node<N>.json
├── reports/progress.node<N>.json
└── control/  # 本次attempt的claim、failed、done、finalize
```

数据和各阶段结果继续写入原来的`nodes/node<N>/`；内部worker日志追加保留历史，
`checkpoints/`只记录逐case完成和依赖校验。重新运行的记录在JSONL中替换，不重复计数。
新的规划日志会打印`total / reused / empty_masks / pending`和`planned 已处理/总数`；
编辑、审核也打印复用与待执行数量。不要按追加日志的行数累计完成量，应读取当前JSONL或summary。

```bash
run_root="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/samtok-derived-4n-20260925"
tail -f "$run_root/attempts/resume-001/logs/bootstrap.node0.log"
# 实际规划worker的模型输出和进度：
tail -f "$run_root/nodes/node0/pipeline/logs/plan_0.log"
```

本轮故障定位：`000408_gres_r480_m0_remove.png`的原始RLE面积为0；
10,633个region中仅此1条为空，和4,666条No target是不同的检查。
之前在构造crop时抛出异常，且调度器顺序等待其他worker，导致错误很晚才向上传播。
现在空mask在模型加载前记录并跳过；任一worker异常会及时上报，其他已完成case仍可用于下一轮续跑。

## 4. clone、安装和真实路径

每台机器执行以下操作，bootstrap 已自动串好，无需再手工重复：

```bash
git clone --branch samtok-derived-edit-labeling --single-branch \
  https://github.com/Tangent0308/sam3-crispedit.git \
  "/opt/tiger/tanyue/labeling_runs/$SAMTOK_RUN_ID/node$ARNOLD_ID/repo"
cd "/opt/tiger/tanyue/labeling_runs/$SAMTOK_RUN_ID/node$ARNOLD_ID/repo"
python3 -m pip install --user --index-url https://bytedpypi.byted.org/simple/ 'uv==0.11.32'
export UV_BIN="$(python3 -c 'import site; print(site.getuserbase())')/bin/uv"
bash scripts/labeling/setup_env.sh
```

本地仓库默认按run/node隔离，不共享可变venv，不在四机同时写同一repo。
与训练指南“node0安装共享单环境”不同，这里有三个依赖不兼容的已验证运行环境：

| 环境 | 核心版本 | 用途 |
|---|---|---|
| `.runtime/sam` | Python3.12.13、torch2.8.0/CUDA12.8、transformers4.57.6 | 源数据准备、SAM区域解析、CPU调度 |
| `.runtime/mllm` | torch2.13.0+cu129、vLLM0.28.0+cu129、transformers5.17.0 | 27B规划和审核 |
| `.runtime/editor` | torch2.13.0+cu129、vLLM0.29.0、diffusers0.40.0、transformers5.14.1 | Qwen-Image-2.1编辑 |

SAM源代码固定 `fff5ca124cf2551dd73c0de2af9c64bdadeea0b3`；Omni固定
`44ea27c8094095bbffd88fa3befdfaa55ba4bc50`，与当前跑通的官方Qwen2.1实现一致，不拉latest改变方法。
本仓库的regional扩展仍通过官方Omni接口加载。H100上仍使用当前已验证的 `TORCH_SDPA`，不换注意力实现。

`requirements/labeling-*.lock.txt` 保存已验证环境的完整版本快照，安装时使用 `--no-deps` 防止解析器
改写CUDA组合；不运行根目录历史 `pip install -r requirements.txt`。两个指定源均可信且所有包版本固定，
安装用 `unsafe-best-match` 解决PyTorch索引遮蔽普通包问题。SAM的无CUDA后缀版本使用严格 `===`，避免
`==2.8.0` 误选另一CUDA构建。环境会占用较多本地磁盘，预留至少100GB用于环境、依赖源码和缓存。
安装结束逐环境检查版本、CUDA、SAM/Omni API import，写 `.runtime/environment.json`；失败不进入生成。

OpenCV只允许安装`opencv-python-headless`（SAM 4.11.0.86；MLLM/editor 5.0.0.93）。
不要同时安装`opencv-python`或contrib变体：这些wheel都覆盖同一个`cv2`目录，混装会使实际加载的
二进制不确定。预检同时检查包唯一性与`cv2.getBuildInformation()`中的`GUI: NONE`，记录到environment报告。
官方Omni依赖元数据包含GUI包，但本流程使用其图像推理API，按固定lock和`--no-deps`安装，
由同版本headless提供`cv2`；不要随后运行普通`pip install -e .runtime/omni-source`重新引入GUI包。

四机各自通过环境检查后写`control/environment.node<N>.ok.json`，必须四份齐全且无失败marker，
才能进入权重复制。复制与读回校验、源数据分批准备也检查peer失败，失败的临时文件保留供排查，
不会发布成功缓存清单或data.ok。共享存储可见性、单次阻塞I/O仍会影响失败传播延迟。

随后默认将Qwen-Image-2.1逐文件复制、SHA256校验到每台机器本地的
`/opt/tiger/tanyue/labeling_model_cache/$SAMTOK_RUN_ID/qwen21`，再启动8个编辑进程。
这是字节相同的权重，不改dtype、模型内容或采样方法。源和目标的校验记录存于
`staging_manifest.json`，四机还会比对该清单。首次共享盘复制仍有成本，但只读一次，
避免8个进程在FUSE共享文件上同时mmap权重触发启动超时。
建议本地预留150GB（环境、下载缓存和31GB编辑模型副本）；可用`SAMTOK_MODEL_CACHE_ROOT`覆盖缓存父目录，
四机应使用相同的节点本地路径。已有高性能存储且明确无需复制时才设`SAMTOK_STAGE_EDITOR_MODEL=0`。

部署依据：[vLLM GPU安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)；
[本轮固定的Omni源码版本](https://github.com/vllm-project/vllm-omni/tree/44ea27c8094095bbffd88fa3befdfaa55ba4bc50)。
为避免ABI变化，优先复现本轮实测环境，不把文档的滚动latest版本当固定依赖。

四台机器都必须可读：

```text
源 parquet:
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet
27B:
/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B
编辑模型（共享路径，不再依赖旧机 /tmp）:
/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1
SAM:
/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt
```

训练用的2511不是这里的编辑模型。模型不联网补下载；不能读共享数据/权重时必须先修挂载。
Arnold镜像需已有git、curl、flock、可运行的python3/pip、NVIDIA驱动；无需sudo apt大范围修改宿主。

## 5. 日志、输出与进度

默认正式运行根目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID
├── logs/bootstrap.node0.log ... node3.log    # clone、安装、准备与调度输出
├── logs/pipeline.node0.log ... node3.log     # 各节点规划/解析/编辑总入口
├── logs/audit.node0.log ... node3.log        # 审核总入口
├── reports/topology.json                    # 四节点一致性验收
├── reports/partition.json                   # 输入/各分片SHA、全部case数
├── reports/progress.node<N>.json            # 当前阶段/耗时
├── reports/final.json                       # 最终计数，成功时才有
├── control/                                # claim、done、failed、finalize
├── inputs/node<N>/                         # 按源图分组的manifest
├── nodes/node<N>/pipeline/                  # 原流程所有中间证据及内部日志
├── nodes/node<N>/audit/                     # 审核输入、prompt、完整回复
└── results/
    ├── all_cases.jsonl                     # 包含fail和no_output，不隐藏分母
    ├── audit.jsonl                         # 全部成图的审核
    └── model_pass.jsonl                    # 模型通过的候选；不是人工质量认证
```

这些日志路径全部位于experiments；不会散落到repo或旧 `.artifacts`。模型缓存和venv可在本地。
图片仍在各node的`pipeline/editing/context_grounded_v4_qwen21/edited`，汇总清单有绝对路径，避免复制几万张图。
不删失败图片、输入、规划拒绝或中间诊断图；磁盘容量应按全部尝试估计，不只按最终pass估计。

```bash
export SAMTOK_RUN_ID="samtok-remove-4n-20260924-001"
export RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID"
tail -F "$RUN_ROOT/logs/bootstrap.node0.log"
tail -F "$RUN_ROOT/nodes/node0/pipeline/logs/plan_0.log"
tail -F "$RUN_ROOT/nodes/node0/pipeline/editing/logs/editor_0.log"  # 模型tqdm
tail -F "$RUN_ROOT/nodes/node0/audit/worker_0.log"                 # audited x/y
cat "$RUN_ROOT/reports/partition.json"
cat "$RUN_ROOT/reports/progress.node0.json"
cat "$RUN_ROOT/reports/final.json"
```

将node0换成node1--3可看其他机器。不同机器可以处于不同阶段，各自使用自己的卡，不需要全机阶段同步。
只有全部node完成、ID/审核覆盖检查通过，node0才写`control/finalize.ok.json`和最终清单。
`pass`是审核通过；`fail`是已成图但审核失败；`no_output`是规划或区域解析未放行，不是成功生成。

## 6. 防覆盖与故障

新run ID只初始化一次；续跑显式设置`SAMTOK_RESUME=1`并使用新的attempt ID。
每个attempt中每个节点以独占claim认领，同rank的运行使用文件锁互斥，重复启动会拒绝。
四节点比对源manifest SHA、代码SHA、完整冻结profile、包版本、GPU数、共享模型路径/配置。
默认要求4个不同hostname和每节点8个GPU槽位。`--local-test`只用于开发模拟，正式bootstrap不传。

任一节点发生异常会写`control/*.failed.json`，其他已进入调度的节点发现后终止自己的模型进程组，
不会单独把半批结果标为完成。被外部SIGKILL的进程无法写marker，因此仍依赖等待超时；当前没有心跳租约恢复。
环境准备期间其他节点可能仍在安装，安装是有限操作；安装后进入门禁会看到失败，不继续生成。

2026-09-25的`samtok-remove-4n-20260925`在node1/node3报`libxcb.so.1`，根因是旧editor锁文件
同时包含GUI与headless OpenCV，不能仅凭此断定四台系统镜像不同。旧预检只读取headless包版本，
未检查实际cv2构建；即使列出的版本一致也可能加载到GUI二进制。当前已删除GUI包并加强检查。
修复后应从新环境安装，使用新的run ID（例如`samtok-remove-4n-20260925-r1`），并复制第3节更新后的
完整入口或使用已更新的共享bootstrap。仅拉新repo但继续使用旧的内联入口，会缺少新环境同步逻辑。
不要删除旧run的失败标记来强行续跑，也不要仅卸载GUI包后复用混装环境：卸载可能同时移除共享cv2文件。

默认节点加入等待3小时，单阶段/最终等待7天。大任务按实际预算评估；支持第3.2节的显式断点续跑，
不自动重启Arnold任务、换seed重试、对质量fail重采样或补齐到100k。
故障后保留目录，同run ID加新attempt ID恢复；不要手动伪造done文件或清空旧失败标记。
`SAMTOK_REPO_DIR`、`SAMTOK_RUN_ROOT`、模型位置和数据位置可覆盖，但四机共享数据/输出位置必须一致。

## 7. 本次验证边界

2026-09-25空mask与断点续跑修复：新增11项恢复测试，覆盖中断后保留正确规划、
空mask跳过、坏JSON重算、PNG损坏重生成、上游图片变化使审核失效、有效fail保留、
四个本地逻辑rank使用新attempt恢复、旧失败标记隔离、worker及时报错和模型缓存修复；相关回归通过。
真实模型smoke使用原任务的3条输入（已有正确规划、模拟坏规划、真实空mask各1条）：
正确规划保持一致，坏规划重算1次，空mask跳过，2条完成40步编辑。
再次执行同一流程后规划调用0次、SAM查询0次、编辑调用0次，两张PNG的SHA256完全不变；
生成流程恢复检查约12.86秒。真实27B审核2条得到1 pass / 1 fail，无解析错误；
审核续跑复用两条结果、调用0次、约2.00秒，有效fail没有被重采样。
这些测试验证恢复行为，不将2条小样本解释为质量通过率评估，也未重启正式四机任务。
完整数字存于当前run的`reports/resume_fix_validation.json`。
停止后的旧进度盘点为3,512条已落盘规划：3,484条可复用、28条格式/字段错误需重算。

当前交互会话的真实配置为1台8卡，因此不能声称已完成真实四台主机32卡的网络/共享存储联调。
已覆盖的本地测试和真实模型链路结果记录在统一的 `docs/SAMTOK_QUALITY_ITERATION.md` 最新部署小节；
部署验证产物统一位于experiments下的`deployment_validation_20260924`。
单机4个逻辑worker、每个2卡的实测已完成：8源图9region，全部成图并完成审核汇总，四worker退出0、
`run_localweights/control/finalize.ok.json`存在。9条均为模型pass，本轮未新增逐图人工验收。
链路最长约9.1分钟（不含安装/数据准备/首次缓存复制），编辑本体平均15.43秒/条。
最终安装脚本也从空目录跑通三套环境，单元测试341 passed，另5项融合测试在新环境通过。
首次真实四机建议先用8张源图做smoke，确认四台bootstrap日志与finalize完成，再换run ID令limit=0正式运行。
小批次冷启动时间不能外推为100k稳定吞吐，也不保证四机刚好达到四倍加速。

2026-09-25 OpenCV修复的验证记录见统一迭代文档最后小节，证据目录为
`experiments/SAMTok_Derived_Edit_Labeling/opencv_fix_20260925`。新锁文件已从空目录安装，
全部环境实际使用headless；回归测试348 passed / 6 skipped，5项融合测试另在新SAM环境通过。
修复后单机8卡模拟四worker的9个region已全部完成出图、审核及汇总，四worker退出0，
`run/control/finalize.ok.json`存在。模型审核9 pass，未新增人工质量验收。原四机任务没有自动重启。

## 8. 2026-09-25 正式四机运行结果

本节记录一次已经完成的 Arnold 四机正式运行，便于后续核对实际吞吐、数据分母和最终产物。

### 8.1 运行身份与目录

```text
run_id:       samtok-derived-4n-20260925
final attempt: resume-002
code commit:  165bfaf2a57bce1563686b1e2e786093851c3ba8
```

共享运行根目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/samtok-derived-4n-20260925
```

整理后的正式数据交付根目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling
```

该目录不是重新生成的一份独立 run，而是正式 run 的稳定交付视图：

```text
SAMTok_Derived_Edit_Labeling/
├── final/                         # 复制的最终清单、审核结果和 HTML
│   ├── all_cases.jsonl
│   ├── audit.jsonl
│   ├── model_pass.jsonl
│   ├── audit_gallery.html
│   ├── inspection.html
│   ├── inspection_assets/
│   └── edited_by_node -> .../experiments/.../nodes/
├── intermediate/                  # 正式 run 的完整中间阶段视图
│   ├── data_source -> .../data/source/
│   ├── inputs -> .../inputs/
│   ├── nodes -> .../nodes/
│   ├── attempts -> .../attempts/
│   ├── reports -> .../reports/
│   ├── control -> .../control/
│   └── logs -> .../logs/
└── run_root -> .../experiments/.../samtok-derived-4n-20260925/
```

`final/`中的 JSONL/HTML 是正式结果的交付副本；大体积的 source、edited、planning、
audit、checkpoint 和日志通过 `intermediate/` 保持原始目录结构，避免重复存储。所有
续跑和运行状态仍以 experiments 下的 canonical run root 为准。

最终 attempt 目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/samtok-derived-4n-20260925/attempts/resume-002
```

本轮曾经使用过`resume-001`，该 attempt 在外部任务生命周期终止时停止；正确的规划、SAM、编辑和审核检查点均保留，之后使用新的`resume-002`续跑。`resume-002`最终产生四个节点的`node<N>.done.json`，并写入：

```text
attempts/resume-002/control/finalize.ok.json
```

### 8.2 数据规模与阶段统计

本轮配置为 GRES-8k / VER-4k 派生数据，任务类型为`remove`，每个索引到的 source mask 形成一条单 region case，同一 source 的多个 mask 一图多用；空 mask 在规划阶段单独拦截。

| 阶段 | 总数 | GRES | VER | remove | 其他类型 |
|---|---:|---:|---:|---:|---:|
| 正例 source 图像 | 7,671 | 3,671 | 4,000 | — | — |
| 计划处理 case | 10,633 | 5,017 | 5,616 | 10,633 | 0 |
| 规划后接受并进入出图 | 9,436 | 4,458 | 4,978 | 9,436 | 0 |
| 规划阶段 `no_output` | 1,197 | 559 | 638 | 1,197 | 0 |
| 实际生成 edited PNG | 9,436 | 4,458 | 4,978 | 9,436 | 0 |
| 最终审核通过 | **7,990** | **3,689** | **4,301** | **7,990** | 0 |
| 最终审核失败 | 1,446 | 769 | 677 | 1,446 | 0 |

本轮正式任务是 **remove-only**：add、replace、attribute 均为 0，不应将本轮统计
解读为四种编辑类型的均衡实验。GRES/VER 的 source 数量来自正例索引，case 数量则是
每个 source 的全部 mask 展开后的数量；因此一个 source 可以贡献多条 case。

按来源分别计算，GRES 的规划接受率为 `4,458/5,017=88.86%`，出图后审核通过率为
`3,689/4,458=82.75%`，相对计划最终通过率为 `3,689/5,017=73.53%`；VER 对应为
`4,978/5,616=88.64%`、`4,301/4,978=86.40%` 和 `4,301/5,616=76.58%`。

#### 规划阶段 resolution status

| resolution_status | 总数 | GRES | VER |
|---|---:|---:|---:|
| `accepted` | 9,436 | 4,458 | 4,978 |
| `defer_unresolved_keep` | 517 | 219 | 298 |
| `defer_unresolved_auxiliary` | 339 | 135 | 204 |
| `defer` | 330 | 199 | 131 |
| `defer_support_conflict` | 6 | 4 | 2 |
| `invalid_input_empty_mask` | 1 | 1 | 0 |
| `invalid_relation_bbox` | 1 | 0 | 1 |
| `invalid_decision` | 1 | 0 | 1 |
| `invalid_reconstruction` | 1 | 1 | 0 |
| `defer_target_point_outside_mask` | 1 | 0 | 1 |

`no_output` 是规划阶段拒绝或无法确定执行关系的结果，不代表进程崩溃；本轮没有
OOM、CUDA error 或 worker failure。

#### 多实例 source 分布

| 每张 source 的原始 mask 数 | 唯一 source 图像数 | GRES | VER |
|---:|---:|---:|---:|
| 1 | 5,244 | 2,370 | 2,874 |
| 2 | 2,177 | 1,280 | 897 |
| 3 个及以上 | 250 | 21 | 229 |
| 合计 | 7,671 | 3,671 | 4,000 |

这说明本轮共有 2,427 张 source 含至少两个实例，生成时会对同一 source 的每个 mask
分别建立独立 case；`num_masks` 不会把多个区域合并成一次编辑。

#### 审核阶段模型与确定性规则分布

| 项目 | 总数 | GRES | VER |
|---|---:|---:|---:|
| 审核输入 | 9,436 | 4,458 | 4,978 |
| 模型原始 `pass` | 8,037 | 3,693 | 4,344 |
| 模型原始 `fail` | 1,349 | 745 | 604 |
| 模型结果未解析 | 50 | 20 | 30 |
| 最终 `decision=pass` | 7,990 | 3,689 | 4,301 |
| 最终 `decision=fail` | 1,446 | 769 | 677 |
| pixel veto 触发 | 214 | 67 | 147 |

其中模型原始 `pass` 中有 47 条被最终规则拒绝（GRES 4 条、VER 43 条）；50 条未解析
结果也不进入最终通过清单。`model_pass.jsonl` 的 7,990 条应理解为模型和确定性规则
共同通过，不等同于人工逐图 ground truth。

最终比例为：

```text
规划接受率       = 9,436 / 10,633 = 88.74%
出图后审核通过率 = 7,990 /  9,436 = 84.68%
相对计划最终通过 = 7,990 / 10,633 = 75.14%
```

### 8.3 运行时间与节点状态

四节点从分片完成（约 13:38:51 UTC）到最终汇总（约 17:56:34 UTC）墙钟时间约 **4 小时 18 分钟**。各节点报告如下：

| 节点 | 输入 case | pipeline 秒数 | audit 秒数 | 总秒数 |
|---|---:|---:|---:|---:|
| node0 | 2,659 | 9,167.32 | 5,922.27 | 15,093.43 |
| node1 | 2,659 | 6,321.05 | 6,155.55 | 12,480.48 |
| node2 | 2,657 | 7,039.00 | 5,796.57 | 12,836.22 |
| node3 | 2,658 | 6,396.61 | 5,957.49 | 12,357.81 |

本轮没有发现 OOM、CUDA error、Traceback、worker failed 或失败 marker。node0 是最后完成 audit 的节点，随后由协调器写入`finalize.ok.json`。

### 8.4 结果文件与图片路径

汇总结果：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/final/all_cases.jsonl
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/final/audit.jsonl
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/final/model_pass.jsonl
```

上述三个文件分别包含全部 10,633 条 case、9,436 条实际出图并审核的 case，以及
7,990 条最终模型审核通过的 case。canonical run 中的原始对应文件仍位于
`$RUN_ROOT/results/`。

编辑图片仍按节点保存，示例目录为：

```text
nodes/node0/pipeline/editing/context_grounded_v4_qwen21/edited/
nodes/node1/pipeline/editing/context_grounded_v4_qwen21/edited/
nodes/node2/pipeline/editing/context_grounded_v4_qwen21/edited/
nodes/node3/pipeline/editing/context_grounded_v4_qwen21/edited/
```

从交付目录访问全部节点图片：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/final/edited_by_node/
```

完整性检查对`audit.jsonl`引用的 9,436 张 edited PNG 逐张执行了存在性检查、PNG 解码和像素加载：

```text
缺失文件       0
解码失败       0
空文件         0
图片模式       9,436 张全部为 RGB
```

其中有 1 张结果接近纯白，是白色背景上的 iPhone 被移除后的有效结果，不是损坏图片。

### 8.5 可视化结果

推荐查看新版审核画廊：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/final/audit_gallery.html
```

canonical run 中的原始画廊仍保存在 `$RUN_ROOT/results/audit_gallery.html`。

本次整理前 `datasets/SAMTok_Derived_Edit_Labeling/` 下的 85 个 pilot、规划、生成和
质量迭代目录已经删除；它们不再是当前正式结果的一部分。历史说明文档中的旧实验路径
仅用于描述迭代背景，不能作为当前输入或续跑路径。

该 HTML 包含 20 个代表性 case，每条同时展示：

1. `BEFORE + MASK OUTLINE`：原图和 mask 邻域 crop，mask 外侧以黑白轮廓标注，mask 内保留原始像素；
2. `AFTER`：相同坐标的编辑结果；
3. Qwen3.8-27B 模型审核的`target_removed`、`quality`、`instruction_match`、像素证据和理由；
4. Assistant review：对这 20 条样例逐条查看后的人工判断和原因；
5. 按“模型/人工通过、像素误拒、编辑失败、instruction/mask 不匹配”筛选。

20 条样例中，5 条模型与人工均通过，4 条模型视觉判断通过但被像素阈值误拒，8 条存在目标残留或背景质量问题，3 条存在 instruction 与 mask 语义不匹配。HTML 图片已经内嵌，不依赖外部图片路径，适合直接在预览器中打开。

### 8.6 代表性最终通过 case（10 条）

下面直接嵌入 10 条 `model_pass.jsonl` 中的代表性样例，GRES 和 VER 各 5 条，覆盖四个
节点。每张卡左侧是带原始 mask 轮廓的 BEFORE，右侧是 AFTER；红色轮廓只用于可视化，
不是输入给编辑模型的红色覆盖。这里的“通过”表示模型审核和确定性规则通过，仍不等同于
人工逐图 ground truth。

| GRES case | GRES case |
|---|---|
| ![000004 GRES](assets/formal_results/formal_pass_01_000004_gres_r5_m0_remove.jpg)<br>`000004_gres_r5_m0_remove`：bottom-left dark sofa/armchair | ![000025 GRES](assets/formal_results/formal_pass_02_000025_gres_r31_m0_remove.jpg)<br>`000025_gres_r31_m0_remove`：frame-truncated white mug |
| ![000002 GRES](assets/formal_results/formal_pass_03_000002_gres_r2_m0_remove.jpg)<br>`000002_gres_r2_m0_remove`：center standing lamb | ![000008 GRES](assets/formal_results/formal_pass_04_000008_gres_r11_m0_remove.jpg)<br>`000008_gres_r11_m0_remove`：right cat silhouette |
| ![010631 GRES](assets/formal_results/formal_pass_05_010631_gres_r12334_m0_remove.jpg)<br>`010631_gres_r12334_m0_remove`：left bicycle | — |

| VER case | VER case |
|---|---|
| ![000000 VER](assets/formal_results/formal_pass_06_000000_ver_r0_m0_remove.jpg)<br>`000000_ver_r0_m0_remove`：right black tripod | ![000001 VER](assets/formal_results/formal_pass_07_000001_ver_r1_m0_remove.jpg)<br>`000001_ver_r1_m0_remove`：wall-mounted light fixture |
| ![000010 VER](assets/formal_results/formal_pass_08_000010_ver_r18_m0_remove.jpg)<br>`000010_ver_r18_m0_remove`：green striped barrier pole | ![000003 VER](assets/formal_results/formal_pass_09_000003_ver_r4_m0_remove.jpg)<br>`000003_ver_r4_m0_remove`：walking man on left walkway |
| ![000018 VER](assets/formal_results/formal_pass_10_000018_ver_r28_m0_remove.jpg)<br>`000018_ver_r28_m0_remove`：airborne soccer player | — |
