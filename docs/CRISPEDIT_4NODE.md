# CrispEdit 四机完整打标入口

四台 Arnold worker 各 8 GPU。每台在本地 clone `crispedit-labeling` 并安装仓库环境，按质量过滤 → 细粒度过滤 → Qwen3.8 grounding → SAM3 mask → 校验运行。共享盘只保存源数据、模型、结果、日志和协调标记。方法、正式统计及可视化见 [CRISPEDIT_MASK.md](CRISPEDIT_MASK.md)。

## 新任务：四台 worker 使用相同入口

四台机器使用相同 `CRISPEDIT_RUN_ID`；每次新任务换唯一值。Arnold 提供 `ARNOLD_ID=0..3`、`ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8`。如果希望锁定精确代码，可四台额外设置同一 `CRISPEDIT_COMMIT` 完整 40 位 SHA。

```bash
#!/usr/bin/env bash
set -euo pipefail
export CRISPEDIT_RUN_ID="crispedit_next_20260926"
export CRISPEDIT_BRANCH="crispedit-labeling"
export CRISPEDIT_REPO_URL="https://github.com/Tangent0308/sam3-crispedit.git"
: "${ARNOLD_ID:?Arnold must supply node rank}"
: "${ARNOLD_WORKER_NUM:?Arnold must supply worker count}"
: "${ARNOLD_WORKER_GPU:?Arnold must supply GPU count}"
[[ $ARNOLD_WORKER_NUM == 4 && $ARNOLD_WORKER_GPU == 8 && $ARNOLD_ID =~ ^[0-3]$ ]]
[[ $CRISPEDIT_RUN_ID =~ ^[A-Za-z0-9._-]+$ ]]

export CRISPEDIT_DATA_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling"
export CRISPEDIT_RUN_DIR="$CRISPEDIT_DATA_ROOT/runs/labeling_4node_${CRISPEDIT_RUN_ID}"
export CRISPEDIT_INPUT_DIR="$CRISPEDIT_DATA_ROOT/source/CrispEdit-2M"
export CRISPEDIT_QUALITY_DIR="$CRISPEDIT_DATA_ROOT/prefilter/quality"
export CRISPEDIT_SCENE_DIR="$CRISPEDIT_DATA_ROOT/prefilter/scene"
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
if [[ ! -e $CRISPEDIT_REPO_DIR ]]; then
  git clone --single-branch --branch "$CRISPEDIT_BRANCH" "$CRISPEDIT_REPO_URL" "$CRISPEDIT_REPO_DIR"
else
  [[ -d $CRISPEDIT_REPO_DIR/.git ]] || { echo "Existing path is not a Git clone" >&2; exit 2; }
fi
cd "$CRISPEDIT_REPO_DIR"
[[ $(git remote get-url origin) == "$CRISPEDIT_REPO_URL" ]]
git diff --quiet && git diff --cached --quiet || { echo 'Existing clone has tracked changes' >&2; exit 2; }
git fetch --no-tags origin "$CRISPEDIT_BRANCH"
target_commit=${CRISPEDIT_COMMIT:-$(git rev-parse FETCH_HEAD)}
[[ $target_commit =~ ^[0-9a-f]{40}$ ]]
git merge-base --is-ancestor "$target_commit" FETCH_HEAD
git checkout --detach "$target_commit"
bash scripts/bootstrap_crispedit_4node.sh
```

启动前确保每台可访问 GitHub/PyPI/PyTorch 下载站点、本地约 40 GB 空间以及共享模型路径。bootstrap 在本地 clone 中安装 `.uv-python/` 和 `.venv-crispedit/`，预检查 8 卡 CUDA、OpenCV 与一次真实 Qwen 双图推理；四机代码与环境摘要相同后才生成 `plan.json`。node0 扫描五类源 shard、检查已完成的 `audit/manifest` 行号、只补跑缺失的两轮过滤；随后汇总双 PASS，grounding 与 mask 按新任务完整打标。不会删除已有正式结果。

## 日志、输出与续传

`CRISPEDIT_RUN_DIR/logs/entry.nodeN.log` 从 clone 开始记录；`bootstrap.nodeN.log` 记录安装与协调；`preflight.nodeN.log` 记录真实模型预检查；`quality.nodeN.log`、`scene.nodeN.log`、`grounding.nodeN.log`、`mask.nodeN.log` 均带 tqdm；`validate.node0.log` 是最终结构校验。`reports/` 存阶段汇总，`labels/{grounding,mask}/` 存标注，`work/` 存续传用的节点中间结果。成功标记为 `control/<attempt>/complete.ok`。可直接运行：

```bash
tail -f "$CRISPEDIT_RUN_DIR/logs/mask.node0.log"
```

失败后先确认四台旧进程均已退出，保留原 `RUN_ID`、计划、`work/`、标签及相同代码/参数/源数据；四台在上述完整入口的 `bash scripts/bootstrap_crispedit_4node.sh` 之前共同增加：

```bash
export CRISPEDIT_RESUME=1
export CRISPEDIT_RESUME_TOKEN="retry_01"  # 每次失败重试换新 token，四台一致
```

原运行 `plan.json` 冻结输入路径和源数据快照，因此已完成运行的旧目录保留兼容链接；不要擅自移除链接或改变历史运行的参数。完整 shard 会复用，缺失 shard 才补算。代码变化通常应新建 RUN_ID，不能把 `CRISPEDIT_ALLOW_CODE_CHANGE_ON_RESUME=1` 当常规开关。

## 已完成的正式运行

`runs/labeling_4node_crispedit_full_localenv_20260925/` 在 `retry_validate_fix_02` 尝试下已出现 `complete.ok`；其结果为 38,971 条双 PASS，37,728 条 `OK`、910 条 `MASK_REVIEW`、333 条 `GROUND_FAIL`，运行时错误 0。`initial` 与 `retry_grounding_fix_01` 目录内的失败标记是历史尝试记录。正式结果、统计和 35 例画廊见主文档；这个已完成的 RUN_ID 不应再次作为新任务入口使用。
