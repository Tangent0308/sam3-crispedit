# ScaleEdit 四机完整打标

四台机器各8卡：质量 → 细粒度 → 编辑单元观察/grounding → SAM3 → 合并校验。下载单独完成，不包含在此入口内。
每台通过git clone获得 `scaleedit-labeling`，安装本clone的Python和环境。共享盘仅放数据、模型、结果/日志和同步标记；没有跨节点模型并行，不需要NCCL跨机通信。

## 1. 完整入口

四台Arnold worker执行同一段脚本，设置相同唯一RUN_ID；Arnold提供 `ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8`、`ARNOLD_ID=0..3`。
需要Git、curl、Python3/pip、flock、可访问GitHub/PyPI/PyTorch下载站点的网络、本地约40GB以上环境空间。

```bash
#!/usr/bin/env bash
set -euo pipefail
export SCALEEDIT_RUN_ID="scaleedit_full_20260925_a"
export SCALEEDIT_BRANCH="scaleedit-labeling"
export SCALEEDIT_REPO_URL="https://github.com/Tangent0308/sam3-crispedit.git"
export SCALEEDIT_RUN_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/ScaleEdit/labeling_4node_${SCALEEDIT_RUN_ID}"
export SCALEEDIT_INPUT_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-100k"
export SCALEEDIT_MODEL_DIR="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B"
export SCALEEDIT_SAM3_CHECKPOINT_PATH="/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"
export SCALEEDIT_FILTER_BATCH_SIZE=4
export SCALEEDIT_GROUNDING_BATCH_SIZE=4
: "${ARNOLD_ID:?Missing Arnold rank}"
export SCALEEDIT_REPO_DIR="/opt/tiger/tanyue/workspaces/sam3-crispedit-scaleedit-labeling-${SCALEEDIT_RUN_ID}-node${ARNOLD_ID}"

# 先获取独立入口；它负责git clone、安装环境、预检查和完整四阶段运行。
bootstrap_dir=$(mktemp -d /tmp/scaleedit-bootstrap.XXXXXX)
mkdir -p "$SCALEEDIT_RUN_DIR/logs" "$SCALEEDIT_RUN_DIR/bootstrap_control/${SCALEEDIT_ATTEMPT:-initial}"
exec > >(tee -a "$SCALEEDIT_RUN_DIR/logs/launcher.node${ARNOLD_ID}.log") 2>&1
trap 'code=$?; if (( code != 0 )); then printf "launcher exit=%s\n" "$code" > "$SCALEEDIT_RUN_DIR/bootstrap_control/${SCALEEDIT_ATTEMPT:-initial}/node${ARNOLD_ID}.failed"; fi' EXIT
curl --fail --location --retry 3 \
  "https://raw.githubusercontent.com/Tangent0308/sam3-crispedit/${SCALEEDIT_BRANCH}/scripts/launch_scaleedit_4node.sh" \
  --output "$bootstrap_dir/launch_scaleedit_4node.sh"
bash "$bootstrap_dir/launch_scaleedit_4node.sh"
```

入口脚本：[launch_scaleedit_4node.sh](../scripts/launch_scaleedit_4node.sh)。如果平台不能下载raw文件，可把该文件完整内容直接作为入口；它不要求预先存在clone。
已有正确、干净clone可复用，不会因目录已存在而重复clone失败；部分clone/错误origin/本地修改会拒绝。可设置 `SCALEEDIT_COMMIT` 为完整40位SHA固定代码；四节点commit、代码、依赖、GPU与独立hostname必须一致。
安装：[setup_scaleedit_env.sh](../scripts/setup_scaleedit_env.sh)；预检：[preflight_scaleedit_env.py](../scripts/preflight_scaleedit_env.py)。预检包含headless OpenCV、spawn导入、8卡CUDA算子与GPU0真实Qwen双图请求，通过后才开始数据规划。

## 2. 数据范围、复用与输出

- 源目录当前1,155 shard / 299,633行。全局类别在质量阶段直接DROP，不调用模型。
- node0动态生成 `plan.json`，冻结源快照、选择行、代码及参数；每阶段根据上游PASS数量重新均衡shard。
- 新RUN处理完整源范围，不自动导入先前512条开发实验的稀疏结果。同一RUN恢复时复用签名一致的完整shard，只重跑未完成shard，包括grounding/mask。
- 质量/scene输出binary判决；后续仅双PASS进入模型。每个shard保持原始行号，阶段结果可能为空但schema完整；无待处理行时不加载模型。
- 每节点只写自身 `work/STAGE/nodeN/`，node0验证后硬链接到 `RUN/STAGE/`；不覆盖其他run结果。共享挂载必须支持POSIX锁、原子rename与同挂载硬链接。

```text
RUN/
  plan.json                         源快照/运行配置
  quality_plan.json, scene_plan.json 阶段分配与实际eligible行数
  grounding_plan.json, mask_plan.json
  selections/                       分节点源行选择
  quality/{audit,manifest}/          质量结果
  scene/{audit,manifest}/            细粒度结果
  grounding/*.parquet                观察与框/多边形、原始回答
  mask/*.parquet                     PNG、实例RLE、QC
  work/STAGE/nodeN/                  节点原子输出和恢复缓存
  logs/STAGE.nodeN.log               quality/scene/grounding/mask tqdm
  logs/entry.nodeN.log               clone、安装和协调总日志
  logs/preflight.nodeN.log           环境与真实Qwen探针
  control/ATTEMPT/                   joined、ready、done、merged、failed
  reports/run_manifest.json          最终统计、hosts、代码摘要
```

每阶段完成四节点屏障后才进入下一阶段。校验覆盖源身份、严格PASS join、audit/manifest一致、PNG尺寸/二值性、实例RLE并集及面积。`complete.ok`表示执行和结构校验完成；各阶段`counts.errors`和mask QC仍需查看，不能当作语义质量认证。
默认质量/scene每机8×TP1、grounding每机4×TP2、SAM3每卡一个worker；batch=4。tqdm分母为节点分配的源选择行，含上游DROP跳过量；模型实际输入看对应阶段plan的`selected_rows`。

## 3. 进度与恢复

```bash
RUN=/mnt/bn/strategy-mllm-train/user/tanyue/experiments/ScaleEdit/labeling_4node_scaleedit_full_20260925_a
tail -F "$RUN/logs/quality.node0.log"
tail -F "$RUN/logs/scene.node0.log"
tail -F "$RUN/logs/grounding.node0.log"
tail -F "$RUN/logs/mask.node0.log"
find "$RUN/control" -type f -name '*.failed' -print
cat "$RUN/reports/run_manifest.json"
```

节点进程失败/异常退出会写`.failed`并让其他节点停止当前模型进程组；找最早失败的节点日志。掉电或SIGKILL无法写标记，其他节点到达等待上限后退出，需结合日志更新时间检查。

恢复前确认旧任务全部退出。四台使用相同RUN_ID、源路径、参数、原commit，额外设置：

```bash
export SCALEEDIT_RESUME=1
export SCALEEDIT_ATTEMPT=retry1    # 每次恢复都换新名称，四台一致
export SCALEEDIT_COMMIT="原运行的完整40位commit"
```

再执行第1节入口。旧失败日志和标记保留；新attempt避免旧失败标记误触发。代码、源快照、选择或推理参数变化会拒绝复用，改用新RUN_ID。恢复按完成shard复用，正在写入的未完成shard重做。

## 4. 验证范围

本机具备8张H100；真实模型验证使用四个本地rank、每rank独占2卡（`--local-test`），覆盖全部四阶段和共享文件协调。
这不等价于四台物理机器32卡实测；正式入口强制检查四个不同hostname及每节点8卡。

```bash
cd /opt/tiger/tanyue/sam3-crispedit-scaleedit-labeling
.venv-scaleedit-current/bin/python -u scripts/smoke_scaleedit_4node.py \
  --run-dir /mnt/bn/strategy-mllm-train/user/tanyue/experiments/ScaleEdit/labeling_4node_smoke_20260925
# 原运行完成或退出后，验证完整shard复用：
.venv-scaleedit-current/bin/python -u scripts/smoke_scaleedit_4node.py \
  --run-dir /mnt/bn/strategy-mllm-train/user/tanyue/experiments/ScaleEdit/labeling_4node_smoke_20260925 \
  --resume --attempt retry1
```

2026-09-25 实测：14个shard的20对样本，质量15 PASS/5 DROP → scene 10 PASS/5 DROP → 10 grounding OK → 10非空mask、26实例；全部阶段执行错误为0。四rank全部退出0，`control/initial/complete.ok`已生成。

同RUN以 `retry1` 恢复成功：84个结果Parquet的SHA256和mtime均不变，追加日志无模型加载/推理；`control/retry1/complete.ok`已生成。报告为 `reports/resume_verification.json`。
92项CPU测试通过，覆盖身份/二值判决、分配、同伴失败终止、恢复发布冲突、稀疏join、PNG/RLE一致性、四机环境报告。安装脚本在当前本地clone运行成功，8卡CUDA与spawn导入预检通过；完整smoke另实际执行了Qwen和SAM3。

[实跑画廊](/mnt/bn/strategy-mllm-train/user/tanyue/experiments/ScaleEdit/labeling_4node_smoke_20260925/review/index.html)展示两轮各3 KEEP/3 DROP及全部10个mask；`review/validation.json`为 `errors=[]`。已目视检查全部10个mask，人物、文字、多实例和新增花瓶等结果大体符合当前策略；scene原有chef/长椅边界误保留仍在，未把工程验证当作语义问题已修复。
未启动全量299,633条推理；四台物理机32卡运行与吞吐仍需在正式集群确认。
