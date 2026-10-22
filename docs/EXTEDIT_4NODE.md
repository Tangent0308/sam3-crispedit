# extedit 四机入口（ARNOLD）

把 CrispEdit-2M、ScaleEdit 的原始编辑对打标成协议 2.0 训练数据（格式同 D3 / D4）的正式四机入口。方法、规则与验证结果见 [EXTEDIT_PROTOCOL2.md](EXTEDIT_PROTOCOL2.md)。

- **当前任务：**第一批 10 万个原始对（用户 2026-10-21 确认）。
- **代码：**本分支 commit `aad18e332075397f120a5255cfa3e37085e7a784`（10-22 起分割改为 3 个进程；之前的 `73cac43` 只有 1 个）；之后的提交只改了文档。
- **v3：**SAMTok codec 与 CUDA 检查用 SAMTokEdit v3 的 `31fb96605f68acb1254ec2e6b0e9ed22761093dd`。

## 1. 一页摘要
- **机器：**4 个 worker × 8 卡 H100 80G，互相独立，没有节点间通信。
  - 每个 worker 取打乱后分片的 1/4：每台的打乱顺序相同，轮流分配，互不重叠；
  - GPU 0–6 跑 Qwen3.8-27B（vLLM 0.31，数据并行 7 份），GPU 7 上跑 3 个 SAM3 + SAMTok 进程（`EXTEDIT_SEGMENT_DEVICES`，默认 `7,7,7`）；
  - 各步流式并行：取数 → 像素预检 → 27B 规划 → 分割与 token → 把关与贴回 → 27B 审核 → 导出。
- **续跑：**用同一个 `EXTEDIT_RUN_ID` 重新提交（或只重启某个 worker）即可从中断处继续。已经完成的 worker 会直接退出。
- **预计：**
  - 每个 worker 约 2.5 万个原始对、5–8 小时，另加装环境和拷模型约 15 分钟；
  - 四台合计预计产出约 3.9 万条。
- **产出：**在 `$EXTEDIT_ROOT/$EXTEDIT_RUN_ID/node<k>/final/`。合并发布到 `datasets/` 是跑完、抽查之后的单独一步（§6）。

## 2. 提交前检查

| 项 | 要求 |
|---|---|
| 机器 | 4 个 worker，每个 8 张 H100 80G；ARNOLD 注入 `ARNOLD_ID`、`ARNOLD_WORKER_NUM`、`ARNOLD_WORKER_GPU` |
| 节点本地盘 | `/tmp` 至少 100 GB：27B 模型 52 GB、两个环境约 20 GB、uv 缓存约 11 GB、vLLM 编译缓存 |
| 共享盘 | `/mnt/bn/strategy-mllm-train` 已挂载；运行目录约 450 GB（解出的原图约 300 GB，贴回后的 PNG 约 140 GB） |
| 网络 | GitHub（https clone，vLLM 0.31.0 的 release wheel）、bytedpypi、`download.pytorch.org/whl/cu129` |
| 代码权限 | `sam3-crispedit` 是公开仓库；`samtok_edit`（v3）是私有仓库，和 round2 入口一样用 https clone，需要集群上的 GitHub 凭据 |
| 模型 | 27B：`/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B` |
| | SAM3：`/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt` |
| | SAMTok：`/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok` |

第一次在集群上运行，建议先跑 §4 的冒烟，约 30 分钟，用来确认网络、凭据和本地盘都没问题。

## 3. 启动命令
把 3.1 和 3.2 两段依次粘贴成 ARNOLD 的 worker 启动命令，四个 worker 相同。

### 3.1 运行设置（每次运行只改这一段）

```bash
# ===== 运行设置 =====
export EXTEDIT_RUN_ID=extedit_protocol2_100k_20261021
export EXTEDIT_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs
export EXTEDIT_COMMIT=aad18e332075397f120a5255cfa3e37085e7a784
export EXTEDIT_V3_COMMIT=31fb96605f68acb1254ec2e6b0e9ed22761093dd
export EXTEDIT_CRISPEDIT=replace=3750,color=2000,remove=1250,add=625
export EXTEDIT_SCALEEDIT=object_surface_text_editing=3750,building_surface_text_editing=3750,object_replacement=3750,color_change=2000,material_change=0,object_removal=1250,object_addition=625
```

- **`EXTEDIT_RUN_ID`：**四个 worker 必须相同。换一个新值就是新任务；同一个值就是续跑。
- **配额：**按每个 worker 计（原始对行数），四台合计：

  | 类型 | 来源 | 四台合计（原始对） |
  |---|---|---:|
  | 写字 | ScaleEdit 物体表面文字、建筑表面文字 | 各 1.5 万 |
  | 替换 | CrispEdit replace、ScaleEdit 物体替换 | 各 1.5 万 |
  | 改属性 | CrispEdit color、ScaleEdit 改色 | 各 0.8 万 |
  | 改属性 | ScaleEdit 改材质 | 约 0.9 千 |
  | 删除 | 两个来源 | 各 0.5 万 |
  | 添加 | 两个来源 | 各 0.25 万 |

  改材质写 `material_change=0`，表示取尽自己那份分片里的改材质行。
- **代码版本：**`EXTEDIT_COMMIT` 和 `EXTEDIT_V3_COMMIT` 固定了两个仓库的版本。每次实际使用的 commit 记在 `node<k>/commits.attempt<N>.json`。

### 3.2 入口脚本（原样粘贴，不用改）
- 这段就是仓库里的 `scripts/extedit_bootstrap_arnold.sh`，测试会检查两者逐字一致。
- 它依次完成这些步骤：
  1. 检查 ARNOLD 注入的变量和共享盘写权限；
  2. 按固定 commit 把两个仓库 clone 到节点本地 `/tmp`；
  3. 按锁定文件装 vLLM 和 worker 两个环境（`scripts/extedit_setup_env.sh`），检查 8 张卡；
  4. 把 27B 拷到本地盘；
  5. 起服务；
  6. 跑两轮流水线，第二轮只重试调用失败的行。
- 27B 服务一旦中途退出，worker 立即报错退出，等重启后续跑。

<!-- code: scripts/extedit_bootstrap_arnold.sh -->
```bash
#!/usr/bin/env bash
# extedit production run on ARNOLD.  Paste this whole file as the start command of every worker (4 workers x 8 GPUs)
# after editing the run settings; docs/EXTEDIT_PROTOCOL2.md section 4.4 has the ready-to-paste settings.
#
# Workers are independent: worker k takes every n-th shard of the shuffled shard lists (the same order on every
# worker), labels its rows with the streaming pipeline (Qwen3.8-27B data parallel on GPUs 0-6, SAM3 + SAMTok workers
# on GPU 7) and exports <run>/node<k>/final.  A restarted worker resumes where it stopped: every stage skips the rows it
# already wrote.  Merging the nodes and publishing to datasets/ is a separate step after review (publish).
exec 2>&1  # ARNOLD's log page shows stdout; route every shell error there as well.

# ===== run settings (edit these) =====
# Identical on all workers; a new ID starts a new run, the same ID resumes it.
export EXTEDIT_RUN_ID="${EXTEDIT_RUN_ID:-extedit_protocol2_100k_20261021}"
# Shared root of production runs: the run goes to $EXTEDIT_ROOT/$EXTEDIT_RUN_ID.
export EXTEDIT_ROOT="${EXTEDIT_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs}"
# Raw rows per type ON EACH WORKER (0 = every row of the worker's shards).
export EXTEDIT_CRISPEDIT="${EXTEDIT_CRISPEDIT:-replace=3750,color=2000,remove=1250,add=625}"
export EXTEDIT_SCALEEDIT="${EXTEDIT_SCALEEDIT:-object_surface_text_editing=3750,building_surface_text_editing=3750,object_replacement=3750,color_change=2000,material_change=0,object_removal=1250,object_addition=625}"
# Code at exact commits: the labeling branch, and SAMTokEdit v3 for the SAMTok codec and the CUDA check.
export EXTEDIT_REPO_URL="${EXTEDIT_REPO_URL:-https://github.com/Tangent0308/sam3-crispedit.git}"
export EXTEDIT_BRANCH="${EXTEDIT_BRANCH:-extedit-protocol2-labeling}"
export EXTEDIT_COMMIT="${EXTEDIT_COMMIT:?set EXTEDIT_COMMIT to the labeling commit to run}"
export EXTEDIT_V3_URL="${EXTEDIT_V3_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export EXTEDIT_V3_BRANCH="${EXTEDIT_V3_BRANCH:-qwen-image-2.1-v3}"
export EXTEDIT_V3_COMMIT="${EXTEDIT_V3_COMMIT:-31fb966}"
export EXTEDIT_MODEL="${EXTEDIT_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}"
# Number of workers; 1 runs the same pipeline on a single 8-GPU machine.
export EXTEDIT_NNODES="${EXTEDIT_NNODES:-4}"
# SAM3 + SAMTok workers, one per entry, all on GPU 7 by default: one worker is CPU-bound (12 cores, GPU 7 about 55 %
# busy) and falls behind the 27B server; three share the GPU (about 19 GB each).
export EXTEDIT_SEGMENT_DEVICES="${EXTEDIT_SEGMENT_DEVICES:-7,7,7}"

# ===== ARNOLD checks =====
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$EXTEDIT_NNODES" ]] || { echo "Expected $EXTEDIT_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM"; exit 2; }
[[ "$ARNOLD_WORKER_GPU" == 8 ]] || { echo "Each worker needs 8 GPUs (7 for the 27B server, 1 for SAM3), got $ARNOLD_WORKER_GPU"; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$EXTEDIT_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((EXTEDIT_NNODES - 1))"; exit 2; }
# torch launcher variables would make the vLLM engines join a foreign process group.
unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE RANK LOCAL_RANK WORLD_SIZE
NODE="$ARNOLD_ID"
RUN="$EXTEDIT_ROOT/$EXTEDIT_RUN_ID"
WORK="/tmp/extedit-${EXTEDIT_RUN_ID}-node${NODE}"
NODE_RUN="$RUN/node$NODE"
mkdir -p "$RUN/bootstrap" "$NODE_RUN" "$WORK" || { echo "Cannot create $RUN"; exit 2; }
# A full NAS directory quota rejects every new file while df still reports free space.
PROBE="$RUN/bootstrap/.write_probe_node${NODE}"
( head -c 1048576 /dev/zero > "$PROBE" && rm -f "$PROBE" ) || { echo "Cannot write to $RUN (quota?)"; exit 2; }
if [[ -f "$NODE_RUN/SUCCESS.json" ]]; then
  echo "Worker $NODE of run $EXTEDIT_RUN_ID already finished ($NODE_RUN/SUCCESS.json); nothing to do."
  exit 0
fi
exec > >(tee -a "$RUN/bootstrap/node${NODE}.log") 2>&1
ATTEMPT=$(( $(cat "$NODE_RUN/attempts" 2>/dev/null || echo 0) + 1 ))
echo "$ATTEMPT" > "$NODE_RUN/attempts"
[[ -f "$NODE_RUN/failure.json" ]] && mv "$NODE_RUN/failure.json" "$NODE_RUN/failure.attempt$((ATTEMPT - 1)).json"
phase() { echo "$1" > "$NODE_RUN/phase"; echo "[$(date -u +%FT%TZ)] node $NODE attempt $ATTEMPT: $1"; }
SERVER="" PIPELINE=""
stop_children() {
  [[ -n "$PIPELINE" ]] && kill -TERM -- "-$PIPELINE" 2>/dev/null || true
  [[ -n "$SERVER" ]] && kill -TERM -- "-$SERVER" 2>/dev/null || true
}
failed() {
  local result=${1:-$?}
  trap - ERR TERM INT
  stop_children
  printf '{"error":"failed during %s; see bootstrap/node%s.log","exit_code":%d,"attempt":%d}\n' \
    "$(cat "$NODE_RUN/phase" 2>/dev/null || echo unknown)" "$NODE" "$result" "$ATTEMPT" > "$NODE_RUN/failure.json"
  exit "$result"
}
set -Eeuo pipefail
trap failed ERR
trap 'failed 143' TERM
trap 'failed 130' INT
echo "[$(date -u +%FT%TZ)] node $NODE of $EXTEDIT_NNODES: start, run $EXTEDIT_RUN_ID, attempt $ATTEMPT, host $(hostname)"

# ===== code: exact commits in node-local /tmp =====
phase code
clone() {  # url branch commit dir
  if [[ ! -d "$4/.git" ]] || [[ "$(git -C "$4" rev-parse HEAD)" != "$(git -C "$4" rev-parse "$3^{commit}" 2>/dev/null)" ]]; then
    rm -rf "$4"
    git clone --quiet --branch "$2" --single-branch "$1" "$4"
    git -C "$4" checkout --quiet "$3"
  fi
  git -C "$4" rev-parse HEAD
}
LABELING_HEAD=$(clone "$EXTEDIT_REPO_URL" "$EXTEDIT_BRANCH" "$EXTEDIT_COMMIT" "$WORK/labeling")
V3_HEAD=$(clone "$EXTEDIT_V3_URL" "$EXTEDIT_V3_BRANCH" "$EXTEDIT_V3_COMMIT" "$WORK/v3")
printf '{"labeling":"%s","v3":"%s"}\n' "$LABELING_HEAD" "$V3_HEAD" > "$NODE_RUN/commits.attempt$ATTEMPT.json"
REPO="$WORK/labeling"

# ===== environments, CUDA, model =====
phase environment
export EXTEDIT_V3_REPO="$WORK/v3" UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv-cache}"
bash "$REPO/scripts/extedit_setup_env.sh" vllm "$WORK/env-vllm"
bash "$REPO/scripts/extedit_setup_env.sh" worker "$WORK/env-worker"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WORK/v3/src" "$WORK/env-worker/bin/python" -B \
  -m samtok_edit21.distributed.cuda_readiness --output "$RUN/bootstrap/node${NODE}-cuda" --expected 8 \
  --timeout 600 --interval 15
phase model
MODEL="${EXTEDIT_LOCAL_MODELS:-$WORK/models}/$(basename "$EXTEDIT_MODEL")"
"$WORK/env-worker/bin/python" - "$EXTEDIT_MODEL" "$MODEL" <<'PY'
import shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for f in src.rglob("*"):  # copy what is missing or of another size (a copy cut by a restart)
    if f.is_file():
        out = dst / f.relative_to(src)
        if not out.exists() or out.stat().st_size != f.stat().st_size:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out.with_name(out.name + ".part"))
            out.with_name(out.name + ".part").replace(out)
print(f"model at {dst}")
PY

# ===== 27B server =====
phase server
PORT=8011
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT="$WORK/vllm-cache" \
  PATH="$WORK/env-vllm/bin:$PATH" \
  "$WORK/env-vllm/bin/vllm" serve "$MODEL" --served-model-name qwen38-27b --data-parallel-size 7 \
  --max-model-len 20000 --gpu-memory-utilization 0.93 --max-num-seqs 64 --enable-prefix-caching \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 6}' --host 127.0.0.1 --port "$PORT" \
  > "$NODE_RUN/vllm.attempt$ATTEMPT.log" 2>&1 &
SERVER=$!
for _ in $(seq 1 180); do
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q qwen38 && break
  kill -0 "$SERVER" 2>/dev/null || { echo "vLLM server exited; see $NODE_RUN/vllm.attempt$ATTEMPT.log"; false; }
  sleep 10
done
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q qwen38 || { echo "vLLM server not ready after 30 minutes"; false; }

# ===== pipeline: two passes, the second retries rows whose calls failed in the first =====
for pass in 1 2; do
  phase "pipeline pass $pass"
  setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" "$WORK/env-worker/bin/python" -B \
    "$REPO/scripts/extedit_pipeline.py" run --run "$NODE_RUN" --whole-shards --node "$NODE/$EXTEDIT_NNODES" \
    --crispedit "$EXTEDIT_CRISPEDIT" --scaleedit "$EXTEDIT_SCALEEDIT" \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" --plan-concurrency 168 --audit-concurrency 112 \
    --segment-devices "${EXTEDIT_SEGMENT_DEVICES:-7,7,7}" --workers 48 --v3-repo "$WORK/v3" \
    >> "$NODE_RUN/pipeline.log" 2>&1 &
  PIPELINE=$!
  while kill -0 "$PIPELINE" 2>/dev/null; do  # the pipeline needs the server: stop at once if it dies
    if ! kill -0 "$SERVER" 2>/dev/null; then
      echo "vLLM server exited during the pipeline; see $NODE_RUN/vllm.attempt$ATTEMPT.log"
      false
    fi
    sleep 30
  done
  wait "$PIPELINE"
  PIPELINE=""
done

phase finished
"$WORK/env-worker/bin/python" -B "$REPO/scripts/extedit_pipeline.py" status --runs "$NODE_RUN" | tee "$NODE_RUN/status.txt"
cp "$NODE_RUN/final/summary.json" "$NODE_RUN/SUCCESS.json"
stop_children
echo "[$(date -u +%FT%TZ)] node $NODE: done"
```

## 4. 冒烟（可选，先确认集群环境）
- **怎么跑：**只把 3.1 换成下面这段（每个 worker 约 40 行），3.2 不变。
- **用时：**每个 worker 约 15 分钟准备，加约 10 分钟流水线。

```bash
# ===== 运行设置：四机冒烟 =====
export EXTEDIT_RUN_ID=extedit_smoke_20261021
export EXTEDIT_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/04_current_validation
export EXTEDIT_COMMIT=aad18e332075397f120a5255cfa3e37085e7a784
export EXTEDIT_V3_COMMIT=31fb96605f68acb1254ec2e6b0e9ed22761093dd
export EXTEDIT_CRISPEDIT=replace=6,color=3,remove=3,add=2
export EXTEDIT_SCALEEDIT=object_surface_text_editing=4,building_surface_text_editing=4,object_replacement=6,color_change=3,material_change=2,object_removal=3,object_addition=2
```

- **通过标准：**
  - 四个 `node<k>/SUCCESS.json` 都在；
  - `bootstrap/node<k>.log` 里每个 worker 都走到了 `finished`；
  - `node<k>/status.txt` 里各步都有数。
- **之后：**冒烟的运行目录不进 `datasets/`，看完可以删。

## 5. 运行中：日志、进度、失败处理

### 5.1 目录
运行目录是 `RUN=$EXTEDIT_ROOT/$EXTEDIT_RUN_ID`：

| 路径 | 内容 |
|---|---|
| `bootstrap/node<k>.log` | 第 k 个 worker 的完整日志，每次尝试追加在后面 |
| `bootstrap/node<k>-cuda/` | CUDA 就绪检查的记录 |
| `node<k>/phase` | 当前阶段（见 5.3） |
| `node<k>/attempts` | 第几次尝试 |
| `node<k>/commits.attempt<N>.json` | 这次尝试实际用的两个 commit |
| `node<k>/vllm.attempt<N>.log` | 27B 服务日志 |
| `node<k>/pipeline.log` | 全流程汇总日志 |
| `node<k>/logs/<步骤>.log` | 各步日志 |
| `node<k>/ingest/`、`precheck/`、`plan/`、`segment/`、`gate/`、`audit/` | 各步记录；每步结束写 `DONE` |
| `node<k>/images/`、`node<k>/pasted/` | 解出的原图、贴回后的 AFTER |
| `node<k>/final/` | 导出结果（字段同 D3 / D4）、token 缓存、拒收清单、`summary.json` |
| `node<k>/status.txt`、`node<k>/SUCCESS.json` | 完成时写出 |
| `node<k>/failure.json` | 失败时写出，下一次尝试时改名为 `failure.attempt<N>.json` |

### 5.2 看进度

```bash
RUN=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/extedit_protocol2_100k_20261021
cat $RUN/node*/phase                       # 各 worker 当前阶段
ls $RUN/node*/SUCCESS.json $RUN/node*/failure.json 2>/dev/null
# 各步行数（需要 worker 环境；也可以让 Claude 在开发机上查）
PYTHONDONTWRITEBYTECODE=1 <worker env>/bin/python -B scripts/extedit_pipeline.py status \
  --runs $RUN/node0 $RUN/node1 $RUN/node2 $RUN/node3
```

### 5.3 各阶段与常见失败

| phase | 做什么 | 失败时先看 |
|---|---|---|
| `code` | 把两个仓库 clone 到 `/tmp/extedit-<run id>-node<k>/` | 网络；`samtok_edit` 的 GitHub 凭据 |
| `environment` | 装两个环境，检查 8 张卡 | bytedpypi、PyTorch 源、GitHub release 是否可达；`bootstrap/node<k>-cuda/` |
| `model` | 把 27B 拷到本地盘（拷到一半重启时只补缺的文件） | 本地盘空间 |
| `server` | 起 27B 服务，最多等 30 分钟 | `node<k>/vllm.attempt<N>.log` |
| `pipeline pass 1` / `pipeline pass 2` | 流式全流程；第二轮重试调用失败的行 | `node<k>/pipeline.log`、`node<k>/logs/*.log` |
| `finished` | 写 `status.txt`、`SUCCESS.json`，关服务 | — |

### 5.4 失败与续跑
- **怎么续跑：**任何阶段失败，worker 都会写 `node<k>/failure.json`（失败阶段和退出码），然后关掉服务和子进程再退出。用同一个 `EXTEDIT_RUN_ID` 重新提交即可续跑：
  - 已写过的行全部跳过，只做剩下的；
  - 已经有 `SUCCESS.json` 的 worker 直接退出。
- **不要换 `EXTEDIT_RUN_ID` 续跑：**换了就是新任务，会从头取数。
- **修了代码再续跑：**把 `EXTEDIT_COMMIT` 改成新 commit，`EXTEDIT_RUN_ID` 不变，重新提交。已完成的行不会重算，每次尝试用的 commit 都有记录。
- **27B 服务中途退出：**worker 立即退出（`failure.json` 里的阶段是 `pipeline pass <n>`），重启后续跑。

## 6. 跑完之后（由 Claude 执行，结果给用户确认）
1. 四个 `SUCCESS.json` 都在后，用 `status` 汇总各 worker 的行数。
2. 抽查审阅图：

   ```bash
   scripts/extedit_pipeline.py review --run $RUN/node<k> --out $RUN/node<k>/review --accept 60 --reject 40
   ```

3. 确认后合并发布，只有这一步写 `datasets/`：

   ```bash
   scripts/extedit_pipeline.py publish --runs $RUN/node0 $RUN/node1 $RUN/node2 $RUN/node3 \
     --out /mnt/bn/strategy-mllm-train/user/tanyue/datasets/ExtEdit_Protocol2_20261021
   ```

4. 发布后，运行目录里解出的原图可以删除（需要时能从源分片重新解出），各步记录保留。

## 7. 改参数参考

| 要改的 | 怎么改 |
|---|---|
| worker 数不是 4 | `export EXTEDIT_NNODES=<n>`，ARNOLD 的 worker 数要一致；配额按每个 worker 调整 |
| 单机跑 | 在 8 卡机器上设 `EXTEDIT_NNODES=1 ARNOLD_WORKER_NUM=1 ARNOLD_WORKER_GPU=8 ARNOLD_ID=0`，再执行 `bash scripts/extedit_bootstrap_arnold.sh` |
| 配额 | `EXTEDIT_CRISPEDIT`、`EXTEDIT_SCALEEDIT`，按每个 worker 计，0 表示取尽 |
| 本地盘位置 | 工作目录固定在 `/tmp/extedit-<run id>-node<k>/`；`EXTEDIT_LOCAL_MODELS` 改模型目录，`UV_CACHE_DIR` 改 uv 缓存 |
| 27B 模型 | `EXTEDIT_MODEL` |
| 分割进程 | `EXTEDIT_SEGMENT_DEVICES`，每项一个进程，写 GPU 编号，默认 `7,7,7`；单个进程受 CPU 限制，GPU 7 只用到约 55%，跟不上 27B（本机 10-22 实测：规划每分钟 96 行，单个分割进程 55 行） |

## 8. 这份入口验证到了什么程度（2026-10-21，开发机）
- **环境：**用全新的 uv 缓存，按与集群相同的源（bytedpypi、PyTorch cu129 源、GitHub release）从零装好两个环境并通过校验，分别用时 48 秒和 18 秒。
- **四机模拟：**
  - 设置：用本入口（脚本从标准输入喂给 bash，与 ARNOLD 相同），仓库地址换成本地 clone，依次扮演 4 个 worker，每个约 38 行；
  - 结果：全部成功，各类型配额都准确取满，四个 worker 的行和分片互不重叠，`publish` 合并出 55 条记录，图片和 token 缓存齐全。
- **中断续跑：**给 worker 1 发 SIGTERM，它写出 `failure.json`，子进程和服务都被关掉；重启后从中断处续完，行与各步记录没有重复。
- **代码获取：**不带凭据也能从 GitHub 按 https clone 本分支的固定 commit。
- **未在集群上实测：**ARNOLD 节点的网络、`samtok_edit` 的凭据、本地盘空间。建议先跑 §4 冒烟。
