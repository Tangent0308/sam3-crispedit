# 合板前的统一文字复查（ARNOLD 单机）

用户 2026-10-24 确认：
- 合板数据里给文字类单元加独立标记；
- 合板前对所有文字相关的编辑统一做一次文字复查；
- 单机运行，由用户提交。

这份文档是可直接粘贴的入口。代码在 `extedit/textrecheck.py`，命令为 `scripts/extedit_pipeline.py textrecheck-units` 和 `textrecheck`。

## 1. 为什么要复查
合板前盘点发现，只有 D10、D11 的改字单元做过逐字读回复查，其余文字类单元都没有。抽查结果：
- **D3 的改字：**12 条里有 2 条，指令要的字本身就拼错了，例如 "Stadiiom"、"IVECCO RACINGTEAMS"，来自 ScaleEdit 原指令；
- **带字的新物体：**15 条里 3 条不可用：
  - 字与指令不一致，例如指令要 'EDELTMAN'，图上是 'EDELMAN'；
  - 字是乱码，例如 "Victory Day Celebration" 被写坏。

记录在 `experiments2/SAMTokEdit/data_pipeline/04_current_validation/merged_text_units_20261024/`。

## 2. 复查哪些单元
清单是 `units.jsonl`，共 17,305 个单元，在本机由 `textrecheck-units` 生成。

| 类别 | 来源 | 单元数 | 做什么 |
|---|---|---:|---|
| 改字 | D1、D3、D4 | 1,477 | 指令检查 + 读图复查 |
| 改字 | D10、D11 | 12,309 | 只做指令检查；打标时已读图复查过 |
| 可能带字的新物体（添加、替换） | D1、D3、D4、D10、D11 | 3,519 | 指令检查；模型从描述里找出要写的字，找到才读图复查 |

- **指令检查：**
  - 只看指令文字，27B 开 thinking（low），一次请求两票；
  - 给出两项：要写的字（改字就是 spec 里的值），以及这段字是否乱码；
  - 乱码指随机字母，或常见词拼错，例如 "Stadiiom"。人名、品牌、缩写、编号数字、外文、干净的自造名称都不算。
- **读图复查：**和打标时的文字复查（`textcheck.py`）完全相同：
  - 给 27B 改动前、后两张特写，不开 thinking、贪心解码，不告诉它要的是什么字，让它读出来；
  - 由代码比对：只看字母和数字，用包含关系，同时判断字是否没改、旧字是否还在；
  - 改字的特写围绕 S；带字新物体的特写围绕改动区域。
- **不在这里判定：**复查只记录模型的回答。拒收规则在合板时由代码判定，先用已知样本校准，见 §5。

## 3. 怎么提交
- **规格：**ARNOLD 1 个 worker，8 张卡，8 张都给 27B。
- **操作：**把下面整段（运行设置 + 入口脚本，到代码块结束）原样粘贴为启动命令。
- **注意：**入口脚本和第二批四机用的是同一个，多了任务类型开关：`EXTEDIT_JOB=textrecheck`，默认仍是打标流水线。

<!-- code: text recheck settings + scripts/extedit_bootstrap_arnold.sh -->
```bash
# ===== 运行设置 =====
export EXTEDIT_JOB=textrecheck
export EXTEDIT_RUN_ID=merge_text_recheck_20261024
export EXTEDIT_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs
export EXTEDIT_TEXT_UNITS=$EXTEDIT_ROOT/merge_text_recheck_20261024/units.jsonl
export EXTEDIT_COMMIT=7e472f7a3d39aea03817633821b8f4ea57afc791
export EXTEDIT_V3_COMMIT=31fb96605f68acb1254ec2e6b0e9ed22761093dd
export EXTEDIT_NNODES=1
export EXTEDIT_SERVER_GPUS=0,1,2,3,4,5,6,7

#!/usr/bin/env bash
# extedit production run on ARNOLD.  Paste this whole file as the start command of every worker (4 workers x 8 GPUs)
# after editing the run settings; docs/EXTEDIT_PROTOCOL2.md section 4.4 has the ready-to-paste settings.
# EXTEDIT_JOB=textrecheck runs the unified text re-check instead (one worker; docs/EXTEDIT_TEXT_RECHECK.md).
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
# Node folders of earlier batches, space-separated: their rows are not taken again (empty: none).
export EXTEDIT_EXCLUDE_RUNS="${EXTEDIT_EXCLUDE_RUNS:-}"
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
# Job: "pipeline" labels raw pairs with the settings above; "textrecheck" re-checks the text-related units listed in
# EXTEDIT_TEXT_UNITS (a file on the shared disk) and needs no SAM3, so its 27B server can take all 8 GPUs.
export EXTEDIT_JOB="${EXTEDIT_JOB:-pipeline}"
export EXTEDIT_TEXT_UNITS="${EXTEDIT_TEXT_UNITS:-}"
# GPUs of the 27B server, one data-parallel engine each; the pipeline keeps GPU 7 for SAM3.
export EXTEDIT_SERVER_GPUS="${EXTEDIT_SERVER_GPUS:-0,1,2,3,4,5,6}"

# ===== ARNOLD checks =====
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$EXTEDIT_NNODES" ]] || { echo "Expected $EXTEDIT_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM"; exit 2; }
[[ "$ARNOLD_WORKER_GPU" == 8 ]] || { echo "Each worker needs 8 GPUs, got $ARNOLD_WORKER_GPU"; exit 2; }
[[ "$EXTEDIT_JOB" == pipeline || "$EXTEDIT_JOB" == textrecheck ]] || { echo "Unknown EXTEDIT_JOB=$EXTEDIT_JOB"; exit 2; }
if [[ "$EXTEDIT_JOB" == textrecheck && ! -s "$EXTEDIT_TEXT_UNITS" ]]; then
  echo "EXTEDIT_JOB=textrecheck needs EXTEDIT_TEXT_UNITS, the units file on the shared disk: '$EXTEDIT_TEXT_UNITS'"; exit 2
fi
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
ENGINES=$(awk -F, '{print NF}' <<< "$EXTEDIT_SERVER_GPUS")
setsid env CUDA_VISIBLE_DEVICES="$EXTEDIT_SERVER_GPUS" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT="$WORK/vllm-cache" \
  PATH="$WORK/env-vllm/bin:$PATH" \
  "$WORK/env-vllm/bin/vllm" serve "$MODEL" --served-model-name qwen38-27b --data-parallel-size "$ENGINES" \
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

# ===== job: two passes, the second retries rows (units) whose calls failed in the first =====
if [[ "$EXTEDIT_JOB" == textrecheck ]]; then
  JOB=(textrecheck --units "$EXTEDIT_TEXT_UNITS" --run "$NODE_RUN" --url "http://127.0.0.1:$PORT/v1/chat/completions"
       --concurrency 256 --view-workers 32)
else
  JOB=(run --run "$NODE_RUN" --whole-shards --node "$NODE/$EXTEDIT_NNODES"
       --crispedit "$EXTEDIT_CRISPEDIT" --scaleedit "$EXTEDIT_SCALEEDIT" ${EXTEDIT_EXCLUDE_RUNS:+--exclude-runs $EXTEDIT_EXCLUDE_RUNS}
       --url "http://127.0.0.1:$PORT/v1/chat/completions" --plan-concurrency 168 --audit-concurrency 112
       --segment-devices "${EXTEDIT_SEGMENT_DEVICES:-7,7,7}" --workers 48 --v3-repo "$WORK/v3")
fi
for pass in 1 2; do
  if [[ "$EXTEDIT_JOB" == textrecheck ]]; then phase "textrecheck pass $pass"; else phase "pipeline pass $pass"; fi
  setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" "$WORK/env-worker/bin/python" -B \
    "$REPO/scripts/extedit_pipeline.py" "${JOB[@]}" >> "$NODE_RUN/pipeline.log" 2>&1 &
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
if [[ "$EXTEDIT_JOB" == textrecheck ]]; then
  [[ -f "$NODE_RUN/textrecheck/DONE" ]] || { echo "Units still unchecked after two passes; see $NODE_RUN/pipeline.log"; false; }
  cp "$NODE_RUN/textrecheck/summary.json" "$NODE_RUN/SUCCESS.json"
else
  "$WORK/env-worker/bin/python" -B "$REPO/scripts/extedit_pipeline.py" status --runs "$NODE_RUN" | tee "$NODE_RUN/status.txt"
  cp "$NODE_RUN/final/summary.json" "$NODE_RUN/SUCCESS.json"
fi
stop_children
echo "[$(date -u +%FT%TZ)] node $NODE: done"
```

## 4. 运行与输出
- **预计约 1.5 小时：**
  - 装环境约 2 分钟；
  - 拷 27B 模型到本地盘约 30 分钟；
  - 起服务约 5 分钟；
  - 复查约 45–60 分钟：1.7 万次只看文字的请求（各两票），加约 3,500 次读图。
- **运行目录：**`experiments2/SAMTokEdit/data_pipeline/02_production_runs/merge_text_recheck_20261024/`

  | 路径 | 内容 |
  |---|---|
  | `units.jsonl` | 复查清单，本机已生成 |
  | `node0/phase` | 当前阶段：code → environment → model → server → textrecheck pass 1/2 → finished |
  | `node0/textrecheck/records.jsonl` | 每个单元一行：两票（要写的字、是否乱码）；读图复查读到的改动前后的字，以及按打标规则得出的通过或原因 |
  | `node0/textrecheck/summary.json`、`DONE` | 计数；全部单元都有记录时写 `DONE` |
  | `node0/SUCCESS.json` 或 `failure.json` | 完成或失败 |
  | `bootstrap/node0.log`、`node0/pipeline.log`、`node0/vllm.attempt*.log` | 日志 |

- **失败与续跑：**用同一个 `EXTEDIT_RUN_ID` 重新提交即可，已有记录的单元会跳过。

## 5. 跑完之后（本机）
1. **校准：**
   - 已知结论的样本：抽查里判过的 27 条，加上 "Stadiiom"、"EDELTMAN" 等例子；
   - 再加各类随机样本；
   - 画出复查前后的对照拼图看图，定下拒收规则：例如读图复查不通过的拒收，两票都说乱码的拒收。
2. **合板：**按定好的规则给每个单元写文字标记（改字 / 写字 / 删字）和复查结论，不通过的单元在默认训练视图里排除。
3. 结果记在本文档、合板数据集的 README 和 v3 文档 01。

## 6. 运行结果与判定（10-24）
- **运行：**ARNOLD 单机，10:45 提交，11:32 完成，第 1 次尝试成功。正式复查只用了 9 分钟，17,305 个单元都有记录，没有报错。
- **校准：**
  - 27 条已知样本里，5 条坏的全部拦下；
  - 按结果分组看了 32 条：乱码判定 16 条全对，"判为不需要写字"和"通过"的 8 条全对；
  - 字对不上的 8 条里 6 条该拒。
  - 记录在运行目录 `admin/calibration/README.md`。
- **规则：**至少一票判乱码，或读图复查不通过，单元就不通过。一条记录里有一个文字单元不通过，整条不进默认训练视图。D10、D11 的改字只看乱码一条。
- **结果：**
  - 改字 13,786 个，通过 13,400；
  - 写字（新物体带字）2,181 个，通过 1,768；
  - 共 794 条记录不进默认训练视图；
  - 每个单元的判定在 `admin/calibration/verdicts.jsonl`，合板时写进每个单元的文字标记。
