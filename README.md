# sam3 × CrispEdit 打标说明

本仓库用于为 [samtok_edit](https://github.com/Tangent0308/samtok_edit) 项目进行数据打标。

这个仓库当前主要用于 **CrispEdit-2M 的两阶段打标**：

1. **prefilter**：本地 Qwen3-VL 先判断编辑是否真正达成，并对 `add` 使用更严格的新内容判定
2. **mask**：对 keep 样本使用当前 production/base SAM3 流程生成编辑区域 mask

当前生产路径对应文件：

- `scripts/setup_env.sh`：环境搭建脚本
- `crispedit_mllm_prefilter.py`：第一阶段语义预筛选
- `crispedit_mask_dataset_runner.py`：第二阶段并行 mask runner
- `crispedit_mask_pipeline.py`：单样本 base SAM3 mask 逻辑
- `CRISPEDIT_FILTER_THEN_MASK_PIPELINE.md`：更详细的生产说明与统计

---

## 1. 环境要求

推荐环境：

- Linux
- Python 3.11
- NVIDIA GPU（prefilter / mask 都默认按多卡运行）
- 本地 Qwen3-VL-8B 模型目录
- 可选：本地 SAM3 checkpoint；如果不提供，则默认尝试使用 Hugging Face 缓存/下载

当前脚本默认的 Qwen 模型路径：

```text
/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct
```

可选环境变量：

```bash
export CRISPEDIT_QWEN_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export CRISPEDIT_SAM3_CHECKPOINT_PATH=/path/to/sam3_checkpoint.pt
```

如果不提供 `CRISPEDIT_SAM3_CHECKPOINT_PATH`，则需要本机具备 Hugging Face 访问能力；必要时先执行：

```bash
huggingface-cli login
```

---

## 2. 一键搭建环境

在仓库根目录执行：

```bash
cd /opt/tiger/tanyue/sam3
bash scripts/setup_env.sh --python-bin python3.11
```

这个脚本会完成：

- 创建或复用 `.venv-sam3-crispedit`
- 安装 CUDA 版 `torch` / `torchvision`
- 安装本仓库的 `.[crispedit]` 依赖
- 校验以下模块能否导入：
  - `torch`
  - `torchvision`
  - `transformers`
  - `pyarrow`
  - `cv2`
  - `PIL`
  - `einops`
  - `pycocotools`
  - `crispedit_mllm_prefilter`
  - `crispedit_mask_dataset_runner`
  - `crispedit_mask_pipeline`
  - `sam3`

搭建完成后，推荐进入虚拟环境：

```bash
source /opt/tiger/tanyue/sam3/.venv-sam3-crispedit/bin/activate
```

也可以直接用解释器：

```bash
/opt/tiger/tanyue/sam3/.venv-sam3-crispedit/bin/python
```

快速检查脚本参数：

```bash
python crispedit_mllm_prefilter.py --help
python crispedit_mask_dataset_runner.py --help
```

---

## 3. 数据组织方式

当前 CrispEdit 原始数据读取方式是：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M
```

注意：

- parquet shard **直接放在根目录下**
- **不是** `.../data/`

例如：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M/add_00000.parquet
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M/remove_00012.parquet
```

runner 会按文件名中的前缀自动识别原始类型，例如：

- `add_00057.parquet`
- `background change_00083.parquet`
- `style_00085.parquet`

---

## 4. 当前打标流程

整体流程：

```text
raw parquet shards
    ↓
Qwen3-VL 本地 prefilter
    ↓
audit parquet + keep manifest parquet
    ↓
SAM3 manifest-aware mask runner
    ↓
keep 样本生成 mask
非 keep 样本写 PREFILTER_SKIP 占位行
    ↓
得到与原始 shard 逐行对齐的最终输出 parquet
```

关键点：

- **prefilter** 负责判断编辑结果是否真的符合 instruction
- **mask 阶段** 不再重新做语义判定，而是直接读取 prefilter manifest
- 最终输出与原始输入 **逐行对齐**，方便后续用 `row_idx` 回查

---

## 5. 第一阶段：prefilter 用法

### 5.1 作用

`crispedit_mllm_prefilter.py` 会读取：

- `input_img`
- `output_img`
- `instruction`
- `type`

输出两类 parquet：

1. **audit parquet**：保存逐行 MLLM 判定结果
2. **keep manifest parquet**：保存 keep / drop 决策，供下一阶段使用

默认决策规则：

- `PASS -> keep`
- `FAIL -> drop`
- `UNSURE -> drop`
- parse/runtime error -> drop

### 5.2 小规模 smoke test

```bash
cd /opt/tiger/tanyue/sam3
source .venv-sam3-crispedit/bin/activate

python crispedit_mllm_prefilter.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --audit-dir /tmp/crispedit_prefilter_smoke/audit \
  --keep-manifest-dir /tmp/crispedit_prefilter_smoke/manifest \
  --devices 0 \
  --max-shards-per-type 1 \
  --limit-rows-per-shard 2 \
  --batch-size 1 \
  --max-new-tokens 220
```

### 5.3 生产用法（8 卡，后台，带 tqdm）

```bash
cd /opt/tiger/tanyue/sam3
source .venv-sam3-crispedit/bin/activate

mkdir -p /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit
mkdir -p /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-manifest

nohup python -u crispedit_mllm_prefilter.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --audit-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-manifest \
  --devices 0,1,2,3,4,5,6,7 \
  --batch-size 4 \
  --max-new-tokens 220 \
  --progress-mininterval 5 \
  > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.log 2>&1 < /dev/null &

echo $! > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.pid
```

---

## 6. 第二阶段：mask 用法

### 6.1 作用

`crispedit_mask_dataset_runner.py` 会：

- 读取 raw shard
- 读取对应的 keep manifest
- 对 `filter_decision == keep` 的样本运行当前 production/base SAM3 mask 逻辑
- 对非 keep 样本写 `PREFILTER_SKIP` 占位行

### 6.2 小规模 smoke test

```bash
cd /opt/tiger/tanyue/sam3
source .venv-sam3-crispedit/bin/activate

python crispedit_mask_dataset_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --keep-manifest-dir /tmp/crispedit_prefilter_smoke/manifest \
  --output-dir /tmp/crispedit_mask_smoke \
  --devices 0 \
  --max-shards-per-type 1 \
  --limit-rows-per-shard 2 \
  --batch-size 2 \
  --preview-dir /tmp/crispedit_mask_smoke/previews \
  --preview-rows-per-shard 2
```

### 6.3 生产用法（8 卡，后台，带 tqdm）

```bash
cd /opt/tiger/tanyue/sam3
source .venv-sam3-crispedit/bin/activate

mkdir -p /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697

nohup python -u crispedit_mask_dataset_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-manifest \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697 \
  --devices 0,1,2,3,4,5,6,7 \
  --batch-size 8 \
  --progress-mininterval 5 \
  > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.log 2>&1 < /dev/null &

echo $! > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.pid
```

---

## 7. resume / overwrite 规则

生产续跑时，**默认不要加 `--overwrite`**。

效果是：

- 已完成 shard：直接跳过
- `.tmp` 文件不会被当作最终完成结果
- 可以从断点继续，而不是把所有 shard 重刷一遍

具体规则：

### prefilter
只有当以下两个文件都存在，并且你**没有加 `--overwrite`** 时，才会跳过该 shard：

- audit shard
- manifest shard

### mask
只有当以下文件存在，并且你**没有加 `--overwrite`** 时，才会跳过该 shard：

- final output shard

因此推荐的正式续跑方式就是：

> **保持原命令不变继续运行，不要加 `--overwrite`。**

---

## 8. 常用监控命令

### 8.1 查看 prefilter 日志

```bash
tail -f /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.log
```

### 8.2 查看 mask 日志

```bash
tail -f /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.log
```

### 8.3 查看后台进程

```bash
ps -fp $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.pid)
ps -fp $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.pid)
```

### 8.4 停止后台进程

```bash
kill $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.pid)
kill $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.pid)
```

---

## 9. 最终输出里有什么

最终 mask parquet 是与原始输入逐行对齐的。常见字段包括：

- `row_idx`
- `raw_type`
- `canonical_type`
- `instruction`
- `phrases_json`
- `qc_flag`
- `qc_status`
- `diff_iou`
- `diff_precision`
- `diff_recall`
- `area_frac`
- `mask_height`
- `mask_width`
- `mask_sum`
- `mask_png`
- `prefilter_verdict`
- `prefilter_confidence`
- `prefilter_prompt_version`
- `prefilter_model_name`
- `prefilter_run_id`
- `prefilter_reason`
- `filter_decision`
- `filter_reason_codes`
- `filter_mismatch_score`
- `filter_version`

其中 `mask_png` 可以直接解码：

```python
from PIL import Image
import io

mask = Image.open(io.BytesIO(row['mask_png'])).convert('L')
```

对于被 prefilter 拦下来的样本，最终输出仍然保留一条对齐记录，但会写成：

- `canonical_type = PREFILTER_SKIP`
- `qc_flag = PREFILTER_SKIP`

这样后续回查时不会丢失 `row_idx` 对应关系。

---

## 10. 相关文档

如果需要更详细的实现说明、生产统计和可视化示例，请继续看：

- `CRISPEDIT_FILTER_THEN_MASK_PIPELINE.md`

当前仓库 README 只覆盖主线生产流程：

- `crispedit_mllm_prefilter.py`
- `crispedit_mask_dataset_runner.py`
- `crispedit_mask_pipeline.py`
- `CRISPEDIT_FILTER_THEN_MASK_PIPELINE.md`

如果后续再做 add-mask 实验，建议放到单独分支维护，而不是混入当前主线说明。
