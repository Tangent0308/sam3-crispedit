# ScaleEdit mask labeling

本分支实现 ScaleEdit 图像编辑区域打标：Qwen3.5 先通过 planner → bbox locator 两阶段确定
编辑对象和位置，SAM3 再生成 source 坐标系下的最终二值 mask。Qwen 推理由 vLLM 批处理加速，
普通局部样本调用 MLLM 2 次，全图编辑调用 1 次；仅在解析失败时做一次带错误反馈的纠错重试。
第二轮只接收精简候选描述并使用 Qwen 原生 `bbox_2d` JSON；损坏图片按行记录并跳过。

完整方法、数据格式、mask 后处理、生产数据路径以及当前 66-case 回归结果见
[ScaleEdit 详细文档](docs/SCALEEDIT_MASK.md)。

## 快速开始

要求 CUDA GPU、本地 Qwen3.5-35B-A3B 模型和 SAM3 checkpoint。以下脚本从零创建一个同时支持
Qwen/vLLM grounding、SAM3 mask 和校验的 CUDA 12.9 环境；若目标路径已存在会直接退出，不会复用。

```bash
cd /opt/tiger/tanyue/sam3-crispedit

# Qwen/vLLM grounding: fresh Python 3.12 + CUDA 12.9 environment
bash scripts/setup_scaleedit_vllm_env.sh
```

设置本次运行路径：

```bash
SCALEEDIT_DATASET=/path/to/scaleedit-parquet
SCALEEDIT_RESULTS=/path/to/scaleedit-results/current
SCALEEDIT_QWEN=/path/to/Qwen3.5-35B-A3B
SCALEEDIT_SAM3=/path/to/sam3.pt
```

运行 grounding 和 mask：

```bash
.venv-scaleedit-vllm/bin/python -u scaleedit_mllm_grounding.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --output-dir "$SCALEEDIT_RESULTS/grounding" \
  --model-path "$SCALEEDIT_QWEN" \
  --inference-backend vllm \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --batch-size 8 \
  --request-batch-size 4 \
  --planner-max-new-tokens 2048 \
  --locator-max-new-tokens 1024

.venv-scaleedit-vllm/bin/python -u scaleedit_grounded_mask_runner.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --grounding-dir "$SCALEEDIT_RESULTS/grounding" \
  --output-dir "$SCALEEDIT_RESULTS/masks" \
  --checkpoint-path "$SCALEEDIT_SAM3" \
  --devices 0,1,2,3,4,5,6,7
```

校验并生成可视化：

```bash
.venv-scaleedit-vllm/bin/python scripts/validate_scaleedit_masks.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --grounding-dir "$SCALEEDIT_RESULTS/grounding" \
  --mask-dir "$SCALEEDIT_RESULTS/masks" \
  --report-json "$SCALEEDIT_RESULTS/validation.json"

.venv-scaleedit-vllm/bin/python scripts/visualize_scaleedit_masks.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --grounding-dir "$SCALEEDIT_RESULTS/grounding" \
  --mask-dir "$SCALEEDIT_RESULTS/masks" \
  --output-dir "$SCALEEDIT_RESULTS/review-all" \
  --samples-per-task 1000 \
  --rows-per-page 8
```

运行测试：

```bash
.venv-scaleedit-vllm/bin/python -m pytest -q
```
