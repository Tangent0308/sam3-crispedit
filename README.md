# ScaleEdit mask labeling

本分支实现 ScaleEdit 图像编辑区域打标：Qwen3.5 先通过 planner → bbox locator 两阶段确定
编辑对象和位置，SAM3 再生成 source 坐标系下的最终二值 mask。普通局部样本调用 MLLM 2 次，
全图编辑调用 1 次；流程不使用 pixel diff，也没有 crop refinement。

完整方法、数据格式、mask 后处理和当前 200-case 可视化结果见
[ScaleEdit 详细文档](docs/SCALEEDIT_MASK.md)。

## 快速开始

要求 Python 3.11、CUDA GPU、本地 Qwen3.5-35B-A3B 模型和 SAM3 checkpoint。

```bash
cd /opt/tiger/tanyue/sam3-crispedit

bash scripts/setup_env.sh \
  --python-bin python3.11 \
  --qwen-model-path /path/to/Qwen3.5-35B-A3B \
  --sam3-checkpoint-path /path/to/sam3.pt

source .venv-sam3-crispedit/bin/activate
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
python -u scaleedit_mllm_grounding.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --output-dir "$SCALEEDIT_RESULTS/grounding" \
  --model-path "$SCALEEDIT_QWEN" \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --batch-size 8 \
  --request-batch-size 4 \
  --max-new-tokens 1024 \
  --fail-fast

python -u scaleedit_grounded_mask_runner.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --grounding-dir "$SCALEEDIT_RESULTS/grounding" \
  --output-dir "$SCALEEDIT_RESULTS/masks" \
  --checkpoint-path "$SCALEEDIT_SAM3" \
  --devices 0,1,2,3,4,5,6,7 \
  --fail-fast
```

校验并生成可视化：

```bash
python scripts/validate_scaleedit_masks.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --grounding-dir "$SCALEEDIT_RESULTS/grounding" \
  --mask-dir "$SCALEEDIT_RESULTS/masks" \
  --report-json "$SCALEEDIT_RESULTS/validation.json"

python scripts/visualize_scaleedit_masks.py \
  --input-dir "$SCALEEDIT_DATASET" \
  --grounding-dir "$SCALEEDIT_RESULTS/grounding" \
  --mask-dir "$SCALEEDIT_RESULTS/masks" \
  --output-dir "$SCALEEDIT_RESULTS/review-all" \
  --samples-per-task 1000 \
  --rows-per-page 8
```

运行测试：

```bash
python -m pytest -q
```
