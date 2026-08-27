# SAM3 × CrispEdit

本仓库用于 CrispEdit-2M 的三阶段处理：

```text
原始 source/target parquet
    ↓
fact prefilter（Qwen3-VL + 代码裁决）
    ↓
keep manifest
    ↓
Qwen3.5 grounding（ref + bbox）
    ↓
SAM3 hybrid mask（PVS + bbox-filtered PCS / box 兜底）
```

prefilter 的方法、实现、运行命令、8 卡实验、全量结果和边界统一见：

- [docs/CRISPEDIT_PREFILTER.md](docs/CRISPEDIT_PREFILTER.md)

## 主要入口

- `crispedit_mllm_prefilter.py`：从原始 parquet 生成 audit 和 keep manifest；
- `crispedit_prefilter_policy.py`：事实归一化与确定性决策；
- `crispedit_mask_dataset_runner.py`：读取 manifest 并生成 mask；
- `crispedit_mask_pipeline.py`：单样本 SAM3 mask 逻辑。
- `crispedit_mllm_grounding.py`：新 mask 流程 S1，Qwen3.5 输出 ref + bbox；
- `crispedit_grounded_mask_runner.py`：新 mask 流程 S2，PVS + bbox-filtered PCS → box；
- `crispedit_grounded_mask_pipeline.py`：无 pixel-diff 的单样本 mask 合成逻辑。

新 mask 流程的设计、字段和 8 卡命令见
[docs/MASK_GROUNDING_PIPELINE.md](docs/MASK_GROUNDING_PIPELINE.md)。旧的
`crispedit_mask_*` 入口保留用于回归对照，不再是推荐生产路径。
47 条历史 mask bad case 的 8 卡回归结果见
[docs/MASK_GROUNDING_EVAL_20260827.md](docs/MASK_GROUNDING_EVAL_20260827.md)。

## 环境

要求 Python 3.11、CUDA GPU、本地 Qwen3-VL-8B 模型和 SAM3 依赖。默认模型路径：

```text
/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct
```

可用环境变量覆盖：

```bash
export CRISPEDIT_QWEN_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export CRISPEDIT_SAM3_CHECKPOINT_PATH=/path/to/sam3_checkpoint.pt
```

安装脚本：

```bash
bash scripts/setup_env.sh --python-bin python3.11
```

## 全量运行

先按 [docs/CRISPEDIT_PREFILTER.md](docs/CRISPEDIT_PREFILTER.md) 运行 8 卡 fact prefilter，
再依次运行 grounding 和 mask：

```bash
python crispedit_mllm_grounding.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-prefilter/manifest \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-qwen35-grounding \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2

python crispedit_grounded_mask_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --grounding-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-qwen35-grounding \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-grounded-mask \
  --devices 0,1,2,3,4,5,6,7
```

全量模式下 grounding 和 mask 输出都与原始 shard 逐行对齐。prefilter drop 行保留
`PREFILTER_SKIP` 占位，不调用 Qwen3.5 或 SAM3。

## 测试

```bash
python -m pytest -q
```
