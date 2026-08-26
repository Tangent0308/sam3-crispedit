# SAM3 × CrispEdit

本仓库用于 CrispEdit-2M 的两阶段处理：

```text
原始 source/target parquet
    ↓
fact prefilter（Qwen3-VL + 代码裁决）
    ↓
keep manifest
    ↓
SAM3 mask
```

prefilter 的实现、运行命令、8 卡实验结果和边界说明统一见：

- [docs/PREFILTER.md](docs/PREFILTER.md)

## 主要入口

- `crispedit_mllm_prefilter.py`：从原始 parquet 生成 audit 和 keep manifest；
- `crispedit_prefilter_policy.py`：事实归一化与确定性决策；
- `crispedit_mask_dataset_runner.py`：读取 manifest 并生成 mask；
- `crispedit_mask_pipeline.py`：单样本 SAM3 mask 逻辑。

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

先按 [docs/PREFILTER.md](docs/PREFILTER.md) 运行 8 卡 fact prefilter，再运行 mask：

```bash
python crispedit_mask_dataset_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-manifest \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-mask \
  --devices 0,1,2,3,4,5,6,7 \
  --batch-size 8
```

mask 输出与原始 shard 逐行对齐。prefilter drop 行保留 `PREFILTER_SKIP` 占位，不生成 mask。

## 测试

```bash
python -m pytest -q tests/test_crispedit_prefilter_policy.py
```
