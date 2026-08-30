# SAM3 × CrispEdit

本仓库用于 CrispEdit-2M 的三阶段处理。`prefilter_mask_improved` 分支中的下述流程是
当前最终生产方案：

```text
原始 source/target parquet
    ↓
fact prefilter（Qwen3-VL + 代码裁决）
    ↓
keep manifest
    ↓
Qwen3.5 grounding（realized edit → region ref/bbox → small-region crop refinement）
    ↓
SAM3 dual-prompt mask（bbox PVS + phrase PCS + phrase/bbox PCS + region fusion）
```

两部分的详细设计、代码和运行说明见：

- [docs/CRISPEDIT_PREFILTER.md](docs/CRISPEDIT_PREFILTER.md)
- [docs/CRISPEDIT_MASK.md](docs/CRISPEDIT_MASK.md)

## 主要入口

- `crispedit_mllm_prefilter.py`：从原始 parquet 生成 audit 和 keep manifest；
- `crispedit_prefilter_policy.py`：事实归一化与确定性决策；
- `crispedit_mllm_grounding.py`：mask 流程 S1，Qwen3.5 输出 realized edit、region ref
  与 recall-first bbox，并对小区域进行局部高清重定位；
- `crispedit_grounded_mask_runner.py`：mask 流程 S2，bbox/phrase 双提示与 region fusion；
- `crispedit_grounded_mask_pipeline.py`：无 pixel-diff 的单样本 mask 合成逻辑。
- `scripts/export_grounding_outputs.py`：将 Qwen grounding parquet 导出为 JSONL/CSV/Markdown；
- `scripts/build_category_previews.py`：从原图重建按类别高清 mask review 图。

`crispedit_mask_dataset_runner.py` 和 `crispedit_mask_pipeline.py` 保留用于旧流程回归对照，
不再是推荐生产入口。

mask 阶段的最终设计、8 卡运行方式、两组最终评测统计和分类可视化统一见
[docs/CRISPEDIT_MASK.md](docs/CRISPEDIT_MASK.md)。

## 环境

要求 Python 3.11、CUDA GPU、本地 Qwen3-VL-8B、Qwen3.5-35B-A3B 模型和 SAM3 依赖。
默认模型路径：

```text
/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct
/mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B
```

可用环境变量覆盖：

```bash
export CRISPEDIT_QWEN_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export CRISPEDIT_GROUNDING_MODEL_PATH=/path/to/Qwen3.5-35B-A3B
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
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-grounding \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --grounding-mode two-pass \
  --bbox-refinement small

python crispedit_grounded_mask_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --grounding-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-grounding \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask \
  --devices 0,1,2,3,4,5,6,7
```

全量模式下 grounding 和 mask 输出都与原始 shard 逐行对齐。prefilter drop 行保留
`PREFILTER_SKIP` 占位，不调用 Qwen3.5 或 SAM3。

## 测试

```bash
python -m pytest -q
```
