# CrispEdit-2M labeling

本仓库提供 CrispEdit-2M 的完整三阶段打标流程：先过滤无效编辑，再用 MLLM 定位实际
编辑区域，最后用 SAM3 生成 source 坐标系下的 mask。当前生产方法不使用 pixel diff。

```text
raw parquet
  -> fact prefilter (Qwen3-VL + deterministic rules)
  -> keep manifest
  -> two-pass grounding (Qwen3.5)
  -> dual-prompt mask (SAM3)
```

详细设计见 [prefilter 文档](docs/CRISPEDIT_PREFILTER.md) 和
[mask 文档](docs/CRISPEDIT_MASK.md)。

## 代码组织

```text
crispedit/
├── prefilter/
│   ├── runner.py              # Qwen3-VL 推理、调度与 parquet I/O
│   └── policy.py              # 事实归一化和确定性 keep/drop 裁决
├── mask/
│   ├── grounding_runner.py    # Qwen3.5 两轮定位与多卡调度
│   ├── grounding.py           # 类别路由、prompt、bbox 与 JSON 解析
│   ├── runner.py              # SAM3 多卡 shard runner
│   └── pipeline.py            # bbox/phrase 候选与最终 mask 融合
└── legacy/                    # 旧 pixel-diff 流程，仅用于回归对照
```

根目录的三个脚本是稳定的生产入口：

- `crispedit_mllm_prefilter.py`
- `crispedit_mllm_grounding.py`
- `crispedit_grounded_mask_runner.py`

`scripts/` 保存评测、导出和可视化工具，`tests/` 保存单元测试。

## 当前打标方式

1. **Prefilter**：Qwen3-VL 分别提取 source、target 和盲对比事实；代码按编辑类别检查
   change、主体一致性、构图和无关区域保持情况。模型不直接决定 keep/drop，最终采用
   fail-closed 裁决并输出逐行对齐的 audit 与 manifest。
2. **Grounding**：Qwen3.5 第一轮根据两张图和 instruction 描述实际编辑，第二轮输出
   SAM-friendly 短语和 recall-first bbox；小目标会在高清 crop 中复核。background 改为
   审计稳定前景：有前景时分割后取反，无稳定前景时显式生成全图 mask。style 直接使用
   全图 mask。
3. **Mask**：SAM3 同时生成 bbox-only、phrase-only 和 phrase+bbox 候选，再按区域密度、
   空间约束和语义一致性融合。输出 mask 始终映射回 source 坐标系。

Prefilter drop 行保留 `PREFILTER_SKIP` 占位，不调用后续模型；所有输出 parquet 与原始
shard 同名、逐行对齐。

## 运行

要求 Python 3.11、CUDA GPU，以及本地 Qwen3-VL、Qwen3.5 和 SAM3 checkpoint。

### 1. 安装环境

```bash
cd /opt/tiger/tanyue/sam3-prefilter_improved
bash scripts/setup_env.sh \
  --python-bin python3.11 \
  --sam3-checkpoint-path /mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt
source .venv-sam3-crispedit/bin/activate
```

### 2. Prefilter

```bash
python -u crispedit_mllm_prefilter.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --audit-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-prefilter/audit \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-prefilter/manifest \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct \
  --devices 0,1,2,3,4,5,6,7 \
  --batch-size 16 \
  --max-new-tokens 512 \
  --slot-cache-size 20000 \
  --confidence-threshold 0.6 \
  --boundary-review-fraction 0.05 \
  --fail-fast
```

### 3. MLLM grounding

```bash
python -u crispedit_mllm_grounding.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-prefilter/manifest \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-grounding \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --grounding-mode two-pass \
  --background-observation-mode foreground-audit \
  --bbox-refinement small \
  --batch-size 16 \
  --request-batch-size 8 \
  --max-images-per-generate 20 \
  --max-new-tokens 512 \
  --fail-fast
```

### 4. SAM3 mask

```bash
python -u crispedit_grounded_mask_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --grounding-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-grounding \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask \
  --checkpoint-path /mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt \
  --devices 0,1,2,3,4,5,6,7 \
  --preview-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-previews \
  --fail-fast
```

不加 `--overwrite` 时会跳过已完成 shard；策略或 prompt 改变后应使用新输出目录，避免混合
不同版本的结果。运行前可用 `python <入口脚本> --help` 查看抽样和类别过滤参数。

## 验证

```bash
python -m pytest -q
python -m py_compile crispedit/**/*.py crispedit_*.py scripts/*.py
```
