# ScaleEdit + CrispEdit 统一 mask 训练集

## 1. 目标与路径

本数据整理步骤将 ScaleEdit 和 CrispEdit 的最终图像编辑 mask 标签做严格筛选，并导出为
字段完全一致、可直接用于训练的自包含 Parquet schema。处理只读取原始数据，不会在
源目录中新增、覆盖或删除文件。

输入：

```text
CrispEdit 原始图像对
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M

CrispEdit mask sidecar（首批和新增 100k 的统一结果，985 shards）
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask

ScaleEdit 原始图像对（100,000 rows）
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-100k

ScaleEdit mask sidecar（当前最新全量运行）
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-100k-vllm-labeled/masks
```

输出：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-CrispEdit-mask-train
├── crispedit/data/*.parquet
├── scaleedit/data/*.parquet
├── audit/crispedit/*.parquet
├── audit/scaleedit/*.parquet
├── audit/shard_reports/{crispedit,scaleedit}/*.json
├── manifest.json
├── schema.json
├── shards.parquet
├── run_config.json
└── _SUCCESS
```

CrispEdit 新增的 100k 已经是 `CrispEdit-2M` 和 `CrispEdit-2M-mask` 中 394 个不重复
shard 的子集，因此导出时不会再次拼接 `CrispEdit-2M-additional-100k-input`，避免重复样本。

## 2. 严格筛选策略

发布口径为 `strict_qc_and_integrity_v1`。一个样本只有同时满足以下条件才会进入 `data`：

1. CrispEdit 的 prefilter 必须是 `prefilter_verdict=PASS` 且
   `filter_decision=keep`；ScaleEdit 当前没有独立 prefilter 字段，不伪造该结论。
2. 最终 mask sidecar 必须存在，`qc_flag` 必须严格等于 `OK`。因此
   `GROUND_FAIL`、`BOX_FALLBACK`、`AR_MISMATCH`、`SEMANTIC_QC`、`EMPTY_MASK` 和
   `PREFILTER_SKIP` 都不会进入训练数据。
3. `mask_sum > 0` 且 `0 < area_frac <= 1`。这会额外排除 CrispEdit 首批 legacy
   background 路线中被旧逻辑误标为 `OK` 的 57 个空 mask。
4. source、edited 和 mask 都必须有有效 bytes，并且可以完整解码。
5. source 与 edited 的原生尺寸可以不同，但 aspect-ratio delta 必须不大于原打标流水线
   使用的 `0.02`，且必须与 sidecar 中的 `ar_delta` 一致；mask 尺寸必须等于 source
   尺寸，也必须与 sidecar 记录的宽、高一致。训练预处理应将 source/edited 分别缩放到同一
   训练分辨率，不能假设其原生宽高完全相等。
6. mask 必须是 8-bit 单通道 PNG，像素值只允许 `{0, 255}`；非零像素数与
   `mask_sum`、`area_frac` 必须一致。
7. `ground_json` 必须是可解析 JSON；样本的 `row_idx`、ScaleEdit `sample_id` 或
   CrispEdit instruction 必须与原始 shard 对齐。
8. `full_image` 样本必须确实是全 255 mask。
9. 文档中已经人工确认的坏例通过 denylist 排除，即使其流水线 `qc_flag=OK`：
   `add_00690.parquet#195`、`remove_00070.parquet#133`、
   `remove_00845.parquet#64`。

这里没有使用统一的 mask 面积上下限来做额外启发式过滤。细小文字/物体的正确 mask 可以非常
小，而 style/background 等合法全图编辑的面积可以等于 1；仅按面积阈值筛选反而会系统性误删
合格任务。未人工审阅、但通过全部结构检查的 `OK` 样本仍然代表“严格自动 QC 通过”，不应被
误解为具有像素级 GT 的人工验收结果。后续新增人工坏例时，追加到
`scripts/unified_mask_dataset_denylist.json` 后重建或另做版本化过滤。

所有拒绝样本会写入 `audit/<dataset>/*.parquet`，包含 `primary_reason` 和
`all_reasons_json`，但不重复保存大图 bytes。`manifest.json` 汇总每个数据源、编辑类型、mask
模式、mask 来源和拒绝原因；`shards.parquet` 记录逐 shard 行数、大小与 SHA-256。

## 3. 统一字段

两个数据源的 `data/*.parquet` 使用相同的 `edit_mask_train_v1` schema。训练最常用的字段是：

| 字段 | 类型 | 含义 |
|---|---|---|
| `sample_id` | string | 带 `crispedit:` / `scaleedit:` 命名空间的全局 ID |
| `source_dataset` | string | `crispedit` 或 `scaleedit` |
| `edit_type` | string | 统一后的训练任务类别 |
| `raw_edit_type` | string | 原数据类别，便于回溯 |
| `instruction` | string | 训练使用的最终指令；ScaleEdit 使用修正后的 `final_instruction` |
| `original_instruction` | string | 原始指令；CrispEdit 与 `instruction` 相同 |
| `source_image` | binary | 原始 source 编码 bytes，不做有损重编码 |
| `edited_image` | binary | 原始 edited 编码 bytes，不做有损重编码 |
| `mask_png` | binary | 8-bit 单通道 PNG，255=编辑、0=保留 |
| `source_width`, `source_height` | int32 | source 尺寸 |
| `edited_width`, `edited_height` | int32 | edited 原生尺寸，可能与 source 略有不同 |
| `source_edited_aspect_ratio_delta` | float64 | 两张图的宽高比相对差异，严格不大于 0.02 |
| `mask_area`, `mask_area_fraction` | int64 / float64 | mask 像素数和面积占比 |
| `mask_coordinate_space` | string | 固定为 `source_image` |
| `mask_mode` | string | `regions` / `protect_foreground` / `full_image` |
| `mask_source` | string | SAM/box 融合路线 |
| `quality_status` | string | 发布数据固定为 `strict_pass` |

完整 schema 还保留 source shard/row、grounding 结果、instance RLE、模型与 prompt 版本、
CrispEdit prefilter 字段及数据源特有 metadata，见输出目录中的 `schema.json`。

CrispEdit 的七个粗粒度类别映射如下：

| CrispEdit | 统一 `edit_type` |
|---|---|
| `add` | `object_addition` |
| `remove` | `object_removal` |
| `replace` | `object_replacement` |
| `color` | `color_change` |
| `motion change` | `action_editing` |
| `background change` | `background_replacement` |
| `style` | `style_transfer` |

ScaleEdit 直接保留其 `final_task` 作为统一 `edit_type`，原始 `edit_task` 放在
`raw_edit_type`。

## 4. 构建与断点续跑

```bash
cd /opt/tiger/tanyue/sam3-crispedit

.venv-scaleedit-vllm/bin/python scripts/build_unified_mask_dataset.py \
  --workers 16 \
  --output-root \
  /mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-CrispEdit-mask-train
```

导出以 source shard 为原子单元，先写临时文件再原子 rename。中断后使用完全相同的参数加
`--resume`；只有同时存在 data、reject audit 和 `complete=true` shard report 的 shard
才会跳过。脚本不会因为 `--resume` 修改任何输入目录。

开发时可用 `--limit-shards 1` 对两个数据源各跑一个真实 shard。当前真实 smoke test 输出在：

```text
/tmp/codex-tanyue-unified-mask-smoke-20260913
```

该 smoke test 共读取 506 行，严格保留 357 行：CrispEdit 122/256，ScaleEdit
235/250；三张图解码、schema、稀疏 ScaleEdit row-index join 和 rejection audit 均通过。

另用 `motion change_00000.parquet` 验证了原生尺寸略有差异的样本：source 为
1820×1024、edited 为 1813×1024，26 个 prefilter/QC 合格样本全部保留，230 个 prefilter
未通过样本全部拒绝。

## 5. 当前全量结果

2026-09-13 使用 16 个 CPU worker 完成全量构建。最终 1337/1337 个 source shard 均有
data、rejection audit 和 shard report，没有临时文件残留：

| 数据源 | 原始行 | 严格保留 | 拒绝 | 保留率 |
|---|---:|---:|---:|---:|
| CrispEdit | 250,421 | 68,496 | 181,925 | 27.352% |
| ScaleEdit | 100,000 | 98,849 | 1,151 | 98.849% |
| **合计** | **350,421** | **167,345** | **183,076** | **47.756%** |

CrispEdit 的拒绝原因为：prefilter 非 PASS 181,641；`GROUND_FAIL` 174；
`BOX_FALLBACK` 47；`AR_MISMATCH` 1；legacy 空 mask 57；人工 denylist 3；二次解码后
source/edited 宽高比差异超过 0.02 的样本 2。ScaleEdit 的拒绝原因为：源图损坏而没有 mask
记录 34；`SEMANTIC_QC` 790；`GROUND_FAIL` 212；`BOX_FALLBACK` 95；
`AR_MISMATCH` 12；`EMPTY_MASK` 4；二次宽高比检查不通过 4。

最终全局校验结果：schema 一致、167,345 个 `sample_id` 全部唯一、所有 accepted 行均为
`quality_status=strict_pass` 且 `qc_flag=OK`。详细计数以输出目录的 `manifest.json` 为准，
训练前还应确认 `_SUCCESS` 存在。

## 6. 训练读取

```python
import pyarrow.dataset as ds

root = "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-CrispEdit-mask-train"
dataset = ds.dataset([
    ds.dataset(f"{root}/crispedit/data", format="parquet"),
    ds.dataset(f"{root}/scaleedit/data", format="parquet"),
])

columns = [
    "sample_id",
    "edit_type",
    "instruction",
    "source_image",
    "edited_image",
    "mask_png",
]
for batch in dataset.to_batches(columns=columns, batch_size=32):
    # Decode image bytes and build the training batch here.
    pass
```

也可以只读取某一个子数据集，或通过 `edit_type` 做 Arrow predicate pushdown。训练前应检查
根目录 `_SUCCESS` 存在，并使用 `manifest.json` 中的最终行数作为数据加载计数基准。
