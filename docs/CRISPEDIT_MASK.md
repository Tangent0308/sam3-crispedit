# CrispEdit mask 打标

本文说明 prefilter 后的最终 mask 打标流程。它不使用 pixel diff，而是先由 Qwen3.5 理解
source/result 中实际发生的编辑并给出保守区域框，再由 SAM3 同时使用 bbox 与指代短语
生成候选 mask。

本文所述方法及默认参数已经固化为当前生产方案。历史 one-pass、仅放大 bbox、不同
region fusion 阈值等实验路线不再作为生产入口。

## 流程概览

```text
raw parquet + prefilter manifest
  → Qwen3.5 第一轮：source + result + instruction → realized edit specification
  → Qwen3.5 第二轮：realized edit → region ref + conservative bbox
  → 小区域局部复核：高清 context crop → corrected complete-object bbox
  → grounding parquet
  → SAM3 bbox-only PVS
             + phrase-only PCS
             + phrase+bbox PCS
  → density-aware fusion / nearby tiny-region coverage
  → source-coordinate union mask parquet
```

第一轮不输出坐标。instruction 只作为编辑意图，图片对是事实来源；模型需要明确实际修改
的对象、修改前后外观、空间布局和完整范围。color/material 类型额外提供四组 source/result
匹配局部放大图，并检查 face/head、neck、arms/hands、可见 legs/feet，避免 instruction
只提到手臂时漏掉同方向变化的脸部。

第二轮按“空间编辑区域”而不是按像素点出框。bbox 是 SAM 的 recall-first 搜索范围，必须
完整包含编辑区域并留有安全边距。相邻花朵、穿孔、花瓣、纹身、斑点等小元素使用一个
`aggregate_region` 框，不逐点出框；语义类别不同或空间明显分离的对象仍分别输出。

由于两张完整大图共同输入时，帽子、嘴、手、手持小物等小目标获得的视觉 token
有限，第二轮可能出现语义正确但框偏移，或只框住上/下半部。默认对短边小于
220（`[0,1000]` 坐标）的候选裁出带语境的局部图，放大后让模型独立复核完整边界。
新旧框明显重合、或者一框高度包含另一框时，取带小安全边距的并集，防止局部复核只看到
锤头等显著子部件；两框明显错位时只保留局部复核框，
避免将误定位的鼻子等区域并入嘴部框。若局部输出无法解析，则回退为对原框做较大的
recall-first 扩展。background/style 不使用这一复核，因为 background 的框表示需保护的前景，
不是编辑区域。原框、crop、局部原始输出和最终框都保存在 `bbox_refinement` 字段中便于审计。

每个 grounding item 包含：

```json
{
  "ref": "facial piercings",
  "bbox_2d": [220, 180, 750, 850],
  "region_mode": "aggregate_region",
  "mask_density": "sparse"
}
```

坐标固定为 `[0, 1000]`，与推理 resize 无关。`ref` 必须是可直接给 SAM3 的可见名词
短语；不能使用 change verb、抽象 absence，或把 hand 和 tablet 等不同语义类别混在
同一个短语中。

## SAM3 候选与融合

对每个 region 同时运行三条路径：

1. **bbox-only PVS**：SAM3 visual prompt，返回 multimask；候选的 mask bbox 与 prompt
   bbox IoU 至少为 0.60，且至少 90% mask 位于局部 containment 区域内。
2. **phrase-only PCS**：SAM3 concept prompt；MLLM bbox 用于过滤同类但非编辑实例，至少
   80% mask 像素需要位于 containment 区域内。
3. **phrase+bbox PCS**：同时提供语义短语和 positive bbox，补回 phrase-only 漏掉的
   局部实例。

普通 object 优先空间明确的联合提示。aggregate region 在两路 PCS 互补时取并集；如果
某一路明显退化成包围物、背景或少量低置信 speck，则根据 fill ratio、候选数、置信度和
两路 IoU 拒绝异常候选。

对局部紧凑的 flower/petal/piercing/tattoo/spot 等小元素组，如果 PCS 找到至少 6 个实例
和 6 个小连通分量，则用实际语义 mask 的凸包生成实心连通区域。bbox 超过半图的全局
散布不做凸包；boat、chain 等重复大对象也不会进入该规则，避免跨背景形成巨大多边形。

其它实现细节：

- bbox 默认按自身尺寸外扩 2.5%；只有某一维小于图像短边 5% 时，该维才使用图像短边
  1.5% 的最小 margin，兼顾 tiny earring/finger 与正常 face/limb；
- color 编辑中的明确人体部位会转换为 `exposed human ... skin` 概念提示，由 bbox 锁定
  具体人物，减少整个人或衣服被当成 arms 的情况；
- target mask 映射到 source 坐标后，普通 mask 膨胀短边 1.5%，已连通区域膨胀 0.3%；
- 三条 SAM 路径全部失败时才回退为矩形；最终 parquet 会通过 `qc_flag` 标明 fallback；
- style 直接使用全图 mask；background 先分割需保护的前景，再取反得到背景。

不同编辑类型的路由如下：

| type | grounding | 最终 mask |
|---|---|---|
| add | target 新增区域；若同时有明确移除，也补 source | mapped target ∪ source collateral |
| remove | source 删除区域；若同时有明确新增，也补 target | source ∪ mapped target collateral |
| replace | source 旧物 + target 新物 | source ∪ mapped target |
| color/material | source 中所有实际变色部位 | source regions |
| motion change | source/target 动作部位和直接交互物 | source ∪ mapped target |
| background change | source 中需保护的前景 | NOT(dilated foreground) |
| style | 无需 grounding | full image |

## 代码结构

| 文件 | 作用 |
|---|---|
| `crispedit_grounding.py` | 两轮与小区域复核 prompt、编辑类型路由、bbox 融合、JSON parser 与 region schema |
| `crispedit_mllm_grounding.py` | 8 卡 Qwen3.5 runner、局部细节/框复核图、manifest 对齐和 grounding parquet |
| `crispedit_grounded_mask_pipeline.py` | 单样本 SAM3 三路候选、融合、映射和连通区域逻辑 |
| `crispedit_grounded_mask_runner.py` | 8 卡 SAM3 shard runner、最终 parquet 与逐样本 preview |
| `scripts/export_grounding_outputs.py` | 将模型两轮输出导出为 JSON/JSONL/CSV/Markdown |
| `scripts/build_category_previews.py` | 从原图重建按类别 review 图，避免放大低清 runtime preview |
| `scripts/evaluate_grounded_mask_bad_cases.py` | 小批量输出完整性、QC、来源和面积统计 |

grounding 和最终 mask parquet 都与原始 shard 的 `row_idx` 对齐。最新 prefilter manifest
中的 `prefilter_evidence_schema`、`filter_reason_codes`、`filter_mismatch_score` 等审计字段
会继续传递到最终 mask；drop 行写入 `PREFILTER_SKIP` 占位，不调用 Qwen3.5 或 SAM3。

最终 mask 主要字段包括 `ground_json`、`mask_png`、`instance_masks`（含 COCO RLE）、
`mask_source`、`area_frac`、`qc_flag`、`grounding_status`、模型信息和 prefilter 审计信息。

## 使用方法

### 1. 运行 prefilter

先按 [CRISPEDIT_PREFILTER.md](CRISPEDIT_PREFILTER.md) 生成逐 shard 对齐的 manifest。

### 2. 8 卡生成 realized edit 与 region bbox

Qwen3.5-35B-A3B 使用四个 TP=2 replica：

```bash
python -u crispedit_mllm_grounding.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-fact-prefilter/manifest \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-grounding \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --grounding-mode two-pass \
  --bbox-refinement small \
  --batch-size 1 \
  --request-batch-size 2 \
  --max-new-tokens 512 \
  --fail-fast
```

`--bbox-refinement small` 是默认生产策略；可用 `off` 做旧路线对照，或用 `all` 复核所有
bbox。阈值和 crop 范围可通过 `--bbox-refine-threshold`、`--bbox-refine-min-context`
与 `--bbox-refine-context-scale` 调整。

不加 `--overwrite` 时完整 shard 会被跳过。修改 prompt 或策略后应使用新的输出目录，避免
把不同策略的 parquet 混在一起。

### 3. 8 卡生成 SAM3 mask

```bash
python -u crispedit_grounded_mask_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --grounding-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-grounding \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask \
  --devices 0,1,2,3,4,5,6,7 \
  --preview-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-previews \
  --fail-fast
```

可通过 `CRISPEDIT_SAM3_CHECKPOINT_PATH` 或 `--checkpoint-path` 指定 SAM3 checkpoint。

### 4. 导出可读模型输出和分类预览

```bash
python scripts/export_grounding_outputs.py \
  --grounding-dir /path/to/grounding \
  --output-dir /path/to/model_outputs \
  --write-json-array

python scripts/build_category_previews.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --mask-dir /path/to/masks \
  --selection-file docs_assets/mask_pipeline/eval_selection.json \
  --output-dir /path/to/previews_by_category \
  --panel-width 520 \
  --panel-height 325 \
  --columns 1
```

## 最终实验结果

### 历史 mask 难例：47 条

使用仓库内 [eval_selection.json](../docs_assets/mask_pipeline/eval_selection.json) 的 47 条历史
mask 难例，在 GPU 0–7 上分别运行 Qwen3.5 和 SAM3。该评测专门检查 mask 方法，因此没有
应用 prefilter drop：

- grounding：47 rows，0 runtime error，0 `GROUND_FAIL`，46 `OK` + 1 `PARTIAL_OK`；
- region bbox：113；小区域复核 33 requests / 52 candidates，0 refinement parse failure；
- 第一轮 observation 有 1 条长输出 JSON 截断；第二轮 grounding 和最终 mask 正常完成；
- mask：47/47 `OK`，0 runtime error，0 rectangle fallback；
- 样本级 mask source：PCS 40、PVS 3、connected group 4；
- mask area fraction：min 0.0126、median 0.1070、mean 0.1646、max 0.7510；
- 单元测试与 manifest 集成测试：51 passed。

这组样本没有像素级 GT，因此面积和来源统计只用于发现异常，最终仍需人工检查。当前已知
边界包括：如果第一轮把完整对象错误改写成材质/纹理短语，SAM 可能只分割其轮廓。例如
`remove_00070.parquet row=133` 的完整兔耳被描述为 `tufts of fur`，当前结果主要覆盖耳缘；
这属于 realized-edit 语义错误，不是 bbox 漏框。

### Prefilter keep 均匀抽样：56 条

从最新 prefilter 全量输出的 keep 数据中按 7 个类别各抽 8 条，并使用同一最终流程运行：

- grounding：56 rows，0 runtime error，0 `GROUND_FAIL`，48 `OK` + 8 `STYLE_FULL_IMAGE`；
- region bbox：97；小区域复核 30 requests / 35 candidates，0 refinement parse failure；
- 第一轮 observation 有 5 条长输出 JSON 截断，集中在不依赖局部 grounding 的
  style/background 路线，均未影响最终 mask；
- mask：56/56 `OK`，0 runtime error，0 rectangle fallback；
- 样本级 mask source：PCS 22、PVS 26、style full-image box 8；
- mask area fraction：min 0.0054、median 0.1474、mean 0.3161、max 1.0000。

下面六张图是当前最终流程在 47 条历史难例上的分类可视化。每行依次展示 source 与 MLLM
bbox、target 与 MLLM bbox、映射到 source 的最终 mask overlay，以及二值 mask。

### Add

![add mask review](../docs_assets/mask_pipeline/add.jpg)

### Background

![background mask review](../docs_assets/mask_pipeline/background.jpg)

### Color

![color mask review](../docs_assets/mask_pipeline/color.jpg)

### Motion

![motion mask review](../docs_assets/mask_pipeline/motion.jpg)

### Remove

![remove mask review](../docs_assets/mask_pipeline/remove.jpg)

### Replace

![replace mask review](../docs_assets/mask_pipeline/replace.jpg)
