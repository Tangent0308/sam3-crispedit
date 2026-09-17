# SAMTok 派生细粒度编辑数据流水线

本文档描述 `samtok-derived-edit-labeling` 分支当前实现的数据策略、指令设计、
运行方式、质量控制和实测结果。目标不是重新生成原图，而是复用 SAMTok
GRES-8k/VER-4k 的原图与实例 mask，构造具有定位难度的单区域图像编辑对。

## 1. 目标与约束

每条训练 case 只编辑一个 region；同一 source 若有多个 mask，则一图多用，
每个原始 region 独立生成一条 case。这样既保留同图中其他实例作为干扰项，
也避免一次编辑 5 个区域带来的任务纠缠。当前实现遵循以下硬约束：

- 仅排除答案为 `No target` 或没有有效 mask 的行，不做生产级候选过滤。
- source image 只物化一次，通过 `source_image` 被多条 case 共享。
- 选中 source 的所有 mask 必须各出现一次，缺失或重复会直接报错。
- 每条 case 恰好一个 region，不超过用户要求的双 region 上限。
- 直接复用 SAMTok COCO RLE；不重新运行 SAM/SAM2，不改变实例语义。
- mask 随 Qwen 一百万像素画布做最近邻缩放，再重新编码为 COCO RLE。
- 输出文件名对应 edit case，输入文件名通过独立的 `source_image` 字段解析。

## 2. 为什么具有细粒度定位难度

当前 pilot 专门采样 `num_masks == 2` 的 source。两个标注 region 在同一原图内，
经常对应同类多实例、关系中的两个实体、局部部件或互相遮挡的对象。每个 region
分别编辑时，另一个 region 天然成为需要保留的干扰项。

为了同时覆盖小物体与复杂的大目标，GRES 和 VER 各自按两个 mask 中较小者的
面积比例排序，并划分为 5 个等频层。每层等量抽取 source，而不是只选最小物体：

- 第 0 层：最小、最局部，重点考验小目标定位和编辑可见性。
- 第 1--3 层：中小至中大目标，覆盖遮挡、关系、拥挤和相似实例选择。
- 第 4 层：较大目标，用于验证身份、姿态和场景保持，而非仅测试像素级小区域。

面积分层只是 pilot 的覆盖策略，不是严格的数据准入条件。正式扩量时可以将
小面积层、同类多实例和遮挡关系提高采样权重，但不应把面积当作质量标签。

## 3. 四类编辑任务

100-case pilot 对四类任务严格均衡，每类 25 条。类型分配由固定 seed 和 source
顺序决定，同一 source 的两个 mask 会得到相邻但不同的类型。

### 3.1 Add

给目标增加一个小而清晰、语义合理的附属物或局部细节，例如领结、丝带、贴纸、
灯、徽章或帽子。约束为只给指定实例添加，不生成目标本身的第二个副本；小目标
优先选择高对比、编辑后仍可观察的变化。

### 3.2 Remove

完整移除指定实例，并自然补全被遮挡背景。必须保留同类别的其他实例和所有无关
场景内容。这类任务重点检验在拥挤或遮挡场景中是否删除了正确对象。

### 3.3 Replace

将指定实例完整替换为一个姿态、位置和尺度相近但视觉上不同、且符合场景语义的
对象。只允许生成一个替代对象，避免数量变化；同类非目标实例必须保留。

### 3.4 Attribute

只改变目标一个明显的局部属性，如颜色、材质或纹理，同时保持身份、几何、姿态、
数量和周围内容。这类任务通常是最细粒度的局部变化，也最容易出现变化过弱的问题。

## 4. 流水线阶段

```text
SAMTok parquet
  -> positive-only index (remove No target)
  -> two-mask, subset-balanced, area-stratified sampling
  -> Qwen3-VL regional instruction generation
  -> instruction validation/retry/fallback
  -> source/mask/crop/overlay materialization
  -> 8-card persistent Qwen-Image-Edit + MIRAGE composition
  -> deterministic pixel-locality metrics
  -> batched Qwen3-VL vLLM visual audit
  -> manual review of every case
  -> paginated contact sheets + HTML gallery
```

### 4.1 Positive-only index

`prepare_samtok_data.py` 只扫描轻量字段 `source/problem/answer/masks`，生成
`positive_rows.jsonl`。原始 12,337 行中保留 7,671 行，剔除 4,666 条 GRES
`No target`。索引保存 parquet 行号、子集、原始指代问题/答案和所有 RLE，暂不复制
内嵌图片。

### 4.2 自动采样与指令生成

`generate_samtok_plan.py` 以固定 seed 采样，并给 Qwen3-VL-8B-Instruct 同时提供：

1. 干净 source；
2. 仅当前 region 红色高亮的 source；
3. SAMTok 原始 referring question/answer；
4. 当前必须生成的 edit type 及类型专属约束。

模型返回三个字段：

- `refer_object`：靠位置、外观或关系区分同类实例的视觉指代表达；
- `editing_instruction`：供全图编辑分支使用的完整指令；
- `new_instruction`：供 MIRAGE region/crop 分支使用的短指令。

输出必须是 ASCII English JSON。校验器拒绝空字段、过短指令及 `mask`、`overlay`、
`marked`、`bbox`、`coordinate` 等标注泄漏词；最多重试三轮。少量模型持续复制
模板时，确定性 fallback 只复用模型已给出的具体 `refer_object` 与短编辑，不创造
新语义，并补充“仅改变该实例、保留其他内容”的约束。成功重跑会清除旧失败清单。

100-case 人工审查后又增加了三项保护：明确告知模型红色只来自 overlay、不得把它
误认为 source 属性；add 的新内容不得已存在；replace 必须改变身份/类别/型号而非
只改颜色或衣服。校验器还会拒绝泛化的“add a plausible accessory”模板和与任务
类型动作词不一致的指令。它们作用于后续新计划，本节实测的 100 条没有追溯重生成。

### 4.3 数据物化

`prepare_samtok_data.py` 按计划从 parquet 解码 50 张 source，统一到与 Qwen 编辑器
一致的约一百万像素、边长为 32 倍数的画布。每个 region 生成：

- 原始 COCO RLE 的画布版 RLE；
- 二值 mask PNG；
- 红色 target overlay；
- MIRAGE crop instruction 及 bbox；
- 含原始问题/答案、hash、尺寸和面积的 provenance。

核心 `annotations.jsonl` 结构为：

```json
{
  "image": "000_gres_r296_m0_auto_auto_add.png",
  "source_image": "source_gres_r296.png",
  "editing_instruction": "Add a small brass bell ...",
  "refer_object": ["the red steam locomotive on the platform"],
  "mask": [{"size": [896, 1184], "counts": "..."}],
  "task_type": "add",
  "source_subset": "gres",
  "parquet_row_index": 296,
  "mask_index": 0
}
```

### 4.4 Qwen/MIRAGE 编辑

`run_qwen_edit_pool.py` 在每张 GPU 上保留一个 Qwen-Image-Edit-2511 worker，worker
从共享队列动态领取 case，避免每条样本重复加载模型，也减少不同画幅导致的长尾。
每条 case 都重新创建相同 seed 的 generator，所以动态调度不会改变单条输出。

MIRAGE 使用 source 全图分支和基于 mask/crop 的区域分支，在扩散 latent 中合成，
当前关键参数为 `patch_ratio=0.2`、40 steps、`true_cfg_scale=4`、
`guidance_scale=1`、seed 0。8 个 worker 仅改变并行调度，不改变生成算法或参数。

### 4.5 自动审核

`audit_edit_pairs.py` 对每条 case 计算 mask 内、mask 膨胀保护区外和全图的平均绝对
像素差及变化像素比例。它同时把 source、红色 mask overlay、edited 三张图和指令
送入 Qwen3-VL-8B，检查：

- 请求的变化是否可见；
- 是否命中正确实例；
- 同类非目标是否保留；
- 背景、构图和无关物体是否基本保持；
- 是否存在严重伪影或广泛场景重绘。

默认使用 vLLM continuous batching，temperature 为 0，并以输入内容 hash 支持安全
resume。vLLM 只加速审核推理，不改变审核模型和 prompt。

### 4.6 人工逐条审核

MLLM 审核不能替代人工检查。本 pilot 对 100/100 条逐一查看 source、target mask、
edited 和绝对差异图，并记录 `pass/review/fail`：

- `pass`：变化清晰、命中正确 region、其他实例和背景保持、无明显伪影；
- `review`：目标大概率正确，但变化过弱/部分完成，或有中等漂移与歧义；
- `fail`：未编辑或编辑错误实例、非目标也变化、目标严重畸变、场景大范围损坏。

另记录该 case 是否体现所需定位难度（同类实例选择、小目标、局部部件、拥挤或遮挡）。
人工结果写入 `manual_review.jsonl`，分页 gallery 同时显示自动与人工结论。

## 5. 100-case pilot 的精确运行方式

```bash
export DATA_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_100_v1
export PARQUET=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet
export PYTHON=/opt/tiger/tanyue/.venvs/mirage_official/bin/python
```

生成 50-source/100-region 均衡计划：

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON synthesis_pipeline/generate_samtok_plan.py \
  --parquet "$PARQUET" \
  --positive-index /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/positive_rows.jsonl \
  --output-dir "$DATA_ROOT/planning" \
  --num-cases 100 --seed 20260917 --batch-size 16 \
  --vlm qwen8b-vllm --vlm-device cuda:0 --vlm-dtype bf16 \
  --max-new-tokens 384
```

物化输入资产：

```bash
$PYTHON synthesis_pipeline/prepare_samtok_data.py \
  --parquet "$PARQUET" --output-dir "$DATA_ROOT" \
  --plan-jsonl "$DATA_ROOT/planning/generated_plan.jsonl"
```

8 卡编辑：

```bash
$PYTHON synthesis_pipeline/run_qwen_edit_pool.py \
  --image-root "$DATA_ROOT/sources" \
  --instruction-jsonl "$DATA_ROOT/annotations.jsonl" \
  --crop-dir "$DATA_ROOT/crops" \
  --results-full-dir "$DATA_ROOT/edited" \
  --model-id Qwen/Qwen-Image-Edit-2511 \
  --gpus 0,1,2,3,4,5,6,7 --python "$PYTHON" \
  --dtype bf16 --cpu-offload none --patch-ratio 0.2 \
  --num-steps 40 --true-cfg-scale 4 --guidance-scale 1 --seed 0
```

vLLM 审核和可视化：

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON synthesis_pipeline/audit_edit_pairs.py \
  --annotations-jsonl "$DATA_ROOT/annotations.jsonl" \
  --source-dir "$DATA_ROOT/sources" --edited-dir "$DATA_ROOT/edited" \
  --out-dir "$DATA_ROOT/audit_vllm" --batch-size 16 \
  --vlm qwen8b-vllm --vlm-device cuda:0 --vlm-dtype bf16 \
  --max-new-tokens 256 --no-resume

# 人工看完每条 case 后，将结论写入 decisions JSON，再物化并强制校验 100% 覆盖。
python3 synthesis_pipeline/materialize_manual_review.py \
  --annotations-jsonl "$DATA_ROOT/annotations.jsonl" \
  --decisions-json "$DATA_ROOT/manual_review_decisions.json" \
  --output-jsonl "$DATA_ROOT/manual_review.jsonl"

python3 synthesis_pipeline/summarize_manual_review.py \
  --annotations-jsonl "$DATA_ROOT/annotations.jsonl" \
  --manual-review-jsonl "$DATA_ROOT/manual_review.jsonl" \
  --output-json "$DATA_ROOT/manual_review_summary.json"

$PYTHON synthesis_pipeline/build_pilot_gallery.py \
  --annotations-jsonl "$DATA_ROOT/annotations.jsonl" \
  --source-dir "$DATA_ROOT/sources" --overlay-dir "$DATA_ROOT/overlays" \
  --edited-dir "$DATA_ROOT/edited" \
  --audit-jsonl "$DATA_ROOT/audit_vllm/edit_audit.jsonl" \
  --manual-review-jsonl "$DATA_ROOT/manual_review.jsonl" \
  --out-dir "$DATA_ROOT/gallery" --page-size 10
```

对长期任务可将上述命令放入 tmux；恢复时编辑池会跳过已经存在的 output，审核则按
内容 fingerprint 复用未变化结果。不要在改变模型、prompt 或图片后手动伪造 resume。

## 6. 输出目录

```text
pilot_100_v1/
  positive_rows.jsonl
  annotations.jsonl
  provenance.jsonl
  prepare_run.json
  planning/
    sampled_source_rows.jsonl
    generated_plan.jsonl
    instruction_responses.jsonl
    instruction_summary.json
    instruction_sources/
    instruction_overlays/
  sources/                 # 50 shared source images
  masks/                   # 100 binary masks
  overlays/                # 100 target overlays
  crops/crop_instruction.jsonl
  edited/                  # 100 generated edit outputs
  edited.worker_logs/      # one log per GPU worker
  audit_vllm/
    edit_audit.jsonl
    summary.json
  manual_review.jsonl
  manual_review_summary.json
  gallery/
    cases/                 # 100 full-resolution comparison rows
    contact_sheet_000_009.jpg
    ...
    contact_sheet_090_099.jpg
    index.html
```

## 7. 实测结果

<!-- PILOT_RESULTS_START -->
测试日期为 2026-09-17，硬件为 8 张 H100 80GB（编辑）和 1 张 H100
（指令生成/审核）。最终产物位于：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/
  SAMTok_Derived_Edit_Labeling/pilot_100_v1
```

### 7.1 数据覆盖

- 50 张 source：GRES 25、VER 25；每张恰好有两个 mask。
- 一图两用后得到 100 个独立单 region case，所有 source mask 覆盖一次且仅一次。
- add/remove/replace/attribute 各 25 条。
- 5 个面积层各 10 张 source，即各 20 个 region case。
- 100 条 `editing_instruction` 全部唯一，且没有 `mask/overlay/marked/bbox`
  等标注泄漏词。
- 最终 50 source、100 mask、100 overlay、100 edited 均存在；100/100 图片可解码，
  source/overlay/edited 的尺寸逐条一致。

mask 面积比例分布如下：

| min | P25 | median | P75 | max |
|---:|---:|---:|---:|---:|
| 0.022% | 0.931% | 4.800% | 11.865% | 86.954% |

97/100 条被人工确认具有至少一种困难定位因素：同类多实例选择、小目标、局部部件、
拥挤/遮挡或复杂背景中的特定 region。3 条不满足者主要是宽泛背景区域或占据大部分
画面的单一主体。这说明采样目标基本达到，但面积层 4 不能直接等同于困难样本。

### 7.2 运行速度

| 阶段 | 冷启动/加载 | 推理或总墙钟 | 吞吐 |
|---|---:|---:|---:|
| Positive-only index | -- | 0.59 s | 12,337 行 |
| Qwen3-VL 指令生成 | 41.15 s | 33.97 s inference；100.85 s wall | 176.63 case/min inference |
| 50 source / 100 region 物化 | -- | 114.46 s | 0.874 case/s |
| 8 卡 Qwen/MIRAGE 编辑 | 含在总时间中 | 989.93 s | 6.06 case/min |
| Qwen3-VL vLLM 审核 | 41.52 s | 40.00 s inference；105.80 s wall | 150.01 case/min inference |

编辑 worker 的单 case 推理耗时为 62.65--81.22 秒，中位数 75.10 秒、均值
73.47 秒。按修复后的代码从头执行，自动阶段（指令、物化、编辑、审核）合计约
1,311 秒，即 21 分 51 秒；人工逐条审核不计入该数值。

首次完整性检查发现同一 source 的 2 条输出为 `864x1216`，而预处理资产为
`864x1248`。原因是 Qwen 的一百万像素 32 倍数取整公式在少数长宽比上第一次
取整后不是固定点，模型内部会再缩放一次。`qwen_canvas_size` 已改为迭代至固定点，
相关 source/mask/bbox 被重新物化并只重生成了这 2 条；旧结果保存在
`rejected_dimension_mismatch/`。修复后全量尺寸检查为 0 错误。该纠错额外花费的
114.46 秒重新物化和 104.56 秒两卡重生成不计入上表的正常单次流水线耗时。

### 7.3 100/100 人工质量审查

最终人工结论为 72 pass、7 review、21 fail。`review` 表示视觉结果可用，但需要
修正文案、重分类或接受部分完成；将其计入候选时可用率为 79%。

| 类型 | pass | review | fail | pass rate | pass + review |
|---|---:|---:|---:|---:|---:|
| add | 16 | 0 | 9 | 64% | 64% |
| remove | 19 | 1 | 5 | 76% | 80% |
| replace | 17 | 5 | 3 | 68% | 88% |
| attribute | 20 | 1 | 4 | 80% | 84% |
| **总计** | **72** | **7** | **21** | **72%** | **79%** |

按来源看，GRES 为 38 pass / 4 review / 8 fail（76% pass），VER 为
34 pass / 3 review / 13 fail（68% pass）。VER 的关系/局部 mask 更细、更难，
但指代语义与 mask 不一致的风险也更高。

| 面积层 | pass | review | fail | pass rate |
|---|---:|---:|---:|---:|
| 0（最小） | 17 | 1 | 2 | 85% |
| 1 | 13 | 1 | 6 | 65% |
| 2 | 15 | 1 | 4 | 75% |
| 3 | 13 | 0 | 7 | 65% |
| 4（最大） | 14 | 4 | 2 | 70% |

本次小目标层反而最好，说明“小”本身不是主要失败原因；更重要的是指令是否与
source/mask 一致，以及任务能否在目标空间尺度内清晰呈现。

人工记录的主要问题为：

- edit missing：9 条；请求变化不存在或弱到不可判定；
- prompt/source mismatch：5 条；例如 prompt 错称原物体已是红色，导致模型先大幅
  改色再添加小配件；
- partial edit / wrong operation：各 4 条；例如 remove 退化为改色，replace 只改局部；
- collateral edit / prompt-mask mismatch：各 3 条；包括误改非目标小羊、指令说镜柱
  而 mask 实际对应前景花坛；
- generic instruction / taxonomy mismatch：各 2 条；模板 fallback 不够具体，或
  `replace` 标签对应实际 attribute 指令。

### 7.4 vLLM 审核与人工审核的差异

自动审核给出 97 pass / 3 fail；3 条自动 fail 都被人工确认，因此 fail precision
为 100%。但人工最终发现 21 条 fail，自动审核只命中 3 条，fail recall 仅 14.3%。
在 97 条自动 pass 中，人工判为 72 pass、7 review、18 fail。

结论是 vLLM 很适合做快速、低成本的明显失败预筛，但当前 prompt/模型过于宽松，
不能直接作为生产数据的最终准入器。特别需要新增确定性检查：指令与 source/mask
语义一致性、add 前目标属性是否已存在、replace 是否退化为 attribute，以及
非目标实例变化检测。

本次提交已经先落地其中可由文本规则可靠处理的部分：红色 overlay 防误认、add
已存在属性提示、replace 身份约束、泛化 add 模板拒绝和任务动作词校验。pilot 中
可被新文本校验直接拦下的已有 3 条；其余视觉/语义一致性问题仍需额外的 source-mask
审核模型或更强的最终过滤器。

### 7.5 可视化与逐条结论

- `gallery/contact_sheet_000_009.jpg` 至 `contact_sheet_090_099.jpg`：10 张分页对比图；
- `gallery/cases/`：100 张单 case 大图；
- `gallery/index.html`：可滚动 HTML 索引；
- `manual_review.jsonl`：100 条逐 case 的人工布尔指标、结论、问题标签和原因；
- `manual_review_summary.json`：按任务和数据子集聚合的统计；
- `audit_vllm/edit_audit.jsonl`：自动审核原始结构化结果与像素 locality metrics。

总体判断：当前采样设计很好地覆盖了所需的细粒度/困难定位场景，且最终通过的
72 条中包含大量同类多实例、小物体、遮挡和局部编辑；但原始出图不能不经筛选直接
入训。正式扩量前应优先修复指令-source-mask 一致性和 add/replace 类型约束，并把
人工发现的失败模式加入自动审核。按本次严格标准，保留 pass、隔离 review、丢弃或
重生成 fail 是更稳妥的数据策略。
<!-- PILOT_RESULTS_END -->

## 8. 已知限制与扩量建议

- “有两个 mask”只保证有 region 干扰，不保证两个对象一定同类；正式生产可增加
  同类实例/关系型 referring 的软评分，但不需要把多样性做成严格去重硬门槛。
- 极小目标的 add/attribute 可能低于模型可稳定呈现的空间尺度，应基于实测失败率设置
  类型与面积的联合采样权重，而不是简单删除所有小目标。
- remove 对背景补全最敏感，replace 对身份与数量保持最敏感；建议按类型维护独立阈值。
- Qwen3-VL 审核可能漏掉细微的错实例或背景漂移。生产流程应保留人工抽检，并持续用
  人工结论校准自动审核，而不是仅以 VLM 的 `pass` 作为准入标准。
- 当前单 case 固定 seed 保证可复现，但不提供同一指令的多随机候选。若以后做 best-of-N，
  必须在 provenance 中记录每个 seed 和选择规则。
