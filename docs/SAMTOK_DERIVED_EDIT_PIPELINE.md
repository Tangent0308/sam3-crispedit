# SAMTok 派生细粒度编辑数据流水线

本文档描述 `samtok-derived-edit-labeling` 分支当前实现的数据策略、指令设计、
运行方式、质量控制和实测结果。目标不是重新生成原图，而是复用 SAMTok
GRES-8k/VER-4k 的原图与实例 mask，构造具有定位难度的单区域图像编辑对。

## 1. 目标与约束

每条训练 case 只编辑一个 region；同一 source 若有多个 mask，则一图多用，
每个原始 region 独立生成一条 case。这样既保留同图中其他实例作为干扰项，
也避免一次编辑 5 个区域带来的任务纠缠。当前实现遵循以下硬约束：

- positive index 仅排除答案为 `No target` 或没有有效 mask 的行；规划阶段另做
  mask--instruction 兼容性过滤，不把不适合指定 edit type 的 region 送入生成。
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

给目标增加一个小而清晰、语义合理的附属物或局部细节。新增内容必须由模型根据
当前干净图像的场景和目标语义自行决定，prompt 不提供物体清单或具体例子，避免
模型反复生成示例中的有限类别。约束为只给指定实例添加，不生成目标本身的第二个
副本；小目标优先选择高对比、编辑后仍可观察的变化。

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

`generate_samtok_plan.py` 以固定 seed 过采样候选，并给 Qwen3-VL-8B-Instruct
提供三张无标注色视觉输入：单独放大的 clean target cutout、包含 clean context/cutout/
binary mask 的定位 panel，以及 clean source 全图。只有 cutout 定义可编辑对象，全图只
用于场景和同类实例关系。SAMTok 原始 question/answer 往往同时描述两个 mask，曾导致
planner 在两个合法对象间串实例；现在它们只保留在 provenance，不再进入单-mask prompt。

模型除 `refer_object`、`editing_instruction` 和 `new_instruction` 外，还必须返回：

- `masked_content`：只描述当前 mask 内的完整连通内容；
- `edit_unit_status`：`complete_object`、`complete_part` 或 `incomplete`；
- `outside_dependencies` 与 `mask_compatibility`：判断指定 edit type 能否严格在当前
  mask 范围内成立。

完整 instruction 限制为 4--22 words，regional instruction 为 3--16 words；不允许
背景重建配方、默认保持条款、解释和无关对象列表。校验器检查 masked-content、
refer-object 和 instruction 的主体一致性、动作类型、replace 的具体替代身份以及
generic add 模板。remove/replace 只接受完整独立物体；局部部件保留给 add/attribute。
无法在 mask 内自洽完成、格式持续失败或类型退化的候选会标为 incompatible，不会
终止整批任务，也不会用坏 instruction 凑数。

默认先规划目标 case 数的 3 倍，再以整张 two-mask source 为单位筛选；只有两个 region
都兼容才保留该 source。最终仍严格保持 GRES/VER 均衡、一图多用和四类数量均衡。
候选不足会明确提示增大 `--candidate-cases`。

规划和审核 VLM 都不接收红色 overlay。新计划记录
`planning_visual_input=clean_crop_cutout_binary_mask_grounded_v3`。remove 不再通过文字
扩展到 mask 外的骑手、手持物或其他依附内容；若只编辑 mask 会留下不合理依附关系，
该 region/type 应判 incompatible。add 不提供候选物体例子，只要求直接命名适合场景的
具体新增物；replace 必须改变身份、类别或型号，单纯颜色、材质、文字或图案变化属于
attribute。

### 4.3 数据物化

`prepare_samtok_data.py` 按计划从 parquet 解码 50 张 source，统一到与 Qwen 编辑器
一致的约一百万像素、边长为 32 倍数的画布。每个 region 生成：

- 原始 COCO RLE 的画布版 RLE；
- 二值 mask PNG；
- 供人工可视化使用的红色 target overlay；该图不进入规划或自动审核 VLM；
- MIRAGE crop instruction 及 bbox；
- 含原始问题/答案、hash、尺寸和面积的 provenance。

核心 `annotations.jsonl` 结构为：

```json
{
  "image": "000_gres_r296_m0_auto_auto_add.png",
  "source_image": "source_gres_r296.png",
  "editing_instruction": "Add a small brass bell ...",
  "refer_object": ["the steam locomotive with the number 178 on its front"],
  "masked_content": "the steam locomotive",
  "edit_unit_status": "complete_object",
  "outside_dependencies": "none",
  "mask_compatibility": "compatible",
  "mask": [{"size": [896, 1184], "counts": "..."}],
  "task_type": "add",
  "planning_visual_input": "clean_crop_cutout_binary_mask_grounded_v3",
  "source_subset": "gres",
  "parquet_row_index": 296,
  "mask_index": 0
}
```

### 4.4 Qwen/MIRAGE 编辑

`run_qwen_edit_pool.py` 在每张 GPU 上保留一个 Qwen-Image-Edit-2511 worker，worker
从共享队列动态领取 case，避免每条样本重复加载模型，也减少不同画幅导致的长尾。
每条 case 都重新创建相同 seed 的 generator，所以动态调度不会改变单条输出。

MIRAGE 使用 source 全图分支和区域分支，在扩散 latent 中合成。旧实现只把 mask
转成 bbox，矩形内非目标像素也可能被重写。当前 remove/replace/attribute 会把原始
binary mask 下采样到 latent grid，并以两格 feather collar 做精确写回；add 的 mask
表示放置锚点而不是新增物轮廓，因此仍使用 bbox。关键参数为 `patch_ratio=0.2`、
40 steps、`true_cfg_scale=4`、`guidance_scale=1`、seed 0。8 个 worker 仅改变并行调度。

### 4.5 自动审核

`audit_edit_pairs.py` 对每条 case 计算 mask 内、mask 膨胀保护区外和全图的平均绝对
像素差及变化像素比例。审核输入同样不使用彩色 overlay，而是按如下顺序提供三张图：

1. 同尺度并排的紧致 SOURCE/EDITED crop，用细黑白边界标出完全相同的 exact-mask
   footprint，避免把边界外同类实例误当成目标残留；
2. 对齐的 SOURCE crop、binary mask、EDITED crop 和带白色 mask 边界的绝对差异图；
3. 左右并排的 SOURCE/EDITED 全图，用于检查全局保持和正常显示尺度下的可见性。

每条 case 仍只调用一次 Qwen3-VL-8B。prompt 采用 failure-first 结构，先要求模型不
依赖指令中的名词，独立描述 source 与 edited 中实际可见的对象、部件和数量，再填写
五个失败槽：source 描述不一致、编辑未完成、错实例/数量、依附物处理错误、非目标
变化/伪影。任一槽位给出明确证据时，程序会强制判为 `fail`，不能被模型同时输出的
`pass` 覆盖。审核只有 `pass/fail` 两类：无法确认、变化太弱或有明显歧义都判 fail，
大体完成且自然的结果直接 pass，不再输出 review。

四种编辑类型使用不同标准：

- add：新增内容必须在 source 不存在，在 edited 中可明确识别，并且不能替换或破坏
  原目标；
- remove：要求当前 mask 对应实例的全部 masked 可见范围和残留轮廓完整消失；mask 外
  同类实例应保留，不能误作目标残留；有 mask 外依附物的候选在规划阶段直接剔除；
- replace：旧身份必须完整消失，新身份必须清楚可辨；仅改色、弱纹理变化或新旧混合
  均失败；
- attribute：指定属性必须覆盖预期部位且清楚可见，目标身份、几何、姿态、数量以及
  非目标同类实例保持不变。

VLM 结论之后不增加模型调用。像素指标一般只作为诊断 warning，不再因为保护区外
变化较广或 add 变化比例较小把明显成功结果降级。完整 remove 的 mask 内变化比例低于
50% 时直接 fail，用于拦截“人物仍在、只改变面貌/衣服”的漏判。反向只处理一个窄而
可测的矛盾：VLM 唯一失败理由明确声称“完全没改/目标仍完整可见”，但 exact mask 内
超过 95% 像素已变化且保护区外变化不超过 5% 时，取消这一条错误 completion failure；
任何残留、伪影、数量或依附物 failure 都不能被该规则覆盖。

默认使用 vLLM continuous batching，temperature 为 0，并以输入内容 hash 支持安全
resume。vLLM 只改变推理调度，不改变审核模型；单条 case 始终只有一次审核调用。

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
  --num-cases 100 --candidate-cases 300 --seed 20260917 --batch-size 16 \
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
  --max-new-tokens 384 --no-resume

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
    instruction_target_panels/ # clean crop + separate binary mask; no colored overlay
  sources/                 # 50 shared source images
  masks/                   # 100 binary masks
  overlays/                # 100 target overlays
  crops/crop_instruction.jsonl
  edited/                  # 100 generated edit outputs
  edited.worker_logs/      # one log per GPU worker
  audit_vllm/
    edit_audit.jsonl
    summary.json
  audit_v7_failure_first/  # 本轮 failure-first 单次调用审核结果
  manual_review.jsonl
  manual_review_summary.json
  gallery/
    cases/                 # 100 full-resolution comparison rows
    contact_sheet_000_009.jpg
    ...
    contact_sheet_090_099.jpg
    index.html
  gallery_v7_audit/       # v7 audit + 第二轮人工复核后的可视化
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

结合用户对可视化结果的第二轮复核，人工结论更新为 66 pass、7 review、27 fail。
`review` 表示视觉结果可用，但需要修正文案、重分类或接受部分完成；将其计入候选时
可用率为 73%。

| 类型 | pass | review | fail | pass rate | pass + review |
|---|---:|---:|---:|---:|---:|
| add | 15 | 0 | 10 | 60% | 60% |
| remove | 15 | 1 | 9 | 60% | 64% |
| replace | 16 | 5 | 4 | 64% | 84% |
| attribute | 20 | 1 | 4 | 80% | 84% |
| **总计** | **66** | **7** | **27** | **66%** | **73%** |

按来源看，GRES 为 34 pass / 4 review / 12 fail（68% pass），VER 为
32 pass / 3 review / 15 fail（64% pass）。VER 的关系/局部 mask 更细、更难，
但指代语义与 mask 不一致的风险也更高。

| 面积层 | pass | review | fail | pass rate |
|---|---:|---:|---:|---:|
| 0（最小） | 15 | 1 | 4 | 75% |
| 1 | 13 | 1 | 6 | 65% |
| 2 | 13 | 1 | 6 | 65% |
| 3 | 12 | 0 | 8 | 60% |
| 4（最大） | 13 | 4 | 3 | 65% |

本次小目标层反而最好，说明“小”本身不是主要失败原因；更重要的是指令是否与
source/mask 一致，以及任务能否在目标空间尺度内清晰呈现。

人工记录的主要问题为：

- edit missing：10 条；请求变化不存在或弱到不可判定；
- prompt/source mismatch：8 条；例如 prompt 错称原物体已是红色，导致模型先大幅
  改色再添加小配件；
- partial edit：7 条；包括实例残留、多实例少删和依附物处理不完整；
- wrong operation / broad target change：各 4 条；例如 remove 退化为改色、replace
  只改局部，或局部 add 导致整个目标大范围改色；
- collateral edit / prompt-mask mismatch：各 3 条；包括误改非目标小羊、指令说镜柱
  而 mask 实际对应前景花坛；
- generic instruction / taxonomy mismatch：各 2 条；模板 fallback 不够具体，或
  `replace` 标签对应实际 attribute 指令。

### 7.4 vLLM 审核与人工审核的差异

初版自动审核给出 97 pass / 3 fail；3 条自动 fail 都被人工确认，因此 fail precision
为 100%。但复核后人工发现 27 条 fail，自动审核只命中 3 条，fail recall 为 11.1%。
在 97 条自动 pass 中，人工判为 66 pass、7 review、24 fail。

结论是 vLLM 很适合做快速、低成本的明显失败预筛，但初版 prompt/模型过于宽松，
不能直接作为生产数据的最终准入器。特别需要新增确定性检查：指令与 source/mask
语义一致性、add 前目标属性是否已存在、replace 是否退化为 attribute，以及
非目标实例变化检测。

在相同 100 条旧出图上重跑 failure-first v7 审核后，结果为 35 pass / 47 review /
18 fail。47 条 review 是有意设置的保守隔离区，不等价于 47 条失败：其中既包含旧
规划指令可能受红色 overlay 污染的 case，也包含像素变化强度或非目标变化接近边界、
需要人工或更强模型确认的 case。新版审核推理 69.10 秒，吞吐 86.83 case/min；模型
冷启动 40.32 秒，总墙钟 139.44 秒，且明确记录为 100 calls / 100 cases。

用户复查指出的九条 case 在 v7 中均不再自动通过：

| case | v7 | 拦截依据 |
|---|---|---|
| 001 | review | 旧 red-overlay source 描述进入强制复核 |
| 005 | fail | VLM 识别到重叠处仍有长颈鹿身体残留 |
| 025 | review | 伞和包的依附物移除范围无法自动确认 |
| 030 | fail | 只改色而没有生成可辨认的替换对象 |
| 033 | review | 两实例 remove 的区域变化不足，可能少删 |
| 054 | review | replacement 区域变化过弱且保护区外变化偏大 |
| 057 | review | source 红色描述未确认且存在较广区域变化 |
| 058 | fail | source 不存在所谓 red patch，替换也不可见 |
| 092 | review | source 红色墙面描述未获独立确认 |

这组结果说明单靠 8B VLM 仍可能把细微残留误判为完成，但放大对齐 crop、失败证据
优先输出与确定性 review gate 的组合可以防止这些 case 直接进入自动 pass。正式扩量
仍应对 review 抽检或重生成，并持续用人工结论校准阈值。

### 7.5 可视化与逐条结论

- `gallery/contact_sheet_000_009.jpg` 至 `contact_sheet_090_099.jpg`：10 张分页对比图；
- `gallery/cases/`：100 张单 case 大图；
- `gallery/index.html`：可滚动 HTML 索引；
- `manual_review.jsonl`：100 条逐 case 的人工布尔指标、结论、问题标签和原因；
- `manual_review_summary.json`：按任务和数据子集聚合的统计；
- `audit_vllm/edit_audit.jsonl`：初版自动审核原始结构化结果与像素 locality metrics；
- `audit_v7_failure_first/`：本轮 failure-first v7 审核结果与精确速度统计；
- `gallery_v7_audit/index.html`：v7 自动结论和第二轮人工结论的更新版可视化。

总体判断：当前采样设计很好地覆盖了所需的细粒度/困难定位场景，且最终通过的
66 条中包含大量同类多实例、小物体、遮挡和局部编辑；但原始出图不能不经筛选直接
入训。正式扩量前应优先修复指令-source-mask 一致性和 add/replace 类型约束，并把
人工发现的失败模式加入自动审核。按本次严格标准，保留 pass、隔离 review、丢弃或
重生成 fail 是更稳妥的数据策略。

### 7.6 Fresh clean-mask 100-case 复跑（seed 20260918）

修复后的 pipeline 从头生成了另一批 100 条，产物位于：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/
  SAMTok_Derived_Edit_Labeling/pilot_100_v2_cleanmask
```

该批包含 50 张 source（GRES/VER 各 25 张），每张两个 mask 分别形成独立单-region
case；add/remove/replace/attribute 各 25 条，五个 mask 面积分层各 20 条。全部 100 条
annotation 都记录 `planning_visual_input=clean_crop_cutout_binary_v2`，文本预检未发现
red-mask、overlay、panel 方位或 generic-rule 泄漏。

一次成功生产运行的速度如下（不含开发阶段失败重试和人工检查）：

| 阶段 | 时间 | 吞吐 |
|---|---:|---:|
| 指令规划 | 104.72 s wall；40.23 s inference | 149.15 case/min inference |
| source/mask 物化 | 110.99 s | 0.90 case/s |
| 8×H100 Qwen-Image-Edit | 1,008.57 s | 5.95 case/min |
| Qwen3-VL v7 审核 | 142.30 s wall；68.94 s inference | 87.03 case/min inference |
| 自动阶段合计 | 1,366.58 s（22 分 46.6 秒） | — |

编辑输出通过 100/100 文件集合、PNG 解码和 source/edited 尺寸一致性检查。v7 自动
审核为 53 pass / 36 review / 11 fail，严格保持一次 VLM 调用/条。随后对 100/100
逐例人工查看 source、mask、edited 和差分图，结论为 72 pass / 3 review / 25 fail：

| 类型 | pass | review | fail | pass rate |
|---|---:|---:|---:|---:|
| add | 18 | 1 | 6 | 72% |
| remove | 12 | 1 | 12 | 48% |
| replace | 20 | 0 | 5 | 80% |
| attribute | 22 | 1 | 2 | 88% |

这批 100 条都来自有两个已标注 region 的 source，人工均确认具有困难定位背景；通过
样本中可见同类多实例选择、极小目标、局部部件、遮挡和拥挤场景。主要失败仍集中在
remove：目标/依附物残留、错实例以及不自然背景补全。自动审核与人工结论的混淆中，
9 条自动 pass 被人工判 fail，另有 7 条自动 review 被判 fail；同时有 2 条自动 fail
实际上人工判 pass，说明 8B VLM 仍不能代替人工抽检。

最终查看入口为 `gallery_v7_manual/index.html`；`manual_review.jsonl` 覆盖 100/100，
`manual_review_summary.json` 保存上述按类型和子集统计。自动 fail 可直接隔离，自动
review 不应全部丢弃：本批 36 条自动 review 中有 27 条经人工确认可用。

### 7.7 Mask-grounded v3 / exact-edge audit v14 定向回归（2026-09-18）

针对人工指出的 19 个 case 重新检查后，确认主要根因不是单一生图失败，而是三层问题：

1. 原始 SAMTok question/answer 同时描述两个 mask，单-mask planner 会串到另一个实例；
2. 指令把 mask 外依附物纳入编辑范围，而区域分支实际只能可靠处理当前 mask；
3. v7 的像素 warning 把大量明显成功结果降为 review，同时 8B VLM 又可能漏掉“原目标
   仍在但外观变化”的 remove。

修复后对 019/020/021/030/033/040/049/053/081/090 做了三轮规划回归，并实际重生成
其中 7 条。019 从“改 mask 外滑雪板”变为只改倒立者雪服；081 从错误要求移除两名
球员变为只移除 mask 对应的 5 号球员，并完成自然补全；090 的硬黑矩形变为融合更自然
的新标牌。040 的小花瓶也能在目标椅垫区域辨认。021 仍留下牙刷/牙膏残余，049 只是
改变人物外观，说明这两类必须由兼容性/二值审核剔除，不能靠冗长 instruction 掩盖。

7 条回归图位于：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/
  pilot_100_v2_cleanmask/regression_v8/edit_gallery_v14/index.html
```

最终 exact-edge v14 审核得到 5 pass / 2 fail：019、020、040、081、090 pass，021
因牙刷残留 fail，049 因 mask 内仅 24.91% 像素变化而触发完整 remove 硬门槛 fail。
081 的 mask 内变化为 96.78%，VLM 唯一理由却声称“完全没改”，因此命中上述窄矛盾
规则并恢复为 pass。审核保持 1 call/case；7 条的 VLM inference 为 7.98 秒、52.64
case/min，含 43.71 秒冷启动的总墙钟为 54.80 秒。

最终 planning smoke test 使用默认 3 倍过采样，从 60 个 region 候选筛出 20 条：
GRES/VER 各 5 个 source，四类各 5 条；55/60 候选通过当轮兼容性校验，1 条持续格式
失败被安全丢弃。总墙钟 90.25 秒，VLM inference 33.74 秒、106.71 candidate/min。
随后加入 partial-section 和 image-language 最终校验后，直接重放同批缓存仍有 49/60
compatible，并能完整选出 20 条，说明更严格规则不需要用坏指令补足类型配额。

早期规划 smoke test 从 80 个 region 候选筛出 20 条：GRES/VER 各 5 个 source，四类
各 5 条；70/80 候选通过当时的兼容性校验，2 条持续格式失败被安全丢弃。总墙钟
104.50 秒，VLM inference 43.29 秒、110.88 candidate/min。加入严格 complete-object、
partial-unit 和 generic-replacement 校验后，重放这批缓存响应只剩 63/80 compatible，
固定 pair bucket 有一项短缺；因此默认过采样从 2 倍提升到 3 倍，避免用坏指令补数。
7 卡单条冷启动编辑回归为 109.58 秒；该速度包含每卡各自模型加载，不代表长任务
steady-state 吞吐。
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
