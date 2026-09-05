# ScaleEdit 图像编辑区域打标

本流程为 ScaleEdit 单独实现，不修改源数据，也不把 23 个 `final_task` 生硬映射到
CrispEdit 的 7 类规则。输入只读扫描 `part-*.parquet`，输出写到显式指定的新目录。

## 为什么需要独立策略

ScaleEdit 的任务名不能直接决定 mask 形状：

- `style_transfer` 同时包含全图风格化和只改变墙面、招牌、屏幕的局部风格；
- `part_extraction` 同时包含“保留主体、背景变白”和局部去壳/显露内部；
- `tone_adjustment` 既有全图滤镜，也有只改变天空或局部环境光；
- `perceptual/scientific/social/symbolic_reasoning` 内部混合增、删、修复、属性和绘线；
- 四类文字编辑需要覆盖实际字形块，SAM 的普通物体语义 mask 容易漏掉细笔画。

因此第一阶段先比较 source/result 中实际发生的编辑，再决定下面三种合成契约：

| `mask_mode` | 使用场景 | source 坐标系最终 mask |
|---|---|---|
| `regions` | 局部增删替换、属性、动作、文字、修复等 | source 区域 ∪ 映射后的 target 区域 |
| `protect_foreground` | 背景替换且前景主体在原位稳定保留 | NOT(膨胀后的稳定前景) |
| `full_image` | 真正的全图滤镜、全画面风格或相机视角变化 | 全图 |

`regions` 中的连贯物体/表面使用 MLLM bbox + SAM3 双提示分割；文字、glyph、细线、
路径、裂缝、点状小标记使用带小安全边距的直接框 mask，以召回优先避免细结构消失。
target mask 按归一化坐标映射到 source；两图宽高比差异超过 2% 会写入 QC 标记。
商品式 `part_extraction` 通常还会重新居中、缩放或重构主体，因此按全图重构处理；仅有
原位去壳/揭示内部时才使用局部区域。

`viewpoint_transformation` 还需区分两种几何：相机移动、站位改变或整幅场景换视角使用
全图；明确命名的单个物体转到正面/背面/侧面必须使用 source/target 物体区域。若首轮
误判为全图，grounding 会自动发起一次强约束复核，取得真实物体 bbox 后才交给 SAM3；
复核仍失败时保留安全的全图契约，不使用可能把背景反选为前景的整图 bbox。迷宫路径类
会强制搜索框覆盖起止端点，避免 MLLM 只框住路径中段。

## 输入数据

每个 `part-*.parquet` 的必需字段为：

- `sample_id`；
- `source_image` / `edited_image`（binary，亦兼容 Arrow image struct）；
- `final_task`；
- `final_instruction`。

`source_relative_path`、`edit_task` 和 `original_instruction` 是可选审计字段；存在时原样传到
输出，不参与实际 mask 决策。打标只使用清洗后的 `final_task` 和 `final_instruction`。输入发现
逻辑只扫描 `part-*.parquet`；`matched_rows.parquet`、统计 JSON 和尚未落盘的下载内容都不会
被当成图像 shard。所有输入图片均从 parquet 内存解码，不会回写源记录。

## 端到端数据流

一次完整打标由两个独立的可恢复 stage 组成。第一个 stage 内含两轮固定的 MLLM 对话，
少量满足条件的样本还会进入一次附加复核；第二个 stage 根据前一 stage 的空间契约生成
最终 mask。校验和可视化只读 raw、grounding、mask 三类产物，不参与 mask 生成。

```text
只读 ScaleEdit part-*.parquet
  └─ Stage A: scaleedit_mllm_grounding.py
       ├─ Round 1: 实际编辑审计（what changed）
       ├─ Round 2: 空间打标契约（where/how to mask）
       ├─ Round 2R: 条件式物体视角复核（仅命中时运行）
       └─ deterministic post-policy
            └─ grounding/part-*.parquet + run_config.json + run_summary.json
                 └─ Stage B: scaleedit_grounded_mask_runner.py
                      ├─ direct box，或 SAM3 PVS/PCS 候选生成与选择
                      ├─ target mask 映射回 source 坐标系
                      └─ 按 mask_mode 合成
                           └─ masks/part-*.parquet + run_config.json + run_summary.json
                                ├─ validate_scaleedit_masks.py -> validation.json
                                └─ visualize_scaleedit_masks.py -> review JPG/summary/index
```

### Stage A：两轮 MLLM grounding

入口为 `scaleedit_mllm_grounding.py`。它只扫描 `part-*.parquet`，按 shard 分配到 Qwen3.5
worker，并保持输入 shard 名、原始 `row_idx` 和 `sample_id`。以下“轮”指同一个样本的模型
对话轮次，不是重新读取或改写数据集。

#### Round 1：实际编辑审计

目的不是立即画框，而是先回答“图中实际实现了什么编辑、它属于哪种 mask 路由”。

每行输入：

- `source_image`：标记为 `Image 1 (source, full image)`；
- `edited_image`：标记为 `Image 2 (edited result, full image)`；
- 归一化后的 `final_task`；
- 清洗后的 `final_instruction`；
- 当前 task 的专用指导，例如 `count_change` 只记录新增/消失实例，文字任务只记录实际
  glyph block，不能直接照搬 CrispEdit 类别规则。

模型输出并解析为：

```json
{
  "realized_edit": "对实际编辑的一句描述",
  "mask_mode": "regions | protect_foreground | full_image",
  "changes": [
    {
      "source_ref": "source 中旧的或发生变化的实体；纯新增时为空",
      "target_ref": "target 中新的或发生变化的实体；纯删除时为空",
      "change": "before to after",
      "geometry": "semantic_object | dense_region | sparse_marks",
      "source_extent": "source 中的大致位置",
      "target_extent": "target 中的大致位置"
    }
  ],
  "protected_foreground": [
    {"ref": "背景变化时保持不变的前景实体", "extent": "source 中的完整范围"}
  ],
  "confidence": "high | medium | low"
}
```

这一轮输出是语义审计，不含最终 bbox。prompt、模型原始文本、解析结果、`parse_ok` 和
错误信息全部保存在最终 `ground_json.observation` 中。JSON 或 schema 解析失败时按
`--parse-retries` 重试（默认 1 次）；该重试只保证可解析性，不保证语义或坐标准确。

#### Round 2：生成空间打标契约

Round 2 延续 Round 1 的完整对话，而不是启动一个无上下文请求。

每行输入：

- 与 Round 1 相同的完整 source/edited 图片；
- Round 1 的模型原始回答，以及解析后的 audit JSON；
- `final_task`、`final_instruction`；
- bbox、region 拆分、文字/稀疏区域、三种 `mask_mode` 的严格输出规则。

模型必须先复核 Round 1，再输出：

```json
{
  "prompt_version": "scaleedit_realized_edit_grounding_v2",
  "mask_mode": "regions | protect_foreground | full_image",
  "source": [
    {
      "ref": "SAM 可理解的 source 可见实体短语",
      "bbox_2d": [100, 120, 600, 800],
      "mask_method": "sam | box",
      "region_mode": "object | aggregate_region",
      "mask_density": "object | dense | sparse"
    }
  ],
  "target": [],
  "protected_foreground": []
}
```

`bbox_2d` 均为命名图片完整画布上的 `[x1,y1,x2,y2]`，范围归一化到 `0..1000`：
`source` 项使用 source 坐标系，`target` 项使用 edited-result 坐标系，不能跨图复用像素坐标。
parser 会裁剪越界值并拒绝非有限、退化或 schema 非法的框。

三种契约的列表约束如下：

| `mask_mode` | `source` | `target` | `protected_foreground` |
|---|---|---|---|
| `regions` | 删除前、替换前、属性变化前等可见区域；纯新增可空 | 新增、替换后、属性变化后等可见区域；纯删除可空 | 强制清空 |
| `protect_foreground` | 强制清空 | 强制清空 | source 图中需要从背景 mask 排除的完整稳定前景 |
| `full_image` | 强制清空 | 强制清空 | 强制清空 |

`mask_method=sam` 用于连贯物体和表面；`mask_method=box` 用于文字、glyph、线、路径、裂缝、
点状小标记等容易被语义分割漏掉的结构。`region_mode=aggregate_region` 表示附近的重复小实体
作为一个区域处理，`mask_density` 用于后续选择 sparse/dense/object 策略。

#### Round 2R：条件式物体视角复核

这不是固定的第三轮，只在以下条件同时成立时运行：

1. task 为 `action_editing` 或 `viewpoint_transformation`，instruction 能提取出一个明确的
   孤立物体；
2. Round 2 却返回 `full_image`；
3. instruction 不包含 camera、entire scene 等明确的相机/全场景视角表达。

输入为前两轮完整对话、识别出的 `object_ref` 和强制 `regions` 的纠错要求；期望输出该物体
在 source/target 中的真实局部框。只有复核结果为 `regions` 且至少一侧有框时才替换 Round 2
契约，否则保留原来的安全 `full_image`。本轮 prompt、原始回答、是否接受和错误信息记录在
`ground_json.grounding.route_retry`。

#### 确定性 post-policy 与 Stage A 输出

模型轮次结束后，代码还会用 `final_task + final_instruction + grounding payload` 应用窄范围
确定性规则；这里不再调用模型：

- 商品摄影、product mockup 或“提取到白底”式 `part_extraction` 强制为 `full_image`，原合同
  写入 `route_override`；
- `symbolic_reasoning` 的 maze path/line 使用 `[50,100,950,900]` 的召回优先框覆盖两端，
  原因写入 `box_override`。

`scaleedit_mllm_grounding.py` 已经在内部执行该 post-policy，正常新运行不需要再调用
`scripts/apply_scaleedit_grounding_policy.py`。后者只用于“模型结果不变、确定性 policy 更新”
时，把旧 grounding 只读转换到一个新的输出目录。

Stage A 为每个输入 shard 写一个同名 grounding parquet。每行主要输出：

| 字段 | 含义 |
|---|---|
| `row_idx`, `sample_id`, `source_relative_path` | 与原始行对齐和审计 |
| `edit_task`, `final_task`, `original_instruction`, `final_instruction` | 原始和清洗后的任务信息 |
| `ground_json` | 两轮原始/解析记录、最终 contract、条件复核和 policy override |
| `ground_parse_ok`, `grounding_status`, `qc_flag` | contract 的结构状态 |
| `source/target_width/height` | 两张输入图片的真实尺寸 |
| `mllm_model`, `prompt_version`, `grounding_seconds` | 模型、prompt 和耗时版本信息 |

`grounding_status` 可能为 `OK`、`FULL_IMAGE`、`PROTECT_FOREGROUND`、`PARSE_ERROR`、
`RUNTIME_ERROR` 或 `GROUND_FAIL`。Stage A 的 `qc_flag=OK` 仅表示最终 contract 可以进入 mask
stage；它不代表 bbox 已经经过人工检查。目录另含完整运行参数 `run_config.json` 和汇总
`run_summary.json`。

### Stage B：SAM3 mask 生成与 source 坐标合成

入口为 `scaleedit_grounded_mask_runner.py`。它按同名 shard 读取原始 parquet 和 Stage A
grounding parquet，首先核对行数、`row_idx` 和 `sample_id`，再在 source、target 各自原始
分辨率上处理 contract 中的区域。

每行输入：

- 原始 `source_image`、`edited_image`；
- 对齐的 Stage A `ground_json` 和状态字段；
- SAM3 checkpoint；
- 当前 `mask_policy_version`。

本文沿用代码中的命名：PVS 指由 bbox 驱动的 `predict_inst` 候选，PCS 指由可见实体短语驱动
并受 bbox 空间约束的候选。

单个 region 的生成方式：

| contract | 处理 |
|---|---|
| `mask_method=box` | 不调用 SAM3；把归一化 bbox 转为该图像像素框，加极小栅格边距并填充矩形 |
| `mask_method=sam` / PVS | 将 bbox 小幅扩张后作为 SAM3 box prompt，生成多个候选并检查 box IoU 与 containment |
| `mask_method=sam` / PCS | 同时生成 text-only 和 text+box phrase 候选，按 bbox 空间约束过滤，再按 object/aggregate 与 sparse/dense 策略融合 |
| 候选选择 | 根据语义细节、区域模式、候选填充率等在 PVS/PCS 间选择；两者均不可用时退化为 box mask |

每个语义实例都会记录 `mask_source`、`predicted_iou`、`box_iou`、`inside_ratio`、
`selection_reason`、实际 `sam_prompt` 和异常列表，随后编码为 source 尺寸的 COCO RLE。

各 `mask_mode` 的最终合成不同：

- `regions`：分别生成 source regions 和 target regions；target mask 先 resize 到 source
  尺寸，再按 source 短边膨胀 1.5%（宽高比差异超过 2% 时再增加 2%），最终取
  `union(source masks, mapped target masks)`；
- `protect_foreground`：只在 source 上分割稳定前景，合并并按短边膨胀 1.5%，最终取反得到
  可编辑背景；
- `full_image`：不调用 SAM3，直接生成与 source 同尺寸的全 1 mask；
- `GROUND_FAIL`：写出空的失败占位行，不调用 SAM3。

target 到 source 的映射是分辨率归一化 resize 与安全膨胀，不是 optical flow 或像素级配准；
因此所有最终 `mask_png` 和 `instance_masks` RLE 都严格位于 source 像素坐标系。

Stage B 每行输出：

| 字段 | 含义 |
|---|---|
| `mask_png` | source 原分辨率二值 union mask |
| `instance_masks` | 各区域的 source 坐标 RLE、bbox、来源、质量指标和选择原因 |
| `mask_source` | `pvs`, `pcs`, `hybrid`, `box`, `direct_box`, `full_image` 或 `inverse_foreground` |
| `area_frac`, `mask_sum`, `mask_width/height` | union mask 面积与尺寸 |
| `qc_flag`, `qc_flags_json`, `ar_delta` | 机器 QC 和宽高比差异 |
| `mllm_model`, `prompt_version`, `sam_version`, `mask_policy_version` | 可复现实验版本 |
| `mask_seconds` | 该行 mask 推理耗时 |

`qc_flag` 的成功值为 `OK`，可见的异常值包括 `EMPTY_MASK`、`BOX_FALLBACK`、`AR_MISMATCH`、
`GROUND_FAIL` 和 `ERROR`；更细的 `FULL_IMAGE`、`DIRECT_BOX`、`INVERSE_FOREGROUND` 等生成
路径保存在 `qc_flags_json`。这些都是运行与结构信号，不是人工语义评分。

输出目录同样含 `run_config.json` 和 `run_summary.json`。不加 `--overwrite` 时，已存在的完整
shard 会被跳过；新 shard 先写 `.tmp`，完成后再原子替换为正式 parquet。

### Stage C：结构校验与人工 review 产物

`scripts/validate_scaleedit_masks.py` 同时读取 raw、grounding 和 mask 三套同名 shard，检查：

- shard/行数、`sample_id`、`row_idx` 是否严格对齐，sample ID 是否重复；
- `mask_png` 是否与 source 和记录尺寸一致，`mask_sum`、`area_frac` 是否可重算；
- `full_image` 是否确实全 1，非全图成功行是否非空；
- 每个 COCO RLE 是否可解码为 source 尺寸，RLE 面积是否与实例元数据一致。

输出为 `validation.json`；存在任何结构错误时脚本以非零状态退出。它不判断“框中了正确
对象”或“mask 是否泄漏到背景”，因此必须结合人工 review。

`scripts/visualize_scaleedit_masks.py` 输入同样的三套目录，输出 source bbox、target bbox、
source mask overlay 和二值 mask 四列 contact sheet，以及选样和面积统计 `summary.json`。
`--coarse-category-groups` 会把 23 个任务分到本文的五个 review 目录，并额外输出根
`index.json`。

## 验证集运行

以下命令将所有产物写到 workspace 中独立的 `ScaleEdit-results/validation-v1`，不会向验证集
目录写入任何内容。

```bash
cd /opt/tiger/tanyue/sam3-prefilter_improved

python -u scaleedit_mllm_grounding.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-200-v5 \
  --output-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/grounding-v2 \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --batch-size 8 \
  --request-batch-size 4 \
  --fail-fast

python -u scaleedit_grounded_mask_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-200-v5 \
  --grounding-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/grounding-v2 \
  --output-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/masks-v3 \
  --checkpoint-path /mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt \
  --devices 0,1,2,3,4,5,6,7 \
  --fail-fast

python scripts/visualize_scaleedit_masks.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-200-v5 \
  --grounding-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/grounding-v2 \
  --mask-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/masks-v3 \
  --output-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/review-v2-v3 \
  --coarse-category-groups

python scripts/validate_scaleedit_masks.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-200-v5 \
  --grounding-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/grounding-v2 \
  --mask-dir /opt/tiger/tanyue/ScaleEdit-results/validation-v1/masks-v3 \
  --report-json /opt/tiger/tanyue/ScaleEdit-results/validation-v1/validation-v2-v3.json
```

## 恢复、定向复核与策略升级

所有 stage 都以 shard 为恢复单位。不加 `--overwrite` 时会跳过已存在输出；prompt 或 mask
策略升级后应使用带版本的新输出目录，避免把不同版本静默混合。全量数据仍在下载时不要
启动生产运行，避免把尚未落稳的 parquet 当成完整分片。

人工 review 后只需修复少量 grounding 时，可重复传入 `--sample-id`，生成仅含指定样本、但
保留原始 shard 名和 `row_idx` 的定向结果。该稀疏结果不能直接送入 mask runner，必须先用：

```text
scripts/merge_scaleedit_grounding.py
  base complete grounding + targeted grounding updates
  -> new complete grounding directory
```

merge 按 `sample_id + row_idx` 覆盖，拒绝重复 ID、错位行、未知样本和原地覆盖。合并后的完整
grounding 再写入一个新的 mask 输出目录。若只修改了 deterministic post-policy，则使用
`scripts/apply_scaleedit_grounding_policy.py` 将旧 grounding 转换到新目录，无需重跑 Qwen；
之后仍需重新运行 Stage B，不能沿用旧 mask。

## 200 条验证集结果

归档结果来自 `ScaleEdit-filtered-balanced-final-task-200-v5`，共覆盖 23 个任务。最终运行目录
使用 `grounding-final-v4` 和 `masks-final`；版本后缀记录验证期间的定向复核，不是生产路径的
硬编码要求。该快照在人工复核过程中合并过 prompt v1 和定向修复后的 prompt v2 grounding，
所有最终 mask 均为 `scaleedit_sam3_hybrid_mask_v3`；新生产运行应直接使用当前 prompt v2，
不要把这种验证期的混合版本当成默认做法。机器校验结果归档在
[`validation_summary.json`](../docs_assets/scaleedit/validation_v1/validation_summary.json)。

| 指标 | 结果 |
|---|---:|
| 样本 / 唯一 sample ID | 200 / 200 |
| 任务数 | 23 |
| instance RLE 数 | 488 |
| 结构校验错误 | 0 |
| `regions` / `full_image` / `protect_foreground` | 178 / 17 / 5 |
| PVS / PCS / hybrid | 81 / 15 / 33 |
| direct box / full image / inverse foreground | 49 / 17 / 5 |
| mask 面积比例，中位数 / 均值 | 0.1156 / 0.2725 |

这里的 `qc=OK` 表示输出对齐、JSON/RLE、尺寸、非空性和 fallback 等机器规则通过，不代表
人工确认语义完全正确。人工抽查已经发现一个 `count_change` 样本中，MLLM 的两个 target
bbox 明显向桌面方向下移；PVS 又接受了与错误框自洽但包含桌面的 mask。该问题应在 100k
生产运行前通过 bbox 复核和语义质量阈值继续改进，不能把本次 200/200 `OK` 解读为
200/200 人工验收通过。

## 粗粒度分类可视化

每个细任务选择一个代表样本。每行从左到右依次为 source 与 bbox、edited result 与 bbox、
source 坐标系 mask overlay、二值 mask。图片和对应选样统计均保存在
`docs_assets/scaleedit/validation_v1/`；五组的任务映射和相对产物路径见
[`index.json`](../docs_assets/scaleedit/validation_v1/index.json)。

### 1. 对象增删与构图

包含 `object_addition`、`object_removal`、`object_replacement`、`count_change` 和
`compositional_editing`。对应的
[选样与统计](../docs_assets/scaleedit/validation_v1/01_object_composition/summary.json)。

![ScaleEdit object and composition review](../docs_assets/scaleedit/validation_v1/01_object_composition/scaleedit_mask_preview_page_1.jpg)

### 2. 属性、材质、尺度与动作

包含 `action_editing`、`color_change`、`material_change` 和 `size_change`。对应的
[选样与统计](../docs_assets/scaleedit/validation_v1/02_attribute_action/summary.json)。

![ScaleEdit attribute and action review](../docs_assets/scaleedit/validation_v1/02_attribute_action/scaleedit_mask_preview_page_1.jpg)

### 3. 文本与符号

包含四类表面文字编辑和 `symbolic_reasoning`。对应的
[选样与统计](../docs_assets/scaleedit/validation_v1/03_text_symbol/summary.json)。

![ScaleEdit text and symbol review](../docs_assets/scaleedit/validation_v1/03_text_symbol/scaleedit_mask_preview_page_1.jpg)

### 4. 推理、修复与美化

包含 `perceptual_reasoning`、`scientific_reasoning`、`social_reasoning` 和
`visual_beautification`。对应的
[选样与统计](../docs_assets/scaleedit/validation_v1/04_reasoning_repair/summary.json)。

![ScaleEdit reasoning and repair review](../docs_assets/scaleedit/validation_v1/04_reasoning_repair/scaleedit_mask_preview_page_1.jpg)

### 5. 场景、全局变化与主体提取

包含 `background_replacement`、`part_extraction`、`style_transfer`、`tone_adjustment` 和
`viewpoint_transformation`。对应的
[选样与统计](../docs_assets/scaleedit/validation_v1/05_scene_global_extraction/summary.json)。

![ScaleEdit scene, global edit, and extraction review](../docs_assets/scaleedit/validation_v1/05_scene_global_extraction/scaleedit_mask_preview_page_1.jpg)

## 全量切换

100k 数据下载完成后，仅需将三个命令中的 `--input-dir` 换成：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-balanced-final-task-100k
```

并使用新的输出目录。不要把任何输出放进源数据目录。
