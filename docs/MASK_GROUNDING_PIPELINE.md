# CrispEdit grounding → SAM3 mask pipeline

这条路径只替换 mask 阶段，prefilter 的判断和 manifest schema 不变。旧实现仍保留用于
A/B，但新路径不调用 `robust_diff`、不解析 instruction head noun，也不让 text PCS 决定
编辑实例。

## 数据流

```text
raw parquet + keep manifest
  → S1 Qwen3.5: source/target pair → selected-image ref + bbox_2d
  → grounding parquet（可独立 resume / 抽检）
  → S2 SAM3: PVS + spatially filtered PCS → rectangle fallback
  → src-coordinate union PNG + per-instance COCO RLE
```

路由如下：

| type | S1 grounding | S2 union |
|---|---|---|
| remove | source 删除物 | source mask |
| color | source 完整改色物/指定部件 | source mask |
| replace | source 旧物 + target 新物 | source ∪ mapped target |
| add | target 新增物 | mapped target + dilation |
| motion change | source + target 局部动作/交互区域 | source ∪ mapped target |
| background change | source 中需保护的前景 | NOT(dilated foreground) |
| style | 不 grounding | 全图 |

`replace` 允许一侧为空，以覆盖 “replace the absence of hands with hands” 等不存在可框旧物的
合法非对称替换；其它需要 grounding 的类型缺框时标 `GROUND_FAIL`。

## S1：Qwen3.5 grounding

入口：`crispedit_mllm_grounding.py`。

- 每次请求都提供 source 和 target，但只允许在路由指定的单张图输出坐标；
- 坐标固定为 `[0,1000]`，只在 S2 按原图尺寸映射；
- thinking 关闭，greedy decode；
- 输出会严格解析、clip 并检查非退化 box；原始 response 一并写入 `ground_json`；
- prompt v2 约束 `ref` 为 2–8 词的 SAM-friendly 视觉短语，并要求稀疏/空心目标在
  8 个实例以内逐实例出框；
- parser 会恢复 truncated response，也会恢复 Qwen 偶发的单个 object 内重复
  `bbox_2d` key（标准 JSON parser 会静默只保留最后一个 key）；
- `absence of ...` 等不可见抽象框会丢弃，replace 仍可使用另一侧的真实 footprint；
- BF16 35B 默认以 4 个 TP=2 worker 使用 8 卡；当前 runner 用 Transformers/Accelerate，
  生产吞吐部署可保持 parquet schema 不变，替换为 vLLM/SGLang replica；
- `--selection-file` 支持稀疏评测，输出保留原始 `row_idx`。全量模式仍逐行对齐。

关键字段：`ground_json`、`ground_parse_ok`、`grounding_status`、`qc_flag`、
`mllm_model`、`prompt_version`。

## S2：SAM3

入口：`crispedit_grounded_mask_runner.py`。构建 SAM3 时开启
`enable_inst_interactivity=True`，从而使用仓库已有但旧 CrispEdit pipeline 未启用的 PVS。

每个 MLLM box 同时评估几何和语义候选：

1. PVS box prompt 返回 multimask 和 predicted IoU。候选需满足 mask bbox 与 prompt box
   IoU ≥ 0.60，且至少 90% mask 位于再外扩 5% 的 box 内；合格候选按 predicted IoU 选；
2. PCS 默认先做 text-only 检测，再用 MLLM box 的 5% containment 区域过滤，至少 80%
   mask 像素需落在区域内；这样 bbox 负责实例消歧，text 负责分出 frame、piercing、petal、
   arm 等 PVS 容易选中父物体的目标。text-only 为空时才重试 `ref + positive box`；
3. 对稀疏集合、装饰和部件语义优先 PCS；其它目标保留 PVS 主路径，但 PVS 相对 box
   过密/过稀且 PCS 更合理时允许语义候选覆盖。每个实例记录 `selection_reason` 和
   `semantic_mask_source`；
4. 两条 SAM 路径都失败才使用矩形 box；
5. 80% directional coverage 针对原始 MLLM box（不是额外外扩后的 prompt box）。稀疏
   语义集合不填充内部空白；其它候选覆盖不足时仍与原始矩形取并，保证宁多勿漏。

PCS threshold 为 0.3，小目标的低分召回由 bbox containment 约束；这是 47 条抽检中
`piercing`、`petal` 等目标在 0.5 threshold 下明显漏实例后的实测选择。

原 bbox 先按 box 尺寸外扩 2.5%，并为极小框提供短边 0.25% 的最小 margin。target
mask 用 nearest resize 映射到 source，再膨胀 source 短边的 1.5%。source/target aspect
ratio 相对差异超过 2% 时标 `AR_MISMATCH` 并额外膨胀 2%。

最终字段包括 `ground_json`、`mask_png`、`instance_masks`（带 COCO RLE）、
`mask_source`、`area_frac`、`qc_flag`、`mllm_model`、`prompt_version` 和 `sam_version`。
多实例混用不同路径时，`mask_source` 记录最弱路径（`box > pcs > pvs`）。

## 47 条 mask bad-case 评测

先从 review doc 自动生成选择文件：

```bash
python scripts/build_mask_bad_case_selection.py \
  --output /tmp/crispedit_mask_improved_eval/selection.json
```

S1 使用 8 卡（4 × TP2）：

```bash
python -u crispedit_mllm_grounding.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --output-dir /tmp/crispedit_mask_improved_eval/grounding_v2 \
  --selection-file /tmp/crispedit_mask_improved_eval/selection.json \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --batch-size 1 \
  --request-batch-size 2 --max-new-tokens 512 \
  --overwrite --fail-fast
```

S2 使用 8 个单卡 SAM3 worker：

```bash
python -u crispedit_grounded_mask_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --grounding-dir /tmp/crispedit_mask_improved_eval/grounding_v2 \
  --output-dir /tmp/crispedit_mask_improved_eval/masks_final \
  --devices 0,1,2,3,4,5,6,7 \
  --preview-dir /tmp/crispedit_mask_improved_eval/previews_final \
  --overwrite --fail-fast
```

两阶段分别写 `run_config.json` 和 `run_summary.json`。preview 包含 source/target box、
source-coordinate union overlay 和 raster mask，便于按类型检查选错实例、漏框及 fallback。

若只改 parser，可复用已保存的 raw response，不重跑 35B 模型：

```bash
python scripts/reparse_grounding_outputs.py \
  --grounding-dir /tmp/crispedit_mask_improved_eval/grounding_v2
```

## 人工检查导出

parquet 是训练/流水线的交换格式；人工检查时可将 Qwen 输出导出为 JSONL、CSV 和
Markdown（`requests[].raw_text` 保留模型原始 response）：

```bash
python scripts/export_grounding_outputs.py \
  --grounding-dir /path/to/grounding_v2 \
  --output-dir /path/to/model_outputs \
  --write-json-array
```

若需要按类别查看高清缩略图，脚本会重新读取原始图像 bytes，而不是放大旧 preview：

```bash
python scripts/build_category_previews.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --mask-dir /path/to/masks_final \
  --selection-file /path/to/selection.json \
  --output-dir /path/to/previews_by_category \
  --columns 1
```

输出目录中每个类别对应一张 PNG；默认每个面板宽 640px、每个类别一张纵向图，便于
放大检查 MLLM box 和最终 source-coordinate mask。

本次 47 条回归已从临时运行目录复制到持久路径：

- selection：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/selection.json`
- grounding：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/grounding_v2`
- mask：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/masks_final`
- collage：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/masks_final/preview_collage.png`
- 分类高清图：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/previews_by_category`
- Qwen 可读输出：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/model_outputs`
- report：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/evaluation_final.{json,md}`

最终 47/47 有输出、0 runtime error、0 `GROUND_FAIL`；共恢复并保存 181 个 instance mask。
样本级 `mask_source` 为 PCS 36、PVS 3、box 8。box 计数按最弱实例聚合，因此一个样本中
只要 1 个小实例触发 coverage/rectangle fallback，样本级即记为 box。
