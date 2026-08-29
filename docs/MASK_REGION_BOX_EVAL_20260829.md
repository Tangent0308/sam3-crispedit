# Region-box + dual-prompt mask 评测（2026-08-29）

## 目标

本轮从 `CrispEdit-mask-improved-eval-20260828-two-pass` 对应提交 `cdee395` 重新开发，
不使用 pixel diff，也不修改 prefilter。改进点是：

1. 第一轮 Qwen3.5 同时观察 source/result/instruction，以图像对为事实，输出实际发生的
   精确编辑区域；
2. 第二轮按空间区域输出 recall-first bbox 和 SAM-friendly ref。相邻的小散点使用一个
   aggregate box，不逐点框；bbox 必须完整包住编辑区域；
3. SAM3 同时运行 bbox-only、phrase-only 和 phrase+bbox，按 object/aggregate 与
   dense/sparse 策略融合；紧凑的小元素组可生成基于语义 mask 的实心连通包络。

## 运行范围与结果

- selection：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260828-two-pass/selection.json`
- 样本：`BAD_CASE_REVIEW_20260826.md` 的 47 条 mask bad case
- Qwen：`/mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B`
- S1：GPU 0–7，4 个 TP=2 replica
- S2：GPU 0–7，8 个单卡 SAM3 worker
- grounding：47 rows，0 error，0 `GROUND_FAIL`，46 `OK` + 1 `PARTIAL_OK`
- mask：47/47 `OK`，0 error，0 rectangle fallback
- mask source：PCS 39，PVS 3，connected group 5
- bbox 数：two-pass baseline 209 → region-box 114
- mask area：min 0.0062，median 0.1046，mean 0.1594，max 0.7510
- 单测：43 passed

没有像素级 GT，因此 baseline mask IoU 只用于发现变化，不能当质量真值。与 baseline 的
median IoU 为 0.7034、median area delta 为 +0.0023；最终结论以逐类别预览和指定 case
人工检查为主。

## 关键 case

| case | 最终行为 |
|---|---|
| `add_00000.parquet row=17` | 用单个 boats region box，SAM 保留码头附近的船，不再受后续多轮碎框影响 |
| `add_00045.parquet row=72` | 多朵相邻花先由一个 aggregate box 定位，再形成底部花圃的实心连通区域 |
| `remove_00009.parquet row=15` | 面部穿孔/红色痕迹形成一个连通包络；空间分离且语义不同的两侧耳环单独保留 |
| `color_00008.parquet row=185` | 第一轮补出 instruction 外的脸/颈变化；surface-aware phrase 避免把衣服/整个人当作 arms |
| `color_00060.parquet row=217` | phrase-only 微小低置信点退化时，由 phrase+bbox 恢复完整右下花圃 |
| `color_00074.parquet row=104` | 第一轮独立审计检测到脸/颈也变黑，最终 mask 同时覆盖脸/颈和双臂/手 |
| `remove_00035.parquet row=247` | chains 不进入小散点凸包，保留链条语义形状，避免跨背景三角形 |

## 产物

统一输出根目录：

`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260829-region-box-v10`

- `grounding/`：parquet 与两轮原始 response
- `masks/`：最终训练格式 mask parquet、`run_config.json`、`run_summary.json`
- `previews/`：逐样本 source/target/box/mask PNG
- `previews_by_category/`：每个类别一张 520×325 panel、单列、不重叠的 review PNG
- `model_outputs/`：便于直接打开的 JSON、JSONL、CSV、Markdown
- `evaluation.{json,md}`：完整性与 QC 汇总
- `comparison.{json,md}`：相对 20260828 two-pass baseline 的 A/B

最终 prompt/policy version：

- observation：`qwen35_realized_edit_spec_v4`
- grounding：`qwen35_realized_edit_region_ground_v6`
- SAM：`sam3-dual-prompt-region-fusion-v5-surface-aware`
