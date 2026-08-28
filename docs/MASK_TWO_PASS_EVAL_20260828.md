# 两轮 MLLM observation → grounding 实验（2026-08-28）

## 结论

流程在工程上可行：Qwen3.5 第一轮先读取 source、target 和 instruction，输出实际变化
观察；第二轮在同一多轮 conversation 中读取上一轮回答，再输出 ref + bbox。47 条 mask
bad case 已用 8 卡完成 Qwen 和 SAM3 端到端运行，0 runtime error、0 observation parse
failure、0 `GROUND_FAIL`。

但这次小样本不能证明质量整体提高。它让模型的判断过程更可见，也召回了一些 instruction
外的明显变化；与此同时，motion 类出现了显著扩框，且关键样本
`color_00074.parquet row=104` 仍漏掉变黑的人脸。结论是：**两轮拆分值得保留并继续 A/B，
但不能把它本身当作 subtle-change 漏检的充分解法。**

## 实现

- `--grounding-mode two-pass` 为本分支默认值；`single` 可复现原 prompt v2。
- Pass 1 输出：

  ```json
  {
    "edit_summary": "...",
    "checked_regions": [{"ref": "...", "changed": true}],
    "changes": [{
      "source_ref": "...",
      "target_ref": "...",
      "change": "...",
      "instruction_aligned": true
    }]
  }
  ```

- Pass 2 仍输出原有 `[{'ref', 'bbox_2d'}]` schema，所以 S2 SAM3 和现有训练字段无需改动。
- `ground_json.observation` 保存 pass-1 prompt、raw response、parsed object、parse status；
  旧的 `ground_json.boxes` 与 `requests` 保持兼容。
- 解析失败会按 `--parse-retries` 重试；仍失败时保留原始文本给第二轮，同时记录
  `observation_parse_fail`，不会悄悄丢失证据。
- 人工导出新增 `change_observations.csv`，Markdown 同时展示两轮原始输出。

## 8 卡运行

评测集来自 `docs/BAD_CASE_REVIEW_20260826.md` 的 47 条 mask bad case。Qwen 使用
4 个 TP=2 worker，占用 GPU 0–7；SAM3 使用 8 个单卡 worker。

```bash
python -u crispedit_mllm_grounding.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --output-dir /opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260828-two-pass/grounding \
  --selection-file /opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260828-two-pass/selection.json \
  --model-path /mnt/bn/strategy-mllm-train/common/models/Qwen3.5-35B-A3B \
  --devices 0,1,2,3,4,5,6,7 --tensor-parallel-size 2 \
  --grounding-mode two-pass --batch-size 1 --request-batch-size 2 \
  --max-new-tokens 512 --max-pixels 1310720 --overwrite --fail-fast

python -u crispedit_grounded_mask_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --grounding-dir /opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260828-two-pass/grounding \
  --output-dir /opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260828-two-pass/masks \
  --devices 0,1,2,3,4,5,6,7 \
  --preview-dir /opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260828-two-pass/previews \
  --overwrite --fail-fast
```

## 运行结果

| stage | rows | errors | 关键结果 |
|---|---:|---:|---|
| pass-1 observation | 47 | 0 | 47/47 parse OK |
| pass-2 grounding | 47 | 0 | OK 46，PARTIAL_OK 1，GROUND_FAIL 0 |
| SAM3 | 47 | 0 | OK 38，BOX_FALLBACK 9 |

第一轮共输出 126 个 `changes`，其中 30 个被模型标为 `instruction_aligned=false`。
第二轮一共发出 62 个 grounding request，加第一轮 47 个 observation request，共 109 次
生成。最终样本级 mask source 为 PCS 33、PVS 5、box 9。

与 2026-08-27 单轮结果的无 GT A/B：

| type | boxes 单轮→双轮 | ref/count 改变样本 | mask IoU 中位数 | area delta 中位数 |
|---|---:|---:|---:|---:|
| add | 19→19 | 7/9 | 0.4430 | -0.0024 |
| background | 23→24 | 4/4 | 0.9053 | -0.0438 |
| color | 11→15 | 7/7 | 0.6998 | -0.0000 |
| motion | 25→59 | 8/8 | 0.4744 | +0.0344 |
| remove | 85→72 | 11/12 | 0.7772 | +0.0000 |
| replace | 18→20 | 7/7 | 0.9066 | +0.0007 |
| all | 181→209 | 44/47 | 0.6998 | +0.0001 |

这里的 IoU 是新旧 pipeline 输出之间的一致性，不是相对 GT 的准确率。变化最大的是
motion：第一轮把头部、躯干、双侧肢体和交互物体都解释成实际 pose/reconstruction
变化，符合“宁多勿漏”的倾向，但分类图上部分样本已经接近整个人体，必须由人工 GT
判断是正确的 realized-change coverage 还是过度 mask。

## `color_00074.parquet row=104`

双轮流程未解决这个漏检。Pass 1 的实际输出为：

```json
{
  "checked_regions": [
    {"ref": "man's face", "changed": false},
    {"ref": "man's neck", "changed": false},
    {"ref": "man's arms and hands", "changed": true},
    {"ref": "man's blue t-shirt", "changed": false}
  ],
  "changes": [{
    "source_ref": "man's arms and hands",
    "target_ref": "man's arms and hands",
    "change": "skin tone changed from medium to darker",
    "instruction_aligned": true
  }]
}
```

即使 prompt 强制逐项检查 face/head，并明确说明同方向肤色变化不能轻易归为光照，模型
仍把 face 判断为 unchanged。Pass 2 忠实使用了错误 checklist，只输出一个
`man's arms and hands` 框 `[456, 312, 963, 694]`，SAM3 mask 也只覆盖手臂。

问题定位因此更清楚：这条 case 的首要瓶颈不是 bbox 生成或 SAM3，而是 Qwen3.5-35B
对两张完整图中的细微局部肤色变化比较失败。两轮结构把失败从“最终漏 mask”前移成了
可直接审计的 `checked_regions=false`。

## 产物

- 根目录：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260828-two-pass`
- grounding parquet：`grounding/`
- SAM3 mask parquet：`masks/`
- 每类一张预览：`previews_by_category/`（560×350 panel，单列，PNG）
- 人工可读模型输出：`model_outputs/grounding_outputs.md`
- 第一轮 CSV：`model_outputs/change_observations.csv`
- 第二轮 CSV：`model_outputs/grounding_requests.csv`
- 完整 JSON：`model_outputs/grounding_outputs.json`
- A/B 报告：`comparison_vs_single.{json,md}`
- 单次运行汇总：`evaluation.{json,md}`

## 下一步建议

1. 先对 47 条建立区域级人工标注（至少标漏框、过框、可接受），不要用与旧 mask 的 IoU
   代替质量指标。
2. 对 subtle color spillover 增加独立的局部验证：先由 pass 1 枚举 subject parts，再对
   face/head 等候选做成对 crop/zoom 判断；只在该验证通过后送入 grounding。
3. motion 类增加粒度约束或第三轮 verifier，区分“真正需要覆盖的重绘区域”与仅由姿态
   变化连带产生的整人重建差异。
4. 当前第一轮能记录跨类型 collateral change，但 S2 仍按原 type route 合成；例如 add
   中观察到 source-only removal 时不会自动加入 source mask。若目标确定为覆盖所有非预期
   重绘，需要给 observation 增加 `change_kind`，再按 change item 动态选择 source/target
   grounding，不能只依赖样本级 type。
