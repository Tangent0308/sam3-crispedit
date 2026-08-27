# Mask grounding bad-case 回归（2026-08-27）

评测集来自 `BAD_CASE_REVIEW_20260826.md` 的 `## 2. Mask 打标这一侧 bad case`，
共 47 条：add 9、background change 4、color 7、motion change 8、remove 12、replace 7。
本次只验证新 mask 流程，不改 prefilter，也不使用 pixel diff。

## 运行配置

- S1：Qwen3.5-35B-A3B BF16，4 个双卡 replica（TP=2 语义），覆盖 GPU 0–7；
- S1 prompt：`qwen35_grounding_v2`，`max_new_tokens=512`，输入上限约 1.31 MP/图；
- S2：8 个单卡 SAM3 worker，PVS interactivity 开启，hybrid policy v2；
- source/target mapping：nearest resize + 1.5% 短边 dilation；
- 最终结果：47/47，0 runtime error，0 `GROUND_FAIL`，181 个 per-instance RLE。

S1 状态为 46 `OK` + 1 `PARTIAL_OK`。`PARTIAL_OK` 是合法的
`replace the absence of hands ...`：source 侧抽象的 “absence” 被丢弃，target hands 正常出框。

S2 样本级来源为 PCS 36、PVS 3、box 8；39 条 `OK`、8 条 `BOX_FALLBACK`。
样本级来源按最弱实例聚合，因此多实例样本只要一个 tiny box 使用矩形，该样本就计为 box，
并不表示 union 全部来自矩形。

## 关键修复效果

| case | 旧面积 | 新面积 | 人工复查结论 |
|---|---:|---:|---|
| `remove_00001:127` | 0.9654 | 0.0778 | 从整张花束改为窄矩形花框 |
| `remove_00009:15` | 0.0015 | 0.0052 | 13 个面部 piercing 逐实例覆盖 |
| `remove_00012:88` | 0.0017 | 0.0652 | 花冠与周边花簇一起恢复 |
| `remove_00059:248` | 0.0342 | 0.1433 | 7 组被删除 mushroom 均覆盖 |
| `remove_00083:133` | 0.0017 | 0.2361 | 19 组包围女孩的花，不再覆盖女孩本身 |
| `add_00040:96` | 0.0015 | 0.0753 | 分散花瓣全部覆盖，不再把沙地当目标 |
| `color_00008:185` | 0.0036 | 0.0096 | 5 只抬起的手臂逐实例覆盖 |
| `color_00074:104` | 0.0273 | 0.0476 | 精确覆盖双臂，去掉 0.225 的大矩形 |
| `motion change_00012:40` | 0.8302 | 0.0670 | 从近全人/场景收敛到手、手臂和电钻 |
| `motion change_00035:182` | 0.0081 | 0.0244 | 同时覆盖 source 原手位与 target 抬手位 |
| `replace_00080:77` | 0.0403 | 0.0444 | 不再分割不存在的 source “absence”，只映射 target hands |

旧 pipeline 的典型 severe under-mask（约 0.1%–0.4% 面积）已经由 MLLM 多实例框和
SAM3 语义 mask 取代；两个历史 motion over-mask（0.70、0.83）也收敛到局部交互区域。

## 残余边界

- `add_00014:156`（0.3226）和 `add_00047:51`（0.3108）的 dense flowers 与 basket/arch
  接触紧密，SAM3 会把整组装饰连同容器的一部分一起分出；结果满足“宁多勿漏”，但仍偏大。
- `BOX_FALLBACK` 主要来自 tiny instance 的 directional coverage 或细长 clipboard/lamp；
  它们已在 instance metadata 中保留 `selection_reason`、coverage 和 RLE，可直接筛选复查。
- 当前不做 MLLM judge/验证。若继续优化上述 dense-group 边界，优先增加 crop/zoom
  重定位或独立 mask judge，而不是重新引入 pixel diff。

## 产物

- grounding：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/grounding_v2`
- masks：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/masks_final`
- previews：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/previews_final`
- collage：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/masks_final/preview_collage.png`
- machine-readable report：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/evaluation_final.json`
- markdown report：`/opt/tiger/tanyue/CrispEdit-mask-improved-eval-20260827/evaluation_final.md`
