# 2026-09-19：生成边界、双图审核与独立指令重建

本文记录实际跑过的实验，不把模型的 pass 当成人工验收结果。
优化前快照为 `13c4149`，已推送到
`https://github.com/Tangent0308/sam3-crispedit/tree/samtok-derived-edit-labeling`。
`origin` 保留 MIRAGE upstream；业务分支推送 remote 为 `sam3`。

## 数据和实验纪律

- 基线：`pilot_100_audit27_seed20260922`，100 条，四类型各 25 条。
- 实验根目录：
  `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/quality_iteration_20260919`。
- 生成开发集：16 条，四类型各 4 条；另外抽取 source 不重叠的 20 条留出样本。
- 所有生成对照保持原图、原始指令、40 steps、seed=0、CFG=4；不同算法的
  输入尺寸/扩散过程不同，不声称逐像素输出一致。
- 人工标签指 assistant 直接查看原图/结果的逐图视觉复核，并非另一个人类标注团队。
  生成复核先于读取新 VLM 决策。旧 100 条标签保持冻结；存在争议的旧标签不偷偷修改。
- 原始数据、旧结果、失败输出、失败实验均保留。数据/模型/运行日志不进入 Git。

## 1. 生成端问题与实现

### 1.1 原 MIRAGE 的约束为什么可能造成残留

原流程先对局部分支去噪，再将分支 latent 写回全图，并在全图去噪中按 write map
混合参考图。SAMTok 的实例分割边界并不等于编辑影响范围：替换的新轮廓、移除后的
背景、阴影、支撑部件可能超过原 mask。反复恢复参考区域可能把原轮廓或附件带回来。
这与错误指代、生成模型本身的形状错误是不同问题，不能单靠放大 mask 全部解决。

还发现分支预测曾使用全图 timestep，而分支 scheduler 的 dynamic shift 可因 token
数量不同而不同。新增 `correct_branch_schedule` 实验开关，默认不改旧路径。
没有做这个单因素的独立质量对照，因此不能把全部质量改善归因于 scheduler 修复。

### 1.2 实际测试的四种生成方式

`inference_mydemo_qwen2511.py` 和 `run_qwen_edit_pool.py` 支持 `--edit-method`：

| 方式 | 实现与定位 |
|---|---|
| `mirage` | 保留原始默认路径，便于复现 |
| `mirage_relaxed` | 更大分支上下文/写入范围、更多局部步数、独立 scheduler timestep 的组合消融 |
| `official_full` | 官方 Qwen-Image-Edit 对干净全图直接编辑 |
| `context_edit` | 带上下文的干净 crop 直接编辑，完成去噪后仅一次像素域回贴 |
| `context_edit_v2` | 同上，但 remove 使用整个上下文窗口回贴；扩大依附物编辑能力，也可能误删邻近实例 |
| `context_adaptive` | 实验性：仅 attribute 使用双参考定位 + 严格原 mask 回贴，其余使用 context_edit；本轮整体效果未超过 context_edit，不作为通用推荐 |

`utils/context_edit.py` 的具体参数：

1. mask bbox 每侧扩展 bbox 对应边长的 75%，至少 128 像素，裁到原图边界。
2. 输入 Qwen 的是干净矩形 crop，**没有红色 overlay，也没有抠空背景**。
3. 使用原有短编辑指令，40 steps，生成后缩回相同 crop 尺寸。
4. `context_edit` 的 remove/replace/attribute：回贴 bbox 加一圈安全边界，
   边界为 `max(32 px, 20% × mask bbox 短边)`，Gaussian blur 10 px。
5. add 的 mask 是放置 anchor，使用整个 context window；v2 的 remove 也如此。
   只有**图像内部裁剪边界**做最多 24 px 的线性羽化。
6. 原图最外缘绝不能混回旧像素：已修复此前羽化使边缘目标残片恢复的 bug，
   用缓存 raw crop 对 9 条移除案例重新合成，未新增扩散调用。
7. crop 外像素原样保留。crop 内仍可能有不必要修改，必须审核。

重要：保留的 SAMTok `mask` 是**原始目标/anchor 的实例标注**，不是新图的
精确变化分割。更宽的编辑区域不能伪称 mask 外像素完全不变；需要严格变化 mask 的
训练配方，应另行定义和输出 change/support mask，而不是误用原始实例 mask。

### 1.3 逐图复核结果

“画面合格”包括目标区域、边缘、物理支撑以及无关实例保持；“原指令也合格”进一步
要求旧指令的对象/操作正确。自然但语义不同的图允许进入独立重建阶段。

| 方法 | 开发 16：画面合格 / 原指令也合格 | 留出 20：画面合格 / 原指令也合格 |
|---|---:|---:|
| 原 MIRAGE | 8 / 8 | 12 / 9 |
| 放宽 MIRAGE 组合消融 | 6 / 5 | 未扩展，首轮较差 |
| 官方全图直接编辑 | 10 / 9 | 8 / 7 |
| context_edit | **15 / 14** | **15 / 12** |

不是所有样本均改善。例如 037 同时把背景成年象变亮，056 将旧指令所说的细条变成整块木墙（真实 mask 范围另见 6.3），
070 新铃铛悬空，094 勺子与耳朵脱离。新生成图不能直接全部作为合格训练数据。

宽窗口 remove 复查 9 条有 8 条通过。033 棒棒糖的棍子残留得到修复；但 067
额外删了另一把椅子。因而 **v2 不自动替代 context_edit 为通用默认**，保留显式开关。
36 条混合验证集含这个失败反例，不能只选最好输出后报告自动方法成功率。

可视化（相对于实验根目录）：

- `generation_gallery/index.html`：16 条 × 基线和三种新方法。
- `holdout/generation_gallery/index.html`：20 条 × 基线、全图、context。
- `remove_iteration/generation_gallery/index.html`：9 条移除的原图边界 bug 修复前后。
- `validation_v2/report/index.html`：新图的所有模型理由、原始响应、prompt、人工理由。

## 2. 双图输入与简化审核

### 2.1 full、crop、overview 的区别

所有模式均只有两张输入，IMAGE 1 是原图，IMAGE 2 是结果：

- `full`：原生画幅的完整图，原图标注黑白外轮廓，结果干净。
  VLM processor 仍有 min/max pixel 预算；“完整图”不代表内部不缩放。
- `crop`：两图同一坐标区域，保留上下文，每侧 bbox 的 75%、至少 80 px，
  longest side 1280；没有抠空背景。
- `overview`：每张输入是一张 1024×1344 的双尺度图，上方完整场景，下方目标上下文。
  仍只有两张图片，但每张含两个视图。目的是让指令里的左右位置基于全图，
  不把 crop 中心误当作全图中心。它是实测候选，不预设比 crop 更好。

实际送入模型的 PNG 保存到 `inputs_crop/`、`inputs_full/`、`inputs_overview/`。

### 2.2 审核只输出三个字段

```json
{"visual_quality":"pass|fail","instruction_match":"pass|fail","reason":"具体的前后视觉证据"}
```

- `visual_quality`：清楚可见、局部、基本自然；残留、重影、明显接缝、错误支撑、
  明显无关修改 fail。原图本来模糊不是新增缺陷。
- `instruction_match`：原指令的实例、操作、身份/属性、数量是否吻合。
- `reason`：用于定位审核错误和完整可视化；不再要求模型生成长检查表。
- 不输出 review。无法解析不能当作 pass。
- 旧指令的总 pass 要求两个字段都 pass；**重建资格只看画面质量**，这是有意区分。
- remove/replace 仍保留像素低变化检查，不能用它取代语义判断。

实际完整 prompt 为 `synthesis_pipeline/audit_quality_v3.py::COMPACT_PROMPT`，
每条调用同时保存实际渲染的 prompt，报告可展开查看。

### 2.3 原 100 条上的非 thinking 对照

| 输入/prompt | 与冻结人工总判断一致 | 与画面判断一致 | 画面误放行 / 39 个坏图 |
|---|---:|---:|---:|
| 原复杂 prompt + crop | 72% | 70% | 15 |
| 原复杂 prompt + full | 66% | 68% | 24 |
| 三字段 compact + full | 73% | 77% | 21 |
| 三字段 compact + crop | **80%** | **81%** | **11** |
| 不给旧指令，quality-only + full | 不适用 | 71% | 26 |
| 不给旧指令，quality-only + crop | 不适用 | 74% | 19 |
| 更长的 grounded/evidence-first + crop | 61% | 64% | 36 |

compact crop 在 source-disjoint 的旧 40 条留出集画面一致率 85%，误放行 3 条。
**full 没有更好，长 prompt 也没有更好**，因此不据直觉切换默认。
新生成 36 条的 compact crop 仍漏掉 5/6 个画面失败案例；小样本上相同 prompt
并不保证可靠。不能把旧数据的 81% 一致率当成最终训练数据的 81% 合格率。

## 3. 独立指令重建：一轮额外调用

实现 `--rewrite-after-audit`：同一常驻 Qwen3.8-27B 实例先审核，随后只为
`visual_quality=pass` 的图再调用一次。没有加载第二个模型，也没有把旧 instruction
喂给改写模型，防止模型追随计划而不是图片。

输出仅：

```json
{"task_type":"add|remove|replace|attribute","instruction":"One concise imperative sentence."}
```

要求原图可明确定位、优先 <25 words、硬上限 35；不写 mask/outline/selected，
不补写“其他地方不变”；无有用变化、残留、错实例或坏图返回两个 null。
重新判断操作类型：颜色变化用 attribute，不因为旧文件名有 replace 就保留 replace。

`corrected_annotation()` 将新的 `task_type` 和 `editing_instruction` 真正写入
`model_accepted_annotations.jsonl`，并在 `instruction_revision` 保留原指令、类型、
审核理由和 `model_accepted_not_manually_verified` 标记。不覆盖原始 annotation。

改写不能挽救坏图。实测模型仍会把已有物体认作新增、把起重机认作十字架、用 crop
相对方位指代全图，或把残留解释为有意保留。因此模型候选与逐图验收后的
`assistant_verified_annotations.jsonl` 严格分开；不能把 candidate 数当作正确条数。

## 4. 复现和运行

### 4.1 生成（8 GPU 常驻 worker）

下面 `$DATA`、`$OUT` 是调用者指定的原始数据目录和**新的**输出目录：

```bash
python -m synthesis_pipeline.run_qwen_edit_pool \
  --python /opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  --image-root "$DATA/sources" \
  --instruction-jsonl "$DATA/annotations.jsonl" \
  --crop-dir "$DATA/crops" \
  --results-full-dir "$OUT/edited" \
  --model-id /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511 \
  --gpus 0,1,2,3,4,5,6,7 --edit-method context_edit \
  --num-steps 40 --seed 0 --true-cfg-scale 4
```

用真实 pool 在 3 卡完成留出 20 条，墙钟 557 秒（含 worker 加载），约 2.15 条/分钟。
常驻单卡直接上下文编辑约 76–78 秒/条。这不是严格同负载吞吐基准；已有共享 GPU
进程未被终止，8 卡全量速度应单独测量，不能把 3 卡小批次速度线性承诺为 SLA。

### 4.2 审核与重建（独立 audit GPU）

先让输出数据目录具有 `sources/`、`annotations.jsonl`、`edited/`，或通过
`--edited-dir` 指定新图位置。模型原图文件名使用 `source_image`。

```bash
CUDA_VISIBLE_DEVICES=0 \
PATH=/opt/tiger/tanyue/.venvs/qwen38_audit/bin:$PATH \
/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python \
  synthesis_pipeline/audit_quality_v3.py \
  --data-root "$DATA" --edited-dir "$OUT/edited" \
  --out-root "$OUT/audit" --variants compact_crop \
  --rewrite-after-audit --rewrite-scope overview --batch-size 8 \
  --target-change-threshold .02
```

每 case 审核一次，画面 pass 再重建一次。原 100 条 compact crop 的纯模型推理
148 秒（40.5 条/分钟），含图像准备、PNG落盘、指标计算约 305 秒（19.7 条/分钟），
模型加载另计。新 36 条审核纯推理 54.7 秒；31 个画面 pass 的重建纯推理 35.0 秒。
这些是非 thinking 的运行值，不应混用为 thinking 速度。

环境使用 vLLM 0.28.0、Transformers 5.17、bf16、TP=1、batch=8。
必须把 venv/bin 放入 PATH：只使用绝对 python 路径会使 FlashInfer 找不到 `ninja`。
本轮该初始化错误未产生样本输出，修正 PATH 后重跑，原失败日志保留。

可选 `--thinking` 使用本地官方模型卡的 thinking chat template 和采样参数
temperature=1.0、top_p=.95、top_k=20、presence_penalty=0，输出上限默认 2048。
使用 `--reasoning-effort low|medium|xhigh` 控制深度，CLI 默认 low；最早一轮
未显式指定，模型模板默认 xhigh。支持来自本地官方模型卡及 chat_template.jinja。
不增加调用次数，但延迟和 token 消耗更高；仅解析 `</think>` 后的最终 JSON。
原非 thinking 结果使用 temperature=0 的确定性消融配置，不宣称它是官方推荐采样。

## 5. 工程安全和验收边界

- 原 mirage 路径默认保持不变，实验方式显式选择。
- 不覆盖原图/旧生成结果/旧指令；报告标注每条输出来源和人工理由。
- 自动结果文件明确区分模型候选与视觉复核通过的数据。
- 单元测试覆盖回贴外部像素不变、真实原图边缘不恢复旧像素、解析拒绝无效输出、
  thinking 中间 JSON 不被误用、坏图不能通过改写门控、原标注不被原地修改。
- 该实验已改善生成质量，但在确认自动审核精度之前，不将本轮结果包装为可无人值守
  接收全部输出的 100k 生产配置。

## 6. 后续迭代、失败分支与最终交付

### 6.1 没有采用的生成方案

继续做了 8 次双参考出图：干净 crop + 同一 crop 的黑白轮廓参考图。
虽然部分目标选得更准，模型会把轮廓直接画入结果，形成贴纸边；8 条均未达到
本轮画面验收标准。因此普通 add/remove/replace 不采用这种输入。

进一步只对 attribute 使用该参考，再在最终像素域仅回贴原 mask 内部：
mask 腐蚀 1 px、Gaussian blur 1.5 px，最后乘原始 binary mask，使 mask 外
严格不变。9 条中 3 条复用缓存扩散结果、6 条重新扩散；逐图复核 6 条画面合格。
它修复了小象误改背景成年象、箱子误改前景箱的问题，但有新的退化：

- 019 颜色变化不够明确；
- 023 方向盘本来就在人物 mask 外，严格回贴后根本不能完成该指令；
- 053 蓝色衣服变成缺少布料细节的平涂区域。

所以 `context_adaptive` 保留为实验开关，没有因为两个成功案例就推广到全部属性任务。
相关结果见 `guided_iteration/guide_only_gallery/index.html` 和
`guided_iteration/attribute_gallery/index.html`。

最后又对 8 条属性任务增加内部的纹理/光照保留要求和反平涂、反轮廓负向提示，
仍保持训练指令不变。`context_guided_attribute_v2` 的画面合格为 6/8，
对应前一版同 8 条是 6/8：053 的蓝色衬衫恢复布料层次，但 043 出现浅色细边和
不自然焦斑，019 的变化仍不清楚。因此不将它作为通用默认，也不靠挑图报告净提升。
完整对照与逐图理由：`guided_iteration/attribute_v2_gallery/index.html`。
每次实际内部 prompt、negative prompt、参数和输入数量保存于 diagnostics 的
`generation_request.json`。本轮共 **119 次新扩散出图**，覆盖 36 个独立 case，
另有缓存像素重合成；不把重合成重复计算为生图调用。

`validation_adaptive` 的 36 条 medium-thinking 审核仍误放行 7 个坏图中的 5 个；
33 个画面 pass 进入独立重建，得到 32 个模型候选。这些不是 32 条验收合格数据。
自动审核仍会漏掉平涂、弱变化和残留问题。

### 6.2 完整 100 条的 thinking 对照

| 模式，同一 compact crop prompt | 总判断一致率 | 最终 JSON 有效率 | 纯模型推理时间 |
|---|---:|---:|---:|
| non-thinking | **80/100** | 100/100 | **148 秒** |
| thinking low，2048 tokens | 79/100 | 100/100 | 903 秒 |
| thinking xhigh，2048 tokens | 61/100（有效输出内 61/79） | 79/100 | 1686 秒 |

thinking low 的画面误放行 14 条，non-thinking 为 11 条；它没有带来稳定收益。
xhigh 有 21 条未完成最终答案，必须 fail-closed，不能将截断当作正确的坏图判断。
因此不把 thinking 默认打开。已停止另一个 quality-only+xhigh 分支：48 次完成请求
中有 22 次未完成 reasoning；这个分支没有完整 100 条结果，不进入完整基准表。

### 6.3 Mask 内外检查不能省略

仅看轮廓不一定能分清带孔 mask 的内外。`mask_evidence/` 保存了原图、二值 mask
和仅用于人工排查的 cutout（**这些调试图没有作为新的模型输入**）。例如：

- 023 的实际 mask 是人物，方向盘在 mask 外；新图内部变化像素仅 0.64%。
- 056 的 mask 面积约 91%，实际是墙面，不是旧规划文字中的细条。

这说明规划描述、真实 mask、实际修改三者可能不一致。新增可选
`--target-change-threshold .02`：对非 add，实际目标内部变化不足 2% 时拒绝
进入重建/入库，保留模型原始判断及独立几何拒绝原因。add 的 mask 可以作为 anchor，
不应用该硬条件。阈值偏保守，可能拒绝很细小的真实变化，不是万能的语义检查。
几何计算不增加 MLLM 调用。旧基准表未事后偷偷加入这个 gate。

056 的木墙本身可能自然，但旧规划和真实 target 范围冲突；本轮验收版暂不收入。
旧冻结标签仍保留，不能把与冻结标签一致当作客观无争议的真值。

### 6.4 26 条逐图验收版

路径相对于实验根目录：

- `release_verified/index.html`：最终 **26 条验收通过**的对比与新指令。
- `release_verified/cases/*.jpg`：逐条可直接预览的完整场景与目标放大对比。
- `release_verified/annotations.jsonl`：合并后的可用标注，原始 mask 以 COCO RLE 保留。
- `validation_v2/report/verified.html`：其中 24 条主验证集样例。
- `validation_v2/report/index.html`：全部 36 条，包括失败原因和所有审核/重建候选。
- `validation_v2/assistant_verified_annotations.jsonl`：实际可使用的已复核标注。
- `validation_v2/curation_decisions.json`：每条人工选择哪个模型候选、为什么拒绝其余条目。
- `validation_adaptive/report/verified.html`：另外 2 条局部属性修复（037 小象、083 后排箱子）。
- `validation_adaptive/report/index.html`：该分支全部 36 条及完整模型/人工理由。

最终构成为 add 6、remove 8、replace 5、attribute 7。这是人工逐条选择后的数据，
不是某个单一自动配置取得 26/36 的质量保证；
不同样本选择的重建候选来源明确记录在 `instruction_revision` 中，原图/结果不重画。
未接受任何“给坏图编合理故事”的指令。例如 044 原本要求 DualShock 替换，实际
干净移除了 Nunchuk，因此重建为 `Remove the white nunchuk controller held in the
foreground adult's hand.`；043 则从“稍微烤过”改为实际可见的黑色焦烤表面。

文件名保留旧任务类型以便追溯；训练时以 JSON 的新 `task_type` 和
`editing_instruction` 为准，不能再从文件名猜类型。质量过滤和实际操作重新分类后
四类数量不再严格均衡，不为凑配额接受坏图。

本轮回归测试：42 passed、1 skipped。跳过项需要 diffusers，已在生成环境单独
运行并通过原始 write-weight 等价性和 dilation 检查；审核/合成单元测试不加载大模型。

结论：生成端 `context_edit` 有明确的小规模提升，审核精简后也优于旧基线；
但 **27B 的自动审核和独立重建仍未达到无人工放行的可靠性**。本轮交付有逐图复核
的可用样例、可复现代码和完整失败证据，不宣称已解决所有质量问题或可直接无人值守
启动 100k。后续若要求自动入库，需要在新的 source-disjoint 样本上继续验收，
并考虑专门的审核训练数据；不能只凭更长 prompt 或开启 thinking 宣称问题解决。

## 7. 可视化预览与实际速度

`release_verified/index.html` 已改为内嵌 JPG 图片，不依赖浏览器访问上级实验目录或
跨目录符号链接；各案例另存 `release_verified/cases/*.jpg`，直接点击图片文件也能看。
每条对照上排为完整原图（黑白轮廓表示原始目标 mask）与编辑图，下排为相同坐标的
目标邻域放大图。`validation_v2/report/index.html` 和
`validation_adaptive/report/index.html` 同样内嵌图像，并保留逐条模型理由、原始响应
及人工复核意见。可视化未重新生成或修改训练图像/标注。

以下数据来自这轮保存的运行记录；各阶段批次不同，因此**不是一次完整串行任务的
实测总耗时**：

| 阶段 | 实测配置与数量 | 墙钟耗时 | 折合速度 |
|---|---|---:|---:|
| 原图采样 + 区域指令规划 | 100 个最终 case，含过采样/兼容性校验和 784 次 VLM 请求 | 507 秒 | 11.8 最终 case/分 |
| 原图、mask、crop 物化 | 100 case | 116 秒 | 51.7 case/分 |
| `context_edit` 出图 | 3 GPU 常驻 worker，20 case，40 steps，含模型加载 | 557 秒 | **2.15 case/分** |
| 精简双图 27B 审核 | 36 case，模型已加载；含图片准备和落盘 | 112 秒 | 19.3 case/分 |
| 独立重建指令 | 前一步视觉通过的 31 case，复用同一已加载模型 | 85 秒 | 22.0 被重建 case/分 |

出图单卡稳定阶段约 74–79 秒/条。另一轮开发分片曾记录约 239 秒/卡的模型加载，
但它与本表的 20 条、3 卡 pool 不是同一次运行，不能加在 557 秒上；小批次以 pool
的整体墙钟为准。模型审核/重建首次加载约 87 秒，合并运行时只加载一次。
按这些不同批次的**串行阶段**简单折算，
100 个候选 case 约 66–68 分钟（约 1.5 case/分，含一次审核模型加载）；这只是
同条件的粗估，不包括人工逐图验收，也没有对 8 GPU 全量作吞吐承诺。
目前主要耗时仍是 Qwen-Image-Edit 出图。26 条验收版是跨实验逐条筛选，
不能拿 26/36 估计生产级自动合格率。
