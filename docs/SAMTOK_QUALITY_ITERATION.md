# SAMTok 派生编辑数据：质量优化与迭代记录

本文件是所有质量迭代的唯一主记录，按时间保存实验、失败原因、修复、运行命令与复核结果。后续迭代继续追加在这里，不再新增按日期拆分的记录文档。历史数字保留其当时评判口径，后续修正以较新的章节为准。

- [最新：新源图开发与留出验证 v10](#fresh-v10)

- [历史：语义编辑范围与实例保护 v8](#iteration-v8)
- [当前单一版本 bad case 诊断](#diagnosis-v6)
- [2026-09-20 mask 对齐与审核规则回归](#repair-20260920)
- 2026-09-19 的生成、双图审核与指令重建实验见下文首章。

## 2026-09-19：生成边界、双图审核与独立指令重建


本文记录实际跑过的实验，不把模型的 pass 当成人工验收结果。
优化前快照为 `13c4149`，已推送到
`https://github.com/Tangent0308/sam3-crispedit/tree/samtok-derived-edit-labeling`。
`origin` 保留 MIRAGE upstream；业务分支推送 remote 为 `sam3`。

### 数据和实验纪律

- 基线：`pilot_100_audit27_seed20260922`，100 条，四类型各 25 条。
- 实验根目录：
  `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/quality_iteration_20260919`。
- 生成开发集：16 条，四类型各 4 条；另外抽取 source 不重叠的 20 条留出样本。
- 所有生成对照保持原图、原始指令、40 steps、seed=0、CFG=4；不同算法的
  输入尺寸/扩散过程不同，不声称逐像素输出一致。
- 人工标签指 assistant 直接查看原图/结果的逐图视觉复核，并非另一个人类标注团队。
  生成复核先于读取新 VLM 决策。旧 100 条标签保持冻结；存在争议的旧标签不偷偷修改。
- 原始数据、旧结果、失败输出、失败实验均保留。数据/模型/运行日志不进入 Git。

### 1. 生成端问题与实现

#### 1.1 原 MIRAGE 的约束为什么可能造成残留

原流程先对局部分支去噪，再将分支 latent 写回全图，并在全图去噪中按 write map
混合参考图。SAMTok 的实例分割边界并不等于编辑影响范围：替换的新轮廓、移除后的
背景、阴影、支撑部件可能超过原 mask。反复恢复参考区域可能把原轮廓或附件带回来。
这与错误指代、生成模型本身的形状错误是不同问题，不能单靠放大 mask 全部解决。

还发现分支预测曾使用全图 timestep，而分支 scheduler 的 dynamic shift 可因 token
数量不同而不同。新增 `correct_branch_schedule` 实验开关，默认不改旧路径。
没有做这个单因素的独立质量对照，因此不能把全部质量改善归因于 scheduler 修复。

#### 1.2 实际测试的四种生成方式

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

#### 1.3 逐图复核结果

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

### 2. 双图输入与简化审核

#### 2.1 full、crop、overview 的区别

所有模式均只有两张输入，IMAGE 1 是原图，IMAGE 2 是结果：

- `full`：原生画幅的完整图，原图标注黑白外轮廓，结果干净。
  VLM processor 仍有 min/max pixel 预算；“完整图”不代表内部不缩放。
- `crop`：两图同一坐标区域，保留上下文，每侧 bbox 的 75%、至少 80 px，
  longest side 1280；没有抠空背景。
- `overview`：每张输入是一张 1024×1344 的双尺度图，上方完整场景，下方目标上下文。
  仍只有两张图片，但每张含两个视图。目的是让指令里的左右位置基于全图，
  不把 crop 中心误当作全图中心。它是实测候选，不预设比 crop 更好。

实际送入模型的 PNG 保存到 `inputs_crop/`、`inputs_full/`、`inputs_overview/`。

#### 2.2 审核只输出三个字段

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

#### 2.3 原 100 条上的非 thinking 对照

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

### 3. 独立指令重建：一轮额外调用

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

### 4. 复现和运行

#### 4.1 生成（8 GPU 常驻 worker）

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

#### 4.2 审核与重建（独立 audit GPU）

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

### 5. 工程安全和验收边界

- 原 mirage 路径默认保持不变，实验方式显式选择。
- 不覆盖原图/旧生成结果/旧指令；报告标注每条输出来源和人工理由。
- 自动结果文件明确区分模型候选与视觉复核通过的数据。
- 单元测试覆盖回贴外部像素不变、真实原图边缘不恢复旧像素、解析拒绝无效输出、
  thinking 中间 JSON 不被误用、坏图不能通过改写门控、原标注不被原地修改。
- 该实验已改善生成质量，但在确认自动审核精度之前，不将本轮结果包装为可无人值守
  接收全部输出的 100k 生产配置。

### 6. 后续迭代、失败分支与最终交付

#### 6.1 没有采用的生成方案

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

#### 6.2 完整 100 条的 thinking 对照

| 模式，同一 compact crop prompt | 总判断一致率 | 最终 JSON 有效率 | 纯模型推理时间 |
|---|---:|---:|---:|
| non-thinking | **80/100** | 100/100 | **148 秒** |
| thinking low，2048 tokens | 79/100 | 100/100 | 903 秒 |
| thinking xhigh，2048 tokens | 61/100（有效输出内 61/79） | 79/100 | 1686 秒 |

thinking low 的画面误放行 14 条，non-thinking 为 11 条；它没有带来稳定收益。
xhigh 有 21 条未完成最终答案，必须 fail-closed，不能将截断当作正确的坏图判断。
因此不把 thinking 默认打开。已停止另一个 quality-only+xhigh 分支：48 次完成请求
中有 22 次未完成 reasoning；这个分支没有完整 100 条结果，不进入完整基准表。

#### 6.3 Mask 内外检查不能省略

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

#### 6.4 26 条逐图验收版

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

### 7. 可视化预览与实际速度

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


<a id="repair-20260920"></a>

## 2026-09-20：mask 对齐、回贴范围与审核入库规则回归

本轮已执行，而不是待执行方案。结论是：修复了输入轮廓、重试反馈和部分标签校验问题；
若干具体出图问题改善，但**新生成策略和新审核 prompt 均未达到可直接扩大到 100k 自动入库的质量**。
不把模型 pass 当人工验收，不以挑选成功案例代表整体通过率。

### 数据与复现条件

- 仓库分支：`samtok-derived-edit-labeling`；开始时版本 `07319cf`。
- 输入：`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_100_audit27_seed20260922`。
- 本轮根目录：同级 `quality_iteration_20260920`，下文记作 `$ITER`。
- 24 个固定 region 重跑三版规划；其中包含 8 个未参加此前生成方法对照的 region。
  这仍是开发回归集，不是独立随机测试集。
- **32 次新的扩散生成**：旧指令 16 次、新规划 v6 指令 14 次、v7 指令 2 次。
  一些 region 多次生成，不能将 32 次称为 32 个独立 source。
- 40 steps、seed=0、CFG=4、Qwen-Image-Edit-2511；32 张均由 assistant 看过完整图及局部图。
- 另有缓存 raw crop 的多版回贴实验，**不增加扩散调用**。
- 27B 在旧 36 对上做三种审核，在新 14 对上再审核并改写；均使用 vLLM、非 thinking。
- 未重启全量标注；旧图片、旧标签和旧审核均保留。GPU 上原有的其他进程未被中止。

### 1. 规划：先厘清 mask 实际选中了什么

#### 已实现

1. 保持两张输入：干净完整原图 + 保留背景的上下文 crop，不使用红色 overlay 或 cutout。
2. crop 增加照片外留白，贴图像边界的 mask 也能画出闭合外轮廓。原实现只在照片内部画外轮廓，
   对贴边大区域可能显示成其补集中的细线，尤其容易误认墙面/金属条。
3. 从真实 RLE 计算全图 mask 面积、bbox、触碰的边界、3×3 网格覆盖率，加入输入 prompt。
   这些是几何证据，不是模型生成的语义描述，也不进入训练 instruction。
4. 使用原始数据集的 problem/answer 作为身份交叉核对，去除 SAMTok token；明确可能同时描述多个
   region，不能把其他 region 当当前目标。正常规划主流程可直接读取已有 row；回归脚本使用
   `--reference-jsonl .../planning/candidate_source_rows.jsonl`。
5. 重试现在包含上次实际无效 JSON 和明确反馈。旧代码仅说“上次失败”，没有把上次答案交回模型。
6. 增强类型要求：add 明确附着/支撑，不凭空引入支撑实体；remove 检查属于目标自身的结构部件；
   replace 不额外创造互动关系；attribute 避免仅 slightly 的弱变化。
7. 同类替换校验检查新名词短语的末端中心词，不再因前面多了形容词而把“墙换另一种墙”放过。

#### 实测与未解决问题

规划输出及完整 prompt：`planning/`（v5）、`planning_v6/`、`planning_v7/`。
24 条分别有 20、14、14 条通过当时版本的格式/规则校验，**不是语义合格数**。
请求数分别 47、49、48。v7 运行后补上的同类中心词校验会再拦截 056 的同类墙面替换；旧响应不覆盖。
回归脚本不执行主采样器的最终 deterministic fallback，因此这不是主采样器最终保留率。

- 023：目标由错误的方向盘变成右下角人物；但新的“头部和肩部变红”生成纯红剪影，仍然失败。
- 056：模型能识别整块墙面，而不是细金属条；仍可能提出同类材质变化却标 replace，已增加规则拦截。
- 091：纠正到拱门**左侧**吊臂，v7 生成红色吊臂，逐图复核通过。
- 070：v6 从悬空铃铛改成系在喇叭口的紫色蝴蝶结，新图复核通过。
- 033：即使强调结构部件，模型仍会把球体 mask 当成包含木棍的完整 cake pop。没有宣称该问题已解决。
- 多实例指代仍可能过宽，例如 067 的 left foreground 无法可靠区分多把白椅子。

### 2. 生成：目标 mask 与可写范围分开

`context_guarded_v2` 已接入单卡入口和多卡动态队列，**显式选择才启用，旧默认未改变**。

- 模型只接收干净 context crop，不输入标注轮廓，避免把轮廓复制到结果。
- 内部生成 prompt 加目标在 crop 中的大致位置以及摄影纹理/支撑约束，训练标签保持短句。
- attribute：仅在原 mask 内做一次晚期合成，mask 外逐像素保持原图。
- remove：沿 mask 形状允许有限外扩，不反复注入旧 latent。
- replace：v1 收紧范围会裁掉新轮廓；v2 改成上下文可写范围，容纳新物体。
- 保护同 source 的其他已标注实例 mask；不能据此声称未标注的邻居也得到保护。
- 保存 `source_crop.png`、`raw_edited_crop.png`、`composition_alpha.png`、
  `protected_instances.png` 和 `generation_request.json`，区分生成缺陷与回贴缺陷。
- 原始 `mask` 始终是目标/anchor 标注，alpha 是另一个编辑支持范围，二者不混用。

#### 可复核结果

| 对照组 | 新扩散次数 | assistant 画面复核通过 |
|---|---:|---:|
| 固定旧指令，guarded v1 | 16 | 9 |
| 仅修复 replace 的宽回贴，复用同一批 raw | 0 | 10/16 |
| v6 新指令 + guarded v2 | 14 | 7 |
| v7 新指令补测 083、091 | 2 | 2；其中仅 091 完整匹配原指令 |

画面通过不等于可入库。例如 062 横幅看上去自然但相对 anchor 扩张过大；083 只改变纸箱一面，
不能沿用“整箱全白”的指令。两个新规划组不是相同样本集，不能横向据此报告提升百分比。

016 的 raw 出租车本来有完整车轮，v1 回贴将其截断；扩大 replace 范围后恢复完整落地，
是明确定位并修复的合成缺陷。027 能完整移除前景长颈鹿并保留后方实例；053 衬衣变蓝有自然明暗。
仍失败的典型：037 小象形状漂移后出现双轮廓，047 桌面补全发白模糊，067 额外删椅子，
064 误删原本已存在的另一位女子。

#### 后续缓存合成消融：不作为默认

`recompose_guarded_edits --composition connected` 根据与原目标相交的变化连通域扩展可写区域：
RGB 平均差阈值 0.08，7px 闭运算，保留与 mask 相交的分量，再做小幅羽化及邻居保护。
这能覆盖 033 木棍的变化，但会产生粉色底座上的不规则色块接缝，也可能把邻居的变化连进去。
`connected_poisson` 进一步尝试泊松融合；对触及 crop 边缘的支持范围自动回退，防止恢复原目标边缘。
本批 016/027/033 实际回退，结果与 connected 相同；047/067 有求解变化，但没有解决根本缺陷。
因此两种策略均保留为失败/待改进实验，不能宣传成已解决边缘问题。

### 3. 审核：降低旧 instruction 的诱导，同时保留失败证据

新增 `correspondence_crop/full`：先跟踪同一实例，再判断请求，输出仍为三个字段。
新增 `forensic_overview`：不提供旧 instruction，只观察实际变化、边界、支撑和邻居；
输入仍是两张图，每张上方全景、下方局部，输出 `observed_change`、`reason`、`quality`。
通过画面审核后再独立调用一次模型改写，仍是 **1 次审核 + 仅对通过者 1 次改写**。

旧 36 对的复核修订单独保存为 `adjudication_overrides.jsonl`：
052 人物腿脚缺损，应为画面 fail；056 真正 mask 是整面墙，应为画面 pass、旧指令 match fail；
023 改了人物 mask 外的方向盘，旧自然度 pass 忽略了指定目标，本轮应按目标错位拒绝。
旧 `manual_review.jsonl` 没有覆盖。

| 审核方式 | 正确放行 | 错误放行 | 正确拒绝 | 错误拒绝 |
|---|---:|---:|---:|---:|
| 原 compact_crop | 26 | 5 | 2 | 3 |
| correspondence_crop | 29 | 5 | 2 | 0 |
| correspondence_full | 29 | 5 | 2 | 0 |
| forensic_overview | 21 | 2 | 5 | 8 |

均按上述三条修订后的 assistant 标准计算，并非独立人类真值。
correspondence 总体一致率较高但错误放行数没有下降，不采用它直接替代旧默认。
forensic 能发现 052 腿脚、070 悬空铃铛、094 悬空勺子，但增加误拒；037/067 仍漏判。
在新 14 对上，forensic 与逐图复核仅 **8/14 一致**：错误放行 043/047/067，错误拒绝 027/075/094。
不能称审核已可靠，也不能据旧 36 对的低误放率宣称泛化改善。

#### 不增加模型调用的准入规则

- `--target-change-threshold .02`：非 add 必须在原目标内有实际变化。
- `--max-add-change-ratio 3`：可选的细粒度范围限制，变化面积不超过 anchor 面积三倍。
  它衡量像素变化范围，不是新物体精确面积；不同任务可关闭。
- `--reject-flat-attribute`：原目标 dominant RGB bin <35%、结果 >=85% 时拦截疑似纯色填充。
  新 023、099 确认发生纹理塌陷；这不是通用美学评分。
- 改写跨 add/remove/replace/attribute 类型默认不自动接受；实验可显式
  `--allow-task-type-change`。合法跨类型救回交给人工验证，不把它们说成模型图像失败。
- 拦截单实例变成“two/both...”的计数漂移，以及替换指令凭空引入多个互动实体。
- 拦截“remove the white outline”等将输入标注当实际编辑的标签。
- 原模型回复、规则拦截原因与人工理由分别保存。`model_accepted_annotations.jsonl`
  仍明确标为 `model_accepted_not_manually_verified`，不是直接交付训练集。

`revalidate_saved_rewrites.py` 可用新规则重新检查旧响应，0 次模型调用、不覆盖旧实验。
旧 031/064/091 的错误改写会被保守规则挡住，但这是准入限制，**不是证明 27B 理解正确**。
新 14 对中，7 条进入改写；最新规则保留 5 个模型候选，仍含需人工剔除的错误。

### 4. 速度、运行方法与产物

- 新扩散 32 次，平均约 **77 秒/张/卡**（已加载模型、包含保存诊断），没有额外扩散调用。
- 3 卡理想稳态约 2.3 张/分钟；这不是 8 卡实测，也不是合格数据入库速度。
- 8B v6：24 条、49 次含重试请求，推理 27.88 秒，模型加载 39.82 秒。
  v5 冷加载曾达 355 秒，应与稳态推理分开。
- 27B 新 14 条：审核推理 32.49 秒、含输入处理 52.11 秒；7 条改写推理 3.46 秒、
  含图片处理 13.39 秒；模型加载 83.03 秒。审核和改写共用驻留模型。
- 16 条缓存合成约 12 秒，0 次 GPU 扩散/MLLM 调用。

示例，均在仓库根目录执行；输出使用新目录，避免混入旧结果：

```bash
GEN_PY=/opt/tiger/tanyue/.venvs/mirage_official/bin/python
AUDIT_PY=/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python
DATA=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_100_audit27_seed20260922
ITER=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/quality_iteration_20260920

## 8B 规划回归；所有原始指代与 mask 来自指定数据集。
CUDA_VISIBLE_DEVICES=0 "$GEN_PY" -m synthesis_pipeline.replan_edit_regression \
  --data-root "$DATA" --out-root "$ITER/planning_next" --ids 23,33,70,83,91 \
  --reference-jsonl "$DATA/planning/candidate_source_rows.jsonl" \
  --model-id /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3-VL-8B-Instruct

## 普通多卡入口也支持 --edit-method context_guarded_v2，默认方法不变。
## 回归入口 --shard/--shards 将不同 case 分给不同 CUDA_VISIBLE_DEVICES。
CUDA_VISIBLE_DEVICES=2 "$GEN_PY" -m synthesis_pipeline.experiment_edit_quality \
  --data-root "$DATA" --out-root "$ITER/generation_next" \
  --variant context_guarded_v2 --ids 16,27,33,67

## 下例是实验性严格审核配置，不代表可以免人工审核发布。
CUDA_VISIBLE_DEVICES=1 PATH="/opt/tiger/tanyue/.venvs/qwen38_audit/bin:$PATH" \
  "$AUDIT_PY" -m synthesis_pipeline.audit_quality_v3 \
  --data-root "$ITER/planning_v6" \
  --edited-dir "$ITER/replanned/context_guarded_v2/edited" \
  --out-root "$ITER/audit_next" --variants forensic_overview \
  --target-change-threshold .02 --max-add-change-ratio 3 --reject-flat-attribute \
  --rewrite-after-audit
```

最终总报告：`$ITER/report/index.html`。包含全部新生成对照与回贴消融；所有预览图内嵌，另有独立 JPG。
每条结果分开记录模型完整回复/prompt 和 assistant 逐图理由，按实际 edited 文件关联，
避免把同 case 另一版本的审核显示在错误图片下面。

当前建议：保留安全的输入/校验修复，生成和审核新方法继续通过显式开关回归。
下一步重点是完整编辑单元识别、未标注邻居保护、属性编辑保持几何，以及真正识别原有背景对象。
这些问题没有通过本轮“多写几句 prompt”全部解决；在独立样本通过验收前，不启动 100k 自动入库。


<a id="diagnosis-v6"></a>

## 当前单一版本出图 bad case 诊断

检查对象为上一轮展示候选 `v6 regional planning + context_guarded_v2` 的同一批 14 张图片，
重点重新逐图检查其中 9 张未通过样本。没有新增扩散调用，没有改动生成策略、原始图片或标签。
这批是困难开发回归样本，不能用它估计随机生产数据的整体通过率。

可视化页面：
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/quality_iteration_20260920/badcase_diagnosis_v6/index.html`

页面包含完整场景和局部的原图/结果对照，另有六格诊断：真实 source crop、raw 输出、最终 crop、
原 target mask、composition alpha、已标注邻居保护 mask。每个 case 给出完整生成 prompt、参数和分析。
18 张图全部内嵌 HTML，并保存独立 JPG。

### 本次复核修正

不是 9 张都属于出图画质失败：

| 类别 | 样本 | 结论 |
| --- | --- | --- |
| 画面 / 空间缺陷 | 023、033、047、064、067、099 | 当前成品拒绝 |
| 目标编辑未完成 | 043 | 最终图基本自然，但 raw 错位，最终目标没有明显哑光变化 |
| 类型定义含混 | 052、094 | 不能直接算生成质量失败；可考虑重新定义指令/类型和有效编辑范围 |

052 的指令是把女子替换为穿绿色背心的人，实际完成了换装。
没有可验证的新身份要求时，直接断言原指令不匹配过严；当前四类划分中适合归服装 attribute。
094 要求细枝替换成类似弯枝，实际变粗、弯曲；但原 SAMTok answer 分别称两个目标为 wire 和 stick，
当前 m0 规划直接称 stick，源对象与 mask 的绑定还需核对。若是 wire 变 branch，replace 可能成立；
不能先假定同类替换失败，再强制转 attribute。新增枝条的变化范围也有扩张。
这两条保留原复核记录以便追溯，本诊断不覆盖标签、不直接新增入库样本。

### 中间结果带来的关键因果证据

对 9 张逐一检查：保存的 `source_crop.png` 均与干净原图相应 crop **逐像素一致**；
用保存的 raw 和 alpha 重新计算 `Image.composite(raw, source_crop, alpha)`，回贴后均与最终 PNG **逐像素一致**。
因此可以区分模型生成缺陷与合成缺陷；这里的 raw 是模型输出缩放到 crop 尺寸后、最终合成前的图。

#### 033：木棍是回贴恢复的

raw 已移除蛋糕球与其木棍。原 mask 只有球体，固定外扩 alpha 没覆盖木棍下端，
于是最终成品恢复了一截原木棍。这是直接可复现的合成问题，同时也暴露了规划把球体误当完整 cake pop。
不能只修改移除 prompt，或把它归咎于模型没有移除干净。

#### 043：不是单纯材质变化太弱，而是错位生成

raw 在原目标上方生成更大、较哑光的香肠，原目标仍大体存在。
最终 attribute 只写入原 mask，错位新香肠大部分被排除，看起来就像没有编辑。
以通道平均绝对差 > 20/255 定义粗略变化像素，raw 变化约 86.1% 位于原 mask 外。
该统计只辅助逐图定位，不等于语义错误率。

当前生成只输入一张干净 crop，目标定位依赖文本指代和近似中心百分比。
mask 是在生成结束后控制回贴的，不是扩散过程中强制约束对象定位的条件。
这解释了为什么最终不越界不等于成功编辑了正确实例。

#### 023、099：原始生成已经平涂

023 的人物头部与肩部被要求 bright red，raw 已变成纯红剪影，而且轮廓漂移。
099 的镜面被要求 completely opaque and non-reflective，raw 已把镜面和花纹边框整体变灰。
输入是干净照片，并无红色 overlay；二者不应再归因为输入叠色污染。
任务的属性作用表面不清楚，模型采用平涂的简化解；已存在的保留纹理 prompt 未能防止它。

#### 047：raw 背景错误，合成进一步混合不同结构

raw 删除女孩后，把可见印刷纸张与部分木桌变成大片浅色平面。
最终 alpha 过渡又混合该新平面与原纸张，留下发白、印刷残片及不自然接缝。
应先修正生成的背景结构，扩大羽化或做泊松融合不能修复物体与结构本身的错误。

#### 064、067：raw 误删邻居，保护 mask 不完整

064 的 raw 已删除目标后方原本可见的另一位行人；replace 的宽 alpha 将其带入最终图。
067 的 raw 同时删除多把椅子；局部合成恢复部分远处原图，近邻椅背和桌椅结构错误仍在。
`protected_neighbors` 只合并同 source 已标注的其他 region，不是场景所有实例。
保护图没有覆盖此次被误删的那些近邻可见区域。

可写像素数（alpha > 0）与原 target 像素数之比：064 为 8.01，067 为 3.82，094 为 76.16。
这是允许写入范围的比例，绝非实际改变面积的比例。094 的原目标尤其细小，比例很大。

### 建议优化顺序及验证方法

#### 1. 先统一目标语义、完整编辑范围和邻居保护

保留原始 SAMTok `target_mask`，另外明确 `edit_support_mask` 和非目标实例的可见区域保护。
训练时若要求 mask 表示所有应编辑区域，就必须提供与 instruction 一致的有效 mask，
不能把原球体 mask 配上完整蛋糕棒移除，却暗中编辑木棍。
移除完整物体时补齐其结构部件；无法可靠补齐的 case 改成与现有区域匹配的任务。
这是对任务的适配，不是把拥挤、小目标、遮挡场景整体筛掉。

第一组对照用 033 的现有 raw，只改变经过确认的结构部件支持范围。
它不需要新扩散调用，能单独测合成修复。已做过的简单变化连通域扩张存在接缝问题，
不能直接作为语义分割的替代，也不能宣称已解决。

#### 2. 修正实际 crop 内的实例定位

对 043、067，给编辑模型的内部指令应结合实际 crop 的目标 bbox、邻居关系和简明唯一指代。
全图 instruction 仍保留给训练；生成内部描述需要正确转换到 crop 坐标。
对拥挤小目标测试更紧的 crop，同时保留足够邻居用于区分实例。
近似百分比仅是提示，不是区域控制保证。

固定样本、seed、steps，比对只改生成内部定位 / 只改 crop / 两者组合。
检查 raw 是否真的改变正确实例，再检查最终图；不能因严格回贴把错位变化截掉而通过。
对于仍持续错位的样本，区域条件输入可以作为后续单独实验，但现有带轮廓参考曾有复制标记的风险，
未经对照验证不直接恢复为默认。

#### 3. 按任务决定合成范围，禁止用一个范围兼顾所有类型

- remove：完整目标与必要结构部件，有限背景修复环带，避开独立邻居。
- replace：原物体 footprint 与合格新轮廓的并集，加有限的接触/阴影余量；不默认开放整个 context。
- 同形 attribute：编辑表面范围与基本不变的外形。raw 轮廓漂移时拒绝，而不是依靠裁剪掩盖。
- 形态 attribute：允许必要轮廓变化，不能因为类型改为 attribute 就机械套用旧的严格 mask。

对 064、067 补齐相邻实例保护，单独测试 raw 中邻居保留和合成后保留。
如果 raw 已生成错误的遮挡结构，仅把邻居原像素贴回来也可能产生接缝，仍需重新生成并核验。

#### 4. 收紧属性作用部件，保留可用的换装/形态编辑

023、099 需要重设可编辑表面，避免人物多个材质或镜框与镜面被当成一个统一填色区。
不能仅靠增加“保留纹理”的泛化文字，因为当前 prompt 已有类似要求。
规划时选择适合区域的属性；生成后判断纹理、光照与结构是否成立。
052 可改写为实际换装；094 先核对 wire/stick 与 mask 的原始绑定，再判断属于 replace 还是形态 attribute，
并再次对齐有效 mask 和变化范围。
保持修改版本记录，不能用 instruction 改写为误删、平涂或错误补全开脱。

每组对照同时报告：目标是否完成、邻居是否保留、边界/背景是否自然、最终指令与有效 mask 是否一致。
修复困难样本时加入原来通过的 016/027/070/075/097 作回归，防止只救一类而破坏其他类型。
这份报告给出优化方案和因果诊断，未声称这些新方案已经通过出图实验。

### 复现可视化

结构化逐例分析：`docs/data/BADCASE_V6_20260920.jsonl`。

```bash
ITER=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/quality_iteration_20260920
python -m synthesis_pipeline.build_badcase_diagnostics \
  --data-root "$ITER/planning_v6" \
  --run-root "$ITER/replanned/context_guarded_v2" \
  --review-jsonl "$ITER/assistant_review.jsonl" \
  --analysis-jsonl docs/data/BADCASE_V6_20260920.jsonl \
  --out-root "$ITER/badcase_diagnosis_v6"
```


<a id="iteration-v8"></a>

## 2026-09-20：语义编辑范围、实例定位与邻居保护 v8

所有输出位于 `quality_iteration_20260920/repair_v8/`，原有结果保留。本节同时记录有效修复与否定实验，不把挑选出的最佳结果当成单次自动运行的通过率。

已经实现：

- 原 `sam_target_mask` 与有效训练 `mask` 分开保存，SAM3 细分编辑表面、补全结构部件并分割近邻；分割失败显式标记 unresolved 并阻止扩散。
- 新编辑方法 `context_grounded_v3`：更紧的上下文 crop、基于实际 crop 的定位、按任务控制回贴范围及完整可见邻居保护，已接入普通入口与多卡队列。
- 规划 v8/v9 在原有 MLLM 规划调用中给出分割短语和邻居类别；remove 另给结构部件短语。增加不合理人体平涂/整体镜框变灰、同类别替换与训练指令中数字坐标的校验。校验失败会走既有重试，因此不能声称实际调用次数永不增加。
- 首轮 14 条 SAM3 预处理总耗时 153.81 秒，其中模型加载 134.10 秒；13 条形成候选范围，033 的 cake pop 整体分割无结果，进入明确结构部件补充分割。
- 13 张首轮扩散回归使用 3 GPU、40 steps、seed=0；随后对 027/033/043/047/067 做第二轮 5 张重生成，对 033/043/047/067 做轮廓参考双图第三轮。新输出与所有 raw、实际 prompt、alpha、保护 mask 均保留。

这批修复指令和分割短语由 assistant 根据已看过的失败样本制定，保存在 `docs/data/REGION_REPAIR_V8_PROPOSALS.jsonl`；它是有监督的困难回归，不是全自动规划通过率评测。自动规划验证独立列于下文。

### 实现与实验取舍

1. **原 mask 与有效编辑 mask 分离**：`refine_samtok_regions.py` 保留 `sam_target_mask`，将确认的编辑表面/完整编辑单元写入 `mask`，另存 `region_contract.protected_mask`。033 的整体 cake pop 分割未返回候选，木棍语义分割与球体邻接匹配才完成完整单元，面积变为原来的 1.126 倍。023 头发、099 玻璃、075 圆形标牌分别缩为原区域的约 74.0%、29.1%、28.2%。未解析或有竞争部件的范围阻止出图，不默认为成功。
2. **内部定位与数据指令分离**：给编辑模型的 prompt 包括真实 crop 的目标位置、简明动作与背景/材质要求；训练指令仍为精简的全图唯一指代。第二轮把动作放在 prompt 最前面；remove 恢复更宽上下文，043 单独收紧 crop。027 从首轮残留恢复为移除干净，但 043/067 仍失败，说明 bbox 文字不是可靠的区域条件。
3. **按任务回贴**：attribute 只回贴真实编辑表面；remove 使用完整结构部件加有限修复环带；replace 优先使用旧目标与 SAM3 新实例轮廓并集。邻居保护在回贴时精确保留已分割的可见邻居。`compose_segmented_replacements.py` 在现有 raw 上处理 016/064/094，不重新扩散；094 的枝条截断消失。该步骤暂为显式后处理 CLI，不宣称普通出图入口已自动运行全部后处理。
4. **负向实验也保留**：完整部件 mask 消除了 033 木棍恢复，但粉色底座仍有几何接缝；Poisson 版本反而没有完成目标移除，不能采纳。047 的人物虽消失，木桌/纸张/椅背补全仍不连贯，保护邻居像素不能修复 raw 中错误的背景结构。
5. **默认行为**：新增 `context_grounded_v3` 可显式选择，未整体替换原有生产默认。双图轮廓参考只对回归 row 的 `editor_target_guide=true` 启用，必须检查标记复制、偏移与额外耗时。

### 自动规划单独验证

相同 14 条困难回归输入，8B vLLM v8 共 23 次实际请求（含重试）、11 条通过格式/规则校验；v9 共 27 次请求、9 条通过。v9 模型加载 41.36 秒、推理 16.90 秒、总计 62.22 秒。校验通过不等于指令与 mask 已被人工确认，更不等于出图成功。

自动规划已经会为蛋糕棒补充木棍，但仍出现 generic refer_object 掩盖 stick→branch 同类别替换、重复结构部件等问题；本轮继续补上直接检查指令源对象、类别别名和部件去重。未把这 9 条全部当成高质量计划投入扩散，也未将 assistant 制定的回归指令伪称为全自动生成。

### 时间与复现约束

SAM3 常驻后的额外分割较轻：3 个替换的新轮廓分割加回贴总计 11.99 秒，其中加载 9.44 秒，处理约 2.55 秒；不含扩散调用。初次 SAM3 冷启动受 CPU 线程配置影响达到 134 秒，后续设置 `OMP_NUM_THREADS=8`，不能把两次加载差异归因为算法加速。

每张扩散的实际耗时、模型加载耗时分别见 `generation*/context_grounded_v3/timing_shard*.jsonl` 和 `summary_shard*.json`。不同 crop/输入张数会改变生成结果与速度；本轮不是“输出逐像素不变”的加速试验。

### 三轮出图及逐例结果

共新增 22 张扩散输出：首轮 13、第二轮 5、第三轮 4；另有不调用扩散的范围/融合对照。首轮平均 76.84 秒/张、第二轮 77.28 秒/张；第三轮双图参考平均 131.75 秒/张，慢约 71%。这些是 3 张 GPU 各自加载单卡编辑模型的实际单 case 耗时，不是整个 8 卡系统的生产吞吐，也不包含模型加载和审核。

第三轮 4 张全部未通过：033 仍有粉色底座碎片，043 目标仍高光而非 matte，047 保留女孩并复制白色描边，067 仍有混乱椅背/横杆。轮廓参考不设为默认。

选出的逐例实验快照为 `repair_v8/selected/`，逐图复核在新的 27B 回复之前冻结。14 条中 9 条同时满足画质和指令；4 条生成失败，1 条画质自然但语义标签仍需核定。**这是跨实验挑选后的困难开发集结果，不是固定版本的 64.3% 自动通过率，不能与旧版 5/14 直接作为等条件提升比较。** 指令、mask、类型及算法部分同时改变，逐例来源完整保留。

| Case | 本轮采用输出 | 结论与未解决项 |
|---|---|---|
| 016 | 新旧替换轮廓并集 | pass：警车替换，摩托与其他巴士保留 |
| 023 | 首轮真实头发表面 | pass：铂金发色保留发丝，任务已从混合人物区域改为头发 |
| 027 | 第二轮动作前置、宽上下文 | pass：目标长颈鹿移除，背景实例保留 |
| 033 | 第二轮完整部件范围 | fail：木棍问题缓解，但粉色底座几何接缝仍明显 |
| 043 | 第二轮紧 crop | fail：香肠高光仍在，没有完成明确 matte 编辑 |
| 047 | 第二轮背景结构提示 | fail：女孩移除，但桌面/纸张/椅背补全不自然 |
| 052 | 首轮、改为 attribute | pass：绿色背心和红 T 恤属于换装而非更换人物身份 |
| 064 | 新旧替换轮廓并集 | pass：自行车自然，后方条纹衣行人保留 |
| 067 | 第二轮实例保护 | fail：椅背/横杆残留、桌椅关系不成立 |
| 070 | 首轮 | pass：新增小号蝴蝶结，原先通过项未退化 |
| 075 | 同 raw、仅圆形牌面回贴 | pass：牌面红、支杆不再误变红 |
| 094 | 新旧替换轮廓并集 | 画质 pass、指令 fail：枝条连续，但 wire/stick 原始标签绑定仍需明确 |
| 097 | 首轮 | pass：包主体变红，保留褶皱与光照 |
| 099 | 首轮真实玻璃表面 | pass：玻璃磨砂、雕花边框不再一起涂灰 |

Poisson 负实验还发现 OpenCV 会原地修改传入 mask，导致随后回贴恢复旧像素。已复制 mask 再调用并补单测；修正实现后 033 确实移除了目标，但生成大块粉色突片，说明算法仍不能修复错误背景几何，依然不采用。旧输出保留，不覆写历史证据。

最终预览为 `repair_v8/final_report/index.html`：图片内嵌，也提供独立 JPG；所有 14 条均展示，不隐藏失败。`docs/data/REGION_REPAIR_SELECTION.jsonl` 保存选用版本与完整 assistant 理由；`selected/assistant_verified_annotations.jsonl` 仅含复核通过的 9 条，不与自动模型放行混用。

### 复现本轮

```bash
ITER=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/quality_iteration_20260920
REPAIR="$ITER/repair_v8"
# 这些 out-root 必须是尚不存在的新目录；不要覆盖已冻结结果。
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python -m synthesis_pipeline.refine_samtok_regions \
  --data-root "$ITER/planning_v6" --overrides docs/data/REGION_REPAIR_V8_PROPOSALS.jsonl \
  --out-root "$REPAIR/reproduce_regions"
# unresolved 的 033 需要先完成 structural_parts 分割，不送入扩散。
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 /opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  -m synthesis_pipeline.experiment_edit_quality --data-root "$REPAIR/regions" \
  --out-root "$REPAIR/reproduce_generation" --variant context_grounded_v3 \
  --ids 16,23,27,43,47,52,64,67,70,75,94,97,99 --shard 0 --shards 3
# shard 1/2 分别另用 GPU 1/2；所有 worker 复用各自模型，40 steps / seed 0。
# 精确复现首轮旧 prompt 需使用 diagnostics/generation_request.json；当前代码已是第二轮 action-first prompt，不能声称像素级重现首轮。
python -m synthesis_pipeline.assemble_repair_snapshot --data-root "$REPAIR/regions" \
  --experiment-root "$REPAIR" --selection docs/data/REGION_REPAIR_SELECTION.jsonl \
  --out-root "$REPAIR/reproduce_selected"
```

代码验证：`python -m pytest tests -q`，69 passed、1 skipped；`git diff --check` 与新增模块编译检查通过。新增单测覆盖分割目标不串实例、歧义部件拒绝、完整木棍移除范围、邻居精确保留、新旧替换轮廓并集，以及 Poisson mask 原地修改防护。

自动规划的最后一道修正：remove 不再同时要求“部件完全位于原轮廓内”和“输出轮廓外结构部件”。明确允许 `complete_part + complete refinement + structural_parts` 的暂定计划；SAM3 必须解析完整范围后才能出图。任意碎片仍拒绝。普通入口与实验入口增加执行防护，禁止这种计划绕过分割而交给使用旧 crop mask 的 legacy 方法。最后这组规则通过单测，尚未另跑一次全量 MLLM 规划；前面的 v9 9/14 数字不能作为此修正的实测结果。

### 27B 独立核验与改写：没有被实验支持的方案不升级默认

在 assistant 冻结 14 条判定后，用同一 Qwen3.8-27B / vLLM、temperature=0、每次双图执行以下开发对照。它们是实验分支，不代表生产流程串行运行三轮审核。

| 实验 | 输出 | 与冻结画质判定对比 | 时间（14 条，不含加载） |
|---|---|---|---|
| `audit27/correspondence_crop`：带原指令逐对象对照 | 14 pass | 10/14 一致，漏掉全部 4 个画质失败 | 推理 26.28 秒；含图片处理墙钟 46.78 秒 |
| `audit27/rewrite_overview`：不带旧指令重建描述 | 10 candidate、4 被规则拒绝 | candidate 不是已确认标签，未直接应用 | 推理 13.56 秒；墙钟 33.38 秒 |
| `audit27_blind/forensic_crop`：不带指令只看画质 | 8 pass、6 fail | 8/14 一致；检出 043/047，漏掉 033/067，误拒 016/064/070/094 | 推理 32.87 秒；墙钟 53.03 秒 |

第一组模型加载与预热共 82.16 秒，改写复用该常驻模型；盲核验另启进程加载 85.34 秒。小样本冷启动不能当作稳态每 case 成本。

关键失败证据：带指令审核将 043 未明显变化的高光香肠描述成“高光消失、变干燥”；盲核验虽然对 043 给了 fail，却错误解释为“移除残留”，判定碰巧正确不等于理由可靠。它还把道路远处警车、路面的自行车、小号上蝴蝶结误称为无支撑漂浮物。这次盲核验不是整体改善，因此不升级为默认审核，也不声称审核问题已经解决。

改写也有明确不可靠输出：016 将一个巴士写成两个，已由数量规则拒绝；043 写“移除白色轮廓”，已拒绝；094 建议改为 attribute，因类型变化需要核验而拒绝。033 即使给出了通顺 remove 指令，也不能修复底座几何；099 的大理石称谓与源对象解释不一致，不替换已有复核指令。冻结的 9 条通过清单仍按 assistant 逐图复核，不让模型通过或改写覆盖人工结论。

最终页每条只展示一个选定结果图；详情中的 `27B_instruction_audit`、`27B_blind_quality`、`27B_rewrite_candidate` 分别对应以上三个开发实验，附完整输入 prompt、原始回复和 assistant 理由。并非一条 case 的三轮生产审核。

### 当前推荐与下一步边界

- 可保留：语义表面/完整部件分割、可见邻居保护、新旧替换轮廓并集、所有输入与 raw/alpha 的诊断存档。
- 不采用：双图轮廓参考、Poisson 自动补救、把 blind forensic 判定直接作为更优审核、自动用改写给失败图兜底。
- 仍需解决：033/047 结构化背景修复，043/067 同实例精确区域控制，094 原始区域语义绑定，以及审核器的请求诱导与物理关系误判。应在独立新样本上验证区域条件生成和审核对照；本轮不启动大规模生产，也不把选定 9 条冒充一个已稳定的自动版本。

<a id="fresh-v10"></a>

## 2026-09-20：新源图开发与留出验证 v10

用户要求按优先级继续优化，并使用更多新源图。本轮根目录为 `SAMTok_Derived_Edit_Labeling/fresh_iteration_20260920_v10`。不改写旧实验，不将留出结果用于本轮调参。

### 冻结数据与验证纪律

- 排除历史 1,018 条已规划/出图源记录，并对 parquet 内的原始解码 RGB 像素计算 SHA256，排除同图不同记录。新抽样 20 张源图、40 个 region，GRES/VER 各 10 张，每张图两个 mask 都进入规划。
- 按源图划分开发 24 个 region、留出 16 个 region；开发各编辑类型 6 条，留出各 4 条。按 mask 面积分层，不手选容易成功的图。规划/分割拒绝也计入总分母，不补抽到凑够通过数。
- “留出”仅指本次 pipeline 调试未使用的新源图，不声称是底层模型训练外数据。精确像素去重不是感知近重复去重。
- 先完成开发出图和逐图复核，再固定算法与审核 prompt，最后跑留出。生产默认不因小样本偶然改善就直接替换。

### 实现

1. `reference_binding.py` 按 canonical span 顺序对齐 raw mask：GRES JSON 读取对应 entry 的 label；VER 读取对应 span 前的局部短语。重复标签或同一括号包含多个 span 时明确标为 shared group，不能假造独立指代。span 数量不匹配、重复 token、复杂不明语句显式 unresolved。无需增加模型调用。
2. 规划输入优先只给当前 mask 的绑定短语，保留 canonical span、mask_index、绑定方法和原 answer；将原先“整段答案交给模型猜哪个”的不确定性显式化。纯视觉判断与绑定短语冲突仍需拒绝。
3. 编辑开发对照：`context_grounded_v3` 对比 `context_grounded_v4`。v4 使用同一干净 crop、相同指令/40 steps/seed 0，在每个扩散步骤把非编辑上下文锚定到对应噪声级别的原图 latent；完整目标及修复环带保持自由生成。不是严格旧轮廓内每步恢复，也不使用轮廓参考图片。使用本机安装的 QwenImageEditPlusPipeline `prepare_latents`、`callback_on_step_end` 与 FlowMatch scheduler 的实际代码接口；结果与速度待实测，不预先宣称更优。
4. 审核开发对照增加 `balanced`：不提供期望指令，先判断真实变化与可见边界缺陷，明确不要把远近透视、遮挡导致看不见支撑误判为漂浮；也不能将轮廓消失当成编辑。仍为两张图片、一次调用、三个字段。改写增加拦截“一个目标替换为两个独立实体”的规则，覆盖旧 064 中把露出的原有行人写成新增对象的错误。

### 开发实测：不将规则通过数当作数据通过数

固定开发集共 24 个 region，未补抽：旧较冗长输出的 8B 首轮接受 7 条；精简为单个 instruction 的 8B 接受 8 条（56 次调用、模型推理 23.66 秒）；同样新视觉输入与精简 prompt 的 27B 接受 20 条（39 次调用、推理 97.38 秒）。27B 加载 85.17 秒、总墙钟 186.73 秒；8B 第二轮加载 41.77 秒、总墙钟 70.77 秒。这里还包含少量开发期间规则修正，且两者重试数不同，不能把 4.1 倍的总推理时间差当作严格单请求模型速度比。原始 prompt 与每次回复均存档于对应 `responses.jsonl`。

指令字段删除了重复要求模型生成的 `new_instruction`，由代码从 `editing_instruction` 派生；保留清晰指代和源对象分割短语。修复真实 face mask 被当成 annotation、holding/carrying 从句干扰 replacement 类别判断等规则错误。结构化 span 绑定在整个正样本索引上得到 bound_mention 4,732、bound_label 2,370、bound_shared_group 3,354、unresolved 177；这是语法对齐覆盖率，不是语义正确率。

8B 可执行的 7 条做相同 source、mask、指令、seed、40 steps 的出图对照：assistant 先盲于审核结果逐图冻结判定，v3 画质通过 4/7，v4 通过 5/7。016 加番茄酱时盘子红花被擦除的错误在 v4 缓解；036 裙摆加花仍出现邻人重复头部；038 人替佛像仍有残手和截断。这只是小样本开发结果，不声称 v4 已全面优于旧方法。

进一步分离 raw 与回贴错误：add 新增 `--include-add` 的 SAM3 新物体轮廓回贴，仅允许新轮廓和窄接触边缘写回，避免整块人物 crop 把多余头部带回来。首版发现邻居保护会把盘子穿回新番茄酱内部，已修复为允许新增物体合法遮挡支撑面。036 的重复头部因此消失；无需新扩散调用或新 MLLM 调用。replace 继续用旧/新轮廓并集，但 raw 已有的断头、残手不会被这种后处理修复。

27B 规划的 20 条中，017 的完整食物分割未解析成功，阻止出图；其余 19 条均执行 v4。020 新帽子无法获得可信新物体轮廓，最终组装拒绝。`dev_final` 的 18 条使用统一算法，没有逐 case 挑版本：add/replace 用语义轮廓合成，remove/attribute 保持 v4。assistant 对全部 18 条先于本批审核模型逐图复核：13 条画质大体可用，其中 11 条与原指令同时匹配；按开发集原始 24 个 region 算 11/24，不能写成生产自动通过率。14 的男子未被替换而是抱上熊玩具；39 是雕像身体变银、头仍金色，需区分画质与标签。18 断头、19 衣服白边、29 错误左右指代并有矩形补洞、36 篮子未接到目标手、38 残手仍是失败。

审核开发对照沿用 7 条 v4 固定结果：带原指令 `correspondence_crop` 7/7 全放行，与 assistant 5/7 一致、漏掉两条坏图；不带期望指令的 `balanced_crop` 输出 4 pass / 3 fail，与 assistant 6/7 一致，检出 036 重影和 038 残手，误拒 016 番茄酱。分别模型推理 16.14 / 13.39 秒，含双图处理但不含共同加载为 27.79 / 25.07 秒。此改善需独立留出验证，不能据 7 条直接升级生产审核。

### 冻结与复现

`frozen_candidate.json` 在留出规划前冻结 27B 规划、v4 40 steps/seed 0、add/replace 语义合成、balanced 双图一次审核及同类型改写门控。此时完整开发组正在复核，后续只做验证，不根据留出表现改 prompt 或逐例修图。新增普通运行入口的 `--edit-method context_grounded_v4`，但默认仍是历史 mirage；不静默切换生产。

```bash
FRESH=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_iteration_20260920_v10
# 先对 split 的全部冻结 IDs 做 replan_edit_regression（--vlm qwen38-vllm），
# 再 refine_samtok_regions；只将非 unresolved 的 IDs 送入出图。
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 /opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  -m synthesis_pipeline.experiment_edit_quality --data-root "$FRESH/dev_regions_27b" \
  --out-root "$FRESH/reproduce_generation" --variant context_grounded_v4 \
  --ids 4,12,13,14,16,18,19,20,21,26,27,28,29,30,31,36,37,38,39 --shard 0 --shards 6
# 其他 shard 使用独立 GPU；manifest-only 导出全组 annotations；必须等待全部 shard 完成。
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python -m synthesis_pipeline.compose_segmented_replacements \
  --data-root "$FRESH/dev_regions_27b" --raw-root "$FRESH/reproduce_generation/context_grounded_v4" \
  --out-root "$FRESH/reproduce_composed" --include-add --carry-others
python -m synthesis_pipeline.assemble_fresh_snapshot \
  --raw-root "$FRESH/reproduce_generation/context_grounded_v4" \
  --composed-root "$FRESH/reproduce_composed" --out-root "$FRESH/reproduce_final"
# balanced_crop 一次审核；--rewrite-after-audit 仅对模型画质 pass 再调一次改写。
# 改写输出只是候选，不能覆盖 assistant 的画质 fail。
```

迭代产物均保存在上述数据目录；这里继续追加同一份日志，不新建多份说明文档。

### 扩大开发集后的审核与改写：改善没有稳定复现

18 条 `dev_final` 的 balanced 审核给出 17 pass / 1 fail，与预先冻结的 assistant 画质结论一致 14/18。漏检 018 断头、019 衣缘白边、029 矩形补洞、036 篮子接触关系；仅拒绝 038 残手，且理由混入了不可靠的雕像悬浮判断。不能因为首组 6/7 一致就声称审核已经解决。模型还把 004 原本就存在的笔说成新添加，表明前后对象追踪仍不可靠。

17 条模型画质 pass 执行同常驻 27B 的独立改写，得到 16 条格式/策略候选、1 条被拦截；不代表 16 条可用标签。026 把狗换猫误判为改毛色，被跨类型规则拦截；014 把“原男子仍在、假发消失并抱熊”写成“假发替换成熊”，即使维持 replace 类型也不成立；036 仍臆造篮子挂在手臂上；039 仍写整尊改银色而遗漏金色头部。029 的 right 指代确实修正，但不能修复坏背景。上述改写均未自动覆盖原标签，也未增加 assistant 通过计数。

`assistant_verified_annotations.jsonl` 只导出画质与**原指令**均通过的样本，模型候选另存。dev 通过类型为 add 4 / remove 3 / replace 2 / attribute 2；输入每类 6 条，失败后的类型不平衡必须如实报告。样本导出附 assistant 完整理由与实际源图/结果路径。

### 本轮速度口径

- 同 7 条生成对照：v3 平均 76.989 秒/条，v4 平均 77.077 秒/条。两组各 3 张卡同时跑，差异不足以声称加速或明显变慢；v4 的目的主要是上下文约束。
- 27B 规划后的 19 条 v4：6 个独立 GPU worker，单条平均 77.214 秒（74.182–80.266）；各 worker 加载约 16–19 秒，最慢 shard 的加载加生成约 327 秒。小批次分片不均使有效吞吐低于理想稳态，不把它混为 8 卡端到端速度。
- 开发源分割 20 条：加载 10.03 秒，总 37.36 秒。新物体后分割/合成 18 个当时已完成结果：加载 10.07 秒、总 18.59 秒，0 次新扩散/MLLM；最后一条 attribute 通过固定组装规则从原生成结果补齐，最终总生成 19、合成可用 18。后续规范是全部生成完成后再后分割，组装器会拒绝缺失 raw 结果，避免半成品被误报成最终批次。
- 开发审核 18 条：模型推理 43.91 秒、含图片处理不含加载墙钟 73.80 秒；改写 17 条额外推理 17.36 秒、墙钟 42.32 秒，共用一次 83.30 秒的模型加载。仍然一条双图审核，只有画质 pass 才额外一次改写。

普通入口可选择 v4，但新的语义合成仍需显式执行后处理命令；本轮未启动 100k，也未将实验审核升级为默认自动放行。

### 独立新源留出结果（未据此回调参数）

16 个 region：规划接受 11、SAM3 源范围解析 10、扩散完成 10；新增白腕带 008 和黑相机 024 未通过新物体后分割，最终候选 8。assistant 在读取本组 VLM 审核前逐图冻结结论：6 条画质大体可用，4 条原指令和画质同时通过（000/025/032/033）。按原始输入为 4/16，不是 4/8 自动通过率。

留出特有问题：003 橙色食品过饱和、像贴片；035 纸盒变纯红且丢掉印刷纹理、列指代也不够精确；023 只改了整列中的一个浮标；022 源 reference 确实写 `The tall lattice crane`，但低分辨率外观也像钟塔，assistant 无法可靠确认类别，暂不纳入原标签通过项，而不是断言原数据标错。008 的原 mask 在后方金发球员，规划却指到前方迷彩短裤球员，说明 shared-group 的视觉消歧仍会失败；结果未出现清晰腕带，后分割拦住。024 的相机在结果胸前实际可见、画质大体可用，但长分割短语未解析成功，属于召回损失。为遵守冻结评估，没有针对这些例子改分割短语后重新计通过。

10 条留出生成平均 77.128 秒/条；6 worker 最慢 shard 加载加生成约 173.66 秒。源分割含加载 32.26 秒，后分割/合成含加载 13.15 秒。规划 16 条含重试 29 次调用，推理 85.60 秒、含加载总 172.21 秒。所有时间均为本机小批次实测，不含 assistant 看图和调试等待，也不是 100k 稳态预测。

一张图的两个 mask 都进入规划，拒绝项没有被替换成新抽样；最终通过数会少于输入 region 数，四类也不再均匀。后续应在新的抽样批次上改进 shared-group 视觉定位、表面范围/数量约束与小附件分割，不用修改本批留出标签制造提升。

留出 27B balanced 审核：6 pass / 2 fail，与 assistant 画质判定 6/8 一致，漏放 003 荧光色食物、误拒 022 灯塔，正确拒绝 035 纯红盒。模型推理 18.32 秒、含图片处理不含加载 30.46 秒。6 个模型画质通过项再改写：4 个策略候选、2 个跨类型拒绝；023 错写成“整列替换成单个黄浮标”，033 把正常 remove 错写成 replace，均拦截。改写推理 7.01 秒、墙钟 15.36 秒，共用加载 86.74 秒。未发现能可靠补救本轮原标签失败项的自动改写，不增加通过数。

### 本轮交付与结论

| 阶段 | 开发 | 留出 | 合计 |
|---|---:|---:|---:|
| 新源图 | 12 | 8 | 20 |
| 原始 region（四类均匀） | 24 | 16 | 40 |
| 规划接受 | 20 | 11 | 31 |
| 源分割通过、实际出图 | 19 | 10 | 29 |
| 后分割合成候选 | 18 | 8 | 26 |
| assistant 画质基本可用 | 13 | 6 | 19 |
| assistant 画质及原指令同时通过 | 11 | 4 | 15 |

26 条最终候选的自动画质审核与 assistant 一致 20/26，但漏掉 7 条画质失败中的 5 条，不能用于无人值守放行。上述 15 条是人工门控后的原标签样本，不是自动生产成功率；另有被后分割拒绝的相机原始结果可用但未计入。不同新源图难度差异明显，小开发组上的改进不能代替留出验证。

- [本轮单一候选方案全部 26 条对比（图片内嵌，含完整模型与 assistant 理由）](/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_iteration_20260920_v10/report/index.html)
- [后分割拒绝的 3 条原始出图诊断](/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_iteration_20260920_v10/rejected_report/index.html)
- 小组生成消融在 `dev_report/index.html`；主展示不混用其中更好看的结果。
- `dev_final/assistant_verified_annotations.jsonl` 11 条、`holdout_final/assistant_verified_annotations.jsonl` 4 条；原始标签、失败图、模型候选均单独保留。

优先保留本轮的可追踪 reference 绑定、单 instruction 输出、未解析范围阻断、add 轮廓回贴和统一分母统计。v4 可显式选择，不升级默认；27B 规划提升了规则接受率，但仍会错认 shared-group 的实例及 mask 覆盖范围。审核和改写尚未达到自动生产要求。下一轮优先：实例排序/局部表面范围的可靠验证、与邻人交互的完整部件编辑、纯色纹理坍塌和结构补洞的独立可见缺陷检测；继续换新源批次，而非反复修本轮留出。

最终代码验证：75 passed、1 skipped，`git diff --check` 通过；新模块和普通出图入口编译检查通过。没有提交/推送，也没有启动全量任务；当前工作树保留原有其他迭代改动。

## 2026-09-20：v11 审核与补救迭代（进行中）

用户要求先在更多新源数据上改善审核，再做约 100 条新出图并逐图复核；若大批次仍有明显问题，继续换新源数据迭代，不以完成数量为终点。

### 新样本与预先约定的评估口径

`audit_iteration_20260920_v11` 抽取 30 张新源图、60 个 region，GRES/VER × mask 面积五分位各 3 张源图；按源图划分开发 36 region、留出 24 region，源图两个 mask 都进入规划。排除历史 1038 个源记录，同时比较解码 RGB 哈希，不只比较文件名。种子 20260925。`evaluation_protocol.json` 在查看本批出图和审核结果之前冻结：先独立记录 assistant 画质与原指令标签，再对比旧 balanced_crop 和新方案。初步门槛为坏图拦截率 ≥80%、好图保留 ≥70%、最终自动标签精度 ≥90%；小样本即便达标也不作为无人值守认证，后续约 100 条仍需逐图复核。

### 候选审核 v4：三阶段，不能用改写推翻画质失败

新增 `synthesis_pipeline/audit_quality_v4.py`，仍使用本地 Qwen3.8-27B 与 vLLM。每次只有两张面板图：各自上半部为干净完整场景，下半部为同坐标的放大区域；仅原图局部画黑白 mask 轮廓，完整原图保持干净，避免轮廓遮住细小邻物。图片、实际 prompt、完整原始回复、解析结果均保存。

1. 独立画质判断：不给旧 instruction 和预设类型，先描述实际变化，再检查旧目标全部轮廓、细部残留、顶部截断、接触点、相邻对象及背景连续性。输出 observed_change/reason/quality 三字段。原图已有的模糊和遮挡不算新缺陷。
2. 独立指令重建：仅画质 pass 进入，不给旧 instruction。输出实际 task_type 与一句最多 25 词的 instruction；要求在完整原图中独立可定位，允许描述真实且完整的局部变化，不允许把残肢等缺陷包装为局部编辑。
3. 独立视觉复核：重新看同一图对及原/新两条指令，不给前两步结论；分别核验画质、两条标签、候选范围。原指令正确时优先保留；原指令不符而新指令与图匹配、范围相同时接受改写。范围缩小的候选进入 `needs_mask_realignment`，不静默沿用整物体 mask；另一实例或范围不清拒绝。未完成 thinking、格式错误都不能默认 pass。

这比上轮两次调用多一次独立复核，且仅对初审画质通过项执行后两步。先实验官方本地模型卡所述 thinking low，3072 token 上限；实际效果与代价须以新源评估为准，不预先宣称 thinking 更好。输出 `model_accepted_annotations.jsonl` 明确只是模型核验候选；`evaluate_audit_v4.py` 将其与 assistant 标签对比并另存人工门控导出，坏图漏检、好图误拒、改写真实性分开统计。

### 运行与可追踪性

新增 `run_fresh_cohort.py` 串联规划、源 mask 校准、六 GPU 编辑、add/replace 新轮廓分割及固定规则组装，每阶段独立 log 和 timing；任意 worker 异常必须显式失败，不把半成品当最终批次。编辑使用 `context_grounded_v4`、40 steps、seed 0；规划 GPU 3，SAM GPU 4，编辑 GPU 0/1/2/5/6/7。`prepare_fresh_iteration.py --sources-per-stratum` 可控制新源批量。最终快照新增 sources 链接，审核/可视化无需猜测原图位置。

报告继续使用图片内嵌 HTML 和独立 JPG；各阶段的完整理由直接显示，同时保留可展开的原始记录。所有拒绝项及原始标签保留，不只展示好图。本次尚未改生产默认，也没有提交/推送。

### 首个新源批次及审核消融（开发集，不能代替留出）

60 region → 42 条规划接受 → 37 条源范围可执行并完成扩散 → 36 条通过 add/replace 新轮廓合成，唯一后分割拒绝为 044。其中开发集 24 条已在读取本批审核之前逐图记录 `dev_reviews.jsonl`：20 条画质大体可用，4 条画质失败为 007 纯红填充、013 删除滑雪者导致相邻人断臂、019 红衣连续浅色边、029 手柄残片及手指拼接。033 原指令在多根木支腿中定位不足；036 屋顶广告屏与原要求立面位置不符；048 救生圈实际在船头侧板而非栏杆，属于候选可改写情形。

| 开发集画质方法 | 坏图拦截（共 4） | 好图保留（共 20） | 解析异常 |
|---|---:|---:|---:|
| 原 balanced_crop / non-thinking | 0 | 17 | 1 |
| 新轮廓检查 prompt / tight crop / non-thinking | 1 | 15 | 0 |
| 同 prompt / tight crop / thinking | 3 | 19 | 0 |
| 同 prompt + 固定视觉标定图 / thinking | 4 | 17 | 0 |

纯文字提示与更紧 crop 在历史开发回归中仍漏掉明显的头部截断；不能宣称加 thinking 就解决了审核。新增实验使用**两张待审局部图 + 一张固定标定参考图**，不是继续维持纯双图。参考图由历史开发集的断头、衣缘白线两个负例，以及帽子贴合、展台车辆两个正例构成；与本轮所有新源图不重叠。它不增加这一步模型调用次数，但增加视觉 token。prompt 明确第三张不是当前 case，不得复制其中对象/操作/结论。这使新源 019 的白边被识别出来，代价是误拒 017 醋瓶旁边界、037 留下白底板的 logo 擦除、055 镂空雕塑。以上三个 assistant 原始画质标签保留为 pass，不修改标签制造提升。标定图 manifest 和 PNG 保存在该批数据根目录。

### 指令补救复核的第二次修订

历史回归发现：同时提供旧、新指令时，复核仍可能把旧要求未完成误判成画质坏，例如身体变银但头部保持金色；还可能只验证“假发移除”为真却忽略同时新加了熊。为此新增 `--verification-policy candidate-only` 对照：不给旧请求，只核验新指令是否准确、是否覆盖全部有意义的实际变化，以及范围是否与 mask 一致。输出 reason/quality/candidate_match/candidate_scope 四字段；original_match 明确记为 not_checked，绝不声称复核了未提供的原指令。此模式用新标签代替原标签，因此所有接受的新标签仍要逐条做 assistant 匹配检查，不能仅凭画质通过计为成功补救。

`--quality-from` 可复用已完成的同一图版本审核记录开展标签验证消融，校验 edited_path，保留原 prompt/回复及 reused_from；质量调用计为 0，不重复计算为一次完整端到端速度。`run_audit_pool_v4.py` 可将固定批次分到多个 GPU，每 shard 常驻一个 27B；全部成功且最终记录数量完整后才合并。该并行是模型进程，不是 subagent。为了减少等待，后续 80 张新源/160 region 的源池已抽好并开始单独指令规划；正式扩散仍等待审核候选的留出评估。

### 冻结候选与独立留出检查

开发集固定参考图方案画质通过 17/24，准确拦截 4/4 个坏图，保留 17/20 个好图。候选单独复核最终自动接受 16 条，assistant 逐条核对实际新指令后确认 15/16 正确；033 木支腿仍有多个满足同一描述的实例，模型接受但人工拒绝。036 屋顶广告屏位置改写成功；048 船头救生圈描述正确，但 scope=subset 被送往 mask 重对齐，尚未自动接纳。人工门控导出 18 条（17 条原标签正确 + 1 条改写补救），与自动接受数分开。

`frozen_candidate.json` 记录 audit 模块、VLM 工具与参考 PNG 的 SHA256、完整 prompts、开发指标和已知限制。冻结配置为 quality=tight/contour-aware/thinking-low/2048 tokens，重建与复核各 3072 tokens，verification=candidate-only；第三张参考图只在画质判断出现。后两步每步两张“完整图 + 放大局部”面板。改写最多 25 词，程序校验长度。初步达标不代表坏图零漏检：开发集坏图只有 4 个，且最终仍有 1 条歧义指令。

12 条源图不重叠的留出结果已先逐图写入 `holdout_reviews.jsonl`，再启动冻结审核，不读取模型结论后修改人工标准。独立观察：015 衣服平涂丢失褶皱、051 广告牌整块抹绿且细杆边缘残留，是明显坏图；027 实际改了枕头及被子、031 只改挡泥板没改车轮、000 吊饰在花瓶前面而非瓶颈，是可尝试补救的标签偏差。留出结果不用于回调当前冻结方案。

后续 `fresh100_20260920_v11` 使用 seed=20260926，80 张新源/160 region，排除 1068 个历史及本轮源记录和重复像素，输入四类各 40 条。规划 160 → 114，274 次调用（含重试）；27B 加载 82.69 秒、推理 608.71 秒、总 697.93 秒。实际最终出图数以源范围校验、扩散和后分割结果为准，不将 160 个输入写成 160 条成功数据。

代码回归当前为 82 passed、1 skipped。迭代记录继续集中于本文件。

冻结审核的独立留出结果：12 条、人工画质好 10 / 坏 2；画质审核拦截 2/2 坏图，保留 8/10 好图，031 挡泥板和 043 窄墙面被误拒。最终自动接受 8 条新标签，逐条核对均正确，其中 000 花瓶挂饰的上部位置、027 被子和枕头的共同改色得到补救。人工门控导出 9 条（原标签正确 7 + 补救 2）；031 虽然可改写，但未获模型通过，不计自动补救。四 worker 审核含加载与图处理共 261.55 秒，12 次画质 + 8 次重建 + 8 次复核，GPU 推理时间相加为 546.96 秒，不能混同墙钟。

保持冻结参数后，扩展批次源 mask 校验 114 → 99，耗时 158.00 秒（含加载 9.60 秒）。99 条正式进入 8 卡扩散，不补抽失败项。`run_fresh_cohort.py --reuse-completed-plan --reuse-completed-regions --editor-gpus 0,1,2,3,4,5,6,7` 可复用已完整结束且样本 ID 匹配的准备阶段；这不是复用旧出图。源分割和留出审核准备阶段可并行，扩散开始前对应 VLM/SAM 进程均已退出。

### 约 100 条扩展出图的独立人工检查

99 条扩散全部完成，8 worker 均正常退出；后分割合成 93 条，6 条原始输出保留在 `final/composition_rejected`。扩散单条均值 77.49 秒（74.31–80.82 秒）；最慢 worker 加载加推理 1038.89 秒，近似扩散吞吐 5.72 条/分钟。编排实际编辑阶段约 1050 秒，后分割合成 48.41 秒（编排含启动 53.94 秒）。这些不是包含人工迭代等待的端到端生产速度。

93 条最终图在启动本批模型审核之前已全部逐图检查，`manual_reviews.jsonl` 保存具体理由。人工画质 pass 68 / fail 25；图片和原指令同时 pass 53；另有 15 条画质基本可用但原标签位置、对象范围或实例指代不符。既有中间文件复核 SHA256 与最终文件一致，93 条无漏看/重复。指令匹配与画质是独立标签，7 条虽然动作大体匹配仍因画质失败不能导出。

新暴露的问题包括：032/064/072 新增物体与宿主景深或光照不一致；052 车顶架断杆悬空；062 搅拌杆穿过新锅盖；070 女子变狗但保留牛仔裤腿和手柄；108 巨鸟的尺度及落点错误；146 玻璃楼被旧工地形状截断。016 花瓶落在座垫、068 方巾系在脖子而非帽檐属于可尝试改写；020 长颈鹿铃铛与原 mask 部位不符、080 灯笼偏到推车后的行人，则必须检查范围，不能只改文字绕过 mask。

6 条后分割拒绝的原始图也逐图检查：034 体操服女子和 132 地毯花瓶的原始图基本可用，属于后分割漏召回；030 矩形裁断裁判、012 胸章无有效变化、100 天线短块不清需拒绝，088 新增整束花且离开原花束位置也不能按原目标放行。详见 `rejected_manual_reviews.jsonl`，不混入 93 条自动审核的分母。

后续新源批次 `fresh_followup_20260920_v12` 已抽取 20 张新源、40 region，四类各 10，源 ID 与 RGB 均排除历史及当前源池，seed 20260927。规划提示新增全图同类候选搜索、仅在需要时添加组内序号/稳定邻物，以及移除时检查 mask 内孔洞及外部依附物；不加入固定新物体示例。规划 40 → 26，74 次调用，总 267.34 秒（加载 85.63、推理 176.51）。本轮原有扩散/审核结果保持冻结，改进只用于后续新数据或显式命名消融。

### 扩展集揭示冻结审核未达到自动放行要求

93 条冻结审核全部完成，8 worker 墙钟 636.03 秒；93 次画质 + 68 次重建 + 68 次复核，GPU 推理时间合计 2809.37 秒。初审坏图仅拦截 12/25（48%），好图保留 55/68（80.9%）。最终自动接受 59 条，逐条核对新指令后只有 40 条同时满足画质与新标签，精度 67.8%；原本合格样本的自动保留 36/53（67.9%）。复核未改善物理坏图召回，仍依赖同一个模型的相似盲点，不能把多轮同模型调用视为可靠独立裁判。

人工门控导出 57 条 = 原标签合格 53 + 真正重写补救 4（016 沙发花瓶、068 脖颈方巾、083 上排斜栏、099 花簇）。这是逐图人工门控数据，不是自动 pass 数据。19 条自动误接受中既有画质漏检，也有重写错误：028 把鞍毯叫马鞍、039 忽略车厢罩消失、055 忽略下片面包改色、066 把新增木板误当显露背景、095 船只指代仍歧义、159 未变教堂却写成移除。完整记录与可视化在 `fresh100_20260920_v11/report/index.html`，全部 93 条保留，不能仅展示 57 条来声称改进。

观察到固定参考图造成明显关注偏置：很多理由机械强调头顶完整/白边，忽略新增对象尺度、景深及穿透支撑；甚至把去掉 BEFORE 标注轮廓视为有效编辑。新增显式实验 policy=`photographic`：取消第三张参考图，改为两张完整干净全图+同坐标细节面板，提示检查语义真实变化、旧对象全部部件、接触与同距离尺度、景深/光照/噪声一致性；区分隐私模糊人脸与场景虚焦、自然平面图案与平涂纹理坍塌。重建/复核增加隐藏轮廓后搜索全部同类实例、新增物的真实宿主检查。它是待测修订，不提前宣称优于冻结方案；保留旧 policy 的默认行为，旧模块完整源码已存 `audit_iteration_20260920_v11/frozen_audit_quality_v4.py` 并验证 SHA256 与冻结清单相同。

并行资源说明：约 100 条正式扩散确实使用 8 卡；后续 23 条新源扩散暂用 GPU 0/3/6/7，GPU 1/2/5 同时跑 93 条新审核初审消融，GPU 4 在上一轮审核退出后空出。这个阶段的 4 卡出图是与审核实验并行，不是模型必须占 3 或 4 卡。

另有不进入主结果的纯色纹理诊断：`repair_color_texture.py` 只对发生明显亮度层次坍塌且不要求换材质/去图案的改色执行原图亮度纹理恢复，严格不改变 mask 外像素。99 条扩展图的 93 个最终输出中仅触发 151 唱片封面，恢复了被纯红抹掉的印刷与塑料反光；但原指令同名封面仍歧义，修复图没有进入冻结审核或训练导出。不能用这种修复掩盖扩散或标注本身的问题。

### 第二次新源追测与未奏效的修订

93 条 `photographic/overview` 初审消融已完成：坏图拦截 11/25，好图保留 66/68，解析异常 0；三 worker 墙钟 585.80 秒、GPU 推理合计 1229.38 秒。比冻结方案明显减少误拒，但坏图召回反而略降，不能升为生产默认。典型理由会把虚焦背景上的锐利心形解释成“刻意数字叠加”、把移除后白色残影解释成换白色材质、把丢失床单纹理视为轻微差异。两种初审取拒绝并集只能拦截 17/25 坏图，同时好图降到 53/68；不能靠简单叠加同模型裁判解决。

`fresh_followup_20260920_v12` 的 23 条出图已全部逐图盲审并保存 `manual_reviews.jsonl`，随后才启动 photographic 完整审核。人工画质失败包括 011 西装平涂、017 女子消失后悬空手柄、020 虚焦广告上的锐利箭头、022 换车后原飞机翼影残留、027 灯柱底座纯红平涂；007 斑马仅躯干颈部条纹变蓝、008 书落在座面而非桌面、031 招牌图案也被清除属于指令范围偏差。修改指令不能救回前五个画质问题。所有原始输出及失败记录均保留。

017 的规划存在明确字段矛盾：masked_content 承认目标 holding white game controllers，却把 white game controllers 列为 protected_objects。增加 `protected_dependency_conflicts`，仅在 complete_object/remove 中检查明说 holding/carrying/wearing/supporting 的对象与保留对象的冲突，触发规划重试，仍冲突则拒绝。不能通过把物品从保护清单删去来偷扩 mask；mask 外依附物仍应标不兼容。replace/attribute/complete_part 不套用这个规则，邻近而不依附的物体也不误拒；规划回归 29 passed。

对上述 23 条执行相同、事先固定的纯色修复诊断，仅触发 011。人工确认上部翻领纹理有恢复，但下半部仍平涂、底部残片仍在且颜色偏橙；保持 fail，不导出，不推广此修复。原输入亮度范围 35.36、编辑后 1.85 说明坍塌真实存在，但数值恢复不等于视觉成功。

下一命名实验 `critical` 使用缺陷优先的短提示：先看完整旧足迹、影子和依附物，再看真实接触、印花是否越出宿主、图像光学一致性，明确数字屏幕叠加不能充当照片里的物体；仍只有两张待审图，不加入新物体示例，不增加单阶段调用。旧 photographic/冻结结果不覆盖。下一批 `fresh_followup_20260920_v13` 以 seed 20260928 重新选择 20 张未用源图、40 region，用于验证规划矛盾守卫和继续追测；不会反复只在同一批图上调 prompt。

第二批 23 条 photographic 完整结果：初审及复核均拦截坏图 2/5，好图保留 16/18；最终接受 16，逐条核对新指令后正确 11、错误 5，精度 68.75%。除三张坏图误放行，000 把“加手表”误写成“替换仍存在的黑护腕”、007 仍把仅躯干/颈部变蓝写成整只斑马条纹变蓝。008 原指令混淆桌面/座面，按实际小水平台面改写得到补救。人工门控导出 16 = 15 原标签合格 + 1 补救。审核 4 worker 墙钟 386.40 秒，23+19+19 次调用、推理 GPU 秒合计 785.76。新增图像并未证明 photographic 达标，因此继续 critical 对照。

critical 的末轮复核按重建出的实际操作类型选择专用标准，只提示该类，不复述其他三类。明确 add 的 mask 是宿主范围：在同一宿主上给出精确落点，不应机械当成 subset；真正换宿主/落在指定局部外仍拒绝。attribute 对整物 mask 改写成局部身体范围依然保留 subset 重对齐门，不通过改字偷缩标注语义。

新增可选 `--pixel-evidence` 提供变动内域（去边缘）的亮度 p90-p10 数值，供画质与末轮复核注意三维表面细节坍塌；它不是自动 veto，没有新增模型调用。图像与 mask 尺寸必须一致，非 attribute/无充分内域时不提供。自然平面图案允许低对比，不从数值直接判坏。默认关闭，和无数值 critical 是两个命名实验。

修复导出标签一致性：接受改写时同时更新 editing_instruction 与 new_instruction，旧全局/局部指令只留在 audit_v4 原始生成记录中，避免下游继续读取旧局部指令。人工门控评估导出也同步两字段；原始出图、历史审核回复和原 annotations 不覆写。此修复不改变之前人工评分结论。

第三批源池与此前 1168 条源 ID 和 RGB 均无重合；规划 40→32、58 次调用，加载 83.66 秒、推理 140.85 秒、模块总 229.24 秒；32 条源范围验证后 27 条可执行，5 条 unresolved 保留。新规划回复中未出现 dependency guard 的明确矛盾触发，不能把接受数提高归因于这个守卫；它目前有单元回归和上批反例证据，尚无新批触发收益统计。

### critical 完整评估及新源追测：仍未达到无人值守自动放行要求

93 条扩展图的 critical/overview 完整审核，四 worker 墙钟 884.09 秒；93 次初审、65 次重建、65 次复核，累计 GPU 推理 2658.63 秒。初审拦截坏图 20/25（80%），保留好图 60/68（88.2%）；复核没有额外挡住这 5 个画质漏检，反而再误拒 3 个好图。最终自动接受 61，逐条核对新标签后合格 48（78.7%），原本合格样本保留 44/53。比冻结版本 40/59 有改善，但未达到预先约定的 90% 最终精度，不升为无人值守生产默认。

仍然漏检的画质包括 009 灰色显示器填块、029 模糊斑马、031 红制服边缘、036 无有效新增而误说花瓶缩小、064 锐利玫瑰。标签错误还包括 039 漏掉车厢罩消失、066 新木板被误当原来遮住的背景、126 新盆栽被错误重写成仅删除雕像、095 远处船只指代歧义、140 蝴蝶悬在花前却写成落在花上。人工门控可导出 57（53 原标签合格 + 4 本版本改写合格）；它不是自动接受的 61。

第二批 23 条使用 critical + pixel-evidence，二 worker 墙钟 511.73 秒，23+17+17 次调用、GPU 推理 762.73 秒。初审坏图拦截 3/5、好图保留 15/18；自动接受 15，合格 10（66.7%）。仍漏飞机旧翼影和平涂灯柱；改写又把拼图中的多人描述得不唯一、把斑马局部条纹泛化到全身、额外要求删除实际仍存在的裸枝树。亮度统计没有解决这些问题，不将该可选提示作为默认。

第三批 27 条原始生成、25 条合成输出全部先逐图检查并保存独立标签，2 条后分割拒绝也有单独人工记录。25 条中画质合格 19，原指令和画质均合格 14，另 5 条为标签/范围偏差。六条画质坏图：014 熊旁悬空旧象牙、019 新熊头嵌在原黑轮廓内、027 高木椅红色平涂、030 雪山替换留下悬空岩块与旗、032 虚焦横幅上的锐利星形、038 松果上方旧叶柄残留。后分割拒绝的 020 灯添加同时凭空生成整块驾驶室，024 没有可辨认的新旗帜。

第三批使用与扩展集同参数的 critical（不含像素统计），二 worker 墙钟 560.73 秒，25+17+17 次调用、GPU 推理 852.33 秒。初审挡住 4/6 坏图，保留 15/19 好图；最后接受 15，人工确认 11（73.3%）。027/032 画质漏检，028 把铆钉加到另一宿主却放行，034 桨板替换被误写成皮划艇改蓝。002 从预期长条点心改写为实际圆形巧克力点心得到补救；001 白熊消失后出现其他玩具的自然图还没有成功自动补救。不能仅用“画面自然”来接受错误标签，也不能把这些自然图的指令不符反过来当作画质坏。

### 新增宿主几何守卫与下一轮标签实验

`audit_geometry.py` 使用已有 SAM 新增轮廓与原始宿主 mask，计算原图同坐标的最小欧氏距离。允许 4 像素或图像对角线 0.3%（取较大者）的分割误差；不要求大部分新增物被宿主包含，允许花束、饰品等延伸。只有明确分离的新增物才拒绝；重叠不能证明语义宿主正确，所以不能替代视觉核验。空轮廓记 unmeasured，尺寸不一致报错，不靠 resize 伪造重合。

对已完成审核做只读补充，结果写到独立 `_contact` 目录，原模型理由与画质结论保持：扩展集拦截 020（最小距离 38.6 像素）和 080（35.2 像素），最终 48/59（81.4%）；第二批无新增拦截，仍 10/15；第三批拦截 028（16.1 像素），最终 11/14（78.6%）。没有拒绝这些批次中已经人工确认标签合格的 add，但这是开发回归结果，不是未见数据的保证。第三批 016 错加到前景斑马但二维重叠宿主 mask，几何检查无法发现，仍依赖语义范围核验。额外耗时各批约 1–3 秒，MLLM 调用为零。

新增 `--contact-guard` 可用于未来完整审核；`--label-policy grounded` 是独立命名消融：重建前先清点可见旧/新对象、轮廓厚度和未变化部位，禁止无证据假设新前景物本来藏在旧对象后面；复核先写实际 before→after 观察，再判断给定标签。保持重建 2 字段、复核 4 字段和原调用数，不叠加更多模型轮次。93 条 grounded 对照只复用 critical 的同图质量记录，省去初审调用，计时必须标为标签消融而非端到端速度。

第四批 `fresh_followup_20260920_v14` 以 seed 20260929 再抽 20 张新源、40 region，排除 1188 条源记录和重复 RGB，四类各 10。使用未筛成功图的固定输入清单，继续检验 grounded 标签流程和宿主守卫。此批尚未完成，不提前写入成功率。

### 标签流程对照与并排输入实验

扩展集 grounded + contact 对照完成（初审复用同一 critical 记录）：最终自动接受 57，逐条人工核对正确 49（86.0%），原本合格样本保留 44/53。复核额外拦下 029 模糊斑马与 036 花瓶无效变化，物理坏图累计拦截 22/25；没有新增画质误拒。066 木板、126 盆栽正确改回 replace，140 不再声称蝴蝶停在花头，151 前排唱片封面明确改成纯红画面。仍有 009 假打印机、031 衣缘、064 景深漏检，以及 028 鞍毯叫马鞍、039 漏掉厢罩、083 前后栏杆定位反了、095 船只歧义、111 施工车指代不清。人工门控导出 58（53 原标签合格 + 5 改写补救）。标签消融四 worker 墙钟 596.70 秒，0 次初审、65 次重建、65 次复核，不能作为完整审核速度。

增加 `--verification-policy selective` 对照：初审仍不见旧指令，重建也不见旧指令；最后一轮同时检查两条标签，用四字段 reason/quality/label_choice/chosen_scope 选择 original、rewritten 或 none。正确原指令优先保留；只有真正修正错误才换标签；指令不符不自动等于画质失败。原指令被选中时同样检查 scope，不能通过 keep_original 绕过实例范围。`--reconstruction-from` 可复用同图的精确候选来单独比较复核策略，校验输出图片路径，复用记录保留原回复与 reused_from、调用/推理时间计为零。

第二批 selective：同一初审和候选，17 次新复核，两 worker 240.74 秒；接受 17，正确 12（70.6%），比原 10/15 略有提高，但仍错选 000 护腕替换和 008 桌面位置，未解决 007 范围及两条坏图。第三批 selective：17 次新复核，单 worker 316.66 秒；接受 16，正确 12（75%）。修正 034 桨板被写成仅改色、补救 001 白熊换棕熊，但 002 仍错误保留 eclair 原标签，008 虽正确描述标识替换，却把整车 mask 的标识子部件当 same 直接放行，因此仍人工拒绝。不能因流程名叫“保留正确原指令”就认为模型已经能可靠判断正确与否；继续保留实验属性。

第四批正式出图 40 输入 → 30 规划 → 24 源范围通过并全部生成 → 20 后分割合成，4 条原始拒绝图亦保留。规划模块 258.31 秒、66 次调用（加载 88.19，推理 163.96）；编辑四 worker 编排约 498.73 秒；后分割编排 25.84 秒。20 条人工盲审先于模型：17 条画质基本可用，15 条原指令与画质同时合格，000 胸针实际在背部、015 新裙面料/图案也变了可尝试补救；007 黄餐车平涂、025 灯杆 Y 形天空残影、033 车厢底部条状残留为画质失败。001 中男子肩上手经干净原图复核属于男子弯臂，不是被移除女子的残手，避免人工也误判。4 条后分割拒绝中 012 小哨与 024 细红系带基本可用但未被 SAM 召回，016 锐利金链与原图颗粒模糊不符，032 没有新红旗。四 worker 执行 critical/grounded/contact 完整审核，未读取结论修改人工标签。

第五批 `fresh_followup_20260920_v15` seed=20260930，20 张新源、40 region，排除 1208 条历史源 ID 及 RGB。修正上游一个具体缺口：原 structural_parts 补充分割只对 remove 开放，现在 replace 也可申报同一旧对象缺失的细结构部件；仍须局部分割确认连接，竞争实例或无法确定即 unresolved，独立被持物/邻物不允许借此加入。执行前 guard 同步禁止未经范围分割的 provisional replace 落入旧编辑器。没有在生图 prompt 添加指定新物体示例。此批 30 条规划均通过源分割，没有出现非空 structural_parts；所以只能记录功能与回归修复，不能宣称本批召回提升来自该逻辑。

新增 `--input-layout paired` 是排版消融，不改变模型或多轮调用数：仍为两张图，第一张左 BEFORE/右 AFTER 全图，第二张左 BEFORE/右 AFTER 同坐标细节；仅细节 BEFORE 有 mask 轮廓。直接把原 vertical 的相同四个 1024×672 半面板重排为两个 2048×672 面板，没有改变源像素、缩放、裁切或总输入像素量。单测逐字节验证重排前后相同内容。明确的布局说明替换旧上下说明，实际图片分别保存 full_pair/detail_pair。这个对照检验模型是否更容易直接比较两侧，而非在一张面板内混淆全图与放大图；未完成前不声称有效。

### 并排与保留原标签的消融结论：均未选为当前方案

93 条 selective 最终接受 61，逐条新标签核对正确 50（82.0%），原本正确样本保留 47/53；虽比 grounded 的 44/53 召回高，但最终精度低于 grounded 的 49/57（86.0%）。它错保留了沙发扶手花瓶、蝴蝶停在花头、车厢罩未消失等旧请求，说明把旧请求重新交给末轮审核会造成锚定。单 worker 新复核 65 次、墙钟 997.31 秒；初审与重建均复用，不是完整流程基准。人工门控导出 56。

并排输入在 93 条的纯画质对照中拦截 16/25 坏图、保留 52/68 好图，均弱于原布局 critical 的 20/25 和 60/68；两 worker 898.39 秒。第四批并排完整流程 20 条，两 worker 504.26 秒、20+17+17 次调用；最终 13/16 正确（81.25%），漏检黄色餐车平涂、拖车底盘残留，并把前景斜绳写成未区分后方水平绳的宽泛标签。原布局 grounded 同批为 13/15（86.67%），人工门控 17 条（15 原标签 + 背部胸针、换裙款式 2 条补救）。两方案 worker 数不同，墙钟不可直接当成布局的速度因果证据；排版变化也影响视觉 token 网格与冷启动 kernel，不以总像素相同推断推理耗时相同。

因此当前保留 `critical / overview / vertical / grounded / candidate-only / contact-guard`，不启用固定参考第三图、亮度数值提示、paired 或 selective。这里的“保留”是现阶段开发候选选择，不是宣称达到无人值守生产标准。93 条精度仍只有 86%，继续保留人工门控。

### 第五批全新源图：先盲审，再执行保留候选

第五批 40 输入 → 30 规划及源范围通过 → 30 原始生成 → 29 合成，另 1 条原始拒绝图单独检查。自约 100 条扩展轮开始累计 **203 条原始生成、190 条最终合成**；五个源池之间及与历史按记录 ID 和 RGB 去重，不把各版本复审算成新出图。29 条最终图均先保存 `manual_reviews.jsonl` 再启动审核，新增模型结果不会反向修改盲审标准。

独立逐图观察：3 条明确画质坏图为 008 小斑马铃铛无可信连接、014 人换熊但胸前球及旧手悬空、016 鞍毯碎片状开孔。26 条画面基本可用，其中 7 条原标签需要修正或范围复核：012 胸针落在胸前非翻领；019 黄色餐车同时失去后部格栅和图案；020 牌落在门扇而非门框；027 工具仅一部分改色且前后轮位置表述不准；029 蓝书变为新书脊而非单纯消失；031 只改最上层立板而非全部台阶；039 近景栏杆拱顶仍白。019 不因原指令不符就自动判画质坏，黄色车壳自身仍有光照层次，但颜色指令确实无法覆盖结构变化。原指令及画质同时合格 19 条。

后分割拒绝的 004 新手表基本可见，但指令把白衣女子举着的电话错误归给橙衣男子，原关系指代仍不合格，不能把 SAM 拒绝简单算成纯漏召回。该例保存在 `rejected_manual_reviews.jsonl`，不进入 29 条最终图的审核分母。

第五批实际编排：规划 278.30 秒、源范围分割 52.95 秒、3 卡编辑阶段约 800.72 秒、后合成 28.03 秒；从规划到最终快照共 1162.88 秒（19.38 分钟），不含人工检查与稍后审核。3 卡来自与审核消融并行分配，不是编辑模型的卡数限制；约 100 条扩展轮已用 8 卡完成。

当前候选完整审核运行方式（ROOT 是已完成且不可变的这一批目录）：

```bash
python -m synthesis_pipeline.run_audit_pool_v4 \
  --data-root "$ROOT/final" --out-root "$ROOT/audit_grounded" \
  --gpus 2,3,4,5 \
  --quality-policy critical --quality-scope overview \
  --input-layout vertical --label-policy grounded \
  --verification-policy candidate-only --contact-guard
```

每个 worker 常驻一份本地 Qwen3.8-27B/vLLM，thinking low。第 1 轮不看旧 instruction，仅判断真实变化和画质，输出 observed_change/reason/quality；画质通过后第 2 轮不看旧指令，按实际两图重建 task_type/instruction（最多 25 词）；第 3 轮核对新标签，先描述实际变化，再输出 reason/quality/candidate_match/candidate_scope。根据重建出的编辑类型选择专用标准。所有阶段为两张“全图 + 同坐标细节”的面板，只有 BEFORE 细节绘制黑白 mask 轮廓。最终再用已有新增轮廓做零模型调用宿主接触检查。子区域重写需重新对齐 mask，不能直接按原 mask 导出；画质失败不能靠换一句话挽救。

报告区分 `model_accepted_annotations.jsonl`（模型自动结果）和 `assistant_verified_annotations.jsonl`（逐图及逐标签复核结果），两者不能混称合格训练集。所有阶段完整 prompt、输入图片路径、模型原始回复、最终决策及逐图理由保留。HTML 图片以 base64 内嵌，同时有独立 JPG，避免本地预览相对图片路径失效。

第五批上述候选审核完整结果：四 worker 468.67 秒，29 次初审、23 次重建、23 次复核，GPU 推理合计 1174.38 秒。初审及复核均仅拦截 1/3 画质坏图，保留 21/26 好图；自动接受 21、逐条核对合格 15（71.43%），原标签合格样本仅保留 11/19。明显低于扩展集与第四批，不能用先前较好数字声称已稳定。失败包括把已存在的后方蓝围巾女子描述为新替换对象、把删除笔记本误写为新增双手、放行熊胸前旧手红球和破碎鞍毯、工具子范围未重对齐、栏杆剩余白拱顶未描述。人工门控导出 23（19 原标签 + 012 胸针落点、020 门牌落点、029 新书替换、031 上层立板 4 条补救），不是自动接受21条。

完整第五批报告 `fresh_followup_20260920_v15/report_grounded/index.html` 同时放入 29 条最终合成和 1 条后分割拒绝原图，不隐藏失败。另做扩展集两布局交集的只读诊断：接受 48、正确 42（87.5%），额外丢掉7条原本合格数据仍不足90%，因此没有新增集成审核调用。

继续实验 `--input-layout full`：三个阶段统一使用两张原生尺寸完整照片，第一张轮廓标注、第二张干净结果，不拼接或重复全图/局部视图。相对于旧 `--quality-scope full` 仅改变初审，这次同样改变重建和复核，专门检查重复视图是否诱发对象对应混淆。原生 AFTER 除40像素白色标题栏外每个像素不变，mask 内 BEFORE 像素也不涂色，单测验证；尺寸不一致直接报错。模型、三个阶段字段数、质量标准和调用次数不变。全图实验保持独立命名 `audit_full_native`，原报告不覆写，未完成前不宣称改善。

### 第六批追测与本轮收尾：不要把单次较好数字当成稳定质量

第五批完整原生双图最终 14/23 合格（60.87%），四 worker 459.42 秒，29+24+24 次调用。虽然修正了笔记本删除，但仍将露出的原人物误写成替换，把苹果改色写成换果、台阶上沿改色写成删除，也漏了熊胸前旧手球、破碎鞍毯。细粒度杯内液体指令与整杯 mask 的子范围不匹配也没有被拦截。故不替换默认候选。

同图进一步定位到一类跨阶段自相矛盾：013 的初审正确写出“前景笔记本删除”，但后续重建与复核却一起声称“新增双手”。三阶段实际保存的两张输入路径相同，不是传错了另一个 case；是模型在不同提示下对同一照片产生不同解释。新增两个显式消融，不增加最终流程的调用轮数：

- `verification-policy=observed`：末轮增加初审不看旧指令时写下的 observed_change，只作为可能有错的另一种解读，不提供旧请求或初审 pass/fail，要求以像素解决矛盾。复用原初审和精确候选，单 worker 新复核23次、443.99秒；最终13/19合格（68.42%）。确实拒绝了错误的“新增双手”，却误判背景苹果的mask范围、台阶三角小片，还放行去掉格栅的餐车颜色标签，整体未改善。
- `label-policy=observed` 配合上述末轮：把同一可错观察也用于重建，仍不提供旧请求，输出仍只有type/instruction。两个worker重建23次、复核23次、423.99秒，最终13/20合格（65%）。恢复了前景女子删除和笔记本删除，但把手机边框误说成screen、苹果属性改成替换，仍未解决实质坏图。故两个 observed 策略均保留实验属性，不升默认。

另有 `quality-policy=topology`：在缺陷标准之前要求沿真实关节/支撑逐段检查，不能仅因二维重叠就宣称“握着/系着”，也不能想象额外肢体解释旧残片。第五批纯初审29次、单worker549.27秒；仍仅拦截1/3坏图，好图保留23/26（原critical为21/26）。模型仍把熊胸前残片描述为正常握球，把破碎鞍毯解释成完整扣带。增加文字检查项未解决底层视觉判断错误，不作为可靠性已提升的证据。

第六批 `fresh_followup_20260920_v16` seed=20261001，10张新源、20region，排除1228条历史记录及重复RGB，四类输入各5；20→13规划→10范围可执行→10原始出图→10合成，无后分割拒绝。先盲审再启动同参数critical/grounded：6条画质基本可用、5条原标签和画质均合格；008手表实际戴在男子右腕而非left wrist，属于可补救标签。4条画质失败为001杯子消失却剩吸管和杯底弧片，003列车正面变整块黄填色，012小玩具车远比同距离衣服清晰，018金狮下方沿旧盔甲双腿出现割裂浅色基座。残留杯子也说明“规划complete_object + SAM找到杯子”不能证明整套依附物都已覆盖。

第六批审核两个worker294.43秒，10次初审+7次重建+7次复核；仅拦截1/4坏图、保留4/6好图，最终4/7合格（57.14%）。手表补救正确，杯残片、过锐玩具车、双腿形基座仍误放行。人工门控导出6条（原标签5+手表补救1）。规划编排200.93秒、源范围31.81秒、4卡扩散约260.91秒、合成22.10秒，生成快照总518.30秒；单例扩散均值77.25秒（75.81–79.60）。小批量10条的加载摊销较高，不能以其吞吐直接外推100k。

至此自约100条扩展轮开始共六批新源，**213条原始生成、200条最终合成**，最终图和后合成拒绝case均逐条检查，所有失败记录保留；不是同一批反复出图凑数量。最新未筛选报告为 `fresh_followup_20260920_v16/report_grounded/index.html`；第五批报告包含29最终图和1原始拒绝图。两者均内嵌图片、完整模型回复及逐条理由。

本轮结论：扩展集的最终自动标签精度确实从冻结方案40/59提升到grounded+contact的49/57，但后续新源追测回落到15/21和4/7，尚未达到可无人值守放行的稳定性。当前可保留的工程收益是独立画质与标签、按真实变化重建、错宿主几何门、子范围重对齐隔离、导出字段一致性与完整追溯；不能把这些工程改进等同于27B已能可靠识别所有视觉坏图。没有重启100k自动生产，也没有把观察到的失败强行改成pass。继续盲目增加提示词/同模型重复投票已出现收益平台；大规模自动放行需要进一步独立标注校准/训练或新的可靠视觉信号，不宜用本轮小样本最佳数字作保证。

### 单独交付：104条逐图复核候选（不是自动审核通过率）

新增 `collect_reviewed_cohorts.py` 严格校验每行有assistant画质通过与对应原标签/精确新标签通过，拒绝纯model-pass、过期改写评论和冲突的instruction别名；验证前后尺寸一致、区域无重复，才写新目录。原始输入不覆盖，图像只做引用，不重新压缩训练图片。两项专用单测覆盖未复核误入、指令文本变更及别名冲突。

`reviewed_104_20260920/` 汇总相同critical/grounded审核配置的四个新源批次：扩展轮58、第四批17、第五批23、第六批6，共104条；这是跨批次复核筛选，不是单批无筛选测试，也不是人工标注金标准。类型为add32 / remove18 / replace23 / attribute31；没有为了凑均衡而收进失败样本。来源完整保存在summary与每行reviewed_cohort等字段中。

- `annotations.jsonl`：最终导出指令、mask及审查来源；source/edited目录引用原始图片。
- `gallery/index.html`：104条可用候选的唯一汇总入口，内嵌图片和完整模型/assistant理由；顶部指令是实际导出标签，模型当时的候选可能不同，页面明确说明。
- `release_reviews.jsonl`：顶层instruction_match指本次导出标签；original_review保留原指令及模型改写的实际判断，不覆写历史。
- 原批次完整报告仍保留所有自动接受/拒绝和人工不同意见，不能以104条精选展示计算自动精度。

回归测试最终为103 passed、1 skipped，`git diff --check`通过。本轮未提交推送，因为当前请求不包含新的提交/推送要求；既有工作区修改均保留。

## 2026-09-21：按阶段修复，第一项——指令与源区域的可执行范围

本节集中记录本轮各次尝试，不另拆多篇文档。用户要求每一项先迭代、展示结果及下一项计划，再进入下一项；本项不修改扩散、后合成或成图审核参数。验收对象是**指令、选区与真实依附关系是否一致**，不是整体出图通过率。对无法确认的区域保留明确拒绝，不能把它们计入“成功修复后出图”。

### 冻结样本及实验顺序

- 已知回归：第六批 000/001/002/003/008/011 六条；第五批 012/014/020/027/031/039 六条。包括正常对照，不只看错误样本。
- 新开发源：`planning_scope_20260921_fresh`，seed 20261002，10 张源图、20 个 region，排除 1238 条历史源 ID/RGB。源图按面积分层，不替换失败项。这20条参与了迭代，不能再称为未见测试集。
- 后续未见源：`planning_scope_20260921_unseen20`，seed 20261003，另10张源图、20个region，排除1248条历史源及重复RGB；用于候选流程的完整生成测试。

### A：只改规划规则不足

规划仍用两张图、Qwen3.8-27B/vLLM，初始调用加既有最多两次格式重试，没有新增审核调用。要求先读取实际边界与接触关系再设计任务；禁止未经确认的解剖学左右；已有“保护外部被持物”的自相矛盾检查扩展至replace。数据集指代只能辅助身份判断，不能把局部mask强行解释成整个对象或整组实例。

结果：持球球员换熊被判外部依附物冲突，手表左腕错误触发重写；但杯子仍被声称包括整根吸管，火车正面仍被当作一种材质，台阶仍被称作整组steps。因此没有把A当作完成。

### B/C/D：独立范围核对及其失败

新增 `ground_planning_scope.py`，在昂贵的扩散前查看同一原图与轮廓crop，只核对拟定动作与源mask，不看编辑结果。输出六个字段：reason、decision、refer_object、editing_instruction、segmentation_target、mask_refinement。它可以修正现有类型内的指令，不得改类型或扩大mask。首次新增一轮MLLM调用，格式/文字自相矛盾可再重试一次；视觉拒绝不反复重试。所有输入、prompt、原始回复和无效原因完整保存。

B确实识别了吸管及台阶窄条，但也出现过度拒绝、把正确的灯框局部改色改宽成整个灯改色。C明确“mask是最大允许范围，add/attribute不必修改全部选区”，并给crop增加外侧TARGET INSIDE文字和蓝色指向线。标记终止于边界外，单测验证选区内部像素完全不变；原图始终干净，不增加第三张图，不使用红色填充。指向线没有消除所有内外混淆：新003仍把后方黑衣男子的mask解释成前方白衣男子。

D对“理由已识别局部、指令却仍宽泛”的文本矛盾添加重试，对不明确的服饰左右和“visible strip”要求明确位置；过长分割短语也记录为无效而非放行。修正了火车指令为painted body panels、不存在的lapel落点为胸前。但这些仍只是文字候选，不能以MLLM的accept证明可执行范围正确。

### E：独立的源分割几何证据

`refine_samtok_regions.py --verify-original-scope` 增加可选严格执行检查：

1. 所有attribute以请求的具体源表面做SAM分割，不能因为模型填original就绕过。候选须至少98%落在原mask内；不再允许把明显更大的对象强行裁进选区后沿用原指令。
2. 全图未解决时，再在保留上下文的局部原图上分割同一短语，映回原坐标。仍不成功则unresolved，不使用原mask充作已确认结果。
3. add先验证请求锚点；细部短语未召回时，允许回退到初始规划的宿主查询，但只证明宿主对应，不冒充已分出胸前/鸟喙等具体落点，回退查询写入证据。
4. 对规划清单明确写出holding/carrying/supporting或held/carried的物品，单独分割并检查覆盖。靠近目标、但主体在mask外的被持物会阻止remove/replace，不自动扩大选区。仅以文字明确归属为前提，不从邻近关系推断所有权。

实际诊断：新003白衬衫查询无法落入后方黑衣男子的mask；新019树林查询超出岸边窄带，均被几何约束拦截。新005虽然MLLM声称球棒已包括，独立球棒分割显示几乎全部在mask外，从而拦截删除持有者。局部分割恢复了部分工具、台阶条和儿童衣服表面。小路灯的金属框仍无法可靠分割：这是召回损失，不记作成功。

首次分割执行发生局部变量`evidence`覆盖证据列表的错误，已改名`match_evidence`；失败输出保留在`regions_c`/部分`regions_d`，后续重跑分开存储。被动语态物品解析也补充了回归测试。不得将这些失败运行计入正常吞吐。

完整实验入口新增 `run_fresh_cohort.py --scope-preflight`；目前为显式实验选项，尚未把失败的B/C核对或严格分割直接升为无人工监督生产默认。展示器 `build_planning_scope_report.py` 内嵌图片，包含初始规划拒绝、每轮完整回复、当前源区域和assistant判断。结果数量、完整出图和最终结论在下文补齐，不以开发集最好的一次数字提前宣布过关。

### F：独立配件接触检查，以及本次子项交付边界

第一批新源和后续未见源又发现两种漏检：棒球手清单使用“a bat held ...”的被动语态，原提取器漏掉；黄衣行人的黑色手提包甚至完全没有出现在规划清单里。已补充被动语态解析，并对remove/replace人物在源图额外执行SAM的bag、umbrella、handheld object查询。这些词仅用于独立安全检查，**不是提示生成什么新增物体的例子**。分割高置信、靠近目标且主要在mask外时记录`ambiguous_external_accessory_contact`并阻断；邻近只代表风险，不宣称已证明所有权，不自动扩大mask。此规则可能误拒相邻独立物件，不能当作完美依附关系分类器。

现有已知依附物回归：杯子外吸管由独立MLLM范围核对拒绝；持球运动员由初始依附关系规划拒绝；棒球手外部球棒由独立分割拒绝；黄衣行人漏写的黑手提包由新增配件接触检查拒绝。后续新源中，马的牵引绳约11.7%在mask外，范围MLLM仍accept，但独立绳分割将其阻断。这些是**五个观察到的风险已被提前拦住**，不是五张原坏图被成功修复，也不是大规模准确率。

台阶另外追了E31/F31两次：E31的“selected strip”仍然依赖标注，未采用；F31改成“upper horizontal segment ...”，文字定位更明确，但精确短语SAM无候选，最终仍unresolved。工具例027虽然SAM面积/包含率通过，实际轮廓仍像多个相接黑色工具，单一lug wrench命名存在疑问；不把几何一致冒充语义已确认。两例都不算成功救回。

### 三批新源的完整结果（不补采样替换失败分母）

本项共使用12个旧回归region，加三批互不重复的新源各20region（共30张源图、60region）；不是在同批上重复生成60张。前两批已参与调参，不能称为独立留出集。

| 批次 | 输入region | 初始规划接受 | 范围核对接受 | 可执行 | 实际原始/最终图 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh / seed 20261002 | 20 | 15 | 13 | 开发多轮，不汇成一个终版指标 | 未运行扩散 |
| unseen20 / seed 20261003 | 20 | 14 | 13 | 11（新增黑包检查前） | 11 / 11 |
| confirm20 / seed 20261004 | 20 | 14 | 12 | 9 | 9 / 9 |

confirm20排除前1258条历史源ID及重复RGB，完成后检查全部20条源/规划及9张成图。9张中8条的源范围与对象指代基本对应；019虽然分割到正确的紫色下装，但“pants”与看起来更像围裹裙的源类别不可靠，仍记为需修正，**不能宣布前置阶段全部过关**。11条未生成也并非全部正确拒绝：002把遮挡飞机当漏分割、006声称人物不是physical object是明显误拒；003白色飞机表面未召回；004三次规划解析失败；012独立reference字段丢定位词导致结构误拒；011仍企图将绿色车棚局部改色扩成全车改色；013医疗场景物件类别猜测错误被SAM无候选拦住。

两批20张实际成图均逐张检查，记录分开评估source_scope / visual_quality / instruction_match：

- unseen20：6/11画质和原指令同时基本通过。坏例含000贴片过锐、008叶片落在象鼻弯曲处而非鼻尖、009删除背景发糊、010熊旁旧包残留、011只改上衣未改裤子。010在新门控里已被单独复测阻断，原失败图不删除。
- confirm20：5/9画质和原指令同时基本通过（007、008、016、017、018）。000帽子过锐、001补桌面亮度/硬边不自然、015河水变近纯色块；019新黄色裤子本身自然，但原下装类别不可靠且花纹/结构改变，不按纯改色合格导出。007总体可用仍有少量红色衣缘，保留轻微缺陷说明，未把所有像素差都当严重失败。
- 另重新生成两个旧新增物回归：v16/008改为明确男子胸前银色胸针，成图与落点均基本通过；这是重新规划了任务，不是同一手表任务的参数对照。v15/012也改为胸前落点，原始图可见胸针，但后合成新增物分割无候选，未导出最终图。两条不能合称2/2端到端通过。

本次成功交付的最小子项是**依附物漏检的分层拦截**及其可追溯证据；范围缩小验证也已有可用实现。整个前置阶段尚有语义命名、遮挡误拒和精确表面召回问题，仍保持实验开关，不升为100k无人值守流程。质量判定来自assistant逐图检查，不是新增人工标注员的金标准，更不是本轮自动audit测得的准确率；本项没有运行或更改成图audit来美化数字。

### 速度、运行恢复与可复现入口

confirm20正常完整运行624.14秒（约10.4分钟），从20个准备好的region到9张最终图，不含源parquet准备、人工看图和成图audit。初始规划183.71秒、范围核对136.01秒、SAM范围29.60秒、4卡并发扩散阶段约253.18秒、后合成19.05秒。单例40步扩散均值77.30秒，范围75.16–79.29秒；模型、seed和扩散参数未作本项调优。

独立范围核对使用Qwen3.8-27B/vLLM，14条输入共17次调用（含格式/矛盾重试），纯推理38.72秒、加载82.80秒，**确实增加了出图前一轮MLLM，不宣称免费或调用次数不变**。SAM范围自身wall23.52秒，其中加载10.57秒。小批量加载占比高，不能据此直接估100k吞吐。

unseen20首次4卡出图中我安排的另一个回归任务与GPU2冲突导致OOM，原分片失败日志保留；其他worker正常完成。仅把失败shard2在空闲GPU4重跑，再手动完成composition/snapshot，最终11张均完整。该批端到端耗时不作为正常吞吐指标，单例扩散均值76.62秒。没有停止用户的其他GPU任务。confirm20固定预留GPU0–3，辅助实验不再占它们，该批所有阶段正常退出。

运行方式（从已冻结的新源目录开始；目录不存在时先用`prepare_fresh_iteration`准备，不能覆盖本次输出）：

```bash
/usr/bin/python -m synthesis_pipeline.run_fresh_cohort \
  --root <new_frozen_cohort_root> \
  --planner-gpu 0 --sam-gpu 1 --editor-gpus 0,1,2,3 \
  --scope-preflight
```

单入口最新报告：数据根目录下`planning_scope_20260921_confirm20/report_reviewed/index.html`，内嵌原图、实际轮廓crop、执行mask及9张成图，保留全部20条输入、各轮完整prompt/回复和assistant逐条理由。旧回归展示`planning_scope_20260921_reg16_a/report_reviewed/index.html`；上一批未筛选结果`planning_scope_20260921_unseen20/final_report/index.html`。审查证据统一放在本repo的`docs/data/PLANNING_SCOPE_*_REVIEWS.jsonl`，说明文档仍只更新本文，不新拆多篇doc。

### 下一项顺序

按用户要求，本次先交付上述子项的改进和局限。下一步仍留在前置阶段，优先解决：①遮挡不等于mask不完整、人/动物也是可替换实例；②不确定物件/服饰不凭场景猜功能类别，避免类别词诱发结构改变；③精确表面短语召回及reference与公开指令字段一致性，降低无必要拒绝。不会以放松几何阈值或扩大局部指令到整个对象来提高数量。上述问题收敛后，再单独迭代编辑阶段的过锐新增物、改色纹理保持及删除背景/边界补全，最后才重测audit。

收尾检查：相关7个测试文件共98 passed，`git diff --check`通过；报告20个case卡片、29张内嵌图片、20条源阶段说明及全部9张输出理由均核对存在。新测试已加入gitignore例外，避免后续提交时遗漏。所有本轮推理进程已结束，未启动下一批或100k生产；本次没有提交/推送（当前请求未要求）。

## 2026-09-21：去掉出图前 SAM，固定数据集 mask 的规划迭代

### 结论与边界

按用户要求新增 `--dataset-mask-plan` 路径，移除**出图前**的 SAM 再分割、范围修正及语义验证，直接使用原始 dataset RLE。保留出图后的新物体轮廓合成 SAM，它不是源 mask 质量审核。本次 100 个新 region 的 plan/scope/regions 中原 mask 已逐条核对相等，源 SAM 调用 0。

已完成代码实现、多轮提示/校验迭代和 100 个新 region 的实际生成与逐图检查；**不宣称前置审核全部过关，更不升级为 100k 无人值守生产**。原始分割可用不代表任意编辑任务都与该分割兼容。去掉源 SAM 降低了额外语义门槛，但 Qwen 仍会漏判实例、孔洞和依附物。下述质量结果来自 Codex 的视觉分析，不是人工标注员金标准，也不是 pipeline 成图 audit 的测试准确率。本轮没有修改或重新运行成图 audit。

### 当前实现

入口 `synthesis_pipeline/plan_dataset_regions.py`，当前版本 `dataset_mask_v5_1`：

1. 每个原始 region 一条任务，source 和原 mask 不变。复用其他原始 region 构成保护区域，不通过模型重分割获得。
2. Qwen3.8-27B 经现有 vLLM backend，bf16、batch size 8；规划和源图计划复核共用同一个引擎/模型加载。两轮都只给干净全图和带外轮廓的上下文 crop。crop 为 mask bbox 周围 40% padding、最小 48 像素，保留原图内容；黑白轮廓、外部 TARGET INSIDE 指针，不给红色 overlay 或去背景 cutout。
3. 每轮 JSON 仅 `reason / decision / target_description / editing_instruction`。正常每条两次调用，格式错误每阶段最多重试一次；模型明确 reject 不进行碰运气重试。首轮 accept/reject；复核 accept/revise/reject。这里 revise 是修改计划，不是成图 audit 的 review 档位。
4. 指令通常 8–24 词，硬上限 32 词，必须含清晰的全图实例定位，且完整包含 target_description，防止生成最终指令时丢掉限定语。目标短语建议不超过 18 词，但 V5.1 不再因为它单独超过 18 词而拒绝一条总长合格的指令。避免保留条款、背景重建配方、坐标或标注术语。
5. 类型专用规则：add 选择可见安装面和场景合理新增物；remove 检查可见依附物，不能删人留下悬空持物；replace 要保持 mask 外支撑/接触可成立，并允许人物、动物、遮挡和边界截断；attribute 明确具体表面，不能以颜色变化为由默认抹掉花纹或改整个主人。
6. 共享多实例 referring hint 不再直接给模型作为单 region 身份证据；仅保留绑定到独立 mention 的提示，并声明文字不是选中范围证据。加入由原始二值 mask 计算的中心孔洞/外围关系提示，不引入 SAM 或第三张输入图。
7. 复核要求独立指出选中部位、最近相关的未选中对象/部件，以及该指令是否在全图唯一。可通过措辞修复的歧义应 revise；确有无法满足的外部依附物才 reject。
8. 代码兼容真实 `face mask`，并在其他口罩写法触发标注词检查时明确提示改为 `face covering`，避免旧重试只是重复无用错误代码。

执行记录保存 `plan/`、`scope/` 的完整 prompt、原始回复、每次 attempt 和输入 crop；`regions/` 保存 `source_sam_called=false`、`provenance=dataset_mask_no_source_sam`。当前新入口需要显式开关；旧入口未被静默切换，旧实验文件没有覆盖。

### 迭代数据与结果

新增 50 张源图、每图 2 个 region，共 100 条；采样时分别排除已有 1268、1288、1308 张历史源图，RGB 内容去重，三批相互无重叠。源任务分配为 add/remove/replace/attribute 各 25 条；门控后不为凑比例补采或隐藏拒绝。

| 本轮新批次 | 版本 | 输入 | MLLM 后进入出图 | final | Codex 严格可用 final |
| --- | --- | ---: | ---: | ---: | ---: |
| dataset_mask_20260921_dev40，seed 20261005 | V2 | 40 | 39 | 39 | 18 |
| dataset_mask_20260921_confirm40，seed 20261006 | V3 | 40 | 39 | 34 | 22 |
| dataset_mask_20260921_holdout20，seed 20261007 | V5 | 20 | 19 | 18 | 8 |
| 合计 | 不同版本/不同源图，非受控质量趋势 | 100 | 97 | 91 | 48 |

97 张实际输出（含 6 张未形成 final 的 raw 诊断图）全部逐图查看，并分别记录 plan、output_quality、instruction_match 和具体理由。严格可用要求 plan=pass、画质=pass、指令匹配=pass，且有 final；有定位风险的计划不算严格通过。仅画质和文字同时可用的输出共 55/97，其中存在原始 mask 不匹配或后处理拒绝，不可直接全部导出。

最新 20：16 个计划范围/指代可用，3 个错误被 MLLM 漏放（007、008、009），1 个合理拒绝（013 石门柱底座删掉将导致上方结构悬空）。19 张出图里，13 张视觉质量基本可用，10 张画质和文字同时可用；004 只有 raw，008 与原 mask 不匹配，最终严格可用 8 张（000、003、006、010、011、012、014、017）。这些数字说明仍有明显瓶颈，不能把 19/20 计划接收率说成质量通过率。

另外做了旧 20 条回归与开发 40 条重复规划：

- 旧 C20：V1 接收 19/20，V3 接收 18/20。旧流程只有 9 个可执行 region；新方案恢复了被遮挡背景飞机 C002、前景白机身 C003、被桌子挡住的男子 C006，C010 车架换踏板车无法接原车厢被拒绝。C014 从包含河流缩到森林；C019 使用实际可见的 patterned lower garment。**同时改了提示和门控，并非只关闭 SAM 的单变量消融；旧 9 张最终图和新 18 条计划不可当成画质率对比。**
- 开发 40 的 V4：40/40 放行，但仍漏放击球手/球棒、成年象/幼象、火车双实例与邮票外圈混淆，不采用“放行率提升即成功”的判断。
- V5 屏蔽共享描述、强化全图定位；成年象 018 从“两只象换两只犀牛”变为只替换成年象；026 前景路缘、027 细杆、032 指定文字纸牌和 035 左侧桥梁框架定位改善。V5 因过严目标短语长度和口罩写法仍误拒了 027/031/033。
- V5.1 修正上述纯文本校验与错误反馈；同一开发 40 条 40/40 格式流程跑通（48 次 plan、40 次 scope 调用）。033 成功改写成 `rainbow heart face covering`，027 保留完整位置短语。**40/40 仍不是语义全对**：005 外部球棒、014 错带黄色机车、037 外圈误认白邮票边和 038 整椅/局部范围仍有问题。此重规划未重新扩散；不能拿新指令配此前 V2 图冒充新成图。最新 20 的图仍是 V5 实际计划，V5.1 没有修改这些图或替换对应指令。

### 最新具体 case 与下一项优先级

- 正例 006_gres_r3119_m0_replace：多架同类飞机，仅最近左侧尾部换 United 涂装；011_ver_r6927_m1_attribute：全图极小船只灯条改红；012_ver_r4668_m0_add：对称门柱中右柱加灯；014_ver_r6295_m0_replace：咖啡师换蓝围裙女子并保持操作关系。这些说明不经源 SAM 仍可完成局部、实例选择和遮挡编辑。
- 007_gres_r3119_m1_attribute：原 mask 主要选中间机尾，最右机尾只有零散噪声；MLLM却要求中间和最右同时改色。新几何提示仍未完全解决多实例合并。
- 008_gres_r7496_m0_add：选中人却把 mask 外背包当新增贴片宿主。实际包上贴片看起来可用，但不能作为原 mask 对齐数据；生成器允许新增轮廓扩展并不意味着可任意更换源宿主。
- 009_gres_r7496_m1_remove：人被删、包悬空；开发 005 留球棒、确认 017 留酒杯，都是依附物漏审。应继续改进 MLLM 的选中/排除对象理解，不依赖恢复源 SAM，也不能仅加几句泛化“检查依附物”后宣布解决。
- 004_gres_r4760_m0_add：raw 项圈与银铃自然，后处理 `new silhouette unresolved`。确认 010 替换玩家、022 陶罐换桶也有可用 raw。022 的新对象短语错误提取为 `orange hose by a blue plastic bucket.`，确有文本解析问题；下一步应修正确切新目标提取和小新增物的合成误拒，不是简单放宽所有阈值。
- 删除地毯猫 005、替树篱 018 仍有补丁/树枝残影；019 草坪亮黄平涂、确认 015 床单和 027 屋顶平涂，应单独优化扩散/合成与纹理保持。015 店员只要求上衣，却帽子也红；016 小木牌要求墙面，结果立在墙前地上，应约束子部位变化和新增物落点。

下一步顺序：①继续在两图输入约束下做 mask 内外/单实例识别的受控可视提示对照，冻结部分新数据只做最终评测；②修复出图后新轮廓提取/短语解析误拒；③编辑器删除补全与改色纹理、局部落点；上述质量有所收敛后再重测成图 audit。保留每条失败，不能靠删样本或换指令掩盖画质差/原 mask 越界。

### 速度与运行方式

| 新批次 | 规划+复核进程 wall | 并行扩散 wall（含各卡加载） | 后合成进程 | 总 wall | 单例扩散均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev40 / 4 卡 | 243.03 s | 804.89 s | 31.59 s | 1082.48 s | 77.21 s |
| confirm40 / 3 卡 | 227.63 s | 1033.09 s | 30.24 s | 1293.74 s | 76.86 s |
| holdout20 / 4 卡 | 189.81 s | 421.78 s | 23.50 s | 637.76 s | 77.79 s |

从已准备源数据开始计时，不含 parquet 准备、Codex 逐图复核或成图 audit。最新 20 的规划模型加载 84.73 秒，规划纯推理 46.63 秒、范围复核 39.76 秒，24+20 次调用含格式重试；小批量冷启动占比较高。V5.1 开发 40 重规划内部 wall 254.72 秒，加载 85.32 秒、plan 77.64 秒、scope 72.77 秒。两轮共享模型避免两次冷启动，但扩散约 77 秒/例仍为主要计算开销。以上不是最终合格数据生产吞吐，也不是 100k 时间承诺。

运行已冻结、尚未生成的新源目录：

```bash
cd /opt/tiger/tanyue/samtok-derived-edit-labeling
/usr/bin/python -m synthesis_pipeline.run_fresh_cohort \
  --root <new_frozen_cohort_root> \
  --dataset-mask-plan \
  --planner-gpu 0 --sam-gpu 0 --editor-gpus 0,1,2,3
```

`--sam-gpu` 在新路径仅用于扩散完成后的新轮廓合成，不会调用源 SAM。不要与旧 `--scope-preflight`、`--reuse-completed-plan`、`--reuse-completed-regions` 混用。未准备的数据先用现有 `prepare_fresh_iteration` 冻结，保留 annotations/sources/dev/holdout；此命令不负责自动补采。各 GPU 先规划后扩散再合成，避免同卡竞争。

### 可视化与证据

数据根目录仍为 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling`。

- 单入口：`dataset_mask_20260921_results.html`，默认引导最新 20；前两批收在展开项中，不混作同一版本。
- 最新：`dataset_mask_20260921_holdout20/report_reviewed/index.html`，约 9.5 MB，20 条输入、18 张 final 和 1 张 raw 诊断图。全部图片 data URI 内嵌，下载 HTML 单独打开也不会丢图。
- 先看 Current（实际出图指令），上方左干净全图、右实际规划 crop；下方左原图、右结果，上排全图、下排局部。PLANNED 只是计划接收。流程 MLLM 理由与 Codex 逐图理由分栏；每次 prompt/reply 均可展开；原始诊断明确不是 final。
- 全部新数据的复核证据：`docs/data/DATASET_MASK_DEV40_PLAN_REVIEWS.jsonl`、`DATASET_MASK_DEV40_OUTPUT_REVIEWS.jsonl`、`DATASET_MASK_CONFIRM40_REVIEWS.jsonl`、`DATASET_MASK_HOLDOUT20_REVIEWS.jsonl`。每条保留完整理由。

收尾测试：本项相关 7 个测试文件 **94 passed**，`git diff --check` 通过。三批全部运行成功，无推理报错；所有推理任务完成，未启动 100k、未提交推送。所有改进记录继续统一追加本文。

## 2026-09-22：Qwen-Image-2.1 编辑器受控对照

保持 planning、dataset mask、局部 crop、区域扩散约束、后合成和输出格式不变，只把图像编辑器从 Qwen-Image-Edit-2511 换为 Qwen-Image-2.1。新后端使用官方 `QwenImage21Pipeline`、bf16、40 steps、`true_cfg_scale=1.0`、默认 KV cache，不传该接口不存在的旧 `guidance_scale`，无 CFG 时也不额外传 negative prompt。独立环境为 `/opt/tiger/tanyue/.venvs/qwen_image_21`，旧 2511 环境未改动；本地安装为 PyTorch 2.8.0+cu128、Diffusers 0.41.0.dev0、Transformers 5.18.0.dev0。

2.1 原生支持 RGBA 输出。最初沿用 2511 的长坐标/rectangle prompt 时，模型会把替换对象输出成透明图层；直接转 RGB 会泄漏紫色 matte，简单 alpha 叠加又会留下旧对象。最终改为官方风格的简短自然语言原地编辑 prompt，并先按 RGBA 语义叠到干净 crop，再进入原有 grounded composition。区域 denoising 适配了 2.1 的 16 倍 VAE、64 通道 latent packing 和 step-end callback，未退化成无 mask 的整图编辑。

冻结 `dataset_mask_20260921_holdout20/regions` 中实际可执行的 19 条，固定同一 source、mask、instruction、crop、seed=0、40 steps。Qwen-Image-2.1 19/19 出图、19/19 完成后合成；2511 的同批次有 1 条视觉可用 raw 被后合成拒绝，只有 18 张 final。按同一 assistant 严格逐图口径，要求画质和原指令同时通过且存在可用 final：

| 编辑器 | 可用 final | 严格通过 | 单卡单例均值 | 单卡吞吐 |
| --- | ---: | ---: | ---: | ---: |
| Qwen-Image-Edit-2511 | 18/19 | 9/19 | 77.79 s | 46.28 图/小时 |
| Qwen-Image-2.1 | 19/19 | 15/19 | 17.21 s | 209.22 图/小时 |

2.1 在相同 40 steps 下单例扩散快 4.52 倍，延迟下降 77.88%。8 张同型 H100 的纯编辑阶段理论吞吐约 1674 图/小时，100k 约 59.7 小时；此估算不含 planning、冷加载、后合成和 audit，不能当作完整流水线承诺。对照运行从 `/tmp/tanyue_qwen_image_21` 本地副本加载，8 个 worker 加载均值 13.61 秒；首次从 `/mnt/bn` 读取约 31 GB 权重曾接近 10 分钟，正式长跑应先做节点本地缓存，否则短批次 wall time 会被共享存储冷加载主导。

主要改善：005 移除猫后的地毯纹理连续；014 人物替换不再出现透明 matte/旧人残留；015 只改后方人物上衣；016 木牌正确附着墙面；018 替换树篱后无旧枝残影。仍失败的 001 是指令只写花束但 basket 也被删，007 的原 mask 主要覆盖一架机尾却要求两架，009 的包在 mask 外导致删人后悬空，019 的亮黄色草地仍像荧光平涂。前三项本质是上游目标范围/依附物合同问题，不能仅换编辑器解决；019 仍需属性改色纹理约束。

运行入口：

```bash
cd /opt/tiger/tanyue/samtok-derived-edit-labeling
/usr/bin/python -m synthesis_pipeline.run_fresh_cohort \
  --root <new_frozen_cohort_root> \
  --dataset-mask-plan --editor-backend qwen21 \
  --planner-gpu 0 --sam-gpu 0 --editor-gpus 0,1,2,3,4,5,6,7
```

完整对照报告位于 `qwen21_ablation_20260922_final/report_side_by_side/index.html`，37 个版本卡片均内嵌图片，并写入 `docs/data/QWEN21_HOLDOUT20_COMPARISON_REVIEWS.jsonl` 的逐图判断和理由。适配测试与 composition 回归共 18 passed，`py_compile`、`git diff --check` 通过。本轮未启动 100k，也未提交推送。

## 2026-09-22：Qwen-Image-2.1 扩展到 78 个新 case

为验证 19 条 holdout 的结果是否偶然，继续复用冻结的 dev40 与 confirm40 可执行 region；这些 source 不与上一轮 19 条 holdout 重合。本次不改 source、dataset mask、instruction、crop、seed=0、40 steps、区域 denoising 与 grounded composition，仅将编辑器固定为 Qwen-Image-2.1。本轮实际输入 78 条，add/remove 各19条、replace/attribute 各20条；8 卡全部生成成功，无 diffusion 报错。

单例扩散均值 17.017 秒，范围 16.092–18.259 秒。每批39张在8卡上，按最慢 worker 的加载加串行样本时间约104.4–104.6秒，包含约17.4–17.9秒模型加载后的实际吞吐约1343–1345图/小时；模型常驻稳态理论约1692图/小时。该数字继续支持上一轮相对2511约4.5倍的编辑阶段加速，不包含planning、composition、audit和共享存储首次拷贝。

后合成得到75/78张final。confirm40的010人物替换、022花盆换桶、024小手表新增均在raw中画质和原指令通过，却被`new silhouette unresolved`拒绝；其中022把新目标错误解析成`orange hose by a blue plastic bucket`，混入了应保留的hose。dev40/004的raw存在小logo，但composition返回成功后final近似原图，是更危险的静默no-op。故本轮3个硬拒绝均为后处理误拒，另有1个未被现有门控发现的后处理退化。

assistant逐张检查75张final与3张raw-only，不用模型pass代替人工结论。按“视觉质量+原始指令同时通过”计：final为42/75；把3个合格但误拒raw纳入可恢复候选为45/78。单看画质为56/78。分类型如下：

| 类型 | 输入 | 视觉通过 | 画质+原指令通过 |
| --- | ---: | ---: | ---: |
| add | 19 | 17 | 14 |
| remove | 19 | 10 | 9 |
| replace | 20 | 16 | 13 |
| attribute | 20 | 13 | 9 |
| 合计 | 78 | 56 | 45 |

这不是独立人工金标准，也不等于完整pipeline的最终通过率；原指令定位歧义、mask范围和依附物错误仍被计为失败。45条中有3条只有raw，因而可直接交付的端到端final仍为42条。

主要问题按责任阶段拆分如下：

1. **remove仍是最弱视觉类型。** dev/005删除击球手后保留球棒与残肢；confirm/001保留骑手腿部；009留下大块白色补丁和人物碎片；017留下手、杯子和身体块；025仍有手状残片。这里既有mask外依附物，也有mask内补全失败，不能只靠更强的通用prompt混为一个问题。
2. **attribute容易抹平纹理或只改一部分。** dev/003把纹身变成越界的平蓝色块，015机车红色覆盖过宽，019小象身体变tan但象鼻仍灰；confirm/014、015床单丢失褶皱，027屋顶丢失瓦片纹理，039仪表数字畸形且清晰度与整图不一致。需要按类型加强源纹理/亮度锚定，而不是增加采样步数的泛化尝试。
3. **replace仍有完整替换和边界问题。** dev/030玻璃楼底部像硬切片；confirm/038只在原转速表内嵌一个小油量表，旧仪表没有消失。dev/014与038则是指令语义范围大于mask，属于上游合同问题，不能要求编辑器越过固定mask补齐。
4. **小物体add多数成功，但出现贴图感和合成退化。** dev/000、016、036的星形/心形较像数字图标，但按当前“整体基本可用”口径仍保留；confirm/004帽子既有平面贴图感又有落点疑问。dev/004证明对极小新增物再次SAM分割可能比直接使用局部raw更差。
5. **前置指代仍影响最终可用率。** 7条主要失败来自同类实例定位不足；另有成年象/幼象、整列火车/部分车厢、整条长椅/局部mask等范围不一致，以及把棕榈状结构叫成fence post的类别错误。Qwen-Image-2.1不能修复错误的训练标签。

当前27B audit与assistant画质结论在75张final上61/75一致（81.3%）：45条共同pass、16条共同fail；误拒8条、漏放6条。误拒集中在真实照片上的小贴纸/胸针以及模型认为不够“摄影化”的add；漏放包括平涂机车/床单/屋顶、拥挤犀牛、硬切楼体和未完整替换的仪表。因此现有audit可做triage，不能无人值守放行；candidate-only改写也不能代替对原始instruction的独立一致性检查。

下一轮优化优先级：

1. composition先加入raw局部变化兜底：新轮廓unresolved但raw的mask外变化受控时保留raw；禁止final相对raw的mask内语义/像素变化塌缩。修正replace新目标短语只提取目标head noun，不把保留上下文混入查询。SAM phrase分数不能用单阈值直接拒绝。
2. 生成prompt按编辑类型与目标尺度分支，但保持一次diffusion：remove强调连续纹理重建；attribute强调保留亮度、高频纹理和结构，只改指定属性；small add强调真实材质、透视、焦点和接触关系，避免vector sticker观感；replace要求旧目标在整个允许区域内消失。
3. 在出图前继续处理依附物与mask/指令范围合同；这部分不可通过成图后改写把错误标签包装成成功。audit同时保留“原指令是否完成”和“是否可改写补救”两个独立结论。

统一入口为 `qwen21_expanded_20260922/index.html`，链接两份内嵌图片报告；每个case的raw/final相邻，final保留assistant逐图结论以及27B audit完整理由、prompt和回复。逐图证据写入 `docs/data/QWEN21_EXPANDED78_REVIEWS.jsonl`。本轮仅扩展测试、复核和分析，没有修改生成策略，也没有启动100k。

## 2026-09-22：固定40步的质量修复与新源配对验证

用户明确不降低扩散步数。本轮保持Qwen-Image-2.1、40 steps、seed=0、原dataset mask及区域denoising，修复生成提示与后合成；所有实验目录独立保存。

首先纠正上节两条assistant复核：放大实际raw/final后，dev/004的小圆logo在final中仍存在，之前“合成抹掉”诊断不成立；实际问题是落在上缘而非所要求的胸甲中心。dev/011的冻结指令已经明确要求领带变蓝，实际图符合；将它说成西装/领带不匹配也不成立。两条已在JSONL中保留带revision的纠正理由。相应旧批次视觉通过数由56改为57，画质+原标签通过由45改为46，存在final且两项通过由42改为43；这些仍是assistant判断，不是人工金标准。后续评估不得把自动audit最终admission的quality字段当成独立画质初审：前者也包含标签及scope失败。

实现内容：

- `typed-v1`为2.1增加四类简短自然语言约束，仍只给干净crop，没有坐标/矩形指令或新增示例。add检查接触、光照和清晰度；remove要求完整消除及连续背景；replace要求完整替代而非在旧对象内部加小图；attribute保留未请求改变的纹理、褶皱和细节。旧`legacy`提示可显式复现。
- `guarded-v1`合成先利用冻结source短语提取replacement，区分“pot with hose by bucket”和“man by door with woman”。SAM首次查询无候选时，可在同一图像编码上以精简对象短语重试一次；不增加MLLM或扩散调用。
- SAM无候选时检查raw变化局部性，只有原目标内有可检测变化、窄边缘外变化受控且受保护邻居未改变才允许raw fallback。记录所有测量值，质量状态仍为`requires_visual_audit`。差分支持区不伪称为新增物语义分割，不生成虚假的added_mask。
- 合成变化保留检查对add使用新物体轮廓，对replace使用旧目标mask。第一版按整个宿主比较，误将清除宿主附带再生成视为退化，已在回归中发现并修正；失败实验保留为`composed_guarded_v1`，修正回归为`composed_guarded_v2`。
- 源历史manifest由多次全树扫描改为一次扫描，跳过sources/edited/inputs/diagnostics等图像资产目录，保留cohort、规划、审核目录。检查仍覆盖源ID与RGB重复。

旧78条的修正合成回归：dev保持39/39，confirm从36/39恢复为38/39；花盆换桶由正确短语恢复SAM轮廓，手表通过局部raw回退恢复。红衣男子因窄边缘外仍有4433个显著变化像素而未通过raw回退；不能为了恢复数量直接放宽。此统计发生在添加精简对象查询重试之前，后续结果另记。

新源批次`qwen21_quality_20260922_fresh80`，seed20261042，40张新源/80region，排除1888条历史源，原始四类各20。当前未使用且具备双mask的VER池耗尽，因此按现有采样器规则回填GRES，本轮40张均来自GRES；不能据此宣称在VER上验证泛化。使用当前已有planner `dataset_mask_v6_28_semantic_surface_and_assembly_guard`，80→54可执行（add14/remove12/replace11/attribute17）。两组共享同一冻结计划，均40步、seed0，分别生成legacy与typed-v1；各自再交叉运行legacy与guarded-v1后合成，以拆分提示和合成的作用。

### 本轮完成结果及采用范围

54个冻结计划各生成两版，共108次40步扩散，全部成功。没有降低步数，没有按结果替换seed，也没有在统计时删除失败case。26个前置拒绝保留在80条原始cohort内；下表的54条是可执行计划分母，不是采样分母。

| 指标 | legacy提示 + legacy合成 | typed-v1提示 + guarded-v1合成 |
| --- | ---: | ---: |
| 成功生成 | 54/54 | 54/54 |
| 存在final（不代表质量合格） | 50/54 | 51/54 |
| Assistant：final画质及原指令均通过 | 39/54（72.2%） | 41/54（75.9%） |
| 同一结果以原始80条输入计 | 39/80 | 41/80 |
| 27B独立画质初审pass | 30/50 | 34/51 |
| 27B与Assistant纯画质判断一致 | 34/50（68.0%） | 35/51（68.6%） |
| 单次编辑平均耗时 | 17.309秒 | 17.397秒 |

Assistant的54条逐图理由保存在`docs/data/QWEN21_FRESH54_QUALITY_REVIEWS.jsonl`，是Codex看图判断，不是另一套自动审核，也不是独立人工金标准。结果没有运行指令改写或完整最终准入流程。模型画质初审不带原始指令，不能把它的pass数当成指令一致率。按类型的可用final为：add 11→11/14，remove 8→8/12，replace 8→9/11，attribute 12→13/17。

四格消融的存在final数分别为：原提示+原合成50、原提示+guarded合成51、typed提示+原合成51、typed提示+guarded合成51。新增可交付的`026_gres_r9127_m0_replace`原raw本就可用，主要收益是合成不再误拒；`015_gres_r8059_m1_attribute`才是明确的出图执行改善：原版未改背景人物黄衣，typed版正确改蓝并保留纹理。不能把两条都归因于更好的像素生成。

`077_gres_r8522_m1_remove`中typed版清掉了原版悬空的酒杯，但手腕附近仍有异常连接，因此两版都不计通过，不能为了提高收益把部分改善算成完全修复。新cohort的016/056/058仍被合成拒绝；其raw都有明显超出允许区域的变化，未放宽局部性阈值来救数量。

旧confirm40另做了精简query重试后的三条定向回归（`composed_guarded_v3_regression`）：010红衣男子在完整phrase失败后用`man`得到轮廓；022蓝桶由修正phrase恢复；024手表通过局部raw回退。三条均有输出，不再调用扩散，含模型加载总计13.126秒。对应图片已目视复核；手表仍很小，不应把语义分割恢复等同于显著视觉增强。未把定向三条回归冒充另一轮78条完整重生成。

本轮将Qwen2.1统一入口默认设置为`typed-v1 + guarded-v1`，作为当前质量候选；Qwen2511默认合成仍是legacy。旧版可通过显式两个legacy参数复现。`experiment_edit_quality`等底层实验工具仍保留legacy默认，避免悄悄改变历史实验。所有原图、raw、旧final、失败原因都保留。用户要求不降低步数，本轮和统一入口仍为40步。

```bash
cd /opt/tiger/tanyue/samtok-derived-edit-labeling
# 新的、已准备源数据的cohort；不要使用本轮已完成目录重跑覆盖。
python -m synthesis_pipeline.run_fresh_cohort \
  --root <new_frozen_cohort_root> --dataset-mask-plan --editor-backend qwen21 \
  --qwen21-prompt-policy typed-v1 --composition-policy guarded-v1 \
  --editor-model-id /tmp/tanyue_qwen_image_21 \
  --planner-gpu 0 --sam-gpu 0 --editor-gpus 0,1,2,3,4,5,6,7
```

本次编辑阶段没有获得速度提升：均值增加约0.5%，属于近似持平；不宣称显著加速。后合成的额外query复用同一SAM image embedding，没有额外MLLM或扩散调用。27B画质初审每组使用4GPU，baseline总墙钟444.4秒、typed368.2秒，包含冷启动且样本输出长度不同，不能由此推断新生成提示让审核模型加速。单图17.4秒也不能拿来直接推算完整100k工期。

### 当前仍需优先处理的失败

1. **plan范围错误会压倒生图改进。** 061源mask只圈到麋鹿服装躯干，却要求删整个人，成图留下头和手；067圈定人物不是穿条纹袜的那个人，导致编辑无效。现有planner已有完整性约束，但视觉判断仍漏过，不能再加一句通用“全部移除”就声称解决。本轮未恢复源SAM审核。
2. **反射依赖漏检。** 078把倒影当独立倒挂猫换成狗，079只改本体猫色而不改倒影。现有planner已经有reflection规则，但本批没有识别出来。需围绕视觉证据修前置grounding，而不是重复添加同样规则；在处理好前，这两类不能入库。
3. **生成本身的失败。** 017象群删除几乎无效；023暗场衣服仍平涂；029地面补全有旧阴影/色块。typed提示尚未解决，下一轮应先分别对比干净raw crop与final，区分模型残留和写回重新带回的源像素，再做边缘/阴影支持区实验；不宜统一扩大所有mask，避免伤到邻居。
4. **局部位置与类型不吻合。** 028红心落在座圈而非正面；056蓝莓替换原果饰而非单纯新增。可进入后续可改写性判断，但本轮没有执行改写，仍按原标签不通过统计。
5. **审核未达到无人值守标准。** 模型对小贴纸、标牌、手镯的误拒仍多，并漏掉部分无效编辑；本轮完整保留模型理由和Assistant不同意的依据。不能因为模型pass提高就停止人工复核。

下一步重点应从这批新源的失败机制再抽取新的源数据做确认，尤其是密集实例remove、暗场属性以及本体/倒影或衣物/依附物跨mask问题；本轮同批A/B是开发证据，还不是独立确认集结论。

### 可视化与检查

完整报告：`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_quality_20260922_fresh80/paired_report/index.html`。图片全部内嵌，同时保存独立JPG。每条左为source、中为原版、右为修复版；上排干净全图，下排同坐标局部，只有source局部带mask轮廓。这样不会让两张source视图都被轮廓遮住小脸或细节。没有final的条目标明`RAW ONLY / composition rejected`。每条包含Assistant完整理由、实际生成prompt、合成证据以及27B完整回复，可展开查看。

相关测试44 passed（edit quality guard、Qwen2.1 adapter、context composition、color texture repair、audit v4），`git diff --check`通过。本轮未启动100k，未提交或推送；没有修改原始dataset mask，也没有把实验raw fallback自动认定为审核通过。

## 2026-09-22：删除残留定位、受限写回修复及26条新源确认

本轮继续保持Qwen-Image-2.1、40步、seed=0、一次扩散、原有区域denoising和原始dataset mask，不降低步数，也不恢复源SAM审核。最终采用版本是 **typed-v1 + guarded-v1（add/replace）+ adaptive-remove-v1（remove写回）**。新增的阴影提示被实验否决，没有设为默认。

### 先追踪raw和final，避免把两种失败混为一谈

逐图读取上一轮12条remove的实际`source_crop.png`、`raw_edited_crop.png`、`composition_alpha.png`、受保护邻居及final：

- `077_gres_r8522_m1_remove`：raw已清掉被删除人物的酒杯，最终却在另一个人的腕部出现玻璃/衣物残片。源mask内的孔洞没有完全写入raw，合成把旧像素贴回，是确定的后处理问题。
- `033_gres_r1866_m1_remove`：raw的幼象已删除，最终成年象脚边的小三角源像素被贴回；属于同类小残边。
- `029_gres_r10441_m1_remove`：raw自身仍有坐便器旧投影，final又增加了硬边。不能只修合成就宣称阴影问题全部解决。
- `017_gres_r6513_m1_remove`：raw里的前景象就没删除，合成无法凭空补救。
- `061_gres_r2325_m1_remove`：raw已留着麋鹿头套，源mask也不含完整人物；扩展一点边缘不能修好错误的编辑范围。

本次更大局部、带raw对照的复核还发现，旧`005_gres_r7449_m1_remove`中土豆沙拉右侧的棕色/阴影边块比上一轮小图判断更明显。因此本轮回归将原宽松pass修正为fail，且同一标准同时用于修复前后；旧轮39/41等数字是当时历史复核记录，不应直接拿来拼接本轮统计。修正理由保留在`docs/data/QWEN21_REMOVE_REG12_REVIEWS.jsonl`。

### 实现：自适应但受限的remove写回

新增`utils/remove_support.py`，只改变后合成的可写支持区：

1. 先计算干净source与raw的轻微平滑RGB差分，平均差分至少12才算明确变化证据，抑制重采样噪声。
2. 原目标之外的修补候选必须在窄带内：半径为目标短边的12%，限制在8–40像素。不是整张crop可写。
3. 只保留与原目标连通的差分支持，允许跨越少量差分小孔隙；远处新变化不纳入。原mask内的小孔洞只有存在变化、且距目标足够近才可能恢复，不做任意大范围填洞。
4. 排除已知受保护邻居，再沿支持区使用原有窄羽化过渡。受保护像素逐字节保持source；源target mask完全不变。羽化带在上述核心支持外另有原有窄过渡，不把40像素误写成最终alpha的绝对最大半径。
5. 输出仍需后续视觉审核，差分支持区不是语义分割。统一snapshot导出单独的`edit_support_mask`与`remove_composition`，保留原`mask`用于目标标注。

`edit_context_crop`、实验入口及`run_fresh_cohort`增加`--remove-composition-policy`。统一Qwen2.1入口默认`adaptive-remove-v1`；Qwen2511保持legacy，底层独立实验工具仍默认legacy，以便准确复现旧对照。add、replace和attribute行为不受本修复影响。可用`--remove-composition-policy legacy`显式回退。

### 新源冻结验证及否决实验

新批次：`qwen21_remove_repair_fresh40_20260922`，seed20261103。新取20张源图、每图2region，共40条remove，排除1928个历史source，没有源ID/RGB重叠。此次为专项删除诊断，因此不要求四类均衡。VER可用未用双mask池仍无候选，采样器回填GRES，不能据此宣称已在新VER验证。使用已有planner，40条中26条进入生成，14条前置拒绝保留。

同一冻结指令/源图/mask/seed，8卡生成两组各26条，每条40步，共52次扩散；另在旧005/017/029上生成3条阴影提示定向回归，总新增55次扩散。没有从不同版本中按case挑最优输出。

| 本轮复核口径 | 原提示+旧写回 | 原提示+新写回 | 阴影提示+新写回 |
| --- | ---: | ---: | ---: |
| 新26条：Assistant画质与原指令同时通过 | 18/26（69.2%） | **20/26（76.9%）** | 19/26（73.1%） |
| 同结果以原始40个region计 | 18/40 | 20/40 | 19/40 |
| 新26条：27B独立画质初审 | 19/26 | 21/26 | 未运行 |
| 旧12条：本轮同标准Assistant复核 | 7/12 | **8/12** | 未对全部12条运行 |

Assistant不是另一个流程模型，也不是独立人工金标准；逐条理由位于`docs/data/QWEN21_REMOVE_FRESH26_REVIEWS.jsonl`。27B仅做不带原指令的画质初审，没有在本实验里重写指令或最终准入；它的21条不等于人工口径的20条，不计算混合口径的一致率。

明确修复的新源例子：

- **022（C形甜甜圈）**：旧写回重新带回盘面的弧形残边，新写回清掉残边，盘面连续；fail→pass。
- **030（红脉叶片）**：旧写回残留明显叶形尖角，新写回清掉源像素，墙地与凳腿衔接自然；fail→pass。
- **旧077（持酒杯人物）**：清掉mask孔洞贴回的腕部碎片及下方小残点；fail→pass。
- 旧033的象脚小尖角、新009等边缘也更干净，但原来大体可用的样本不重复计为新增通过。

三条明确fail→pass均来自同一raw的重新合成，没有增加扩散。新26条中没有观察到新写回让原pass变fail；样本量仍小，只能作为本轮采用证据，不是全量无风险保证。

阴影提示`remove-shadow-v2`额外要求删除目标自身投影、保留其他对象阴影。本批038原本能删除上方巧克力甜甜圈，增加该提示后却变成未执行；018船体也出现更明显残留。旧005/017/029三条定向回归未因此解决主要失败。故**不采用该提示**，保留在实验选项中并展示失败证据，不能按case选回原提示后再声称新提示整体变好。

### 速度与运行方式

原提示新26条平均编辑耗时17.3655秒；阴影实验17.3327秒，差异很小且后者已否决。采用的写回修复复用相同raw，扩散耗时不变。对26张图各重复3次、交替顺序的纯合成计时（不含读写盘）：legacy均值12.55ms，新版43.58ms，增量约31ms/条，约为17.37秒编辑时间的0.18%。这是小幅额外CPU成本，不宣称生图加速；没有新增MLLM、SAM或扩散调用。

27B审核两组分别用了3卡和4卡，墙钟305.9秒和309.2秒（含冷启动），并行度和输出长度不同，不据此计算审核加速比。现有审核仍会把轻微残边判断为pass，例如022旧版，所以仍保留Assistant不同意见。

```bash
cd /opt/tiger/tanyue/samtok-derived-edit-labeling
# 新的已准备cohort；不可用已完成的实验目录做覆盖性重跑。
python -m synthesis_pipeline.run_fresh_cohort \
  --root <new_frozen_cohort_root> --dataset-mask-plan --editor-backend qwen21 \
  --qwen21-prompt-policy typed-v1 --composition-policy guarded-v1 \
  --remove-composition-policy adaptive-remove-v1 \
  --editor-model-id /tmp/tanyue_qwen_image_21 \
  --planner-gpu 0 --sam-gpu 0 --editor-gpus 0,1,2,3,4,5,6,7
```

### 结果入口、仍未解决的问题与下一步

主入口只展示本轮采用版本，不把否决提示混进推荐结果：

`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_remove_repair_fresh40_20260922/adopted_report/index.html`

左为干净原图、中为旧写回、右为新写回，上排全图、下排同坐标局部；仅原图局部标mask轮廓。26条均展示，包含Assistant完整理由、实际生成prompt、两组27B完整回复，图片内嵌并另存JPG。完整三方案消融位于同级`comparison_report/index.html`。旧12条raw定位与修复对比位于`qwen21_remove_repair_reg12_20260922/report/index.html`。

新批仍失败的005手部悬空、016长椅部分残留、025披萨范围不符、024盘面补错、029块状残留和018船体/反射问题，需要分阶段处理。**后续逐像素复查纠正本节原归因**：005的手和016的背板已在mask中，raw未完整删除，不能说成源mask没有覆盖；029在raw中已基本删除，棕色块主要由最终写回重新带回，不是raw本身补全失败。025确有指令数量大于mask范围，024则是raw把白盘补成木纹。018新船样结构可能是背景幻觉或替换，现有源图不足以确认其合理性，同时最终保留了旧水面色影。这些不能统一扩大mask，也不能靠事后改写挽救坏像素。

下一步优先把“目标完整可见部分/依附物/承载面”与实际源mask再次对齐，重点避免删除人物却保留其手、删除食物却不保留其盘面、把长椅局部写成整张长椅。对范围正确但raw失败的样本，再单独验证背景承载面约束；不要把它与本轮已经验证的写回修复混在一起。

本轮相关测试 **50 passed**，覆盖小孔洞恢复、远处像素不变、受保护邻居逐字节一致、大孔洞不被整体填满、原mask不变、支持区独立导出、Qwen2.1适配及既有审核/合成。另有pycocotools的NumPy接口弃用warning，不影响结果。`py_compile`和`git diff --check`通过。没有启动100k，没有提交或推送，没有删除历史实验。

## 2026-09-22：优先级一——类别无关的内部残块写回修复

用户担心按特定词汇/物体打补丁。本轮只完成第一项：**raw已经删除、final又贴回内容**的合成修复。没有同时修改规划prompt、生成prompt或审核标准，不把其它阶段的失败归为已解决。采用版本为 `typed-v1 + guarded-v1 + adaptive-remove-v2`，Qwen-Image-2.1仍为40步、seed=0；Qwen2511保持原默认。

### 根因及非词汇化实现

旧029幼象的残块位于目标可见部件之间，既非受保护邻居，也不是raw中未删除的内容。v1的最多40像素距离带截断了一块连续差分区域，导致外侧旧像素被恢复。修复函数只接收source、raw、二值target、protected及policy，**不接收instruction、物体类别或case ID**。没有物体关键词分支、特定坐标或按case挑选版本。

v2保留v1的外侧窄带，额外处理目标凸包内部的小块：

1. 候选必须有source/raw差分证据，与已有支持区相邻，不跨已知protected邻居。凸包仅约束候选位置，不直接填满凸包。
2. 完整接纳或拒绝连通块，不再次在其内部按距离硬切。单块新增面积不超过target面积的10%，总新增不超过15%；超预算保持v1。
3. 大的闭合排除孔洞（超过target面积10%）不进入新增内部修补；不能把明确排除的中心对象整体覆盖。
4. 原始dataset mask保持不变，实际写回支持单独记录。差分并不等于语义分割，仍需成图审核。不存在“所有未标注遮挡物均绝对安全”的保证。

开发记录保留：`reg26`是初版（单块5%、最大距离2倍旧半径），没有解决029；`reg26_v2b`取消重复距离硬切并调整面积限制后清掉残块，但合成测试发现可能填满较大闭合孔洞，遂增加大孔洞保护。最终 `reg26_final`、`reg12_final` 和 `fresh28_final` 使用冻结的同一决策规则。新数据出图前规则已冻结；没有按新数据结果调阈值。

随后将逐块整幅布尔数组循环改为标签向量化，减少开销及碎片多时的内存占用。优化后对全部66张重新计算，final RGB和alpha均与优化前保存结果逐像素一致；这一步没有改变选择规则。

### 独立新源确认及逐图结果

实验根目录：`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_remove_interior_20260922`。

新源 `fresh40` 使用seed20261119，排除1948条历史源记录及对应RGB像素hash，抽取20张新图、每图2mask，40条remove。未用双mask的VER池仍耗尽，本批回填为GRES；不能据此声称已覆盖新的VER或所有物体类别。前置规划保持既有版本，40→28可执行、12条拒绝保留。8卡并行生成28次，全部成功，均40步、seed0、typed-v1；v1/v2共享同一raw，没有额外扩散或MLLM调用来做候选比较。

| Assistant逐图复核：画质及冻结原指令同时通过 | v1 | v2 |
| --- | ---: | ---: |
| 旧26回归 | 20/26 | 21/26 |
| 更早12回归 | 8/12 | 8/12 |
| 新源28可执行计划 | 22/28 | 23/28 |
| 新源按原始40条候选计 | 22/40 | 23/40 |

66张中观察到2条fail→pass，未观察到pass→fail；这是小样本的助手复核，不是独立人工金标准或全量质量保证。本轮未运行额外27B成图审核、指令改写或最终准入，不能把这些数字叫做自动audit通过率。

- **旧029 / gres_r5546_m1**：清除幼象脚边被贴回的棕色块，成年象、栏杆和链条保持，fail→pass。
- **新031 / gres_r11699_m1**：raw已清掉中间斑马，v1贴回地面中央的一截孤立条纹腿；冻结v2清除残腿，周围斑马和犀牛保持，fail→pass。这是新源上的同机制收益，不是加入“斑马”规则。
- **新003 / gres_r9501_m1**：沙发边缘残布缩小，但raw和final仍不干净，仍fail，没有把部分改善算通过。
- 新008掌纹区域仍有明显斜向拼接色阶；016前景头颈删除后镜内仍有悬空头发形状；019毛皮补全呈均匀颗粒块并残留耳状结构；028客车旧影及硬边仍在。两版均fail。
- 旧005手、016椅背的raw执行不完整，以及024白盘补成木纹，仍然fail；本轮没有通过重写标签挽救坏像素。

### 速度、保护及运行入口

本轮28次编辑平均17.389秒；8卡编辑阶段含加载约91.3秒，规划/合并/编辑/旧新对照中v1重合成与报告阶段合计296.9秒，不含采样准备、v2报告和Assistant看图，不等于完整生产pipeline吞吐。

纯合成计时对66张各重复3次、交替顺序，不含图片读写：v1均值45.35ms，向量化v2均值63.78ms，额外18.43ms/条（约为单条17.39秒编辑时间的0.11%）。v2向量化前约101ms，故此前过程更新中的101ms不是最终实现开销。没有宣称扩散生图加速，没有降低步数，没有新模型调用。

全部66张逐像素验证：向量化前后final及alpha一致，已知protected且不属于target的像素与source完全一致。相关测试 **62 passed**，覆盖旋转不变的内部小块、未变化区域、远处变化、大闭合孔洞、被raw误删的受保护细遮挡物、独立支持区导出以及add/replace/attribute不受影响。`py_compile`和`git diff --check`通过；两条pycocotools NumPy接口弃用warning不影响结果。报告校验包含66张内嵌图，不依赖外部图片URL。

统一Qwen2.1入口默认已采用v2；底层实验工具仍保留legacy默认以免改变历史重放，v1也可显式回退：

```bash
python -m synthesis_pipeline.run_fresh_cohort \
  --root <new_frozen_cohort_root> --dataset-mask-plan --editor-backend qwen21 \
  --qwen21-prompt-policy typed-v1 --composition-policy guarded-v1 \
  --remove-composition-policy adaptive-remove-v2 \
  --editor-model-id /tmp/tanyue_qwen_image_21 \
  --planner-gpu 0 --sam-gpu 0 --editor-gpus 0,1,2,3,4,5,6,7
```

### 如何看可视化、泛化边界及下一步

主报告：`qwen21_remove_interior_20260922/reviewed_report/index.html`。66条均展示，左到右为原图、raw、v1、v2，上排全图、下排同坐标局部。图片全部内嵌，也保留独立JPG。每条包含完整Assistant理由、原始instruction和支持区证据。Assistant review明确是Codex看图，不是pipeline调用的另一个模型。逐条记录为 `docs/data/QWEN21_REMOVE_INTERIOR_REVIEWS.jsonl`，报告可由 `synthesis_pipeline.build_remove_interior_report` 重建。

没有关键词分支只避免了一种过拟合；几何规则和阈值仍可能对未见布局失效。本轮新数据覆盖人物、家具、交通、食物、动物、行李及遮挡场景，但规模小且只有GRES。下一项应转向**raw的部件完整性、背景层次与承载面**，使用场景中实际可见的关系生成内部执行提示，不枚举固定物体模板、不增加训练instruction冗余；同样要冻结候选后在新源验证。更远的阴影/反射不能靠无限扩大写回范围解决，应分开做范围与生成实验。

没有启动100k，没有提交或推送，没有删除历史实验或改动源dataset mask。

## 2026-09-22：优先级二——删除出图的执行信息传递与消融

本轮保持40步、seed0、同一干净crop、原mask、区域denoising和adaptive-remove-v2写回。只调整Qwen2.1编辑prompt；训练instruction不变，不新增MLLM调用、不恢复源SAM审核。

检查发现，已有两轮规划中的`visual_grounding.selected_surfaces`和`outside_description`在Qwen2.1适配时没有进入最终prompt：适配器用简短action重建自然语言prompt，丢弃了更早构造的上下文。不能说是API接错图片；图片依旧是干净crop，问题在规划证据与编辑执行之间的信息传递。

增加`removal_execution_evidence`，仅接纳ground和scope都accept的冻结视觉证据，校验字段类型、条目数及长度。没有证据或字段不合法时，候选严格回退到原typed-v1。函数不判断物体类别，不使用case ID或固定对象模板。实际调用和`generation_request.json`均记录所用证据，测试覆盖信息能穿过Qwen2.1的prompt重建路径且仍只用一张干净图。

先在旧26条上测试`remove-evidence-v1`：同时加入实际选中部件、实际排除周边内容，并要求延续直接露出的表面，而不是用更远背景覆盖较近承载面。相同冻结计划的结果是21/26→21/26：024白盘从木纹错误恢复为盘面（fail→pass），026却出现长颈鹿残腿（pass→fail）；018从错误船样补全进一步退化为未执行删除，仍fail；005悬空手、016椅背残留和025范围错误未解决。因此联合提示不能直接作为默认。记录位于`docs/data/QWEN21_REMOVE_EXECUTION_DEV26_REVIEWS.jsonl`。

为避免根据对象词汇修补，拆成`remove-parts-v1`（只传入已选部件）与`remove-context-v1`（只传入已排除周边及直接背景层关系），同一旧26条各生成一次，分别使用4GPU并行。这是机制消融，而不是对每个case选择最优提示。三组候选各26次，均40步。

独立新源在`qwen21_remove_execution_20260922/confirm40`预先冻结：seed20261203，排除1968条历史源及RGB重复，20张新图/40个mask。VER未用双mask池耗尽，本轮仍为GRES；不得宣称跨VER泛化。既有planner保留25条、拒绝15条。先完成typed-v1基线出图，候选按开发消融结果选定后，再对相同25条做配对确认，不依据新图调候选参数。

### 消融及新源确认的最终结果

以下是Codex逐条查看全图和局部后的判断，同时要求画质可用及满足冻结原指令；不是自动audit通过率，不是独立人工金标准。本轮不调用成图审核或指令改写，不通过修改标签挽救坏像素。

| 提示方案 | 旧26条通过 | 相对旧版改善/退化 | 新25条通过 |
| --- | ---: | --- | ---: |
| 原typed-v1 | 21/26 | 对照 | 16/25 |
| 部件+周边remove-evidence-v1 | 21/26 | 024改善、026退化 | 17/25 |
| 仅部件remove-parts-v1 | 19/26 | 无改善，006和010退化 | 未运行 |
| 仅周边remove-context-v1 | 20/26 | 024改善，020和026退化 | 未运行 |

三个候选的旧26例均已逐条看图，记录分别为`QWEN21_REMOVE_EXECUTION_DEV26_REVIEWS.jsonl`、`QWEN21_REMOVE_EXECUTION_PARTS26_REVIEWS.jsonl`、`QWEN21_REMOVE_EXECUTION_CONTEXT26_REVIEWS.jsonl`。部件版006在原白衣人处生成另一名蓝灰衣人，010基本保留原象；周边版020残留悬空象头/象鼻，026残留长颈鹿躯干/腿。抽查这些退化的raw，问题已经存在，不能归罪于最后的合成。

开发结果中组合版最接近基线，因此冻结该版，在未查看的新25例上做确认；没有把另外两个候选也跑新源后择优。新源逐例记录在`docs/data/QWEN21_REMOVE_EXECUTION_CONFIRM25_REVIEWS.jsonl`。观察到1例fail→pass，没有观察到pass→fail；按全部40条原始候选计是16/40→17/40，不能忽略15条前置拒绝。

- **新032 / gres_r10818_m0**：旧版几乎未删目标玩具熊，组合版删除指定熊并保留周边玩具，背景补全基本可用；fail→pass。
- **新012 / gres_r11153_m0**：新提示恢复更多车厢地板，空间感比旧版大块蓝色延伸更好。但原图上方本来有蓝色物体，旧版也没有明确目标残留，不能凭想象把旧版判fail；两版pass。
- **新003 / gres_r3779_m1**：人物删除后留下孤立投影；004滑雪者消失但雪杖悬空；007裁判删除后手持装备残留；019机车消失但原烟囱上方的蒸汽仍在。都是目标与其视觉依附关系没有被完整执行，raw已有问题。
- **新015 / gres_r1103_m1**：墙板上仍有原颈部位置的斑纹状竖条；018列车删去但铁轨在空地前中断；033只改变了部分围巾和毛绒区域，未完整执行局部删除。两版fail。
- **新022 / gres_r5692_m0**：最值得优先修复的上游问题。instruction要删父女两人，但实际mask主要包围父亲。两轮规划都错误声称mask覆盖孩子，并将孩子衣服、裤子、鞋加入selected_surfaces；这些文字虽已accept，仍然是错的。两版raw就留下缺损且悬空的孩子。说明字段验证和accept状态不能代替像素范围核验，把错误视觉证据传下去也不可能自动修正区域约束。

### 本轮采用与不采用的内容

**不修改默认提示**：正式推荐仍为`typed-v1 + guarded-v1 + adaptive-remove-v2`。组合版在两个集合合计仅净增1例，且旧集出现明确退化，样本量不足以说明可靠泛化；另外两个版本更差。所有新提示保留为显式实验选项，不按类别、关键词或case ID做路由，也不把各版最优图片拼成虚假的单一版本收益。

已完成可复用的代码更新：经过类型/长度/双accept检查的证据传递接口、真实调用prompt和证据落盘、冻结计划的多GPU实验入口、同时对照raw/final且图片内嵌的报告工具。缺字段/非法字段回退原typed-v1，add/replace/attribute提示逐字保持原样。增加单测白名单，避免新测试被既有`test_*.py`规则忽略。

运行入口（仅实验，输出目录必须新建，禁止覆盖旧实验）：

```bash
python -m synthesis_pipeline.run_frozen_prompt_experiment \
  --data-root <frozen_regions> --out-root <new_experiment_directory> \
  --policy remove-evidence-v1 --gpus 0,1,2,3,4,5,6,7
```

固定40步、seed0、原数据mask不变，每条只有一次扩散；本轮实验共128次扩散（旧26×3候选，新25×2对照），全部worker正常退出。证据来自已完成规划，不新增MLLM调用。相关回归测试最终86 passed，另有2条已有pycocotools弃用warning；`py_compile`和`git diff --check`通过。

### 速度及可视化

新25例单条编辑耗时：基线均值17.444秒，组合版17.565秒（+0.121秒、约0.7%，单次小批测试不视为确定的性能变化）。8GPU编辑阶段含模型加载基线约91.4秒、候选约93.3秒；计时入口略有差异，不用于精确加速比。本轮没有生图加速收益，也没有降低步数。基线规划、合并、编辑和一次旧合成总计294.3秒，不含采样、额外v2重合成、候选实验、报告、看图或生产audit，不应外推为完整pipeline吞吐。

主展示为`qwen21_remove_execution_20260922/confirm25_report/index.html`，包含全部25条新源配对结果及完整Assistant理由；旧开发报告分别为`dev26_report`、`parts26_report`、`context26_report`下的`index.html`。左为原图、中为typed-v1、右为候选；上排全图、下排相同位置局部，白轮廓仅标记原图mask。展开每条可看raw及完整实际prompt。所有图内嵌，也保留同名独立JPG；Assistant review来自本助手看图，不是流程模型回复。

### 下一优先级

不继续堆叠部件名和周边物体名。优先在已有第二轮plan审核内独立核验“指令每个对象是否确实被mask覆盖、依附物是否在可编辑范围内”，避免仅附和第一轮grounding。对多对象不能仅凭三处正样本定位点推出整组覆盖；无法支持组删除时应在原范围内改成可执行任务，或明确拒绝该任务，不能扩大原标注mask。随后再针对raw内部残留与外部阴影/反射区分处理。下一轮需要新的冻结源，并同时保留此次022、004、007等回归，不使用对象关键词特判。

本轮没有启动100k，没有提交推送，没有删除历史实验。

## 2026-09-22：邻接对象保留与专属附件连带删除（实验，未切换生产默认）

用户补充的语义是本轮判定依据：父女例应删除父亲、保留女孩；目标专属雪橇/滑雪装备则应按场景连带处理。不能把“接触、抱着、遮挡、支持”统一当作删除或保留条件，也不能写对象词表或case ID分支。

### 原因定位与范围设计

1. **计划把几何接触误当成同一对象。** `022_gres_r5692_m0`的旧instruction要求删父女，两轮旧规划都错误认为女孩属于目标。白轮廓和蓝色TARGET点仍未阻止关系规划复现错误；在同一crop内，对另一个数据集mask的可见内部像素增加少量橙色OUTSIDE点后，27B才稳定改为删除父亲、保留女孩。仍只有干净全图和保留背景的轮廓crop两张输入，不涂红、不挖空。OUTSIDE表示原目标范围之外，不自动等同于语义上的KEEP；独立人物应KEEP，专属装备可另行声明REMOVE_TOGETHER。
2. **原mask与应执行的删除范围不是同一概念。** 裁判的手持装备、滑雪者的雪杖等可能不在原始人形mask内。只写“同时删掉”但不提供可编辑范围，会留下附件；无限扩大矩形又会牵连邻人。因此保留原始`mask`，另存`execution_region.mask = 原mask ∪ 已定位的连带附件`及`auxiliary_mask`、定位证据。新范围不得缩小原mask；已知保护区不能被辅助候选覆盖。辅助候选不确定就defer，不做矩形兜底。
3. **latent保护强度不是越大越好。** 旧版保护要求token内保护像素占比超过50%，且不含目标像素。细邻物和混合token可能漏保护；改成见到保护像素就锁，或按目标/保护占比分配混合token权重，可以减小邻物像素变化，但也会把父亲的手、被删孩子的衣物一起锁回去。
4. **遮挡补全本身会出错。** 父女计划纠正后，raw和final都出现女孩三条腿/鞋；不是最后paste才制造的问题。原女孩可见像素没变，并不保证新补出来的身体合理。新源中还出现悬浮支撑物、栏杆结构错接、只删手臂而保留人物等问题，不能靠改instruction掩盖坏画质。

关系规则由27B结合图像判断：独立人物/动物、邻居、共用装备和稳定场景表面应保留；目标专属且移除主体后会无主悬浮/残缺的非生命附件可连带删除；若所选目标本身是附件或局部部件，不能反向删除其所有者。移除承载面但保留其上的独立物体，可能需要改变该物体的支撑/姿态，不能机械承诺所有可见像素不变；本轮甜甜圈和香薰案例说明这一冲突仍未解决。

### 实现与调用边界

- `plan_removal_relations.py`：独立实验规划，每case一次Qwen3.8-27B/vLLM调用，输出decision、target、简短instruction、至多5项keep/remove_together关系及全图归一化定位点、reconstruction。不是生产planner已完成替换，也没有把这次实验说成原pipeline零额外调用。V3增加部件不能扩成整体、每件可见附件分别定位、不虚构不可见装备、训练指令不携带冗长保护条款等提示。实际输出仍有不遵循提示的例子，字段类型验证不等于语义验证。
- `resolve_removal_relations.py`：仅对REMOVE_TOGETHER执行辅助SAM定位；**没有恢复源mask的SAM审核，源mask审核调用为0**。以MLLM的实际场景描述作查询，结合点距、目标接触距离、面积、与已知保护mask重叠和候选歧义选择；不得只凭SAM置信度选另一个实例。保存候选NPZ与完整numeric checks。辅助SAM新增计算和模型加载开销必须计入，不能声称免费。
- 修复了“候选80%以上落在原目标内就拒绝”的错误：小附件已被原mask覆盖并不是坏事。现在区分`original_target_fraction`（候选有多少已在mask内）和`target_covered_fraction`（候选是否吞掉几乎整个目标）；后者防止把整个主体当成附件，含小目标回归测试。最后新增的防整主体门槛用本轮12个保存的实际查询重放，所有resolved/deferred决定不变，无需重跑扩散。
- `utils/context_edit.py`：仅显式提供resolved execution_region的remove实验走新范围；原始row及RLE不变。关系执行上下文传入Qwen2.1实际prompt，diagnostics记录范围与策略。`guard-any-v1`、`guard-fraction-v1`和`relation_geometry_policy=visible-v1`均为显式选项，默认legacy不变。visible-v1只增加“保留可见姿态/位置，仅补全新露出部分，不重复肢体”的内部提示，不改训练instruction。
- `run_relation_cohort.py`：规划分卡、辅助定位、同一Qwen2.1编辑器、各阶段日志和拒绝记录。`build_relation_report.py`保留全部输入，包括未生成项；HTML图片内嵌，另存JPG。Assistant review是Codex逐图判断，不是pipeline模型判定，也不是独立人工金标准。此实验未接生产成图audit/改写及训练准入，不能把resolved当作质量通过。

### 数据、实验与逐图结果

目录前缀：`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_neighbor_guard_20260922`。

固定Qwen-Image-2.1、40步、seed0、typed-v1和adaptive-remove-v2；不降低步数。两种latent方案先在上一轮25条相同source/mask/instruction上消融。关系方案保留6条困难回归；另选10张未见源图、每图两个mask，共20条remove。新源seed20261217，排除1988条历史source ID及RGB重复，全部来自GRES；VER未用双mask池已耗尽，不能宣称跨VER验证。正样本索引有7671条、排除4666条No target。

| 实验 | 输入/生成 | Codex通过 | 结论 |
| --- | ---: | ---: | --- |
| 旧25，原latent基线 | 25/25 | 16/25 | 同冻结计划对照 |
| 旧25，guard-any-v1 | 25/25 | 16/25 | 无新增质量通过，不切默认 |
| 旧25，guard-fraction-v1 | 25/25 | 15/25 | 023退化；混合token保留目标残片 |
| 关系V2，回归6 | 6/3 | 2/6输入，2/3出图 | 007、023可用；022仍重复腿；004/032/033定位defer |
| 关系V2，新源20 | 20/20 | 14/20 | 全部已看；不是旧生产pipeline的配对通过率 |
| V2冻结计划+visible-v1，回归3 | 3/3 | 2/3 | 父女没有解决 |
| V2冻结计划+visible-v1，新20 | 20/20 | 13/20 | 001退化为保留人物，只删板，不采用 |
| 关系V3，回归6 | 6/4 | 3/6输入，3/4出图 | 辅助覆盖率修复后032出图通过；004/033仍defer |
| 关系V3，相同新20 | 20/20 | 11/20 | 001、006、015退化，不采用新提示为默认 |

V2的新20在首轮生成前已冻结，V3看过这20例后迭代，因此V3这组是开发回归，不是新的独立验证集；没有把两版逐case择优拼成“最佳版本”。V3改变计划内容，表内质量比较不是只改扩散一项的严格消融。visible-v1是同冻结计划/范围的严格提示对照。

代表性结论：

- **回归022父女**：旧版女孩躯干大片缺损；关系V2/V3只删父亲、保留女孩，语义改善，但三条腿/鞋仍fail。不能汇报为修复完成。visible-v1也没有消除重复。
- **回归007裁判**：旧final有黑色装备碎片悬在草地上；关系范围包含实际手持装备后，人和装备均删除，球员保留，V2/V3通过。
- **回归004滑雪者**：应删除白衣目标及其专属装备，保留前方红衣人及其装备。模型仍用一个点描述复数雪杖，滑雪板定位失败；本轮保留defer，没有把“语义应连带”偷换成“装备已全定位/全删”。
- **回归032玩具**：原typed-v1几乎没删目标；附件候选覆盖率误拒修复后，目标玩具、帽子和围巾一起消失，邻近玩具保留，木背板补全基本可用，V3通过。
- **新004鹅**：原始reference是左右两只成年鹅，但mask实际覆盖前方幼鹅像素；在全图归一化点(480,880)、(430,880)、(500,900)、(550,900)检查，均落在target内，9×9邻域占比均1。模型据此把幼鹅也纳入主目标，V2/V3都删了幼鹅。不能简单归罪于看错轮廓；这是源语义和几何范围冲突。本轮不改源mask，不把它当通过。
- **新007水箱**：V2把水箱扩成整套马桶；V3主目标收窄为水箱，但仍保留其上香薰，结果悬浮。部件定位与物理支撑需要分别核验。
- **新012/013甜甜圈**：删除下层后上层悬浮；删除上层后有折片样结构与不自然切口。保留邻物像素不能解决失去支撑和被遮挡形状的补全。
- **新018人物**：V2/V3只删手臂和控制器，人体仍在；visible-v1改成悬空头部残留，仍fail。不是可通过改写指令挽救的合格remove。
- **新001/006/015，V3退化**：分别只删长板而保留人物、把座盖翻开而非删除、在原目标位置生成红背心人物而非删除。说明更详细的关系文字仍可能被生成器误执行。

逐条理由全部存入`docs/data/QWEN21_NEIGHBOR_{ANY,FRACTION}25_REVIEWS.jsonl`、`QWEN21_RELATION_V{2,3}_{DEV6,FRESH20}_REVIEWS.jsonl`、`QWEN21_RELATION_GEOMETRY_{DEV3,FRESH20}_REVIEWS.jsonl`。所有新20及V3全部成图已逐张查看全图和局部；latent两组中16条与基线final逐像素相同，其余9条逐张对照。与基线相同的图片保留前次视觉判断并纠正了不准确的旧理由。

### 保护指标、速度与验证

旧25 raw已知保护区，逐case平均RGB绝对误差均值：legacy 1.5407、guard-any 1.4704、guard-fraction 1.2833；按保护像素加权为1.1714、1.1207、0.9931。父女例9.6723→8.5536→4.9989，但仍fail，证明像素指标不能替代视觉结构质量。关系V2的23张及V3的24张final实测已知保护且不属于执行范围的像素与source完全相同；原始RLE逐项相同。

本轮总扩散120次：latent两组50，关系V2为23，visible-v1为23，关系V3为24。关系规划调用77次：首次无OUTSIDE的25，V2共26，V3共26。每次编辑40步，不做多seed择优；全部worker正常退出。

关系V3新20用5卡，单条编辑均值17.5875秒、中位17.5605秒，编辑阶段含加载92.99秒；规划+辅助定位+编辑合计256.40秒，不含采样、报告、Codex看图或生产audit。V2新20用6卡，单条编辑均值17.6241秒、阶段95.39秒、合计394.18秒。GPU数量和计划不同，不从这些小批数据声称扩散提速。

辅助定位发现CPU线程过多；设置`torch.set_num_threads(8)`和OMP_NUM_THREADS=8后，回归集从156.22秒降到16.15秒，新20从146.53秒降到19.17秒。两轮prompt/候选有所变化，这不是严格同输入性能基准，但日志确认主要瓶颈显著减轻。V3两个集合各6个辅助SAM文本查询，分别编码4/5张source；不是源mask审核。后续应合并生产规划/审核已有调用并复用常驻模型，不能把本实验256秒直接外推100k全流程。

相关回归测试119 passed，3条pycocotools NumPy弃用warning。覆盖原mask不可变、执行范围不可缩小、薄邻物/混合token、非remove无变化、已覆盖小附件可用而整个主体不可当附件、OUTSIDE图不修改原数组、实际Qwen2.1 prompt传递、已保存辅助查询重放。

### 展示、复现和下一步

主实验展示：`relations_v3_dev6/report/index.html`（原图/旧生产final/本实验final）及`relations_v3_fresh20/report/index.html`（原图/关系V2 final/关系V3 final）。上排全图，下排同坐标局部；白线仅为原始mask。新20报告同时列两版instruction，便于辨别任务变化；每例展示完整Codex理由，展开有实际双图输入、prompt、关系计划、候选定位及generation_request。NOT GENERATED明确只有原图占位，不是假输出。图片全内嵌，旁边同名JPG可直接看。

```bash
python -m synthesis_pipeline.run_relation_cohort \
  --data-root <frozen_remove_source_regions> --out-root <new_experiment_directory> \
  --gpus 0,1,2,3,4,5,6,7
python -m synthesis_pipeline.build_relation_report \
  --root <experiment_directory> --reviews <complete_codex_reviews.jsonl>
```

**生产推荐仍为typed-v1 + guarded-v1 + adaptive-remove-v2，legacy latent。没有启用本轮关系实验为默认，也没有采用退化的visible-v1/V3提示。** 当前可保留的是关系/原mask/辅助执行区分离、输入排除点、候选记录、辅助门槛修复和CPU线程优化这些实验基础设施；不是宣称整体质量过关。

下一步应先把原始reference绑定与独立实体范围一致性接回关系核验；当前独立实验只凭双图规划，未使用原始reference文本，鹅和水箱案例说明语义证据不能丢。其次对每个连带实体建立实际逐实例定位，复数装备一个点不应放行。最后针对独立保留对象的可见部分、遮挡部分和支撑改变分开设计执行/选择性重试，不能继续一味增强保护或堆长提示。正式集成前，还必须让后续audit和训练导出同时理解原mask与execution mask，验证额外范围并控制新增调用成本。本轮没有启动100k、提交推送、删除历史实验或修改源数据集标注。

## 2026-09-22：简洁训练指令、两例defer及全部失败归因

### 指令与内部执行应分离（设计要求，尚未切换默认）

用户明确要求训练instruction不枚举常识性专属附件及保护对象。例如滑雪者可写`Remove the background skier in the white suit with bib number 5.`，内部计划再记录需要一起删除的可见专属装备，不能把保留女孩与删除雪橇混成同一规则。

查到两个实现问题：`plan_removal_relations.py`明确要求instruction包含必要连带附件，冗余不是偶然输出；`resolve_removal_relations.py`的relation_execution_context只拼KEEP与reconstruction，没有单独传递REMOVE_TOGETHER的语义，后者目前依赖训练instruction和扩大的mask。因此不能只把最终句子截短：应由同次规划分别产生简短dataset instruction与内部执行关系，编辑器从后者独立构造提示，审核/改写/导出也应保留这种分离。该设计不需要在正常成功路径新增一轮MLLM。

### 004和033不是MLLM plan拒绝

两例MLLM都输出accept，后面的附件几何解析才置`defer_unresolved_auxiliary`并跳过编辑。当前defer并无自动重试队列，意味着记录原因、保留源数据、此次不出图、不做成图审核也不导出训练对。报告中的原图占位不能当成编辑结果。

- `004_gres_r2595_m0`：语义方向正确，白衣背景滑雪者及其专属装备应删除，前景滑雪者应保留。但雪板点和候选不匹配：最近候选点距约93.76px，门槛约37.50px；仅一支雪杖候选通过。复数雪杖仍用一个点，选择器每项只选一个mask，不能保证全部装备覆盖。不能直接取消全部门槛，否则会误选前景人的装备；需要逐个可见实例定位、用候选身份/归属校正点，遮挡下不可见的装备不能凭空要求分割。接触距离也不能作为所有专属装备的绝对条件，因为可见片段可能被他人隔开。
- `033_gres_r10818_m1`：源reference指向左侧熊和画面最右侧被截断的熊；m1的mask位于全图最右侧，bbox `[1090,117,1248,781]`（全图1248×832）。实际规划却写central teddy bear scarf，并把同一围巾再列为附件，点落在全图约(749,541)。正确右侧围巾候选存在，score约0.672、约97.9%在原mask内，但离错误点349px而被拒；与错误点重合的中间围巾不接触目标且过大，也被拒。问题主要在实例身份/坐标而非SAM完全找不到围巾。
- 更正此前对033的口头判断：原mask已含右侧部分毛绒肢体，不能声称毛绒胳膊必在mask外、只需再补分割。原始窄mask按横向宽度加padding，产生仅221×832的上下文crop（源坐标1027,0,1248,832），缺乏横向语义上下文。应改善极瘦mask的crop最小宽度/长宽比，并将原reference、全图位置及crop映射输入规划；需校验主目标而非把primary重复当附件。不能要求所有真正的外部附件点必须落在原mask内。

### 最近一轮全部10个已出图fail

统计范围固定为关系V3的dev6+fresh20：26个输入，24个出图，Codex逐图判断14通过、10失败，另外2个defer。均为remove，不能推广为其他类型的通过率。以下错误在raw已可见，不应全部归因于final paste。

| case（均为remove） | 失败表现 | 归因及优先方向 |
| --- | --- | --- |
| 001_gres_r369_m1 | 只删长板，前景男子仍在、脚悬空 | 主动作未执行；内部补全甚至要求恢复被删对象的投影，存在自相矛盾。先独立明确主对象消失，再处理装备与原投影。 |
| 003_gres_r8442_m1 | 人删去但栏杆错接、弯曲不自然 | 背景结构补全失败；需要保留可见栏杆端点/走向，不能只要求纹理自然。 |
| 004_gres_r10193_m0 | 成人鹅与幼鹅一起删除 | 原reference与mask像素范围冲突，模型把幼鹅纳入目标；独立幼鹅不是专属附件。需语义冲突修复或换任务，不能以原mask有效为由认定所有内部实体都该删。 |
| 006_gres_r3785_m0 | 座圈/盖翻开而不是移除 | 部件身份和删除状态未正确执行；不能将删除整套马桶当成同任务的成功。 |
| 007_gres_r3785_m1 | 水箱消失，香薰悬空 | KEEP身份不等于位置固定；缺少移除支撑后的可行性判断。是否连带删除应依场景关系决定，不能自动移动独立对象。 |
| 012_gres_r11067_m0 | 下层甜甜圈删去，上层悬空 | 保留独立对象与移除唯一支撑冲突；若不允许合理落位，应换编辑任务而不是强行通过。 |
| 013_gres_r11067_m1 | 下层甜甜圈出现薄片、异常切口 | 遮挡解除后的形状补全失败，疑似残留；仅锁原可见像素不能恢复自然整体。 |
| 015_gres_r201_m1 | 白衣男子变成红背心男子 | 编辑crop不含右侧红背心男子，却要求KEEP此人，产生上下文错位；下述严格提示对照得到改善。 |
| 018_gres_r1906_m0 | 只删手臂/手柄，头躯干仍在 | 删除被误执行成局部切除；去掉crop外KEEP后仍失败，不能把原因全部归给上下文。 |
| 022_gres_r5692_m0 | 父亲删去、女孩保留，但出现三条腿/鞋 | 关系判断已改对，遮挡补全产生多余肢体；现有visible-v1提示未解决，继续堆保留条款没有验证收益。 |

### 已完成的3例严格提示对照

新增诊断脚本`experiment_relation_context_visibility.py`，冻结V3的source、原mask、execution mask、instruction、crop、seed0、40步、typed-v1及合成策略，仅移除定位点落在实际编辑crop外的KEEP描述；reconstruction保持不变。该点规则仅用于消融，点在crop外不保证整个实体不可见，不能直接作为生产过滤器。

- 001负对照没有删除任何描述，raw与final实测逐像素等于基线；仍fail。
- 015实际crop为`[0,0,536,896]`，右侧红背心男子定位在约(784,161)，不在输入内。移除这条KEEP后，目标男子和木棍消失，墙架地面补全基本自然，其余人物保留；查看全图和raw后判pass。支持提示/输入错位会干扰出图，但不是单例即可证明普遍根因。
- 018去掉crop外拿笔记本者描述后，仍保留残缺目标人体，fail。

本对照为3个已知困难开发例，0/3→1/3，不是新数据通过率、不是完整pipeline质量验证。3卡并行含加载37.63秒，worker均正常退出，没有新增规划或audit调用。完整理由存`docs/data/QWEN21_RELATION_VISIBILITY3_REVIEWS.jsonl`，Assistant review为Codex逐图判断，不是流程模型回复。

报告根目录`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_relation_diagnosis_20260922/visibility_report/`。`index.html`图片内嵌；每例同名JPG左原图、中旧提示、右候选，展开为raw和实际完整prompt。此次未修改生产默认、源标注或重启全量。

后续实施顺序：先分离最终instruction和内部执行计划，并让编辑提示与实际crop中的实体一致；再修复reference绑定、瘦长crop及逐附件身份定位；最后分别解决支撑/遮挡可行性与结构补全。应采用关系和几何规则，而非滑雪、熊等词汇或case ID特判。涉及从execution mask剔除独立对象时，现有“不缩小原mask”实验契约须显式重新设计，不可偷偷改写源RLE；需要新数据和原困难回归共同验证。

## 2026-09-22：关系V4实施、身份纠错、附件归属修复及提示对照

### 结论与实验边界

已实现上一节前两项，并完成多轮实跑。004滑雪者从defer最终得到合格出图，033最右玩具也修正了实例身份；但新方案没有稳定超过旧关系方案，**不切换生产默认**。所有改动均为显式开启的实验选项，不按case ID或物体词表选策略，不按结果逐例拼接“最佳版本”。第三项支撑/遮挡可行性仅加强了规划提示，实验说明尚未解决。

根目录为`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_relation_v4_20260922`。复用了26条困难回归输入，另取10张新source、每图两region，共20条remove；新source排除1998个历史行及像素hash，历史重叠0。此次双mask候选池中VER无剩余符合采样条件的source，回填GRES，因此不能声称验证了VER或其他编辑类型的泛化。新20先冻结再运行旧/新方案；查看全部结果后又做compact和spatial迭代，所以尽管准备目录保留dev/holdout字段，**后续迭代的20条全部是开发数据，不能作为未见holdout**。

本轮不同实验分配到GPU 0–7并行，Qwen Image 2.1、40步、seed0、legacy latent、adaptive-remove-v2不变。未调用生产成图audit/改写/训练导出；以下通过率均为Codex逐图视觉判断，不是MLLM audit，也不是独立人工金标准。

### 实施内容

1. `plan_removal_relations.py --policy relations-v4`同次规划输出简洁`instruction`与独立`relations/reconstruction`。训练句只指代主对象，专属附件、KEEP和补全要求留在内部；原始mask RLE不变，辅助范围仍单独记录在execution_region。输入仍为干净全图和轮廓crop两张图片。
2. 使用原始reference绑定、原mask全图bbox、crop到全图的映射。极瘦crop的短边扩展至长边的至少0.8倍（受图像边界限制），避免画面最右玩具只剩极窄局部。主目标需给出原mask内的target_point；首次点错时仅追加一次纠错调用，提供实际mask内部点，要求重新核对身份，而非只把坐标吸附到mask。正常成功路径不增加调用。
3. 每个可见附件分别给定位点和bbox。编辑前按**实际编辑crop与预测bbox相交**过滤不可见KEEP文字，不再单凭代表点判断对象是否在crop内；部分可见实体保留。REMOVE_TOGETHER语义始终显式传入编辑器，不依赖冗长训练句。预测bbox仅用于提示可见性，不作为执行mask。
4. 新增`repair_removal_grounding.py`：仅对附件解析失败者调用一次27B，展示已有候选的原图轮廓crop，判断身份/归属、选择不同实例，或说明完全不可见。复用已有SAM候选，新增SAM调用0；不得因靠近错误点就选邻居装备，面积/邻物保护/接触等安全门槛未放宽。此脚本仍是独立实验入口，未接为生产自动重试队列。源mask的SAM质量审核始终0次。
5. 严格冻结计划/mask/seed比较两种编辑提示：`relation-compact-v1`将主对象和附件合为一个简短删除动作；`relation-spatial-v1`在typed-v1基础上补充**原目标在实际crop中的自然语言位置**，并明确不能只删附件。两者都只改内部提示，不修改训练instruction，不降步数。

几何与编译逻辑在`utils/removal_relations.py`；编辑接入在`utils/context_edit.py`。新增输入检查及回归测试覆盖窄crop、完整实体bbox可见性、主目标点纠错、原标注不变、内部/外部指令分离、crop坐标转换、候选展示不修改输入。附件复核提示在最后整理中将滑雪装备例句改为通用“不得用不同附件类型替代”；滑雪实跑使用的原始确切prompt仍保存在实验inputs中，未篡改历史证据。

### 配对结果：不只展示改善样本

| 对照数据 | V4规划 + typed-v1 | 同计划 compact | 同计划 spatial |
| --- | --- | --- | --- |
| 困难dev6中的5个已出图 | 4/5 | 3/5 | 3/5 |
| 旧dev20中的17个已出图 | 12/17 | 11/17 | 13/17 |
| 本次新增20条 | 13/20 | 15/20 | 16/20 |
| 滑雪者，附件归属修复后独立对照 | 0/1 | 未跑 | 1/1 |

dev6初始另1条滑雪者defer，后续单独修复；dev20初始3条主目标点错误未出图，增加条件纠错后3条均能出图，但仅1条通过，不能把“更少拒绝”当作“更高质量”。这些未计入上述固定5/17条提示对照的分母。

新20同时重跑旧关系方案得到**17/20**。最新V4+spatial为16/20，对旧方案修好004（画面左侧截断成人），但010显示器底座残留、018黑色低纹理补全为退化，净少1条通过。V4 typed-v1→spatial修好001白色衣服补全、017未删公交车、019误删托盘；compact虽修好001/010，却使旧032玩具及015白衣男子退化为只删附件。**更短提示或增加位置提示都不是普遍提升，不能按单例成功宣布最佳版本。**

### 两个用户重点case

- **004_gres_r2595_m0**：V4先正确分别定位两支雪杖，仍因要求分割雪板而defer。候选复核显示，可见板属于前景红衣人；背景白衣目标的板被遮挡，没有应当强行选取的可见像素。27B输出not_visible，删除该无依据附件义务，保留已定位的目标雪杖。单独用typed-v1生成仍只删杖、人没删；再加实际crop中的upper-center主目标提示后，白衣主体与可见雪杖均消失，前景人和装备保持，雪地补全可用。最终训练句为`Remove the skier in the white suit with the number 5 bib.`。没有用宽松几何门槛接受前景装备。
- **033_gres_r10818_m1**：加入reference、原mask全图位置、宽上下文crop后，识别为最右被截断的熊，而非中间熊的围巾。该对象及围巾/下方毛绒体删除，邻近前排熊和后排熊头保留；露出的后排熊手臂不能当作目标残留。本例V4 typed与spatial均通过。但同图左熊032在spatial中留下无头躯干，必须明确记为退化。

### 仍未解决的问题及下一步

- **主目标删除不完整**：032毛绒躯干、010显示器底座；不能靠把instruction改成“去掉头/显示屏”掩盖。下一步应区分主目标本体与附件的实际可见覆盖，评估有证据的失败重试，而不是普遍增加文字或扩大mask。
- **独立对象失去支撑**：新014删桌面后纸盒悬空、新015饮料杯/瓶悬空、旧012甜甜圈悬空。V4已要求检查支撑，但模型仍accept，说明提示要求并非可靠约束。需要显式记录失去支撑对象与可执行处置；无法合理局部完成时重拟任务或defer，而非自动删掉全部桌上物品。人物与独立动物不可因邻接就连带删除。
- **遮挡补全与边缘**：父女022仍有多余腿/鞋，003栏杆错接，006水箱下沿白色锯齿，018披萨区域补成黑色低纹理块，013甜甜圈形状异常。下一步需要针对结构连续性和邻物可见/新暴露区域区别处理；锁定原像素本身不能保证新露出肢体的结构正确。
- **源语义和几何冲突**：旧004鹅的mask已包含幼鹅像素。纠正主目标点后仍误删幼鹅；原mask不是可以无条件改动所有内部独立实体的语义许可。若要从执行范围剔除独立对象，必须显式修改当前“执行范围不得缩小原mask”的实验契约、保留原标注及变更证据，不能偷改RLE。

优先先解决支撑可行性及独立实体范围冲突，再做结构/残留修复。新候选进入默认前必须继续使用新source对照，并同步验证audit与训练导出的execution-region语义；本轮不改生产审核来配合放宽通过率。

### 速度、调用和工程检查

共151次扩散：初始旧回归22、点纠错3、新20旧/新两组40、compact42、滑雪修复typed1、spatial43。成功规划/纠错/候选复核共73次MLLM调用；辅助SAM文本查询19次，源mask审核0次。附件修复初次启动vLLM失败，保留失败目录；在使用审核虚拟环境PATH的第二次独立目录启动成功，首次未完成模型推理，不计入成功调用。未对失败目录覆盖写。

新20旧关系方案单图编辑均值17.576秒，V4 typed为17.458秒，spatial为17.489秒；差异很小，**无证据表明扩散提速**。最新spatial使用3卡，20条编辑阶段含模型加载144.03秒；同期其他卡执行回归/滑雪对照，因此是8卡分组并行，不是只使用3卡。V4新20规划+辅助定位+编辑315.09秒（4卡），旧方案331.99秒（3卡），卡数不同不能用于宣称端到端加速，也不能直接外推100k或包含audit的速度。条件纠错仅在失败路径增加调用，新20首轮均无需点纠错。

相关测试最终191 passed，5条pycocotools NumPy弃用warning；`git diff --check`通过。本轮模型worker均已结束，未启动全量任务、提交推送或删除既有实验。

### 可视化及复现

- `latest_fresh20_report/index.html`：完整新20，左原图/中旧关系方案/右V4+spatial，上排全图、下排同坐标局部。每条均列完整Codex理由，展开可见关系规划输入/实际prompt/模型输出与生成请求。包含全部失败，不是逐例择优合集。
- `spatial_ski_report/index.html`：滑雪附件修复后的同计划出图对照，左原图/中typed失败/右spatial成功；展开含raw和完整编辑prompt。
- `dev6/report/index.html`：重点033及其它5条，展示V4 typed版本，滑雪初始defer明确为未生成占位。
- `spatial_{dev6,dev20,fresh20}_report/`和`compact_{dev6,dev20,fresh20}_report/`保留严格同计划对照，`point_repair3/report/`保留点纠错3条结果。HTML图片内嵌；同目录同名JPG可直接看。Assistant review是Codex判断，不是流程模型输出。逐条理由在`docs/data/QWEN21_RELATION_*_REVIEWS.jsonl`。

```bash
# 新规划实验；输出目录必须为新目录，默认政策仍为legacy。
python -m synthesis_pipeline.run_relation_cohort \
  --data-root <frozen_remove_regions> --out-root <new_v4_run> \
  --policy relations-v4 --gpus 0,1,2,3,4,5,6,7
# 冻结计划的内部提示对照，保持40步。
python -m synthesis_pipeline.run_frozen_prompt_experiment \
  --data-root <new_v4_run>/regions --out-root <new_spatial_run> \
  --policy relation-spatial-v1 --gpus 0,1,2,3,4,5,6,7
# 可选：仅对已有候选的附件defer运行归属复核，不增加SAM查询。
CUDA_VISIBLE_DEVICES=0 /opt/tiger/tanyue/.venvs/qwen38_audit/bin/python \
  -m synthesis_pipeline.repair_removal_grounding \
  --root <new_v4_run> --out-root <new_repair_regions>
```

## 2026-09-22：显式建模“主目标 + 依附物/支撑物”

### 这次改动解决的具体问题

此前关系V4虽然已经让规划模型输出`remove_together`，但对“目标是支撑面”的情况仍常常只写`keep`：例如删除桌面时保留杯子、调味瓶和纸盒，导致编辑结果中物体悬空；删除显示器时有时也没有把专属底座放入执行区域。问题不在最终训练句是否冗长，而在于**执行区域没有覆盖规划阶段已经可以推断的附着/支撑像素**。

本轮新增`relations-v5`和`relations-v6`（均为opt-in，不改变legacy/V4默认）：

- V5要求模型先模拟“主目标像素消失之后”的状态，然后分别判断专属附件、独立邻物、被目标唯一支撑的物体，并把可见附件逐个写成带点和bbox的`remove_together`或`keep`。
- V6进一步增加支撑面规则：当主目标是桌面、平台、货架、托盘等水平支撑面时，位于待删除区域且没有其它可见支撑的非生命物体必须进入`remove_together`；如果场景包含多个独立物体，无法形成可信的选择性删除，则`defer`，不生成物理上错误的训练对。人物、动物和被保留人物手持的物品不会因接触自动删除。
- `resolve_removal_relations.py`继续只用已有的SAM候选生成独立`execution_region`，原始mask/RLE不改。候选仍需通过面积、点归属、保护区和接触门槛；V6不是通过放宽门槛来“凑”依附物。

因此数据层面仍保存简短instruction，例如`Remove the metal table surface in the foreground.`；内部执行区可以包含纸盒、杯子或瓶子，或者记录明确defer。训练句没有被改成长的“保留/删除清单”。

### 新20条验证

在同一批冻结的新source/20 regions上运行V6，Qwen3.8-27B规划4卡、Qwen Image 2.1出图，40步、seed0不变：

| 结果 | 数量 | 说明 |
| --- | ---: | --- |
| 规划并出图 | 18 | 其中Codex逐图视觉判断17 pass、1 fail |
| 安全 defer | 2 | 两个桌面支撑冲突案例（014、015），不输出错误编辑图 |
| 源mask审核 | 0 | 源标注没有被重新用SAM否定 |

014和015正是旧方案中会出现悬空杯子/瓶子或桌面关系不成立的案例；V6在规划阶段识别出“支撑面承载多个独立物体且无法安全选择性删除”，选择defer，而不是继续生成坏样本。显示器底座等简单专属支撑可进入执行区并正常处理。

但是V6并非整体质量已经解决：004案例仍生成了不相关的小孩幻觉，说明执行区域正确也不能消除Qwen Image的背景内容幻觉；因此不能直接将V6切生产。通过率17/18只对**已接受生成**的案例统计，不能与原20条17/20直接比较，因为V6主动拒绝了两条困难支撑面任务。完整逐图结果（包含defer和失败理由）见：

`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_relation_v6_20260922/review_report2/index.html`

### 结论与下一步

“先规划依附物，再生成独立执行mask”是正确方向，尤其能解决目标外专属附件残留和支撑物悬空；但应把它做成**可拒绝的关系图**，而不是强行扩大mask。下一步优先级：

1. 对`remove_together`候选增加实例级所有权/遮挡核验，避免像004那样出图阶段凭空生成不相关实例。
2. 将支撑面关系细化为“可连带删除 / 可保留且仍有支撑 / 必须defer”，并在更多新source上验证，不仅使用桌面案例。
3. 对通过的执行mask增加成图后结构检查，重点检测主目标残留、悬浮附件和新出现的人/动物；检测失败时保留defer，不改写源mask。

本轮新增V6代码路径及测试，未启动全量100k、未改生产默认、未提交或推送远程。

## 2026-09-22：V7短规划 + 单次双图审核（新source复验）

### 本轮目标和原则

本轮针对“目标与依附物关系没有进入执行区域”和“编辑质量尚可却被过长规则误导/误拒”两个问题继续迭代。要求是减少文字而不是减少判断：最终数据的`instruction`只命名主目标，依附物、支撑关系和补全方式只保留在内部计划/执行提示；不增加每例审核调用次数，不以滑雪、熊、杯子等词汇建立特判。

### V7规划实现

`plan_removal_relations.py --policy relations-v7`使用干净全图和带轮廓的上下文crop。V7 prompt约265个英文词（V6实测约811词），只保留以下通用规则：

1. 依据可见像素、reference和全图位置锁定准确实例，轮廓/标签不是原图颜色；
2. 生成一句短的主目标instruction；内部关系图逐实体选择`keep`或`remove_together`；
3. 仅当非生命体是目标的专属可见附件，或目标被移除后该物体会失去唯一可见支撑并悬空时，才允许`remove_together`；独立物体、共享物体、人物和动物保留；无法形成物理合理结果则`defer`；
4. 每个实体必须有自己的可见点和bbox，原始mask/RLE不可变，外部连带像素由后续独立`execution_region`解析。

编辑器在`relations-v7 + typed-v1`下使用一条短的内部动作句（主目标，必要时加“along with …”，再加自然补全和保留其它物体），不把关系清单写入最终训练instruction。规划输出解析、bbox/target-point约束、关系执行和crop可见性均由测试覆盖。

### 审核实现

新增`audit_removal_concise.py`，每个case只调用一次Qwen3.8-27B vLLM。输入为两张完整尺寸图：BEFORE（仅此图有mask轮廓）和AFTER；不再把多张局部候选图堆给模型。审核prompt约145个英文词，先根据轮廓追踪目标的完整可见主体/附件，再分别判断：

- `quality`：是否存在目标残片、截断/悬浮物、破碎边界、不自然补全或邻物损坏；
- `instruction_match`：被命名的目标是否确实在对应位置完整完成编辑，不能把“只删了一部分”当成成功，也不能把错误区域的成功操作算作匹配。

输出严格为三个字段`quality`、`instruction_match`、`reason`；解析失败按fail处理；准入要求两个字段都为pass。HTML中模型理由标记为“Qwen3.8-27B审核（独立于Assistant review）”，不会与Codex/Assistant逐图判断混淆。一次审核耗时仍主要来自27B模型，不增加调用次数；第一批38例实际38 calls、推理654.8s、墙钟800.3s（单卡含启动），第二批19例实际19 calls、推理312.1s、墙钟423.6s。

### 两批source-disjoint结果

两批均从历史行和像素hash排除后的新source采样；当前可用VER池不足，回填为GRES，因此这轮不能声称验证VER泛化。

| cohort | 输入case | 规划接受 | 安全defer | 有效出图 | Qwen审核pass | Qwen审核fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_relation_compact_20260922` / V7 | 40 | 38 | 2 | 38 | 24 | 14 |
| `fresh_relation_confirm_20260922` / V7 | 20 | 19 | 1 | 19 | 15 | 4 |

首批40条的2条defer均是模型给出的target point落在原mask外，经过保存计划复核后没有强行编辑；第二批没有源mask审核调用，规划辅助SAM查询分别为5个source/6 queries和1个source/2 queries，原始标注未被改写。第二批编辑19例采用Qwen Image 2.1、40 steps、seed0、3卡并行，编辑阶段137.4s；它是验证集而不是生产全量。

工程边界需要说明：首批40条的规划是V7，但最初编辑作业在V7编辑器分支修复前启动，实际使用了旧的relation context；随后已用冻结的36条有效计划完成`qwen21_relation_v7_compiled_20260922`重跑，第二批20条则从规划到编辑均走修复后的V7短执行提示。因而首批审核数字用于检验“短双图审核能否发现坏结果”，不能当作修复后编辑器的严格A/B通过率；第二批才是当前链路的独立新source复验。

### 审核能抓住什么

第一批失败14例中，模型给出了可复核的目标级理由，包括：吃东西的人只删掉头/躯干而手和长条食物残留；背包残片粘在消防栓；人物衣服删掉但脸/手/眼镜仍在；羽毛帽残片；人物删除后影子仍在；斑马主体截断；大象残留断掉的鼻子/不自然填充；白衣儿童只删身体而头发/头融合到邻人；右侧小杯只剩杯盖和袖套残片。第二批4例分别是：左匹诺马被生成成黑马（目标位置仍有新增马）、右匹诺马只删头留下身体、女孩的手和墙上轮廓残影、显示器未发生编辑（no-op）。

其中首批`008_gres_r583_m0_remove`特别说明了旧审核的漏洞：编辑局部像素看起来自然，但mask选中的斑马主体仍在。旧审核可能给pass；V7审核明确要求“轮廓目标与instruction对应且完整消失”，现在给`quality=pass, instruction_match=fail`，避免以“局部看起来干净”掩盖目标错位。

### 可视化入口

- 首批40条（包含2条defer、38条生成、完整Qwen理由）：
  `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_relation_v7_20260922/audit_report/index.html`
- 第二批20条（包含1条defer、19条生成、完整Qwen理由）：
  `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_relation_v7_confirm20_20260922/audit_report/index.html`

每张卡片按原图/编辑前后对照展示，展开可见短审核prompt、输入图片路径、模型原始response和解析后的三个字段；`待复核`只表示没有额外Codex review，不等于模型审核通过。图片已内嵌到HTML，直接打开`index.html`即可预览。

### 结论和边界

V7已经把“关系判断”从最终instruction中剥离，同时保留依附物/支撑物判断；在两批新source上，短审核能稳定拦截明显残留、错实例、no-op和不自然新增。但38例中仍有14例失败、19例中仍有4例失败，主要瓶颈已从文字过长转为Qwen Image 2.1的实例完整移除、遮挡结构补全和目标重绘。因此当前应把V7作为质量更可解释的实验路径，不直接替换legacy生产默认，也不把审核通过率当作人工金标准。

下一轮应优先做两件事：

1. 对`remove_together`增加“附件是否确实在mask/crop中可见、移除后是否留下独立残片”的几何证据，并把无法证明的附件设为defer；
2. 将生成后检测到的残留类别沉淀为通用结构检查（主体完整性、断裂/悬浮、支撑关系、no-op），只在失败时重试或延期，不把具体物体词写入通用prompt。

本轮新增/修改的定向回归测试49 passed（11条pycocotools NumPy弃用warning）；全量`tests`为281 passed、1 skipped（13条同类弃用warning）。`py_compile`和`git diff --check`均通过。没有提交、推送或启动100k全量任务。

## 2026-09-23：仅替换 Qwen-Image-2.1 推理后端为 vLLM-Omni

### 结论

适配已跑通。19个冻结的V7 remove计划，两边均19/19正常输出；保持40步、相同prompt/crop/mask/约1MP尺寸、seed0、逐步区域保护与最终融合，平均编辑耗时从16.720秒降到14.312秒，延迟减少14.4%，等价吞吐提升16.8%。不是此前未对齐试跑中的约6秒。人工逐图判断两边均9 pass / 10 fail，没有观察到后端带来的实质质量提升或退化；本轮仅覆盖remove，不代表其他类型已经验证。

### 官方版本与环境

- 官方说明：https://recipes.vllm.ai/Qwen/Qwen-Image-2.1 。当前支持仍在官方PR #7759，而不是已发布tag；按官方recipe使用该分支与vLLM 0.29.0。
- 第一次19对实验使用`0c82bb1315d186ff9b5126679517b16d819b298e`。运行期间上游增加一个格式修正提交；已快进到当前头`44ea27c8094095bbffd88fa3befdfaa55ba4bc50`，两处运行时代码的AST逐项比对一致。最新头另做交换GPU的3对复测，三个Omni最终PNG与前一提交逐像素相同。
- 独立源码：`/opt/tiger/tanyue/third_party/vllm-omni`；独立环境：`/opt/tiger/tanyue/.venvs/qwen_omni_21`，Python3.12、torch2.13.0+cu129、vLLM0.29.0、diffusers0.40.0、transformers5.14.1，Omni editable install。
- 原Diffusers基线继续使用`qwen_image_21`环境：torch2.8.0+cu128、diffusers0.41.0.dev0、transformers5.18.0.dev0，未更换权重。不同运行时/算子可造成数值差异，不承诺跨后端逐像素一致。
- 当前H100驱动无法执行随包提供的CUDA13 FA3内核，实验使用官方支持的`DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`。没有降精度、FP8、减少步数、近似cache或CPU offload；保留默认compile/CUDA graph及精确prefix KV cache。启动脚本为专用环境补充CUDA库搜索路径。
- 环境纠偏：早期试装曾误升级`qwen38_audit`中的vLLM。已移除其中的Omni，恢复原`vllm0.28.0+cu129`与`flashinfer0.6.16.post3`；CUDA Python wrapper对齐torch要求到12.9系列，`uv pip check`无冲突，vLLM导入及CUDA可用性检查通过。审核模型、prompt、调用逻辑没有改动，本次后端对比未重新调用自动审核。

### 如何保持原方法

`utils/qwen21_omni.py`使用官方`Omni.generate`、`build_image_to_image_prompt`、`OmniDiffusionSamplingParams`。原生Omni API不直接接受Diffusers的Python每步callback，因此使用官方custom pipeline扩展接口：

```python
Omni(
    model=model_id,
    diffusion_load_format="dummy",
    custom_pipeline_args={
        "pipeline_class": "utils.qwen21_omni_regional.RegionalQwenImage21Pipeline"
    },
)
```

扩展继承官方QwenImage21Pipeline，只在scheduler输出后应用现有的`anchor_step`：

```text
source_next = (1 - sigma_next) * source_latent + sigma_next * initial_noise
next_latent = edited_latent * editable_weight + source_next * (1 - editable_weight)
```

mask到token权重仍由同一个`editable_token_weights`计算。Omni已经在相同尺寸编码了干净条件图；VAE采用argmax而非随机采样，因此直接复用该条件latent作为source anchor，避免重复编码。初始噪声仍使用CUDA generator、seed0。对每例检查全部40次anchor均执行；尺寸/shape不一致直接报错，不静默关闭保护。暂不支持该扩展的step batching或多请求batch，遇到不支持的方式明确拒绝。

曾发现并纠正两个不公平条件：初版Omni适配静默关闭区域保护，以及按crop像素大小直接输出（低于原约1MP）。这两项均已移除，旧`editing_omni_smoke2`结果不作为速度/质量证据。当前诊断新增`model_output_size`，19例自动验证prompt、输出尺寸、source_crop、editable_tokens、protected_instances完全对齐。

规划/关系解析、短训练instruction、内部执行prompt、最终adaptive-remove-v2融合、审核与改写规则不变。入口新增显式`--qwen21-backend vllm-omni`；省略时仍为Diffusers，避免静默改动既有作业。`run_relation_cohort`会只向编辑阶段传递该参数。

### 速度与质量结果

机器为H100 80GB，两后端各用一张卡，同时存在其他GPU负载；这是当前机器实测，不是独占硬件峰值。时间包含单例读取/crop、区域保护、推理、融合及结果保存，不包含规划/审核。启动单独统计。

| 测试 | Diffusers平均/例 | Omni平均/例 | 单例延迟减少 |
| --- | ---: | ---: | ---: |
| 19例，同一冻结计划，GPU0 / GPU1 | 16.720s | 14.312s | 14.4% |
| 交换GPU复测3例，最新官方提交 | 17.113s | 14.392s | 15.9% |

19例启动耗时为10.80s / 26.43s，编辑累计317.68s / 271.92s；整个子进程墙钟333.84s / 309.04s。Omni冷启动更慢，3例短作业的总墙钟仍可能更慢，适合常驻加载后连续生成，不能把稳态加速直接套到小批总耗时。

19对最终RGB的平均绝对差逐例为0.014～0.252/255；融合前编辑crop为0.228～0.543/255。像素相近仅用于说明后端差异小，不是任务正确率。Assistant逐图查看全部对照图，并放大疑似悬空附件/残片的case；人工判断独立于pipeline MLLM audit，两边均9 pass / 10 fail：

- 可接受：002、006、007、008、010、013、014、015、016。包括明确背景人物、不同实例雨伞、女孩、小遥控器和左显示器的移除。
- 未移除/重绘代替移除：000中央西兰花、003右侧男子、017右显示器、018后右钟部件、019右钟面。
- 不完整移除：004左马重绘为小黑马，005右马只删头，009女孩的手/手持物及人形阴影残留。
- 附带物失去支撑：011幼儿消失后球棒仍斜悬，012站立女子消失后白色盘状物悬空。011的判断涉及物理合理性，不单以“人物没了”判pass。

这些是两后端共有的问题，不能归因于Omni，也不能继承上一轮自动audit的15/19通过数字作为人工真值。本次没有顺手改变规划、prompt或审核阈值来抬高通过率。

### 可视化与复现

主报告（内嵌图片，不依赖外部图片路径）：`.artifacts/qwen21_omni_ab19/index.html`。从左到右为原图+展示用mask轮廓、Diffusers最终图、Omni最终图。生成模型实际仍接收干净crop，不接收展示轮廓。每例列出耗时、实际prompt、对齐检查以及两边完整的Assistant判断理由，明确不是审核模型的输出。`contact_00.jpg`～`contact_04.jpg`为无需HTML的对比图。PNG原件保留在各后端`edited/`目录。

最新头交换GPU复测：`.artifacts/qwen21_omni_ab3_swapped_latest/index.html`。它是同三例的重复验证，不当作新增样本计算通过率。

从repo根目录执行19例对比，输出目录必须尚不存在：

```bash
/usr/bin/python -u -m synthesis_pipeline.benchmark_qwen21_backends \
  --data-root /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/qwen21_relation_v7_confirm20_20260922/regions \
  --out-root .artifacts/qwen21_omni_ab19_repeat --gpus 0,1
```

已有冻结计划使用8卡Omni独立worker（不是将一张图切到8卡）：

```bash
/usr/bin/python -m synthesis_pipeline.run_frozen_prompt_experiment \
  --data-root /path/to/frozen/regions --out-root /path/to/new/omni_output \
  --policy typed-v1 --qwen21-backend vllm-omni --gpus 0,1,2,3,4,5,6,7
```

测试：全量285 passed、1 skipped。新增覆盖相同输出尺寸/token权重、anchor传递、拒绝静默丢弃callback、逐步anchor数值公式；`py_compile`及`git diff --check`通过。未提交/推送，未启动100k生产任务。建议先将Omni作为显式可选后端；扩大到add/replace/attribute前，不声称所有任务都已无损迁移。

## 2026-09-23：两后端共同失败的定位与 removal 迭代

本轮不再只替换后端，而是按用户要求优化共同失败。实验根目录为`.artifacts/removal_v8_20260923`；所有中间结果独立保留，没有覆盖原对照结果。使用8卡独立worker交错跑规划、生成和审核，没有停止机器上原有的其他GPU任务。保持Qwen-Image-2.1、1024级输出、40步、CFG=1、seed=0、原始source/mask以及adaptive-remove-v2最终融合不变；随机种子和无anchor实验另列，不混为默认设置。

### 根因：不是换推理框架就能解决

1. **执行范围遗漏。** 旧规划漏掉站立女子手中的盘子、幼儿的球棒、人物头顶苹果和球拍。原始mask正确并不代表完整覆盖编辑所需的附带物；后续融合也无法把不可写区域里的附件删掉。新增辅助分割只解析声明的外部附件，不审核或替换原始mask。
2. **支撑关系方向错误。** “目标自己站得稳”和“目标删除后其他生命体仍有支撑”不是一回事。骑手和被抱的孩子不能被当成一般背景，不能靠移动/删除他们来完成原任务。v7/v9会放过部分不可能的任务；v10明确先判断目标正在支撑谁。
3. **单数指令遗漏mask内的实例。** 新源`005_gres_r1438_m1`原mask选中两只小羊，v7/v9却都只写右侧一只。源标注是可信的，但指令并未完整覆盖它。v10先读取全部轮廓中的实例，再命名整个选中组。
4. **生成模型没有完成主动作。** 公交车残留、男子仍在而仅苹果/领带改变、显示器只重绘等问题在raw里已经存在，并不是final把正确结果贴坏了。新数据近乎no-op的原生alpha也未异常；不能把所有失败归因于RGBA透明层。
5. **重复辅助检查导致误拒。** 已完全位于原mask内的小衣物被再次列为附件，辅助SAM分数略低就拒掉整例。修复为：只有点命中且候选完全包含于原mask、并非整个人/整个主体时，允许作为零新增像素的already_covered；任何新增像素仍走原分数、接触、面积和保护区域检查。实际extra mask只保存原mask之外的像素。

### 实际尝试与取舍

- 在旧失败集合上依次测试取消区域anchor、第二张轮廓定位图、灰色擦除条件、传统inpaint预填、改写擦除提示、换seed。这些没有稳定解决共同错误。擦除条件还经常触发近透明输出，合成后恢复原图；保留native RGBA用于诊断，不把隐藏RGB硬当照片，也不将其纳入候选默认流程。
- 轮廓定位图保持干净crop为第一张、对齐轮廓图为第二张。修复了Qwen2.1重建prompt时丢掉第二张图含义的代码问题；区域anchor只绑定第一张干净参考latent。部分显示器删除受益，但新数据不稳定，且单图15.36s对照下双图19.25s（+25%左右）。旧钟面一例在连续双图任务中OOM，保留错误log，独立进程同参数补跑，不降低尺寸/步数。
- 关系规划比较v8/v9/v10。v10用通用的实例完整性、所有权、物理支撑规则，没有物体关键词黑名单，也没有添加物体示例。20条新源上v7/v9/v10的辅助分割query数分别为0/12/4；v10不再大量罗列已覆盖衣物，且正确拒绝抱孩子成人的移除。拒绝不是生成成功。
- 长推理单独测试：20例规划总墙钟约802.85s（含加载83.31s），同批v10非推理208.97s（含加载86.70s）；两者均21次调用，含1次点位置修复。长推理能识别骑手支撑，但又误把完整遮挡人物描述为“头和领带”、把可信mask与原始指代文本的冲突过度解释为拒绝，并重新误保留幼儿球棒。因此不采用长推理为默认。
- 编辑prompt比较已有较详细的`relation-spatial-v1`和较短的`relation-located-v2`。后者将主目标与实际外部附件合在一个动作里，保留crop相对定位和简短完整性提示；不在最终训练instruction里添加通用保护/重建段落。已覆盖的小衣物不再出现在其附件列表中。

### 新源样本与判定口径

新源目录：`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_removal_v8_20260923`。seed=2026092308，10张原图各2个region，共20case，排除2038个历史source、与历史零重合。此次可用双region新源均为GRES，不能声称已验证VER或其他编辑类型。多轮实验看过同一新批次后，它属于开发验证集，不再视为独立盲测。

Assistant逐图查看source/final并放大疑似问题，判定包括图像质量、任务完成和instruction/mask范围一致。缩略图初看通过但放大显示明显坏补全的case已更正，例如v9牙刷底端蓝色矩形、公交车后不自然拱起路沿。所有理由在JSON及HTML中完整保留。`007`即使出图完成删除，其v10指令含不清楚的“and Bus center”，仍按最终数据不合格计，不以图像漂亮掩盖指令问题。

20条新源同源对照，未出图仍在分母里：

| 配置 | Assistant通过 | 已生成失败 | 出图前defer |
| --- | ---: | ---: | ---: |
| v7计划 + 原typed-v1编辑 | 13 | 7 | 0 |
| v10计划 + spatial编辑 | 13 | 6 | 1 |
| v10计划 + located短提示编辑 | 16 | 3 | 1 |

这是小样本探索结果，不是100k产线质量估计，也不是将多版本逐例挑最好图得到的通过率。v10 located的新源成功包括双小羊、绿衣球员、人物连球拍、左侧人物连头顶苹果；仍失败的是前景公交车未删干净、后公交车指令含糊、右侧人物仅删苹果/部分手部。被抱孩子的成人在plan阶段defer，不再生成悬空孩子/断臂图。

人工判断更正记录：`017_gres_r6496_m1`原图有四颗橙子，其中一颗仅在黄苹果和红苹果之间露出弧形部分。缩略图初审错把删除红苹果后露出的第四颗橙子当成新增；原尺寸复核后，baseline、v9双图和v10 located改为pass，v9/v10 spatial反而因连带删除该橙子改为fail。JSON和主HTML均保留更正原因。不能将MLLM结论直接当真值，也不能将人工缩略图初判当绝对真值。

旧失败回归集共11例（原本2 pass/9 fail）：v10 spatial为7 pass/2 fail/2 defer，v10 located为5 pass/4 fail/2 defer。两例骑乘马被拒绝，因为删除马且保持骑手原姿势会失去支撑。短提示在新源更好，但旧显示器、钟面仍不如spatial；不存在已经验证的全场景最优策略，不能在每例出图后择优再宣称单一版本成绩。

### 审核补充：仅依赖MLLM仍有漏检

在v9 spatial的18张结果上，原27B全图双图审核给13 pass/5 fail，其中至少两例是明确误放：双羊只删一只；左侧男子还在却声称已删除。保存的实际输入确实是对应before/after，没有传错图。后者原mask内显著变化像素仅约4%，是可测量的近乎不变，不能相信文字理由。

增加以下**显式可选**检查，不增加每例MLLM调用数，也不增加模型输出字段：

1. `--policy mask-coverage-v2`：先从轮廓识别完整选中组，再看request，禁止单数request重定义mask。4例复核中纠正双羊误放，并保持两个好例通过、断臂例失败。此4例是定向回归，不是总体准确率。
2. `--pixel-veto`：只对remove计算原mask内平滑RGB变化；当平均变化<8/255且变化≥12的像素比例<10%时否决。它只能否决、不能产生pass，保留MLLM原结论、独立像素证据和最终结论。低对比度真删除也可能被否决，所以保持可选，需继续检查误拒，不能当作语义证明或用于attribute。
3. `--policy pixels-first-v3`：第一张实际AFTER、第二张带轮廓BEFORE，要求先看实际结果而不预设编辑已经成功，再定位原目标。只交换顺序并添加通用观察规则，不加入物体特定规则。定向4例（006/017/018/019）均与Assistant一致，纠正了019“人还在却说已删除”的误放，同时合理补全的橙子和左男子删除保持pass。定向复测不足以证明总体有效，另对全部19张结果进行完整复测。

v10 located的mask-coverage-v2审核为17 pass/2 fail，与原尺寸复核后的Assistant判定一致18/19；019仍误放，原mask变化比例18.24%，不满足保守像素否决。没有为这一例临时提高阈值。相同图像的审核差异不得混入出图通过率。

**完整复测结果，不采用定向4例的乐观结论：** pixels-first-v3在全部19张图上为16 pass/3 fail，与Assistant一致17/19。它修复019误放，但新增001牙刷误拒，并放过007含糊指令；001原尺寸对比可见牙刷确实消失，模型将邻近包装/实例误当目标仍在。该方案不替换默认审核。相同4卡设置下，scope方案整批199.53s、after-first方案204.31s（均含各worker冷启动）；累计worker推理时间325.68s/301.87s，不能把worker累计时间当墙钟。均19次调用、0解析错误。这也说明通过数恰好等于人工通过数，并不代表逐例判定一致。

after-first完整实验展示：`.artifacts/removal_v8_20260923/fresh_afterfirst_audit/index.html`；与主报告使用完全相同的图片，仅审核策略不同，保留完整模型理由。原始回复、两张实际审核输入与完整prompt位于`fresh_audit_afterfirst19/`，未将两次独立实验包装成每例两次调用的生产流程。

### 速度、复现与上线边界

同批稳态编辑均值：原版15.23s/case，v10 spatial 15.20s，v10 located 15.12s。可认为质量优化未明显增加编辑延迟，不声称0.1s左右差异是可靠加速。27B规划端v10在同20例的加载后墙钟约122.27s，v7约208.61s；这是batch=4的整批耗时，不是单请求latency。两者输出上限也不同（v10为768，v7为1280 tokens），不能将差异全归因于prompt改写。小批并行反复加载27B有明显冷启动成本，正式运行应常驻模型，不能用这轮冷启动实验的墙钟直接外推100k。

候选流程通过显式参数运行，默认后端、旧策略没有悄悄切换；尚未启动生产或提交推送：

```bash
/usr/bin/python -m synthesis_pipeline.run_relation_cohort \
  --data-root /path/to/new/removal/cohort --out-root /path/to/new/output \
  --policy relations-v10 --editor-policy relation-located-v2 \
  --qwen21-backend vllm-omni --gpus 0,1,2,3,4,5,6,7

/usr/bin/python -m synthesis_pipeline.audit_removal_concise \
  --data-root /path/to/new/output/regions \
  --edited-dir /path/to/new/output/editing/context_grounded_v4_qwen21/edited \
  --out-root /path/to/new/output/audit --input-layout full \
  --policy mask-coverage-v2 --pixel-veto --gpus 0,1,2,3,4,5,6,7
```

实验目录里的`generation_request.json`、native RGBA、raw crop、final、模型原始回复及计时均保留。主可视化使用内嵌图片，不依赖HTML相对图片路径。后续优先处理“主目标残存但附件已删”、与源mask匹配的简短最终指令，以及审核的明显误放；不能通过放宽pass或无限重试掩盖这些问题。

### 本轮交付与后续优先级

- 新源主对照：`.artifacts/removal_v8_20260923/fresh_final/index.html`。左为带展示轮廓的原图、中为baseline、右为候选v10 located；20条全部展示。Assistant是本轮逐图人工判断，Qwen是独立模型判断，二者不一致也原样展示。模型完整判定理由不截断；展开显示实际编辑输入、raw/final crop及prompt。`contact_00.jpg`至`contact_06.jpg`是不依赖HTML的对比图。临时Markdown预览已按用户要求删除，后续多图说明统一使用HTML。
- 旧失败回归：`.artifacts/removal_v8_20260923/regression_final/index.html`。baseline、spatial、located三种出图并列，明确展示短提示在旧例的退步，不做逐例择优。此报告没有新跑的MLLM audit，不将Assistant判断冒充产线审核。
- 修正记录：`fresh_manual.json`、`regression_manual.json`在实验根目录。源mask保持原样，执行范围扩展另存；未修改源数据，也未将实验图片混入生产数据。
- 回归测试：297 passed、1 skipped；`git diff --check`通过。新覆盖规划物理支撑、可信mask内附件零扩张、独立外部附件、定位prompt、RGBA诊断、像素否决及审核输入布局检查。

下一步不按物体名称打补丁：首先将“主目标完整消失”和“附件处理完成”作为分开的观察问题，研究同一次审核内更稳定的证据读取；其次清理public instruction与internal execution的职责（目前部分规划仍把附件写入public instruction，未完全解决），并在全新数据上检验。对大面积主体仍存在的情况，先比较raw与final确认是生成未执行还是融合回贴，再决定是否需要局部再编辑；本轮尚未证明增加一次生成的收益足以抵消速度损失，不默认加重试。

## 2026-09-23：融合、指代与主体删除的连续迭代（run v11）

本轮实验路径`.artifacts/removal_v11_20260924`中的日期字符串是run标识，实际启动日期为9月23日。按用户要求先修复融合，再改规划/编辑提示，随后验证审核。所有记录继续汇总在本文。

### 第一步：校正融合轮廓，不增加生成

重新检查`009_gres_r10622_m1`的spatial版本：raw已删除女孩及附件，墙面较连续；final却出现沿人体mask的色差。之前笼统称为“阴影残留”不准确。合成alpha内部为255、外侧为0；原图外侧背景与raw存在约2–3/255的色调差，回贴后出现目标轮廓。

新增`harmonic-offset-v1`：从可写区域外的可靠像素估计source−raw颜色偏移，在低分辨率网格上求平滑延拓，将偏移应用到raw后仍使用原alpha融合。不扩大可写范围；排除目标像素和大幅变化像素，修正限幅16/255。目标内部保持raw纹理和几何，保护孔洞/区域外像素保持原图。没有任何物体名称规则。`adaptive-remove-v3`将此步骤接到原adaptive-remove-v2支持区域之后。

已复用9张旧回归raw和19张上一批raw，无新扩散调用。Assistant查看全部对照后，女孩例从fail改善为pass，其他27张未发现明显新增损伤；原本未完成主体删除/指令含混的例子仍判fail。9条旧输出由7 pass/2 fail变为8 pass/1 fail；19条新输出保持16 pass/3 fail。候选没有补救计划阶段未出图的例子，不能从分母中删除它们。

算法本体平均0.131s/例（旧9条）、0.145s/例（新19条）；包含读图、存图、生成诊断目录的整批墙钟5.44s/12.57s。保护区域像素一致性逐例检查通过，合成色偏的独立数值检查也通过。

可视化：`lighting_regression_report/index.html`与`lighting_fresh_report/index.html`，位于本轮实验根目录。左原图、中原融合、右新融合；完整Assistant理由保留。原图片没有覆盖。

### 第二步：当前region指代与附件去重

新增`relations-v11`，共享组的原始提问不再直接输入为目标描述；只在有明确绑定时提供当前region的可选提示。模型用干净全图和轮廓crop独立描述选中对象/组，public instruction由target字段编译，外部附件放在relations。编辑`relation-action-v3`从target构造一次主删除动作、追加已解析的外部附件一次，并带入一句背景补全说明。

新采样60张源图、62个region：GRES/VER各30张源图，五个面积档各6张，排除2048条历史记录，像素级零重合。允许一/双mask源图，每条源图的全部region都被使用，恢复了VER覆盖。37条dev、25条预先冻结holdout，路径`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_removal_v11_20260924`。开发和确认结果待本轮完成后在下文续记。

### 开发集第二轮：不能用更短的prompt替代完整执行语义

37条dev的v11规划得到33条输出、4条未出图。固定规划、执行mask、seed=0、40步和adaptive-remove-v3，对照action-v3与spatial-v1。逐张检查并放大可疑区域后：action 27 pass/6 fail，spatial 28 pass/5 fail。早期口头计数26/27有误，已以`dev_reviews.json`及报告汇总校正。未出图不计为成功。`dev_comparison/index.html`展示所有37条，没有逐例择优。

- `010_gres_r2295_m0`：action只改了飞盘、男子保留；spatial完整删除男子。短提示未稳定加强主体动作。
- `005_gres_r5481_m0`、`048_ver_r5112_m0`：缩略图疑似残留，放大后确认是露出原有黑箱、保留独立邻车，均通过。不能只按目标颜色消失或缩略图判定。
- `032_ver_r1541_m0`：出图合格，但开放的伞被写成closed，属于描述错误，不应再生图补救。
- `054_ver_r5385_m0`：建筑填为平面天空并保留悬空墙上碎片；`059_ver_r616_m0`、`061_ver_r10244_m0`：大建筑mask、公共指令和实际删改范围不一致。
- `050_ver_r946_m0`：缩略图疑似切断补出的人头，检查full final后可由原有帽子和举手遮挡解释，没有足够依据判为合成失败。未据此擅自扩大alpha。

新增relations-v12只加强通用约束：按方向检查其他生命的支撑、允许共享物体由其他可见支撑保留、缩短target、按深度延续背景而不是重建目标。非thinking开发34条出图，3条未出图；人工29 pass/5 fail。桌边男子及披萨盒案例的出图改善，机场大范围目标被规划defer，但新增`026_gres_r747_m0`误删后方红车。该版本尚不能据通过率提升直接定版；继续对照同一prompt开启同次调用内的低强度推理。

### 审核开发：区分真正删除、图像质量和文字错误

旧19条上的visual-evidence-v4（左右前后全图+左右前后细节，两张输入）给出18 pass/1 fail，但仍把只删除头顶苹果误读为整个人删除。没有将它晋升为默认审核。

新dev33条旧mask-coverage-v2完整双图审核：30 pass/3 fail，与Assistant一致28/33。主要误差是伞的错误closed描述被照单全收、建筑残留被忽略、将独立空中网球误认为必须连带删除的附件。统计的是逐例一致，不是通过数相同。

新增completion-v5保留一次调用，仅四个字段：target_removed、quality、instruction_match、reason。真正删除失败不能被文字改写挽救。对比native full与stacked输入：后者每张分别是同一时刻的干净全图+目标细节，仍只有BEFORE/AFTER两张输入，避免把四个视图混成多个实例。实验结果待完成后补记。

另实现`rewrite_verified_removals`：只对quality与target_removed均通过且未触发像素否决的图，独立看前后图生成一条简短Remove指令，不提供旧指令，避免复制旧幻觉。输出标记为model_candidate，不能把自述same-target当作独立验证，也不能用于把未删主体改写成删除附件。尚未通过实测的候选不进入训练集。

### 第三轮与冻结确认方案

v12 thinking在37条dev得到30条输出，Assistant逐图检查28 pass/2 fail，7条未出图。`023_gres_r3951_m0`放大确认是保留原有木餐板，不应把棕色餐板当作平涂缺陷。剩余明显问题为人物投影疑似残留和建筑补全。思考模式修复了骑乘马的支撑判断、伞的open状态和后车保留，但也错误地将车内、已位于原mask内的驾驶员当作外部必须保留的人，且为小伞推测了难以定位的额外杆件。旧回归也仍有控制器、球棒等附件残留，不能称thinking全面优于之前版本。

新增v13不按物体名称分支，只限定：支撑冲突针对选区外需要保留的对象；附件必须有清楚可见的外部像素，不能凭功能推测；背景补全不要重建目标的视觉痕迹。在16条选定开发回归中生成14条，Assistant 11 pass/3 fail，2条未出图。修复了公交驾驶员误拒、伞附件误拒和closed误描述；机场大mask仍defer；两条建筑未完整移除和一条投影问题仍失败。它是针对开发问题的回归成绩，不是独立泛化分数。

逐字检查指令后再校正：v13伞例虽将closed修正为open，却仍错误写成above the Schneiderei sign（实际在招牌前并低于首行字），因此完整训练对应为10 pass/4 fail，而非仅检查图像时的11/3。`dev_v13_reviews.json`保留具体原因。v5 adaptive审核拒绝两条建筑，但漏掉该文字错误和疑似残影；不是所有错误都已解决，后续独立确认集必须如实计入误放。

审核v5的native full在旧19条与人工19/19一致，但新dev33上只有27/33；stacked在同33条action输出上流程判定31/33一致（含1条输出截断而保守拒绝，不能称模型语义判断正确），29 pass/4 fail。stacked在旧19条又误放了只删除头顶苹果的主体残留例。故冻结面积驱动的adaptive输入：源mask占全图不足5%用全图+放大细节，大目标用原尺寸前后双图。这是通用可读性规则，不含案例ID或类别名；开发回放不替代新确认集。每例一次调用，token截断按fail处理并在报告单独标注。

改写实验两版各29次调用：grounded-v2补上了网球拼图的right panel，并将小塔改成更可靠的泛称；但对全部pass强制改写仍可能引入无必要的运动状态或扩大目标描述。因此最终默认`--rewrite-scope mismatches`：仅图像合格、完整删除且原指令不匹配时才候选改写；`all`只用于显式消融。历史两次29条实验均为all，运行时尚未加入此参数，原summary不回写。

`holdout_protocol.json`在打开25条holdout结果前冻结：relations-v13 thinking、spatial-v1、adaptive-remove-v3、Omni 40 steps/seed0、completion-v5 adaptive一次审核；仅文字错配候选改写。目标是不同面积、GRES/VER场景上大体自然，人工确认模型接纳样本至少90%可用；小样本不代表100k生产可靠性。未出图也保留在报告，不能挑图提高成功率。

新开发编辑本体均值：action-v11 14.88s、spatial-v11 14.90s、v12 14.88s、v12 thinking 14.77s。保持40步，不把0.1秒波动视为加速。规划thinking增加明显耗时，特别是小批反复冷启动，因此本轮不能宣称完整流程加速。130张新开发输出逐例验证源mask字节保持不变、alpha为0处最终像素与原图完全一致。测试308 passed/1 skipped；融合的3个SciPy数值测试在Omni环境另行通过（原系统Python无SciPy，已声明依赖）。

### 冻结25条确认集：实际结果与失败分解

`holdout25`按事先冻结参数完成：25条输入、23条出图、2条未出图。Assistant对23条输出逐图检查并对照指令，20 pass、3 fail；完整训练对初始可用率20/25=80%，出图后的可用率20/23=87.0%。GRES/VER、小物体、同类多实例、拥挤人群和大面积目标均在其中。不能把23条出图直接记为23条成功。

`holdout_report/index.html`展示全部25条，图片内嵌。`holdout_reviews.json`保存Assistant的完整理由，明确区别于27B回复。三条失败：

- `014_gres_r9683_m0`：棕牛删除，白牛保留，但原目标在mask之外的投影碎块留在路面。主体删除不等于视觉痕迹完整清除；当前小范围融合不能自动识别远处影子属于谁。
- `028_gres_r1055_m1`：raw的盘沿连续，回贴后的final右下盘沿突兀断开。不是色调差，而是生成背景几何与源图边界不一致，harmonic色偏校正不能解决。
- `051_ver_r9594_m0`：选中的稀疏短发、深色夹克男子实际已删掉，拥挤人群补全大体自然，但draft混淆了相邻光头和羽绒服人物的外观。应尝试改写，不应重做图，也不能把审核错认成编辑错对象。

`022_gres_r11374_m0`的三明治插签/橄榄/辣椒组合，以及`045_ver_r302_m0`的外部附件解析未完成，保留未出图记录。它们不是源mask质量不合格，不恢复SAM源mask审核，也不把defer计作质量通过。`052_ver_r10952_m0`圆形物的细分类别yoga wheel不够可靠，虽然编辑本身自然、实例指代清楚，仍提示后续生成应优先使用可见形状和位置，少补分类猜测。

冻结completion-v5审核：22 pass、1 fail，0解析错误，与完整人工判定21/23一致；模型接纳的22条中20条人工可用（90.9%）。它漏掉014投影与028断边，051拒绝结论虽正确，但错误把删除完成度判fail，堵住了本应可尝试的文字补救。这是小规模确认结果，不能声称审核可靠到可无监督放量100k。

确认集编辑23条平均14.998s，median14.926s，范围14.195–16.025s；40步、seed0、同分辨率策略均未改变。5卡编辑阶段含加载/编译/保存墙钟125.56s；规划+区域解析+出图整体墙钟474.52s。随后8卡27B审核含冷启动195.34s，合计约11.2分钟；不含人工检查和后续消融。多卡不同阶段不能用单条15秒直接外推端到端吞吐。

### 确认后开发：边界融合消融与审核字段解耦

确认集已被查看，以下再调参属于开发回放，不再冒充独立确认成绩。

新增completion-v6实验策略，仍一次调用、四字段：target_removed只以原图轮廓选区为基准，draft身份错配只影响instruction_match；quality检查附近接触面上的断开投影与新断边。没有添加具体物体名称/案例ID。并行回测23条新确认输出及19条旧输出，观察是否补救错配同时维持主体残留拒绝。

复用23条raw进行`poisson-diagnostic`梯度域融合，零扩散调用，墙钟14.23s。重点放大014、028、018：028仍有盘沿/色块异常，014阴影仍在且出现明暗扩散，因此不晋升为默认，不称其修复成功；其余20条没有逐图重新验收，不报告整体通过率。保留独立实验结果`holdout_poisson`，未覆盖候选图。下一步几何衔接应从生成约束/受保护背景边缘入手，而不是全局扩大alpha或反复换融合算法。

本轮合计167张主要开发/确认输出检查通过：源mask字段与冻结源标注一致，alpha=0处final与source逐像素一致。最新单元测试312 passed/4 skipped（其中3项因系统环境无SciPy）；Omni环境中该3项数值测试另跑全部通过；`git diff --check`通过。未启动生产、未提交/推送本轮变更，原有无关工作区修改保留。

completion-v6回放结果：新23条仍22 pass/1 fail，改正051的字段归因，仍漏掉014和028；旧19条17 pass/2 fail，漏放019（实际只删苹果、人物仍在），较v5 full的19/19一致退步。没有把v6替换为生产默认；它是文字补救路径的实验策略，不能凭一条修复就推广。新增提示并非可靠解决视觉判断的办法。

051的首轮独立改写（全图+detail两张拼接视图）仍输出dark quilted jacket，指向未删的前景人物，Assistant拒绝。模型原始推理明确表现出全图与放大图的对应混淆。保留原始回复，不自动把合法JSON当作成功；随后在同一候选图上对照简化为两张完整原尺寸前后图，不改变改写任务和模型。该对照是困难例调试，不作为新泛化证据。

完整双图的grounded-v2已能在推理中识别正确被删人物，但最终句子只有“the man in the dark jacket on the left side of the crowd”，对拥挤人群仍缺少足够消歧。grounded-v3仅增加一条通用要求：同一侧还有相似对象时，使用与可见邻居的相对位置，而不只写宽泛方位。同时预选7条不同场景回放（大象、人物、拼图网球、伞、细塔、远处车辆、光轨）；该7条为显式all改写消融，正常流程仍只改写错配。

### 当前候选的运行入口与使用边界

生成侧可复现入口如下；它是remove专项候选，不代表add/replace/attribute已完成同样验证。源数据目录须为已经准备好的、未混入实验输出的源图与标注。输出目录必须为新目录。

```bash
python -m synthesis_pipeline.run_relation_cohort \
  --data-root /path/to/fresh/removal/data --out-root /path/to/new/run \
  --policy relations-v13 --thinking --editor-policy relation-spatial-v1 \
  --remove-composition-policy adaptive-remove-v3 \
  --qwen21-backend vllm-omni --gpus 0,1,2,3,4,5,6,7

python -m synthesis_pipeline.audit_removal_concise \
  --data-root /path/to/new/run/regions \
  --edited-dir /path/to/new/run/editing/context_grounded_v4_qwen21/edited \
  --out-root /path/to/new/run/audit --policy completion-v5 \
  --input-layout adaptive --pixel-veto --gpus 0,1,2,3,4,5,6,7
```

当前不将模型pass直接当作可无人复核入库：新确认集的投影与几何断边仍会漏判。也不能把development上更高通过率或一次改写成功解释为泛化已经解决。质量上已验证的改动保留，失败消融不替换默认；下一优先是受保护背景几何约束、可归属目标的外部投影范围以及附件定位失败的有限重试，不以物体关键词、案例ID或扩大全部可写区域解决。

### 本轮最终交付（含失败尝试）

051的grounded-v3完整双图改写为：`Remove the short-haired person in the dark navy jacket at left foreground, beside the man in the blue puffer jacket.` Assistant确认它区别于保留的前景羽绒服男子、对应实际删除，接受这一条候选。最终25条中21条人工确认可用（含1条改写补救）、2条图像失败、2条未出图；即21/23出图可用，21/25输入得到可用训练对应。最后这个数字包含确认集查看后的定向文字修复，不是冻结流程的独立泛化指标，原冻结成绩仍为20/25。

与此同时7条开发回放只4条改写可接受：008大象、019白裙人物、029右侧网球panel、041远处白货车；032把伞误写成gold oval emblem，034把背景细塔误写成飞机上的部件，058把道路光轨误写成train，均拒绝。泛化证据不足，因此grounded-v3/full仍为显式实验参数，不全面替代原方案，不对所有pass强制改写。

最终主展示为`.artifacts/removal_v11_20260924/latest_report/index.html`：全部25条，只展示一个生成候选版本；顶部说明冻结与后续修复的统计差别。每条左原图轮廓、右最终图；Assistant对原指令判断、27B实验审核完整理由、实际编辑输入/raw/final crop/prompt分别标注。051另显示候选改写及Assistant接受理由，不能把该人工复核当作自动产线能力。所有图片内嵌，约15MB，没有依赖外部图片路径或README。

本轮结束时实验GPU任务均已正常退出，未启动100k正式生产。当前可保留的是生成侧的局部色偏修复和更明确的区域/依附关系规划；通过率不错的普通与细粒度场景不等于困难建筑、投影和拥挤遮挡全部解决，审核误放和改写误指代仍是明确未解决项。

### 附件SAM零候选修复：relations-v14与正框兜底

确认集两条未出图并非候选通过后被面积或保护区规则误拒。保存的NPZ只有query、没有mask；`022_gres_r11374_m0`的`green olive and pepper skewer`在0.05阈值探测时最高0.2695，低于正式0.3；`045_ver_r302_m0`的`white tray box`降到0.05仍为零候选。替换为更贴近画面的`olives`和`wooden tray`分别达到0.7539与0.6641，并通过原有点、接触、面积和保护区规则。根因是复合/错误视觉属性、小而细或遮挡目标，以及SAM候选生成阶段未使用planner已经给出的bbox；不是源mask审核，也不是SAM看到了overlay。

resolver现采用分阶段有限重试，原始RLE仍逐字节不改：先对最多三个短视觉查询执行文本定位；失败时在同一整图视觉编码和同一文本状态上调用SAM3官方positive-box geometric prompt。正框结果必须同时满足score>=0.6、预测框与planner框IoU>=0.15，以及原有point、target contact、area、protected overlap和owner-size门槛。每个查询文本失败后立即尝试其正框结果，一旦安全通过就停止，不把三个别名全部跑完。候选NPZ现在保存query、mode、scores和masks；resolution记录每次尝试、选中模式和数值证据。bbox只引导候选并参与审核，从不直接作为编辑mask。

relations-v14保持v13关系判断不变，只要求每个relation在同一次27B调用中给出2-3个指向同一实体的短视觉名词短语；不确定颜色/材质不进入查询，不使用斜杠类别或多对象合取。它是实验选项，不是当前默认。首轮probe把`holding a box`写入人物target且漏掉附件；第二轮修正人物scope，却把与三明治相邻的薯条加入target，导致公开instruction和成图错配；第三轮修正薯条后，人物target又出现`holding a box`，虽有独立relation仍违反简短公开指令要求。最终prompt明确外部实体只允许出现在relations、绝不进入target，但在更大新数据验证前不晋升。当前推荐组合仍是冻结的relations-v13 planner加新版resolver。

保持旧v13规划完全不变的25条resolver消融：旧实现23/25，15.21s；新版25/25，15.19s，只有原两条进入正框回退，各执行一次文本与一次box grounding。人工查看辅助mask：022仅补入三明治上的两枚腌制物与细签，045仅补入人物手中的浅托盘，没有吞并邻人。V14第二轮的两条SAM解析也均通过；早停后共2次文本grounding、2次box grounding、2次整图编码，SAM阶段13.47s，但上述planner文字回归意味着不能把它计作2/2完整训练对。该小样本证明原始两种SAM失败机制已被修复，不足以把25/25解释为100k泛化率；局部crop回退尚未启用，只有更多新数据仍出现小目标零候选时才继续验证。

两条V14第二轮结果随后用当前Qwen-Image-2.1 Omni、40步、adaptive-remove-v3完整出图，扩散耗时15.586s与14.397s，两卡含冷启动墙钟59.43s。两张图都完整移除了主体及外部附件：三明治签/腌制物无残留，人物及托盘无残留，邻近实例保留。但三明治公开instruction错误要求同时删除右下薯条，结果仍有薯条，因此只算图像生成成功，不算合格训练对；它也是V14不晋升而继续采用V13公开指令的直接证据。

直接单卡启动27B probe时曾因PATH中缺少venv的`ninja`导致FlashInfer首次JIT在进入case前失败；使用正式`run_relation_cohort.py`相同的venv PATH后正常。失败目录被保留为`planner_v14_probe_failed_no_ninja`，不计入结果。专项SAM测试30 passed；完整回归315 passed/4 skipped，`git diff --check`通过。

### 当前方案在100张新源图上的冻结验证（2026-09-24）

本轮响应“使用当前方案测试更多新数据”，不再复用旧25条调参。使用`prepare_fresh_iteration`，seed=2026092407，GRES/VER各50张新源图，每个数据集按原mask面积五分位各抽10张，保留抽中源图的全部region，共101个remove任务。排除数据根目录已记录的2108条历史源记录及其已保存图像哈希；另检查本地1834份历史manifest覆盖的700条源记录/哈希，无重合。实际mask面积分布：小于1%共25条、1–5%共24条、5–20%共37条、不小于20%共15条。该分层集合覆盖不同尺度，但不是原数据自然分布的无偏抽样，也不是四类编辑的综合评测。

数据保存在`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_sam_bbox_confirm_20260924`。运行与全部证据保存在`.artifacts/sam_bbox_fresh100_20260924/`，入口`run.py`；`protocol.json`记录源manifest和关键代码SHA256、固定参数及不在本集合调参的约定。

冻结配置：relations-v13 thinking（Qwen3.8-27B/vLLM）→新SAM文本+positive-box附件回退→relation-spatial-v1→Qwen-Image-2.1/vllm-omni、40步、seed0→adaptive-remove-v3回贴→completion-v5、adaptive双时刻输入、pixel-veto（27B/vLLM）。规划、编辑、审核阶段分别8卡并行，SAM附件解析单卡且复用整图编码。没有恢复SAM源mask审核，没有降步数、改提示词、放松阈值或重跑挑图。V14仍不晋升。

流程正常结束：101条规划，98 accepted、3 defer；附件解析后96条进入出图，2条defer_unresolved_auxiliary；96张图全部生成，审核96次调用，0解析错误，所有编辑worker退出码0。规划共106次调用，含5次既有修复调用。96条原始mask字段不变、最终尺寸不变、composition alpha为0处像素与source完全一致；这只证明工程不变量，不能证明语义正确或邻居没有被可编辑边缘波及。

Assistant先逐例检查101条source/final完整图和局部图，对有疑问的案例额外检查clean-source/raw/final。于UTC 11:08:00保存`assistant_reviews.json`及SHA256冻结记录，之后才读取自动审核结论。此处Assistant看图是本助手的独立视觉判断，不是流程27B输出，也不冒称另请真人标注。

结果：Assistant 65 pass、31 fail、5 no_output。出图后可用率65/96=67.7%；按全部输入得到可用训练对65/101=64.4%。GRES为32/17/2，VER为33/14/3。按面积分别为：<1% 18/5/2，1–5% 11/11/2，5–20% 25/11/1，>=20% 11/4/0（均pass/fail/no_output）。小面积本身不保证简单，1–5%组包括影子、人体接触和建筑局部；不能据小组样本直接推断所有小物体优于大物体。098等近乎整图移除虽然没有明确主体残留，也不作为细粒度定位能力证据。

27B审核85 pass、11 fail；对照Assistant为：双方pass62、双方fail8、Assistant fail而模型pass23、Assistant pass而模型fail3。一致70/96=72.9%；模型放行对应中62/85=72.9%被Assistant认可。以Assistant判断为参照，31条失败仅拦住8条；其中056是像素规则拒绝，模型文字实际为pass，应与模型看对区别。部分边缘轻微瑕疵存在标准差异，但还有003断臂、008雪板、026人物被沙发覆盖、043瓶身、090凳面等明确漏检，不能把差异都归为过严。

主要失败机制（标签可重叠，不相加作为31条的分母）：

- 独立邻居误删10条：003捕手删除牵连裁判前臂；015女孩删除牵连成人手；026沙发补全覆盖后方男孩；044握手男子缺手；048骑手后方旁观者被删；075亭子开口内的人物被抹掉；076竖药盒删除牵连下方横药盒。抽查这些例子的`protected_instances.png`均为零：既有保护依赖已知邻居mask，不能从“原mask不变”推导“所有相邻物被保护”。
- 主体不完整10条：011猫头、043瓶状体、090凳面、095左手、096左半组公寓仍在；056/063/080/089结构性目标删除不足。明显残留不能靠改写指令洗成正确删除。
- 附件残留5条：008雪板、034半个公文包、075悬空木柱等。008的planner错误声称板已在mask内，relations为空。正框回退只解决“已提出附件的定位”，不能弥补规划漏提。
- 投影/反射5条：078推车投影、084人物投影、088楼体倒影等仍在原区域。需要明确归属的有限扩展，不能简单对全部mask大膨胀。
- 回贴边界5条：005盘面沿原三明治边缘的斜线、085盾形浅色接缝在final比raw明显；004细桅杆片段被源图回贴保留。019的车身形状灰块和035后方食物悬空则已涉及生成与背景几何，不是单纯色调融合。
- 指令范围：037图像中的人物删除自然、餐盘保留合理，但instruction把餐盘也写成要删。它可以考虑仅修文字，然而27B也错误地认为mask包括盘子，输出target_removed=fail、instruction_match=pass，阻断既有改写入口。本轮未人工改写后混入冻结成绩。

3条模型拒绝而Assistant接受：024香蕉实际已删，模型却声称仍在；025黑白合影的背景重建存在视觉标准分歧，Assistant认为总体可接受；097只是标题HOYALESWARA与HOYSALESWARA的拼写差异，目标仍可唯一指认、图像正常，不应为无关紧要的文字细节拒绝。完整双方理由均保留，不把Assistant作为绝对真值。

SAM专项结论与上轮小样本不同：98条规划通过中，新解析器得到96条；按保存的同批文本候选和原阈值推算，关闭box回退仅94条（这是已保存候选的反事实统计，不是另跑了一次编辑）。011的两处爪部及025的杆件通过正框回退获得出图机会，但011仍留猫头，只有025被Assistant认可；当前审核还拒绝025，所以不能把解析多2条等同于自动入库多2条。071只解决稻草绳，另一绿环/白绳仍低分而整例暂缓。

007粉伞则不是零候选：文本候选score=0.9414、bbox IoU=0.9137、point_distance=0、contact_distance=1、protected_overlap=0；面积32925像素，而人物mask42800、固定75%上限32100，故被拒。正框score=0.9844仍被同一面积规则拒绝。这是本轮新发现的安全阈值过拒，不能继续笼统称SAM找不到。028骑手依赖马而defer合理；061是同棵树中段与树冠的scope冲突；068把局部椅背杆等同整把承重椅子，有过拒风险。5条no_output均保留，不悄悄跳过。

速度：96条编辑本体均值14.8987秒、中位14.8565秒、范围13.92–16.25秒；8卡编辑阶段含加载/编译/保存230.39秒（3.84分钟），约25张/分钟。SAM阶段23.08秒（14次文本查询、6次正框查询、10次图像编码、0源mask审核）。规划+解析+编辑816.44秒；27B审核343.93秒；全流程1160.37秒（19.34分钟），不含源数据准备、Assistant视觉检查和HTML构建。本批约298输出/小时，按65条Assistant可用折算约202条/小时；不将短批次含冷启动吞吐外推为100k稳定产能。主要端到端时间已经是MLLM规划及审核，而非SAM。

交付主入口`.artifacts/sam_bbox_fresh100_20260924/report/index.html`：摘要、优先问题、速度与12个导览例。另四个分页覆盖000–025、026–051、052–077、078–100全部101条，`all_cases.html`保留完整单页版本。每条左原图轮廓、右final，下方Assistant独立理由；展开可看27B完整理由和原始回复、审核prompt、生成prompt、raw/final局部、原始二值mask及规划/解析证据。图片全部内嵌，分页降低编辑器preview负担；没有添加README。

因此本轮结论不是“总体OK”：方案能生成不少良好同实例/局部编辑，但还不足以仅依赖当前自动审核大规模入库。下一优先为接触/遮挡邻居保护与完成度审核，其次才是附件面积过拒、外部影子/反射和回贴接缝。继续改进必须另开新开发批次，再用未见确认数据检验；不把本101条上后续修复当作新泛化成绩。此次没有改动生成方法，没有提交/推送，也没有启动100k生产；实验GPU进程均已退出，原有用户进程未动。


## 2026-09-24：接触邻居归属、目标优先执行和窄带融合（ownership-v15 实验目录，当前候选 relations-v16）

响应用户三项要求，本轮实际改了规划、区域执行、融合，并在旧困难例和20条新源数据运行。统一入口为 [.artifacts/ownership_v15_20260924/index.html](../.artifacts/ownership_v15_20260924/index.html)。所有可视化图片内嵌；不新增README。本节中的Assistant判定是本助手逐图观察，不是pipeline模型输出。

### 1. 规划和执行区域

在同一次27B/vLLM规划调用内，relations-v15明确检查接触、遮挡、原mask孔洞中的独立邻居：keep为需要保留的实体，remove_together为原mask之外属于目标的附件/碎片。一个简短reconstruction句描述露出的背景和保留对象的连续关系，公共instruction只描述主目标。没有添加物种/物体关键词特判，没有重启SAM源mask审核。

V15暴露两个问题：双布尔support字段容易拼错或矛盾；模型会把普通背景大量列为keep，增加无意义定位拒绝。V16改为单个support_check=coherent|conflict，要求一条关系对应一个实体，普通填充表面只写reconstruction；“接触地面”不能作为手持物失去主人后仍可保持原姿态的依据。后者修正了旧008错误保留雪板的规划。现有目标点修复调用机制没有删除，因此不能声称每例严格只有一次规划调用。

新增 --ground-keeps：先解析独立邻居，再解析连带附件。邻居候选需满足点、框、置信度及与原目标重叠约束，不再只靠文字说“保留”。原始mask保持不变。--auxiliary-policy ownership-v1只在高置信、框/点明确匹配且不侵入保护点时，允许突破旧的附件面积<=主体75%限制；附件选区仍减去实际保护mask。修复了大于人物的雪板以及边界小范围相交的控制器被旧阈值拒绝，不是全面放松SAM阈值。

--keep-fallback box-protect-v1仅对KEEP分割失败提供保守框保护：点必须落在合法框内且位于原目标外；框只能冻结未选中的像素，不能当作删除区域或声称是可靠SAM分割。先使用可靠SAM保护解析附件，再从软保护框中扣除完整执行区域，避免保守框阻止已确定的连带删除。原目标和确定的连带附件保持可编辑；确切SAM邻居仍是硬保护。目标点冲突、目标身份错误不会被这个回退掩盖。

### 2. 生成阶段

保持Qwen-Image-2.1 / vllm-omni、40步、seed0及现有缓存设置，不降步数。固定候选使用单张干净crop，relation-spatial-v1、visible-v1和guard-any-v1，不增加额外轮廓参考图。

对照发现，guard-fraction-v1在同一latent单元同时包含目标和邻居时，按保护比例回注源latent，会连原目标内容一起注入。003捕手出现白色碎片、026女孩手/控制器残留，在“不加轮廓图、仅使用比例保护”的控制组也复现；所以不能只怪额外参考图。guard-any-v1让含目标的单元保持完全可生成，纯非目标邻居单元严格保护，缓解这类残留。像素级最终保护仍生效，但混合单元无法同时做到目标自由生成和邻居全程完全固定，因此复杂接触几何不是已彻底解决。

旧回归8条固定候选：7条出图、1条未出图。图像质量5 pass/2 fail；严格按可训练图文对统计为4 pass/3 fail/1 no_output，因为011虽图像移除完整，指令混入“mask [427,309,848,564]”。003裁判手臂、026男孩、076横药盒得以保留，034人物与包删除；008雪板已删除但邻人肩臂不自然，015腕部边界仍断裂。075把框架主目标误描述为稻草人而未出图，不能当修复成功。初期V15的16条结果也保留在regression/：6 pass、3 fail、7 no_output，不能只报6/9而忽略过拒。044握手残留和048背景旁观者仍失败。

### 3. 融合对照

两种新策略都复用完全相同RAW，未额外调用扩散模型：

- adaptive-remove-v4：在原写入包络内图割找接缝，目标alpha必须1、保护alpha必须0。工程约束成立，但085徽标出现明显盾形硬边，效果退步；不采用。批量重组旧96张不等于已人工重新检查96张，本轮只对选定问题例详细比较。
- adaptive-remove-v5：保留v3的alpha和低频harmonic校正，再对可写区域内部最多12像素窄带做native-resolution screened harmonic RGB offset校正。可靠外侧颜色差必须<40，修正幅度限制24，内部核心边界为0；不复制原目标纹理、不扩张写入范围、不改生成核心形体。005盘面斜线有所减轻但仍可见，085浅色印记仍未彻底解决。它是有限改善，不是结构修复。

recompose_ownership逐例断言：alpha为0处最终像素与source一致；保护区不变；执行目标alpha=255。源码mask不变以及像素不变量不能证明语义归属正确，019的错误保留手臂就是反例。当前20新例的v3/v5语义通过数没有变化，不宣称融合显著提高总体通过率。

### 4. 新源20条完整验证

源目录为 /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_ownership_v15_20260924。seed2026092415，GRES/VER各10张，每个原mask面积五分位各2张；20图20region，均remove。排除历史2208条源记录及已存像素哈希，采样报告零重合。因本轮看过V15规划后继续改V16，该集合定义为新源开发验证，不是未见确认集，不能与前轮101条不同分布的67.7%直接当成提升对照。

固定配置见fresh_protocol.json。V16初始邻居严格分割13条可执行，附件强定位策略14条，KEEP框保护回退后17条可执行；没有逐case混用最好的生成版本。生成8个worker全部退出0，17张输出。Assistant逐图检查全部20条，疑问例另看原尺寸图，再保存fresh_reviews.json（UTC14:21:29，SHA256 e52e481ba98c12d5e88ac6dfa61df0fb932f70f2bcfb98f5488423e06fee2168），之后读取逐例27B结果。此前进程日志已输出14 pass/3 fail聚合数，因此不是完全盲测。

Assistant：14 pass、3 fail、3 no_output；成图可接受14/17=82.4%，输入端可接受14/20=70%。27B completion-v5 + adaptive双时刻 + pixel-veto：17次调用、0解析错误，14 pass/3 fail，与Assistant二元判定17/17一致。这个小批次一致不能外推审核已可靠：019模型主要抓边缘划痕，却把target_removed判pass，未充分识别悬空搭肩手臂。

- 正例：005删右包保住邻包/安全带；007删裤腿雨靴保住狗和另一人；012删中间骑行者并连带自行车、另两人保留；013细灯杆删除、栏杆窗口保留；016多赛车中只删前车。
- 003 fail：右半拼图的球员消失，但其脚部投影断片残留在右下边界；指代还应写清右半画面。
- 006 fail：前排熊头删去，身体区域生成不自然熊身并融入后排熊，保护篮子不等于正确背景补全。
- 019 fail：左男孩删除后，其搭在右男孩肩上的手臂仍悬空，脸颈边沿细线仍在；是归属和局部形体问题。
- 000 no_output：盘子KEEP点落进原目标，定位冲突，有过拒风险。
- 017 no_output：规划判基石移除与保留雕像支撑冲突，defer而非生成报错。
- 018 no_output：大建筑删除的保留路牌等定位冲突；仍应优化定位，不将无输出作为质量通过。

### 5. 速度、测试和运行

本轮17条编辑本体均值15.156秒，13.89–16.053秒；8卡编辑阶段99.64秒含加载/保存。前轮均值14.90秒，仅能说当前质量改动没有显示提速，不能把不同批量阶段时间直接对比。新融合平均0.2648秒/条，离线17张读写重组总12.04秒。27B审核178.53秒；规划及第一次解析482.25秒，最终框回退解析另20.74秒（34文本、16正框、17图像编码、源mask审核0次）。这些是迭代分段实测，不伪称一趟完整pipeline端到端实测。

完整单元测试327 passed、6 skipped；其中系统环境跳过的5项融合测试另在qwen_omni_21环境通过（unittest 5 passed）。git diff --check通过。新增tests/test_ownership_execution.py、tests/test_removal_harmonization.py已加入gitignore豁免，防止测试代码被忽略。所有本轮GPU实验已退出，用户原有进程未动，未启动生产、未提交推送。

显式启用当前候选的运行方式（输出目录必须不存在；默认旧配置未静默覆盖）：

```bash
python -m synthesis_pipeline.run_relation_cohort \
  --data-root /path/to/prepared_dataset --out-root /path/to/new_run \
  --gpus 0,1,2,3,4,5,6,7 --policy relations-v16 --thinking \
  --ground-keeps --auxiliary-policy ownership-v1 --keep-fallback box-protect-v1 \
  --editor-policy relation-spatial-v1 --qwen21-backend vllm-omni \
  --latent-protection-policy guard-any-v1 --relation-geometry-policy visible-v1 \
  --remove-composition-policy adaptive-remove-v5
python -m synthesis_pipeline.audit_removal_concise \
  --data-root /path/to/new_run/regions \
  --edited-dir /path/to/new_run/editing/context_grounded_v4_qwen21/edited \
  --out-root /path/to/new_run/audit --policy completion-v5 \
  --input-layout adaptive --pixel-veto --gpus 0,1,2,3,4,5,6,7
```

下一优先：接触肢体的真实归属和遮挡后连续性（不是更大范围膨胀）；投影/反射的显式有限执行区域；KEEP点冲突时的定位修复而非直接放行；主目标名误绑定及公共指令标注坐标泄露。保留简短通用prompt，不做熊、滑雪、手臂等词表分支。上述未完成点不能通过改instruction把明显坏图包装为合格。

## 2026-09-24：冻结当前方法的四机部署

本轮不调整质量方法，将上述remove候选封装为四节点数据并行入口。详细clone、安装、正式入口和日志命令见
[四机运行指南](SAMTOK_LABELING_四机运行指南.md)。不是DDP，不跨机切分单个模型；每机独立使用8卡。
同一source的全部region分到同一节点，不更改源mask、不重编号、不复制样本凑数量。
relations-v16/thinking、ownership-v1、box-protect-v1、relation-spatial-v1、guard-any-v1、visible-v1、
adaptive-remove-v5、40步/seed0、completion-v5/adaptive/pixel-veto均保持原值，不增加改写或重试调用。

### 部署修复与可复现环境

- 将当前路径依赖改为`SAMTOK_*`环境变量，旧入口默认值仍保留。每台clone对应分支到节点本地。
- 三套互相隔离的环境固定Python、完整包快照、SAM和官方Omni源码commit；安装后检查实际import/CUDA/GPU。
  初测修复了PyTorch索引遮蔽普通包、`torch==2.8.0`匹配到非基线CUDA构建的问题。
  最终脚本从空目录完整安装到`/opt/tiger/tanyue/labeling_runtime_validation_20260924_v3`成功，
  检查结果与真实模型测试使用的v2环境完全一致；SAM为torch2.8.0/CUDA12.8。
- 第一次真实模型测试从共享FUSE目录并发加载编辑权重，触发Omni600秒启动超时。
  保留失败日志，没有通过放宽质量或调大超时掩盖。新增节点本地模型缓存：26个文件、约31GB，
  复制时计算SHA256并读回校验。正式入口默认使用该字节相同副本，模型参数和推理设置不变。
- 控制文件独占认领、防旧run覆盖，节点比对代码/环境/模型/输入/方法，异常传播并终止本节点模型进程组。
  汇总检查所有输入、规划、区域解析、生成和审核的覆盖，保留fail与no_output分母。
  `model_pass.jsonl`仅表示模型判通过，不冒称人工验收。

### 测试证据与边界

本机实际上只有1台8张H100，不是四台32卡。部署测试将8卡分为4个独立worker（每个2卡），
使用真实27B、SAM、Qwen-Image-2.1/Omni运行完整链路；生产入口仍强制四个不同hostname、每台8卡。
输入为源parquet的前8个正例（含GRES/VER），共9个region，分片为3/2/2/2。
四worker均退出0，最终`finalize.ok.json`已产生：9/9出图、9/9完成审核、9条model_pass、0解析错误。
逐条核对ID覆盖无重复、原mask与输入相同、前后图尺寸一致且全部可以完整解码。
这9条没有新增Assistant逐图质量验收，不能将模型9/9通过当作真实准确率100%。
各节点规划+解析+编辑耗时279.25–387.84秒，审核138.67–176.84秒；并发链路最长约546.5秒（9.1分钟），
不含环境安装、源图准备和首次缓存复制。编辑本体均值15.43秒/条（15.09–15.81秒），
各节点编辑阶段58.10–76.42秒（含加载、保存）；日志确认全部40步、seed0、vllm-omni。
源码和测试、文档会一起提交；模型、环境、图片、日志均不进入Git。

证据根目录：
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/deployment_validation_20260924`。
其中`run/`保留第一次共享权重超时，`run_localweights/`为本地权重重测，
`logs/setup_final_locks.log`为最终锁定环境的完整新建安装日志。
本轮全套单元测试341 passed / 6 skipped；另在新建SAM环境执行融合测试5项全部通过。
提交后从Git全新clone验证，发现旧ignore规则漏收7份已有测试，已补入版本控制。
修复后的clone同样341 passed / 6 skipped，全部pipeline/utils源码逐文件SHA一致，
三套环境的SAM、vLLM、官方Omni及regional扩展import/CUDA检查通过；不依赖工作树未跟踪代码。
共享bootstrap副本已安装至指南所列路径，内容与仓库脚本逐字节相同。
不将单机模拟结果说成真实四机网络/共享存储已联调，也不将小批次冷启动时间外推为100k吞吐。
未启动正式全量任务；真实四机首次提交应先限制8张源图smoke，再换run ID执行全量。

## 2026-09-25：修复Arnold编辑环境的OpenCV混装

实际任务`samtok-remove-4n-20260925`四台均clone到`cd9129a`，node1/node3在约03:33 UTC
导入editor环境的cv2时出现`ImportError: libxcb.so.1`，随后`editor environment check failed`。
node0/node2通过旧检查并继续复制模型，尚未开始生成。初步判断为系统库差异不充分，进一步发现
`requirements/labeling-editor.lock.txt`同时包含`opencv-python==5.0.0.93`和
`opencv-python-headless==5.0.0.93`。OpenCV wheel自带METADATA明确要求四种分发包只能选一个，
它们共享cv2命名空间和文件；实际二进制会受到文件覆盖影响。成功节点可能加载headless或具备GUI库，
不能仅凭成功/失败判断系统镜像不同。上一轮本地v3环境实际加载headless，所以旧预检漏掉了混装。

本次修改：

- 编辑环境只保留同版本headless；官方Omni固定源码继续`--no-deps`安装，其cv2图像API由headless提供。
  不改变规划、出图、融合、审核prompt或模型参数。
- 每个环境安装后立即检查OpenCV分发包唯一性；完整预检还检查实际`GUI: NONE`，将实际版本/构建写入报告。
  即便cv2暂时能import，只要GUI与headless混装也会明确拒绝。本地旧v3混装环境已实测被新检查拒绝。
- 四台全部通过环境检查并发布environment marker后才允许复制权重；有失败标记时，即使ready标记齐全也退出。
  大文件复制和读回SHA校验、源数据分批准备增加失败检测；中断时保留临时文件而不写成功manifest。
  单次阻塞I/O和共享文件系统可见性仍影响检测延迟，安装本身仍会运行到结束后检查peer。
- 四机指南完整Arnold入口同步以上实现，并纳入此前已完成但未提交的Arnold说明。
  旧run不自动重启，使用新的run ID和新环境；不能删除failed marker强行继续。

全新三环境安装位置：`/opt/tiger/tanyue/labeling_runtime_opencv_fix_20260925`。
三个环境实际GUI均为NONE，SAM/vLLM/官方Omni/regional扩展及CUDA预检成功。
新editor的cv2.abi3.so与上一轮成功验证环境实际使用的二进制SHA256完全一致：
`9e29605abcc31c9942d0e7dffc53b4ccaa601676c794327b0ea71a570a0c26c2`；ldd未发现X11/xcb/Qt依赖。
新增回归覆盖混装、GUI二进制残留、权重校验期间peer失败以及四节点环境等待和失败优先行为。

本轮证据保存在
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/opencv_fix_20260925/`，
`logs/setup.log`为从空目录安装记录，`run/`为真实模型完整链路验证。
验证已完成：本机8卡分为四worker各2卡，8源图9region，9/9成图、9/9审核，四worker退出0，
`run/control/finalize.ok.json`已产生；模型9 pass、0解析错误，本轮未新增Assistant逐图质量评审。
逐图完整解码、前后尺寸与ID覆盖检查通过，日志确认40步/seed0及原冻结profile。
编辑本体均值15.19秒（14.81–15.69秒），并发链路最长约749.3秒/12.5分钟，不含环境安装。
首次新环境的27B共享权重读取和Triton预热占用明显时间，这不是稳定吞吐或提速对比实验。
完整回归348 passed / 6 skipped，5项融合测试另在新SAM环境通过；21项部署测试包含真实Bash环境门禁。
两个准备CLI也读取原失败run的真实marker验证：均在创建输出目录前拒绝继续。
原四机任务未重启；真实四台主机上的重提验证仍需使用更新后的Arnold入口、新run ID及新环境。
