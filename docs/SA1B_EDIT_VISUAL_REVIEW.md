# SA-1B 当前构造管线人工视觉复核

复核对象是 `disjoint_mllm_plan30/generated_full22_40step` 的 110 个生成 unit 中，经过模型语义审核保留下来的 42 个 pair。复核方式是逐页查看全部 7 页拼图，再放大检查目标区域；这是一轮独立的 Codex 视觉抽查，不等同于正式人工标注金标准。

## 1. mask 与 case 的关系

- 候选池：1,044 张源图、127,123 个原始 dense mask，平均每张约 121.8 个 mask。
- MLLM 选择：22 张源图进入本轮，每张恰好选择 5 个 object/mask，因此得到 110 个 unit。
- 当前不是对每张源图的所有原始 mask 都构造数据，而是对 MLLM 选出的 5 个 mask 分别构造 1 个独立编辑 unit；同一源图被复用 5 次。
- 110 个 unit 的规划分布：attribute 36、replace 22、add 29、remove 23。
- 最终通过：42/110（38.2%），attribute 18、replace 12、add 11、remove 1。
- 42 个视觉 pair 展开为 168 条训练协议行；这 168 行不是 168 个独立视觉 case，而是每个 pair 的协议变体。

## 2. 人工复核结论

### 明显可用的部分

- 大多数较大、边界明确的 attribute/replace 是可用的。例如轮胎变银、雕像变金、雕像替换为马、围栏替换、轮胎替换为圆桶、加红旗、加鸟屋、删除拱窗等，目标完成且邻域基本保持。
- mask overlay 与最终编辑区域大体一致，说明源 mask 在这些样本上能作为有效目标定位。
- 通过样本的源图、目标图、局部 crop 和 instruction 之间通常能对上，指令格式本身不是主要瓶颈。

### 需要人工复核或不应直接放行的样本

以下 case 在总览或放大 crop 中存在明显低置信度：

- `sa1b-sa_4063118/U5`（#9，attribute）：指令是把金属杆改成木质，目标 crop 中材质变化非常不明显，肉眼难以确认。
- `sa1b-sa_11072880/U2`（#19，replace）：指令是把橙色外套人物替换成蓝色牛仔外套，源/目标局部几乎没有可见变化；像是漏编辑，但 27B 审核给了 pass。
- `sa1b-sa_1615968/U1`（#32，add）：给冰淇淋桶加价格标签，新增物非常小，在全图几乎不可见，局部也缺乏稳定的形状/阴影证据。
- `sa1b-sa_4154844/U1`（#36，add）：给栏杆加小铜铃，目标很小，难以确认是否真正存在且附着自然。
- `sa1b-sa_7383524/U5`（#39，add）：给花瓣加水滴，源/目标 crop 中变化极弱，肉眼基本看不出水滴。
- `sa1b-sa_1615968/U5`（#33，add）：条码上的贴纸存在，但呈现为较硬的矩形覆盖，光照/纹理融合一般。

因此，42 个 `semantic_audit=pass` pair 不应直接等价为 42 个“人工确认通过”pair；至少上述小目标样本应进入二次人工或多模型复核。

## 3. 失败统计与症状

| 类型 | 生成数 | 通过 | 通过率 | 主要肉眼失败症状 |
|---|---:|---:|---:|---|
| attribute | 36 | 18 | 50.0% | 颜色改错、目标不变、局部平涂/材质不成立 |
| replace | 22 | 12 | 54.5% | 改错实例、旧物仍在、只在旁边新增替代物 |
| add | 29 | 11 | 37.9% | 没有新增、加到错误实例、颜色错误、附着关系不自然 |
| remove | 23 | 1 | 4.3% | 目标未删、变成青色/灰色占位块、残留或背景补全失败 |
| 合计 | 110 | 42 | 38.2% | — |

模型审核失败证据中，失败原因会重叠统计：

- attribute：18 个失败中约 9 个漏编辑、7 个颜色错误、4 个实例错误。
- replace：10 个失败中约 7 个实例错误、6 个旧目标未消失、3 个明显 artifact。
- add：18 个失败中约 16 个漏编辑、8 个实例错误。
- remove：22 个失败中约 21 个漏编辑、12 个 artifact。

## 4. 问题来自哪一步

### 源 mask/选择阶段

当前选中的每张图有约 86–96 个原始 mask，但选择器只挑 5 个。110 个 selected mask 中，30 个面积占比低于 0.1%，50 个低于 0.3%；13 个是多连通分量，9 个 bbox 填充率低于 0.2，15 个属于很细的结构。这些目标适合做困难 benchmark，但不适合直接交给当前编辑器做稳定的 1024 级局部生成。

`source_pool.py` 只要求一张图至少有两个 mask；没有在进入规划前硬过滤小面积、碎片、截断、依附关系和“mask 只覆盖物体一部分”的情况。拥挤图在选择器可见的候选还被 `_review_objects(limit=40)` 截断，密集图的部分 mask 根本没有进入 MLLM 选择视野。

### 规划阶段

规划 prompt 强制 5 个目标使用四类操作的平衡组合。结果是 22 张图固定产生 23 个 remove，即使很多目标并不适合删除；remove 最终只有 1 个通过。这不是单纯的编辑模型能力问题，而是操作分配策略把不适合的 mask 推进了 remove。

同样，add 被规划到小叶片、细杆、条码、栏杆等 host 上时，新增物尺寸和附着位置很难在局部 crop 中稳定生成。

### 编辑生成阶段

这是当前最主要的结构性问题。`sa1b_pipeline/editor.py` 中，编辑器收到的是“源图 crop + 文本 instruction”；二值 mask 没有作为图像编辑模型的显式控制输入。mask 只在生成后用于把整块生成 crop 通过膨胀/羽化 alpha 粘回原图。

这会导致：

1. crop 内有多个相似实例时，模型按文字猜目标，容易改错实例；
2. replace 可能在旧物旁边添加新物，旧物不消失；
3. remove 容易把目标改成青色/灰色占位块，或根本不删；
4. 小目标在 1024 级 crop 中像素不足，生成后即使语义审核说“有”，肉眼也看不出；
5. add/replace 使用固定半径膨胀区域，可能截断新物，也可能把邻域一起重绘。

当前 prompt 已经明确写了“不要 cyan/gray patch”“只改目标”，但文字约束不能替代 mask-conditioned inpainting。

### 审核阶段

确定性检查主要看 mask 内外的像素变化；它没有判断“颜色是否真的是要求的颜色”“旧身份是否消失”“新物是否真的可见”。add 的新增物还可能自然落在 host mask 外，所以单纯的 inside/outside fraction 对 add 不适用。

最终语义审核是一次 27B MLLM 判定，存在明显漏判：#19 的 replace 几乎没有可见变化，#39 的水滴几乎不可见，但仍被判 pass。小目标和低分辨率 crop 需要单独的高分辨率审计或人工队列。

## 5. 优化优先级

1. **先改编辑输入接口**：使用真正支持 mask-conditioned inpainting 的编辑路径；至少把 source crop、二值 mask/高亮 mask 作为独立控制输入，而不是只把 mask 用于事后粘贴。replace/remove 应使用目标 mask，add 应使用 host mask + 允许新增物的 attachment band。
2. **停止强制每张图都产生 remove**：先做 operation suitability 分类，再分配操作；remove 只接受自包含目标，排除支撑物、人体依附物、被截断物和会留下悬空附件的 mask。
3. **加入 mask QA 门槛**：面积、最小像素、连通分量、边界截断、bbox 填充率、与邻近 mask 的重叠/包含关系都要检查；tiny/fragment/part mask 只在高分辨率专用分支使用。
4. **为 add/replace 建立宿主和依附关系检查**：要求新增物在 host 边界附近，替换物覆盖旧目标并使旧身份消失；不要只依赖固定 dilation 半径。
5. **多候选生成与重排**：每个 unit 用多个 seed 生成，按目标区域变化、非目标变化、目标完成度和审计置信度重排；低置信度或审计不一致的样本不进入训练。
6. **加强审核**：大图全局 + 高分辨率目标 crop 双输入；对 attribute 检查目标颜色/材质，对 replace 检查旧目标消失，对 add 检查新增物尺寸/附着，对 remove 检查目标完整消失和背景连续性。至少对 tiny edits 使用第二个独立审核器或人工复核。
7. **按源图切分数据集**：同一 source image 的多个 mask 是高度相关样本，train/val/test 必须按 source image 切分，不能按 168 条协议行随机切分。


## 6. 本轮质量优先迭代的离线验证

本轮代码把 mask 几何质量门槛前移到规划前，并按 operation 做门控；同时取消默认的四类操作强制混合，编辑器在 Qwen-Image-2.1 Omni 接口可用时发送干净 source crop、坐标提示和 regional latent weights，审核器增加目标局部高分辨率 panel，并对 attribute/replace 的低变化结果直接 veto。

用既有 22 个已接受场景及其缓存的选择/规划结果离线回放（`--min-targets 3 --max-targets 5`）得到 20 个场景、88 个 unit：12 个场景保留 5 个目标，4 个保留 4 个目标，4 个保留 3 个目标；操作计数为 attribute 33、replace 18、add 24、remove 13。这个回放证明多目标约束和质量门控能够同时成立，但没有重新运行图像生成，不能替代新模型输出的视觉验收。

## 7. 新一轮 88 个 unit 的生成复核

使用上述 88 个 quality-gated unit 运行了 Qwen-Image-2.1 Omni 生成，结果保存在外部工作目录 `quality_iteration_v1/generated_guided_88_40step/`，对应全量拼图为 `quality_iteration_v1/visual_all_88/index.jpg`，目标局部拼图为 `quality_iteration_v1/target_panels/page_*.jpg`。确定性检查得到 82 个待语义审核、6 个直接失败；6 个失败包括 5 个 remove 目标变化过弱和 1 个整体没有可见变化。

逐页查看这 88 个结果时发现，之前为了定位 mask 而传入的第二张边界 guide 图会在不少 add/replace/remove 结果中被模型复制成青色或实色矩形，或者留下边界形状。这是条件输入接口的错误语义，不能作为训练数据放行。已在编辑器中移除第二张 guide，保留 regional latent weights 和坐标提示，只传入干净 source crop；但这项修复的 5-unit 对照在当前机器上卡在 Omni 多进程初始化，未能完成新的图像输出，因此还不能声称全量质量已达标。

语义 MLLM 审核也尚未完成：Qwen3.8-27B 的 vLLM 初始化在本机触发 `cudaHostGetDevicePointer ... CUDA driver version is insufficient for CUDA runtime version`，即 CUDA driver/runtime 与 UVA 路径不兼容。该环境问题不改变已完成的确定性检查和人工视觉结论，但需要换兼容运行时或机器后再做最终语义审核。

## 8. 无 guide 全量结果与保守发布集

随后用不带第二张 guide 图的路径重新生成全部 88 个 unit，结果位于外部工作目录 `quality_iteration_v1/generated_noguide_88_40step/`，全量人工复核拼图位于 `quality_iteration_v1/visual_noguide_88/index.jpg`，逐页文件为 `page_01.jpg` 到 `page_11.jpg`。这一轮没有再出现旧版的青色/实色边界 guide 伪影。

确定性检查结果为 80 个待语义审核、8 个失败（6 个 remove 变化不足、2 个低变化目标）。对 16 个低置信度 unit 又用两个新 seed 重采样；这些结果大多仍然漏编辑，说明根因是目标尺寸或操作适配性，而不是随机种子。

我逐个查看了 88 个目标局部，并把没有清楚完成目标、产生错误替代物或无法确认新增物的 unit 拒绝。最终保守发布 66 个 unit、17 张源图：每张保留 3–5 个不同目标，分布为 6 张×3、7 张×4、4 张×5；操作为 attribute 29、replace 15、add 15、remove 7。发布拼图在 `quality_iteration_v1/final_release_visual/index.jpg`，发布 JSONL 在 `quality_iteration_v1/final_release/generated_gate.jsonl`。

发布门槛已固化在 `sa1b_pipeline/release.py`：默认要求确定性检查、人工 `manual_pass` 和完整语义审核同时通过；本轮因 CUDA runtime 阻塞，只使用显式的 `--allow-manual-without-semantic` visual-pilot 模式，不能把它误称为最终语义审计通过集。未通过的 22 个 unit 没有进入发布 JSONL。
