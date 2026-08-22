# CrispEdit-2M 当前完整数据打标 Pipeline（Filter → Mask）

本文档记录当前 repo 中 **正在使用且已经完成生产运行** 的 CrispEdit-2M 打标方案：

> **本地 Qwen3-VL 预筛选（filter） → manifest-aware SAM3 打标（mask）**

这不是历史实验说明，而是当前生产实现、当前生产命令、当前结果统计和当前已知边界的统一参考。

当前对应实现文件：

- `crispedit_mllm_prefilter.py`
- `crispedit_mask_dataset_runner.py`
- `crispedit_mask_pipeline.py`

---

## 1. 当前生产数据与输出位置

### 1.1 输入数据

当前生产输入根目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M
```

注意当前数据布局是：

- parquet shard **直接在根目录下**
- **不是** `.../CrispEdit-2M/data/`

当前生产输入规模：

- shard 数：`399`
- 总行数：`101,697`

### 1.2 当前生产输出

第一阶段 prefilter：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-manifest
```

第二阶段最终 mask：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697
```

当前生产完成状态：

- prefilter：`399` 个最终 parquet，`0` 个 temp 文件
- mask：`399` 个最终 parquet，`0` 个 temp 文件
- 最终对齐输出：`101,697` 行
- 运行错误：`0`

---

## 2. 当前 pipeline 总览

当前流程不是“先全量打 mask 再过滤”，而是先做语义预筛选：

```text
raw parquet shards
    ↓
Qwen3-VL-8B 本地 prefilter
    ↓
audit parquet + keep manifest parquet
    ↓
SAM3 manifest-aware runner
    ↓
对 keep 样本生成 mask
对 drop 样本写入对齐占位行（PREFILTER_SKIP）
    ↓
得到与原始 shard 逐行对齐的最终 mask parquet
```

这条架构的核心目标：

1. **先过滤明显未达成编辑指令的样本**，避免错误 target 进入 SAM3。
2. **保留原有 base SAM3 mask 逻辑作为生产主路径**，不把实验性 add-mask 路线混入生产。
3. **全流程本地、确定性、可后台运行**。
4. **最终输出逐行对齐原始 parquet**，方便后续按 `row_idx` 回查。

---

## 3. 第一阶段：本地 Qwen3-VL 预筛选

### 3.1 当前做法

入口：`crispedit_mllm_prefilter.py`

每条样本读取：

- `input_img`
- `output_img`
- `instruction`
- `type`

当前 runner 的视觉输入是：

- Image 1 = 编辑前 source
- Image 2 = 编辑后 target
- 再拼接 instruction + raw edit type 的文本 prompt

### 3.2 当前实现细节

当前生产版本已经不是早期的逐条串行 + 临时 PNG 文件方案，而是：

- **PIL 内存解码**，不再写 `src.png / tgt.png`
- **真正的 batched multimodal inference**
- **decoder-only 左填充**
- **固定 `do_sample=False` 的确定性生成**
- **`max_new_tokens=220` 保持不变**

对应关键实现点：

- 图像直接以内存中的 PIL `Image` 传给 processor
- `self.processor.tokenizer.padding_side = "left"`
- `apply_chat_template(..., processor_kwargs={"padding": True})`
- `prompt_length = int(inputs.input_ids.shape[1])`
- `trimmed = generated_ids[:, prompt_length:]`

也就是说，当前生产 prefilter **不依赖临时图片文件**，而是直接走内存里的图像对象。

### 3.3 当前判定口径

当前 prompt version：

```text
qwen3vl_edit_prefilter_v4_add_strict
```

判定的不是 raw `type` 是否“纯粹”，而是：

> **target 图相对于 source 图，是否达到了 instruction 想要的结果。**

当前决策规则：

- `PASS -> keep`
- `FAIL -> drop`
- `UNSURE -> drop`
- parse/runtime error -> drop

### 3.4 当前生产配置

来自生产 `run_config.json` 的关键参数：

- 模型：`/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct`
- GPU：`0,1,2,3,4,5,6,7`
- `batch_size=4`
- `max_new_tokens=220`
- `overwrite=false`
- `job_count=399`
- `total_rows=101697`
- `run_id=prefilter_20260813_091022`

### 3.5 当前生产结果（以 manifest 为准）

**注意：** 当前这次 prefilter 是在 resume 后完成的，因此：

- `run_summary.json` 的 `rows=101697` 是正确的；
- 但 `keep/drop/verdict_counts` **低估了最终总数**，因为 resume 时被跳过的已完成 shard 只推进了总进度，没有重新累计 verdict 聚合。

因此，**最终权威统计应以完整 manifest / 最终 mask 输出为准**。

当前完整 manifest 权威统计：

| 指标 | 数值 |
|---|---:|
| total rows | 101,697 |
| PASS / keep | 74,796 |
| FAIL / drop | 24,346 |
| UNSURE / drop | 2,555 |
| total drop | 26,901 |
| keep rate | 73.55% |
| drop rate | 26.45% |

### 3.6 prefilter 示例图

下面两张图来自当前 repo 已保存的审核拼图，可直接用于理解当前 prefilter 的 keep/drop 行为：

#### KEEP examples

![KEEP examples](docs_assets/crispedit_pipeline_workflow/prefilter_keep_examples.png)

这组图可以看到：

- `add`：柠檬被正确加入，判为 keep
- `background change`：背景从暖色星空变成深蓝星空，判为 keep
- `replace`：宇宙飞船被潜水艇替换，判为 keep
- `style`：水彩风格改成更明显的数字绘画风格，判为 keep

#### DROP examples

![DROP examples](docs_assets/crispedit_pipeline_workflow/prefilter_drop_examples.png)

这组图展示了当前 drop 的典型原因：

- instruction 根本未实现
- 变了但变错目标
- 图像本身看起来有变化，但与 instruction 不匹配
- 原图里已经有目标对象，结果图并没有新增请求对象

这也是当前 prefilter 设计的目的：**尽量把“编辑没做对”的样本拦在 SAM3 之前。**

---

## 4. 第二阶段：manifest-aware SAM3 mask 打标

入口：`crispedit_mask_dataset_runner.py`

### 4.1 当前做法

第二阶段不再重新判断语义是否成功，而是直接读取第一阶段 manifest：

- `filter_decision == keep`：进入正常 SAM3 打标
- `filter_decision != keep`：写一条 `PREFILTER_SKIP` 占位行

因此最终输出 parquet 与原始输入是 **逐行对齐** 的。

### 4.2 当前并行方式

当前生产 runner 的并行度来自：

- **一个进程绑定一张 GPU**
- **每个 worker 只加载一次 SAM3 模型**
- **shard 级任务均衡分发到 8 张卡**

当前 mask runner 的 `--batch-size`：

- 是 **parquet 读取/迭代批大小**
- **不是** SAM3 模型的真正 batch inference

当前每个 worker 内部仍然是：

- 一次处理一条样本
- 每条样本内部按 `annotate_one(...)` 的逻辑串行调用 SAM3 查询

这也是为什么当前全量 mask 阶段比 prefilter 更难进一步靠简单调参提速。

### 4.3 单条样本的生产 mask 逻辑

单样本入口：

```python
annotate_one(processor, sample)
```

主流程：

1. `canonicalize_type(...)` 归一化类型
2. `_prepare_workspace(...)` 统一 before/after 尺寸并计算：
   - `local_diff`
   - `motion_diff`
   - `global_diff`
3. `parse_instruction(...)` 做本地确定性 instruction 解析
4. `fuse_mask(...)` 按类型走不同逻辑：
   - `add/remove/replace/color`
   - `motion`
   - `background`
   - `style`
5. `postprocess(...)` 做后处理
6. `quality_score(...)` 生成 QC 指标与 flag

当前生产明确使用的是：

> **原始/base mask pipeline**

而不是后面实验过的 MLLM-guided add-mask 路线。

### 4.4 autocast / dtype

当前 GPU 路径使用：

```python
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = annotate_one(processor, sample)
```

这是为了解决此前实测出现过的 BF16 / Float dtype mismatch 问题，并且当前生产跑通时已经验证稳定。

### 4.5 最终输出字段

当前最终 mask parquet 主要字段为：

- `row_idx`
- `raw_type`
- `canonical_type`
- `instruction`
- `phrases_json`
- `qc_flag`
- `qc_status`
- `diff_iou`
- `diff_precision`
- `diff_recall`
- `area_frac`
- `mask_height`
- `mask_width`
- `mask_sum`
- `mask_png`
- `prefilter_verdict`
- `prefilter_confidence`
- `prefilter_prompt_version`
- `prefilter_model_name`
- `prefilter_run_id`
- `prefilter_reason`
- `filter_decision`
- `filter_reason_codes`
- `filter_mismatch_score`
- `filter_version`

对于被 prefilter 拦下来的样本：

- `canonical_type = PREFILTER_SKIP`
- `qc_flag = PREFILTER_SKIP`
- `mask_png = b""`
- 但相关 prefilter 字段仍然会保留，便于追踪原因

---

## 5. 当前生产运行命令

下面是与当前完成结果一致的生产命令写法。

### 5.1 prefilter（8 卡，后台，带 tqdm 日志）

```bash
cd /opt/tiger/tanyue/sam3
source .venv-sam3-crispedit/bin/activate

mkdir -p /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit
mkdir -p /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-manifest

nohup python -u crispedit_mllm_prefilter.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --audit-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-manifest \
  --devices 0,1,2,3,4,5,6,7 \
  --batch-size 4 \
  --max-new-tokens 220 \
  --progress-mininterval 5 \
  > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.log 2>&1 < /dev/null &

echo $! > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.pid
```

### 5.2 mask（8 卡，后台，带 tqdm 日志）

```bash
cd /opt/tiger/tanyue/sam3
source .venv-sam3-crispedit/bin/activate

mkdir -p /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697

nohup python -u crispedit_mask_dataset_runner.py \
  --input-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M \
  --keep-manifest-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-manifest \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697 \
  --devices 0,1,2,3,4,5,6,7 \
  --batch-size 8 \
  --progress-mininterval 5 \
  > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.log 2>&1 < /dev/null &

echo $! > /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.pid
```

### 5.3 当前生产命令为什么 **不加 `--overwrite`**

当前正式运行是按 resume 方式完成的，因此生产命令默认不加 `--overwrite`。

这样做的效果：

- 已完成 final shard：直接跳过
- 不完整 `.tmp` 不会被当作 final shard，worker 会重新跑该 shard
- 能从断点继续，而不是把全部结果重刷一遍

具体规则：

#### prefilter
仅当以下两个文件都已存在且不加 `--overwrite` 时，才跳过该 shard：

- audit shard
- manifest shard

#### mask
仅当以下文件已存在且不加 `--overwrite` 时，才跳过该 shard：

- final output shard

因此当前推荐的 resume 方式就是：

> **保持原命令不变，继续运行，但不要加 `--overwrite`。**

---

## 6. 当前生产结果汇总

### 6.1 全局结果

最终 mask `run_summary.json`：

- `rows=101697`
- `errors=0`
- `kept=74796`
- `prefilter_skipped=26901`

QC flag 汇总：

| qc_flag | count |
|---|---:|
| PREFILTER_SKIP | 26,901 |
| OK | 72,390 |
| EMPTY_EDIT | 246 |
| LOW_CONF_BG | 1,271 |
| LOW_CONF_BG_FULL | 497 |
| LOW_CONF_COLOR | 4 |
| TOO_LARGE_LOCAL | 388 |

几个直观比例：

- keep rate：`73.55%`
- prefilter drop rate：`26.45%`
- `OK` 占全部样本：`71.18%`
- `OK` 占 keep 样本：`96.78%`

### 6.2 分 edit type 统计

| type | rows | keep | drop | keep rate | OK | 其他 flag |
|---|---:|---:|---:|---:|---:|---|
| add | 14,592 | 8,833 | 5,759 | 60.53% | 8,821 | EMPTY_EDIT 12 |
| background change | 14,592 | 13,295 | 1,297 | 91.11% | 11,527 | LOW_CONF_BG 1,271 / LOW_CONF_BG_FULL 497 |
| color | 14,592 | 11,476 | 3,116 | 78.65% | 11,400 | EMPTY_EDIT 72 / LOW_CONF_COLOR 4 |
| motion change | 14,422 | 10,696 | 3,726 | 74.16% | 10,696 | 无 |
| remove | 14,592 | 7,640 | 6,952 | 52.36% | 7,442 | EMPTY_EDIT 122 / TOO_LARGE_LOCAL 76 |
| replace | 14,592 | 10,644 | 3,948 | 72.94% | 10,292 | EMPTY_EDIT 40 / TOO_LARGE_LOCAL 312 |
| style | 14,315 | 12,212 | 2,103 | 85.31% | 12,212 | 无 |

从这些统计可以读出几个当前生产特征：

- `background change` 和 `style` 的保留率较高
- `remove` 和 `add` 更容易在 prefilter 被筛掉
- 进入 mask 阶段之后，大多数 keep 样本最终都落在 `OK`
- 当前剩余的非 `OK` 主要集中在：
  - 背景分支置信度不足
  - 局部编辑为空
  - 局部区域过大

---

## 7. 结合具体图例理解当前 mask 输出

当前 repo 中已经保留了若干真实预览图，可以直接说明当前方案的输出形态。

### 7.1 多对象 add：当前 base 生产方案的局限

![add row3](docs_assets/crispedit_pipeline_workflow/add_row3_mask.png)

这个例子里，target 图相对 source 图新增了多个彩蛋，但当前 base add 路径只抓到了部分局部变化，overlay 里可以看到：

- 没有把三个新增蛋整体覆盖完整
- 只抓到了局部边缘/局部区域
- 这也是此前实验 MLLM-guided add 路线的主要动机之一

但当前生产仍然选择保留 base 路线，因为更大范围评估后，实验 add 路线整体退化更多。

### 7.2 replace：当前 base 方案的代表性成功例子

![replace row1](docs_assets/crispedit_pipeline_workflow/replace_row1_mask.png)

这个例子里，source 的飞船被替换成潜水艇：

- `diff` 图只提供局部变化线索
- 最终 `mask` 覆盖了新目标主体
- `overlay` 可以看到结果基本贴合潜水艇外轮廓

这类单主体、语义清晰的 `replace` 是当前 base 路线比较稳定的场景。

### 7.3 style：当前生产里 style 分支基本按全局编辑处理

![style row0](docs_assets/crispedit_pipeline_workflow/style_row0_mask.png)

这个例子展示了 style 的当前行为：

- 当 instruction/全局差异满足条件时，style 会走全局分支
- overlay 基本覆盖整张图
- 这符合当前 pipeline 对风格迁移类编辑的设计：
  - 不是去抠某个小物体
  - 而是把风格变化视为整图级编辑区域

### 7.4 小规模并行预览拼图

![sam3 preview collage](docs_assets/crispedit_pipeline_workflow/sam3_preview_collage.png)

这张拼图展示了当前 preview 的统一格式：

- `source`
- `target`
- `diff`
- `mask`
- `overlay`

这也是当前人工抽查 production / smoke 输出时最常用的可视化布局。

---

## 8. smoke / 小规模验证

在进入全量生产前，当前 repo 先做过两类验证：

### 8.1 batched prefilter smoke

batched prefilter 在修复左填充与 prompt trimming 后，最终 smoke 结果为：

- rows: `28`
- errors: `0`
- keep: `19`
- drop: `9`

这一步验证了：

- 真正的 batched Qwen3-VL 推理可用
- 不再需要临时 PNG 文件
- `max_new_tokens=220` 可以保留不变
- parse error 已降为 0

### 8.2 manifest-aware SAM3 小规模链路验证

随后又验证了：

- manifest 和 mask 输出逐 shard 对齐
- `PREFILTER_SKIP` 占位逻辑正常
- preview / preview collage 正常生成
- 两阶段串联能够稳定跑通

这些小规模验证通过后，才继续跑当前的 101,697 行生产版本。

---

## 9. 监控与排查

### 查看 prefilter 日志

```bash
tail -f /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.log
```

### 查看 mask 日志

```bash
tail -f /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.log
```

### 查看进程

```bash
ps -fp $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.pid)
ps -fp $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.pid)
```

### 停止进程

```bash
kill $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-prefilter-audit/run.pid)
kill $(cat /mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M-mask-parquet-101697/run.pid)
```

---

## 10. 当前已知边界

### 10.1 prefilter 的权威汇总来源

这次生产 prefilter 有 resume 行为，因此：

- `run_summary.json` 的总行数可信
- 但最终 keep/drop/verdict totals **不要直接引用该 summary**
- 最终统计请使用：
  - 完整 manifest
  - 或最终 mask 输出

### 10.2 mask 并行度的瓶颈仍在单样本内部

当前 8 卡 mask 已经做到了：

- 一卡一进程
- shard 级负载均衡

但尚未做到：

- 单 worker 内多样本真 batch
- 单样本内部多次 SAM 查询的并发化

因此当前 mask 阶段虽然稳定，但不是极致吞吐实现。

### 10.3 add 多对象场景仍是当前 base 路线的薄弱点

从 `add_row3_mask.png` 这类图可以看到：

- 多实例新增
- 小目标新增
- 相似物体并排新增

仍然可能出现漏检、只抓局部边缘、或只覆盖其中部分实例的情况。

当前生产仍接受这一边界，是因为更大范围对比后，实验 add 路线总体回退更多。

---

## 11. 一句话总结

当前 repo 已完成的正式生产流程是：

> **先用本地、确定性的 Qwen3-VL 对 CrispEdit-2M 做语义预筛选，再把 keep 样本交给 manifest-aware 的 base SAM3 mask pipeline 生成编辑区域，对 drop 样本保留 `PREFILTER_SKIP` 占位，最终得到 399 shards / 101,697 rows / 0 errors 的逐行对齐输出。**
