# CrispEdit mask 优化迭代记录

## 迭代产物路径索引

统一根目录：`/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/`。正式源数据、两轮 prefilter 和最终打标结果不在此处，路径见 [CRISPEDIT_MASK.md](CRISPEDIT_MASK.md)。以下目录均保留原名；旧 `experiments/CrispEdit/`、`datasets/` 路径现为兼容链接，本文链接已指向新位置。

| 分类 | 根目录下的目录 | 内容 |
| --- | --- | --- |
| 四机失败与环境验证 | `experiments/labeling_4node_crispedit_full_20260924_a/`、`experiments/labeling_4node_crispedit_full_localenv_20260924_b/`、`experiments/local_env_fix_20260924/` | 失败日志、环境修复和预检 |
| 四机/筛选小规模测试 | `experiments/pipeline_4rank_smoke_20260924/`、`experiments/pipeline_4rank_verified_20260924/`、`experiments/prefilter_4node_plancheck_20260923/`、`experiments/prefilter_4node_setup_20260923/`、`experiments/prefilter_4node_smoke_20260923/` | 计划、端到端 smoke、画廊 |
| 小批量 mask 实验 | `experiments/mask_fresh65_doublepass_20260923/`、`experiments/mask_newdownload64_20260923/` | 65/64 条复核、日志、可视化 |
| 下载与开发归档 | `experiments/download_200k_20260923/`、`experiments/repo_cleanup_20260924/` | 旧下载记录、迭代图片和清理前备份 |
| 早期筛选与标注 | `datasets/CrispEdit-2M-difficult-local-edit-labeling/`、`datasets/CrispEdit-2M-fact-prefilter/`、`datasets/CrispEdit-2M-grounding/`、`datasets/CrispEdit-2M-mask/` | 旧版迭代输出，非当前正式结果 |
| 旧版画廊与追加批次 | `datasets/CrispEdit-2M-mask-previews/`、`datasets/CrispEdit-2M-mask-run-additional-100k-20260908/`、`datasets/CrispEdit-2M-mask-run-resumed-after-background-audit-20260831/` | 历史预览和增量实验 |

## 2026-09-24：四机环境缺库修复与节点本地安装

生产尝试 `labeling_4node_crispedit_full_20260924_a` 完成 1,908 shard 规划，每节点待质量/场景各 195 shard，
在首次加载 vLLM 时四台均出现 `ImportError: libGL.so.1`。node2 最先写失败标记，随后其他节点退出；
新增结果 0 行、0 parquet，未进入细粒度或 mask。失败日志保留在该运行的 `logs/`。
根因是环境混装 `opencv-python` 与 headless，实际 cv2 为 QT5 构建；缓存路径预检查没有显式导入 cv2。
此前本机自带 libGL，使单机四 rank 验证未暴露容器差异。

修复：仅安装 headless OpenCV 4.11；每节点 git clone、用安装脚本创建本地基础 Python 和依赖，不再分发共享代码/环境包。
预检查先验证 OpenCV 运算和 spawn 导入，再对每张 GPU 做 CUDA 运算、GPU0 真实双图生成；四节点 commit、代码/安装配置与版本一致后才生成数据计划。
更新[四机完整入口](CRISPEDIT_4NODE.md)，删去环境打包/解包脚本。共享部署代码与旧基础 Python/venv 按用户要求清理；
原始数据、正式筛选、打标结果和故障日志继续保留。

验证根目录：`/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/local_env_fix_20260924/`。
`install.log` 为从头安装，`preflight.log` 为已通过的实际 Qwen 双图预检查，`tests.log` 记录 198 项测试通过。
从提交 `b045184` 克隆的干净本地副本独立安装通过（`clone_install.log`），基础 Python 和全部依赖都在 clone 内；
`install_reuse.log` 确认重复执行可以复用完整环境。
在干净 clone 的新装环境中，一机 8×H100、四 rank 重跑：20 质量 PASS → 19 场景 PASS / 1 DROP →
19 非空 mask、35 实例，0 解析/运行错误，四 rank 均 exit 0；`smoke/run/control/initial/complete.ok` 已生成。
逐阶段日志和汇总位于 `smoke/run/{logs,reports}/`；[19 条画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/local_env_fix_20260924/review/index.html)。
已删除两个旧共享 workspace 和本机旧缓存环境，测试 clone 也在完成后清理；Git 代码、验证日志与结果保留。
本轮修复部署依赖，没有更改筛选或 mask 语义策略；物理四机需用户重新提交新入口验收。

## 2026-09-24：CrispEdit 专用整理与完整四节点接入

本轮不改 mask 语义策略，保留当前编辑单元观察 → 单图 grounding → 外扩 crop → SAM3。
新建 `crispedit-labeling` 分支，移除其他数据集、旧 prefilter、撤回的观察/复核模式、临时研究脚本；
将共同 vLLM 推理抽入 `crispedit/inference.py`，由 `crispedit/distributed.py` 统一调度四阶段。
本地目录同步改为 `/opt/tiger/tanyue/sam3-crispedit-crispedit-labeling`。仓库只保留主文档使用的代表图；
大份画廊及历史图片移出仓库，链接指向下述归档或正式实验目录，源数据与结果不动。
SAM3 本身是运行依赖，保留上游源码，不把其内部库误当其他数据集 pipeline 删除。

接入过程中修复了 scene 输入指向根目录而非 `manifest/` 的交接错误；补齐单机/四机一致的
五类 shard 选择、保留历史 background/style 结果的统计、不覆盖其他运行、同 rank 锁及恢复门禁。
恢复完整筛选 shard 不再为跳过任务加载模型。所有推理 prompt、模型与当前候选选择保持原方法。

验证及路径：

- 清理后 189 项回归测试全部通过，Shell 语法与 `git diff --check` 通过。
- 115 + 65 + 64 条历史实验的 244 个有效观察响应回放：编辑单元解析结果全部一致。
- 一机 8 H100、四独立 rank，各 2 卡：五类各 4 条重新过两轮筛选，质量 20 PASS，场景 19 PASS / 1 DROP。
  19 条双 PASS 全部打标，19 非空、35 实例、0 解析/运行错误；四 rank 均正常退出。恢复再次完成。
- 路径：`/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/pipeline_4rank_verified_20260924/`。
  `run/logs/` 是各阶段 tqdm，`run/reports/` 是统计，`provenance.json` 记录本次重编号到原始行号的映射。
- [19 条内嵌画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/pipeline_4rank_verified_20260924/review/index.html)。已逐页查看全部 5 页：
  新增鸭子/孔雀、右侧衣服、酒瓶/南瓜/小盆栽、单人替换等表现基本延续原方法；
  `add_00076:27` 树冠过标、`motion change_00020:181` 眼睛/眼镜只出线状 mask、
  `replace_00186:214` 把保留人物也标进来等已知错误仍在。19 个自动 OK 不代表 19 个语义成功。
- `shared_run/` 用于共享环境复验，依赖导入十余分钟未结束后主动停止。调用栈显示共享盘依赖文件检查和字节码写入开销，不是样本推理或四节点通信。
  增加环境打包与节点本地缓存（SHA256、锁、冲突保护），基础 Python/模型/数据/结果继续共享；复验目录为 `cached_run/`。
  复验四 rank 全部完成：19 非空 mask、35 实例、0 解析/运行错误。
  15 条与首轮 mask 完全一致；4 条观察清单相同、框略有变化，预测间 IoU 0.979–0.996（不是对真值的准确率）。
  [缓存环境画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/pipeline_4rank_verified_20260924/cached_review/index.html)。

下一步：在实际四台 8 卡 worker 验证共享挂载/环境/全量吞吐；mask 质量继续按首轮漏项、局部扩大为宿主、
稀疏候选三个方向做独立消融。本轮没有启动新的生产全量任务，也没有把单机四 rank 称为物理四机验收。
完整入口和限制见[四机指南](CRISPEDIT_4NODE.md)。

本文记录双阶段 prefilter 之后、Qwen grounding 与 SAM3 生成 mask 的优化过程。只讨论 mask 标注，
不包含 prefilter 的开发或重跑。记录保留失败试验、输出路径和待办；自动 `OK` 是结构/QC 状态，
不能当作语义准确率。主流程与当前筛选结果见 [CRISPEDIT_MASK.md](CRISPEDIT_MASK.md)。

2026-09-24 整理说明：本文保留迭代历史，早期命令/开关仅描述当时实验，并非当前可执行接口。
当前代码只保留最终方法，旧实验图片移至
`/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/`，
本文对应链接已更新。原始数据、正式筛选结果和各轮 mask 实验目录未删除；旧实验已归入上述 `iterations/`。清理前完整工作区备份：
`/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/pre_cleanup_worktree.tar.gz`。

## 2026-09-22：早期 mask 迭代索引

以下是移出主文档的实验记录摘要。运行目录都位于
`/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/datasets/CrispEdit-2M-difficult-local-edit-labeling/`，
以双 PASS 小样本验证为主，部分包含筛选跳过对照；未启动当前两阶段筛选后的全量 mask 打标。早期结果使用当时的代码，
不能当作现行实现的质量结论。

| 运行目录 | 改进或对照 | 当轮结果与结论 |
| --- | --- | --- |
| `smoke_20260922` | 双 manifest 严格对齐、跳过占位及 8 卡端到端接入 | 20 条打标、20 条跳过；结构检查通过，但衣服、手臂、彩纸 mask 有明显问题；[选择与汇总](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/two_stage_smoke/validation_summary.json) |
| `local_mask_final_20260922`、`local_mask_qwen38_20260922` | 非 add 仅 source；修正 crop 复核、局部 SAM；Qwen3.5 与 Qwen3.8 对照 | 各 20 条打标 + 20 跳过，均 19 OK/1 REVIEW；Qwen3.8 睁眼框较好，但衣服、人物定位仍有回退；[图](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/local_region/qwen38/motion.jpg) |
| `checklist_verified_20260922` | 改为 Qwen3.8 完整变化清单、按对象定位与身份保持 | 100 条打标 + 20 跳过；98 OK/1 REVIEW/1 GROUND_FAIL，0 最终解析错误；灯串/彩纸仍不可靠；[100 条索引](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/checklist/verified/index.json) |
| `candidate_selection_20260922/verified` | 固定 grounding，只改 SAM 候选与碎片风险判断 | 100 条中 91 OK/8 REVIEW/1 GROUND_FAIL；改善轮胎孔洞、鸟头误分，但自动 OK 不等于语义正确；[对照图](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/candidate_selection/index.html) |

随后按对象范围、crop 和附件漏区分步修复，同一固定 100 条 + 20 条跳过对照：

| 运行目录 | 改动 | 自动 OK / REVIEW / GROUND_FAIL |
| --- | --- | ---: |
| `edit_scope_20260922` | 让模型自由生成 SAM ref，过度泛化，撤回 | 87 / 11 / 2 |
| `single_crop_grounding_20260922` | 每张 crop 独立请求，避免跨图坐标串用 | 94 / 4 / 2 |
| `scope_grounding_verified_20260922` | 改为代码保守生成 SAM ref | 96 / 3 / 1 |
| `scope_grounding_final_20260922` | 补查骑乘对象及共同变化附件 | 96 / 3 / 1 |
| `attachment_completion_20260922` | 固定 grounding，受限补回坐垫等内部附件 | 96 / 3 / 1 |

最终 99 条非空 mask，20 条跳过正确保留，0 runtime/解析/crop 错误；坐垫一例多恢复
76,800 px，其余 99 条 mask 逐字节一致。人头和两匹马得到改善，但骑手腿、手臂与灯串仍漏区，
轮胎范围还出现回退，故继续扩样。[全部 100 条前后对比](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/scope_grounding/index.html)。

扩展到 300 条双 PASS + 20 条跳过对照、80 个 shard 后，逐轮记录如下。分母及集合可能不同，
自动旗标不能横向解释为语义准确率；各目录均含相应 grounding/mask、日志和验证结果。

| 运行目录 | 本轮尝试 | 结构结果 |
| --- | --- | --- |
| `edit_units_20260922` | 完整编辑单位、肢体/道具分开；100 条 | 95 OK / 3 REVIEW / 2 GROUND_FAIL |
| `expanded_evidence_20260922` | 300 条中按 unchanged 删除框；误删，撤回 | 258 / 10 / 32 |
| `expanded_zoom_20260922` | 显式放大 crop；冲突仅标记复查 | 258 / 37 / 5 |
| `matched_instances_20260922` | 固定框，只改 SAM 单实例匹配 | 邻车误并改善；上游漏项未解决 |
| `coverage_review_20260922` | 用原清单做全图复查 | 254 / 41 / 5；增加调用仍会受旧答案锚定 |
| `independent_observation_20260922` | 不给指令/旧答案，独立双图观察 | 257 / 29 / 14；多实例召回提高，部位仍不准 |
| `compact_scope_20260922` | 精简提示、修复截断重试与空侧清单 | 262 / 32 / 6 |
| `mask_quality_20260922` | 固定框，完整候选与受限附件补全 | 262 / 32 / 6；293 条非空 |
| `edit_scope_quality_20260922` | 扩写范围 prompt | 263 / 33 / 4；人工开发集退化，撤回扩写 |
| `verified_masks_20260922` | 撤回扩写、保留附件修复 | 262 / 32 / 6；293 条非空 |
| `reasoned_observation_20260922` | 24 条仅第一轮开启 thinking | 22 OK / 2 REVIEW；语义退化，不默认开启 |
| `short_refs_20260922` | 强制 1–5 词 ref，100 条 | 88 OK / 12 REVIEW；开发质量退化 |
| `tiled_observation_20260922` | 短 ref + 配对放大图，24 条 | 22 OK / 2 REVIEW；不采用 |
| `attachment_contents_20260922` | 容器新增内容物只分内容物，固定框回放 | 261 / 33 / 6；仅 2 条 mask 改变 |
| `unseen_validation_20260922` | 冻结实现后取 50 个新 shard 各一条 | 41 OK / 8 REVIEW / 1 GROUND_FAIL；人工仅 25 可用、19 明显问题、6 歧义 |

独立观察研究目录 `observation_study_20260922` 比较完整双图、附指令与拼接图；附指令容易漏共同变化，
拼接图产生伪变化，故保留独立双图。`difference_observation_study_20260922` 的额外差异图
也未显示稳定收益，没有接入现行 pipeline。上述 50 条新 shard 的逐例人工判定在
[人工复核 JSON](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/expanded/unseen/manual_review.json)，
[全量画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/expanded/unseen/review/index.html)。
下一优先级由此转向 observation 漏项/错项、编辑对象与局部部位范围、容器内容物，再处理 SAM
碎孔；2026-09-23 的实验接续下文。历史细节可从这些运行目录、审计和 Git 历史追溯。

## 2026-09-23：起点与问题

2026-09-22 的人工 bad-case 清单显示两类原因交织：grounding 的 ref/框只指向内容物或一部分，
SAM 于是漏掉实际编辑对象；grounding 已覆盖完整对象时，SAM 仍会漏部件、留下孔洞或碎块。
例子包括 `replace_00570.parquet:203`（应覆盖整只碗）、`replace_00924.parquet:214`
（只改鸟头）、多人/手臂动作及汽车、面包、饼干等不完整 mask。开始本轮时的碗 mask 只有
2,104 px、33 个连通块；原始 observation 把“碗被换成盘子”缩窄成了“pie slice with fork”。

用固定 65 条回归集：报告过的 15 条 + 此前 50 条新 shard 样本。另抽取 25 条未曾人工复核的样本，
每类 5 条；先查看 source/target 并写下期望区域，再运行预测。它们用于发现新失败类型，但因整个
迭代过程中的样本已被查看，不能视为最终盲测。清单、原图预览和期望标注在
[`docs_assets/mask_pipeline/edit_units/`](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/edit_units/)。

## 每轮试验

运行根目录统一为：

`/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/datasets/CrispEdit-2M-difficult-local-edit-labeling/`

### 1. 在双图 observation 里直接加入 instruction（65 条）

让 remove/replace 第一轮带着指令观察，希望明确整件被替换的对象，再把容器名简化为 SAM 短语。
运行：`edit_unit_20260923`。结果 65 条均无运行错误，自动为 56 OK、8 MASK_REVIEW、1 GROUND_FAIL；
这不是人工质量通过。碗的 ref 和框变对了，但 SAM 只得到碗沿/表面，食物仍是开放漏区。

逐图复查发现指令会锚定模型，丢掉未写进 instruction 的共同变化：花盆左侧对象漏掉、玩具群只剩
一件、饼干背景对象漏标、花篮上的共同变化装饰也被漏掉。**撤回直接给 observation 加指令的改法。**
保留结果与运行日志：

- `edit_unit_20260923/{grounding,mask,grounding.log,mask.log,validation_summary.json}`
- 初轮 65 条画廊：`docs_assets/mask_pipeline/edit_units/review/`
- 报告 15 条与此前 50 条对照：`docs_assets/mask_pipeline/edit_units/reported/`、
  `docs_assets/mask_pipeline/edit_units/previous50/`

### 2. observation 后追加受限 scope review（65 条）

恢复独立双图 observation；对 remove/replace 再输入 instruction 与观察清单，只准按既有
`change_id` 修正 `source_ref`。代码不准删条目、改 target_ref、合并或重排，且保留原清单和修正理由。
运行：`edit_unit_verified_20260923`。

26 条触发复核，但模型 **0 条修正**；碗仍是内容物 ref。自动统计为 54 OK、10 MASK_REVIEW、
1 GROUND_FAIL，无解析/运行错误。问题是已有错误的“slice of pie”描述锚定了 scope 模型，导致它
反复判断无需更改。这个范围过宽、没有纠正目标类别的阶段没有收益，下一轮重写任务边界。

日志与审计：`edit_unit_verified_20260923/{grounding,mask,grounding.log,mask.log,validation_summary.json}`。

### 3. 固定正确碗框，只查 SAM 双 PCS（1 条）

把第一轮 ref 固定为 `bowl with pie slice and fork`，只重跑 SAM 内容恢复探针：
`edit_unit_contents_probe_20260923/mask/replace_00570.parquet`。文本 PCS 与 text+box PCS
各自把 pie/fork 缩成碎点，双路交集无可靠增补；主 mask 仍约 20,650 px 的薄碗沿。
自动 QC 被置为 MASK_REVIEW，以免继续把这类退化说成完整。

图和详细 `candidate_audit_json`：
`docs_assets/mask_pipeline/edit_units/contents_probe/index.html`、`replace_00570_203.jpg`。

### 4. 用内容物自己的框做 PVS，并只在容器内恢复（1 条）

当两路 PCS 没有支持时，对文字 PCS 内容物 mask 求自身框，再用该框独立运行 PVS；只有两路内容物
分割共同支持的像素才可补回，且受碗的空间包络限制。第一版拒绝了跨出碗沿的叉子；第二版仅接受
交集在碗内的部分，不把碗外 handle 加进去。

运行：`edit_unit_content_visual_20260923/mask/replace_00570.parquet`。内容物恢复后面积为
27,618 px；pie 区明显补齐，但 fork 被整块拒绝。图与双提示 audit：
`docs_assets/mask_pipeline/edit_units/content_visual_probe/index.html`。
这一轮说明 PVS 可以找回 PCS 漏掉的实体内容；容器边缘不能作为叉子整件的排除条件。

### 5. 将 instruction 规则泛化成“整物替换都标整物”（51 条）

Scope prompt 改成只要整物被替换就要求整件对象，不要求外轮廓发生变化。回归集中运行 26 条
remove/replace，另加预先看过 raw pair 的 25 条新增验证：`edit_unit_final_20260923`。
51 条全部完成，47 OK、3 MASK_REVIEW、1 GROUND_FAIL。碗 ref 成功修正；同时
`replace_00924.parquet:214` 的鸟头被扩大为整只鸟，违反用户要保留局部编辑单位的要求。
因此撤回对所有对象通用的整物扩大，只对明确的容器名词做 scope review。

中间结果留在 `edit_unit_final_20260923/{grounding,mask,grounding.log,mask.log,validation_summary.json}`。
对照图：`docs_assets/mask_pipeline/edit_units/validation_previous50/` 和
`validation_reported15/`。该组新增样本结果后来被最终 90 条重跑取代，不将中间 mask 作为最终结果。

### 6. 收窄为容器，并补齐局部接缝（最终 90 条验证）

当前规则只对 instruction 明确提及 bowl/plate/cup/mug/basket/tray 的 remove/replace 启用
scope review。失败的鸟头案例不进入该复核。scope patch 仍保留所有观察条目、ID 和 target。

容器 SAM 流程先运行主体名词 mask，再恢复 ref 中明确列出的内容物。首选双路 PCS 交集；交集
不能支持时，使用内容物自身框做 PVS。新增像素限制在容器凸包内，凸包只是空间边界，不直接生成
最终 mask。允许器具横跨碗沿，但只接受碗内且双路支持的部分。最后只在新增内容物附近做 1–3 px
接缝闭运算，增加面积不得超过主体 mask 的 5%；无内容物支持的孔洞不填。

运行目录：
`container_scope_validation_20260923/{grounding,mask,grounding.log,mask.log,validation_summary.json}`。
使用 Qwen3.8-27B、8 张 GPU（4 个 TP=2 vLLM worker），再用 SAM3 8 个单卡 worker；batch size 4，
两阶段进度写入 log 并显示 tqdm。90 条（65 回归 + 25 新增）全部完成，0 runtime error、0 最终
解析错误；scope 触发 5 条、总调用 6 次（1 次格式重试），实际更正 1 条。

`replace_00570:203` 修正 ref 为 `bowl with slice of pie and fork`；最终 mask 30,734 px、
一个主连通块，覆盖碗、派和碗内叉子，自动 MASK_REVIEW → OK。目视仍有少量像素级漏点，不能称为
精确 GT。65 条回归中 48 条与历史 mask 逐字节相同。另一个已知鸟头案例没有扩大为整鸟，但鸟颈
仍可能漏标。

自动 QC 汇总：

| 集合 | 行数 | OK | MASK_REVIEW | GROUND_FAIL |
|---|---:|---:|---:|---:|
| 已知回归 | 65 | 55 | 9 | 1 |
| 新增验证 | 25 | 21 | 3 | 1 |
| 合计 | 90 | 76 | 12 | 2 |

这些是流程旗标，不是 mask 质量通过率。逐图看新增 25 条：13 条大体可用，8 条需修复，4 条
属于原编辑歧义或 source-only 无法完整表示。失败包括加勺子后 mask 为空、摩托车同色组漏实例、
肤色只标头部、容器只标表面而漏内容物、换水壶却把人偶整体框住。逐例判断在
[`validation25_review.json`](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/edit_units/validation25_review.json)。

新增样本 mask 画廊：
[HTML](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/edit_units/validation25/index.html)；15 条报告 case 对照：
[HTML](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/edit_units/validation_reported15/index.html)；此前 50 条对照：
[HTML](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/edit_units/validation_previous50/index.html)。图片均嵌入 HTML。

代表性碗修复对照：

![Replace bowl: source/target and before/after mask](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/edit_units/validation_details/replace_00570_203.jpg)

仍有漏内容物的失败例：

![Remaining container omission](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/edit_units/validation25/remove_01100_34.jpg)

## 容器范围实验后的计划（历史）

1. 先修正 observation 对完整编辑对象及内容物的枚举。`remove_01100:34`、`remove_00893:85` 仍漏
   容器内容物；`replace_01134:104` 把红壶错当整个人偶。加一组盘/碗/壶对照，检验容器附件与邻近
   对象不会互相吞并，然后重跑固定 90 条。
2. 再处理多人皮肤/头部和 motion 手臂多实例（如 `color_00530:105`、
   `motion change_00057:138`）；分别检查 grounding 清单/框和 SAM 候选，避免用填洞修复漏框。
3. 汇报时继续人工查看失败与边界 case。自动 `OK` 与本轮 13/25 人工抽样结论都不能替代更大规模
   标注真值；当前不应据此启动全量新 mask 生产。

## 容器范围实验的代码与复现（历史）

本轮实现集中在：

- `crispedit/mask/checklist.py`、`grounding_runner.py`、`scope.py`：双图 observation、容器范围校正和审计；
- `crispedit/mask/matching.py`：有边界的容器 SAM prompt 简化；
- `crispedit/mask/attachments.py`、`pipeline.py`：容器内容恢复、PVS fallback、局部接缝；
- `scripts/review_crispedit_masks.py`、`compare_crispedit_masks.py`：内嵌式 HTML 和前后对比图；
- `tests/test_crispedit_mask_scope.py`、`tests/test_crispedit_mask_attachments.py`、
  `tests/test_crispedit_checklist.py`：范围约束与内容补全用例。

最终 296 项测试通过。复现使用 [8 卡 pipeline 脚本](../scripts/run_crispedit_mask_pipeline.sh)，
指定 `docs_assets/mask_pipeline/edit_units/validation90_selection.json` 与未使用过的运行目录；
模型和数据路径见 [`CRISPEDIT_MASK.md`](CRISPEDIT_MASK.md)。当前机器上的可用环境需显式指定：

```bash
CRISPEDIT_VLLM_PYTHON=/opt/tiger/tanyue/sam3-crispedit/.venv-scaleedit-vllm/bin/python \
CRISPEDIT_SAM_PYTHON=/opt/tiger/tanyue/sam3-crispedit/.venv-scaleedit-vllm/bin/python \
CRISPEDIT_BATCH_SIZE=4 bash scripts/run_crispedit_mask_pipeline.sh \
  /path/to/new-mask-run docs_assets/mask_pipeline/edit_units/validation90_selection.json
```

## 2026-09-23：指令引导的编辑单元 → 单图 grounding → 上下文 crop

### 当前实现

按用户提出的流程重组，全部 MLLM 使用本地 `Qwen3.8-27B`；保留双 PASS 输入门禁，不改 prefilter。
**当前代码是已跑通的实验实现，尚未通过整体质量验收，不用于全量生产。**

1. **第一轮**输入 source、target、编辑指令和类型，输出实际编辑事件 `change`，每个事件包含独立
   分割单元 `units`。每个单元有两侧短名词 ref、两侧实例位置及 single/nearby_group。
   指令只解释意图，不限制实际变化；整物替换与局部改动分开，分别枚举实例、左右肢体、容器内容物。
2. **第二轮**输入单张待标图及编辑描述/该侧 ref/位置，返回 `change_id + bbox_2d`。
   坐标为 0–1000 的 xyxy；代码检查 ID 不重不漏、数值和范围，不让模型重写 ref。
   add 定位 target；其余类型只定位 source。参考
   [Qwen 官方 grounding 示例](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)
   的简短定位指令和 bbox JSON 格式，加入编辑身份约束。
3. **SAM3**使用第一轮 ref（仅做已有确定性规范化），框每侧扩展 25%、至少 8 px 后 crop；
   框面积 ≥65% 时直接使用整图。保留原有 text-only PCS、text+box PCS、PVS 候选选择及保守清理，
   本轮没有新增大范围填洞。mask 映回原图；非 add 不合并 target mask。
4. 常规路径两轮 MLLM；默认关闭旧的额外 scope/evidence/crop-grounding 复核。解析失败仍有重试，
   所选侧无对象时不调用 grounding。`edit_id/change_id` 从描述、框贯穿到 mask 候选审计。

实现：`crispedit/mask/checklist.py`（prompt/单元展开/框校验）、`grounding.py`（解析）、
`grounding_runner.py`（两轮执行）、`pipeline.py`（单元审计）、
`scripts/run_crispedit_mask_pipeline.sh`（8 卡入口）。当时保留的旧诊断开关已于 2026-09-24 清理移除。
每行 `ground_json` 保存实际 prompt、原始回复、重试、ref 与框，避免只根据方法名猜测当轮配置。

### 验证设计与首轮结果

运行根目录同前文 `CrispEdit-2M-difficult-local-edit-labeling/`。使用 8 卡：Qwen 为 4 个 TP=2
vLLM worker，之后 SAM 为 8 个单卡 worker；batch 4，observation 3072 tokens、grounding 1536。

- 90 条固定回归，与 `container_scope_validation_20260923` 对比。
- 另取 25 条双 PASS 样本，5 类各 5 条，seed=202609231；排除此前样本的整个 shard。
  先看原图并记录 [预期区域](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/fresh25_expected.json)，
  再跑预测。新样本用于小规模验证，不是有像素真值的独立测试集。
- 第一轮结果暴露共同变化遗漏与容器镂空；随后仅修正第一轮 prompt，强调全图比较、实际整物替换
  优先于类型标签、容器与内容物明确分开。第二轮及 SAM 策略保持不变，重跑全部 115 条。

| 运行目录 | 样本 | OK / MASK_REVIEW / GROUND_FAIL | 运行/最终解析错误 |
| --- | ---: | --- | --- |
| `container_scope_validation_20260923`（历史基线） | 90 | 76 / 12 / 2 | 0 / 0 |
| `instruction_units_20260923`（首轮） | 90 | 83 / 5 / 2 | 0 / 0 |
| `instruction_units_fresh25_20260923`（首轮新样本） | 25 | 25 / 0 / 0 | 0 / 0 |
| `instruction_units_complete_20260923`（强化完整范围） | 115 | 108 / 6 / 1 | 0 / 0 |
| `instruction_units_scoped_20260923`（收紧局部范围，当前代码） | 115 | 109 / 4 / 2 | 0 / 0 |

首轮 90 条记录到 92 次 observation（含 2 次重试）+89 次 grounding = **181 次 MLLM 请求**；
历史基线至少 403 次，减少至少 55.1%。这是请求数，不是实测吞吐加速倍数。

人工看首轮旧 25 条：12 大体可用、9 需修复、4 编辑歧义/source-only 限制；基线为 13/8/4。
勺子和肥皂漏标改善，但后墙、警察脸部、编织水果碗出现退步，不能把自动 OK 增多当成总体质量提高。
首轮新 25 条：13 大体可用、10 需修复、2 原图变化不明确。逐条判断：
[旧 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/previous25_review_initial.json)、
[新 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/fresh25_review_initial.json)。

首轮图：[90 条前后对照](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/comparison/index.html)、
[旧 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/previous25/index.html)、
[新 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/fresh25/index.html)。HTML 均内嵌图片。

强化完整范围一轮：旧 25 条为 15 大体可用/6 需修复/4 歧义；新 25 条为 14/9/2。
找回后墙、警察脸部、编织碗内容物，但 `color_00530:105` 和 `motion change_00057:138`
扩大成整人，`remove_01360:184` 又漏了马，`color_00469:119` 右椅靠垫再次镂空。
这些退步未被自动 OK 识别。记录与图：
[90 条对照](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/complete_comparison/index.html)、
[旧 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/complete_previous25/index.html)、
[新 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/complete_fresh25/index.html)、
[旧样本逐例评价](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/previous25_review_complete.json)、
[新样本逐例评价](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/fresh25_review_complete.json)。

这一轮实测 260 次 MLLM 请求（115 条），其中旧 90 条 208 次，对比基线至少 403 次减少至少
48.4%。含 30 次仅为补外层 JSON 对象的 observation 重试。随后解析器兼容**完整**事件数组，
仍逐单元严格校验，不修补截断 JSON。回放 30 条：全部可解析，26 条与纠正回复完全一致，
另 4 条仅 change 描述措辞不同，ref/位置/单元不变。这次解析回放不是新 GPU 推理，不能从
上面的实测调用数中扣除重试。

因此再收紧一轮：肤色/发色/局部动作不能推断为整人替换，保留不变衣服/躯干；共同消失的
乘骑物、承载物和内容物单独列出。与新解析器一起在 `instruction_units_scoped_20260923`
重跑同一 115 条。新 25 条此时已经看过，仅属于迭代验证集，不再声称是未见测试集。

### 当前验证结论：流程跑通，但质量尚不稳定

`instruction_units_scoped_20260923` 已于 2026-09-23 10:57 UTC 完成。115 条全部使用 Qwen3.8-27B，
0 运行/最终解析错误、0 格式重试；113 个非空 mask、187 个实例。共 228 次 MLLM 请求；
其中固定 90 条为 178 次，比历史至少 403 次减少至少 **55.8%**。只比较调用数，不声称相同倍数的
吞吐提升。[机器检查汇总](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_validation_summary.json)。

人工逐图检查同一旧 25 条与新增 25 条，另查看用户此前点名的重点案例。50 条成组审阅结果：

| 方法 | 旧 25 条：大体可用 / 需修复 / 歧义 | 新 25 条：大体可用 / 需修复 / 歧义 |
| --- | --- | --- |
| 历史基线 | 13 / 8 / 4 | 未运行 |
| 指令编辑单元首轮 | 12 / 9 / 4 | 13 / 10 / 2 |
| 强化完整范围 | 15 / 6 / 4 | 14 / 9 / 2 |
| 收紧局部范围（当前） | 11 / 10 / 4 | 12 / 11 / 2 |

这是无像素 GT 的定性审阅，不是全数据准确率。**最后一轮没有优于中间轮或历史基线**，不能
因为 109/115 自动 OK 就验收。当前代码保留用于继续研究，不覆盖历史产物；未启动全量。

当前逐例评价：[旧 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/previous25_review_scoped.json)、
[新 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/fresh25_review_scoped.json)。
可视化：[90 条历史基线对照](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_comparison/index.html)、
[旧 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_previous25/index.html)、
[新 25 条](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_fresh25/index.html)。均内嵌图片。

有改善或保住的案例：鸟头与颈部覆盖更完整；骑手和马分别列出后恢复完整覆盖；椅子靠垫不再镂空；
两件共同变白的衬衫均保留。以下鸟头图按 Source / Target / 历史 / 当前 / 两侧二值图排列：

![鸟头与颈部对照](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_details/replace_00924_214.jpg)

仍失败的案例及定位：

- `remove_01000:197`：第一轮又合并成 `pink crocheted bowl with fruit`，未独立列水果；
  SAM 最终只取碗，重新出现大洞。`remove_00893:85` 的 `white plate with soap` 同类。
- `add_00005:57`：复合 ref `black spoon and chocolate garnish`，框扩到甜点区，最终把罐子也标了。
  这不是缺少扩框，而是第一轮没有遵守分割单元粒度。
- `color_00530:105`：恢复脸/头发/四肢，不再整人；但男孩短裤被带入、女孩一条腿漏掉，仍不能验收。
- `motion change_00057:138`：不再整人，脸部较完整，但手臂仍不完整；需检查 SAM 候选与局部部位语义。
- `remove_01100:34`：盘子及甜点较完整，但杯沿共同消失的草莓仍漏列。多实例观察遗漏仍存在。

失败例（碗有框、内容物却未覆盖）：

![仍失败的编织水果碗](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_previous25/remove_01000_197.jpg)

下一步不继续只堆叠通用 prompt：优先为“一个 ref 混多个独立概念”增加单元级检查与有条件的
重拆分，尤其区分新增物与不变的承载物；再对局部部位检查 text 候选与 PVS/联合候选是否扩大到
躯干，避免融合吞入衣物。共同变化漏列则用局部双图补看验证收益，按需调用而非全量恢复多轮复核。
每步都需要同时守住本次容器、皮肤、动作及乘骑物回归，不能用单个好例代替整组比较。

### 复现本轮

实际运行目录各自包含 `run.sh`、`pipeline.log`、`grounding/`、`mask/`、`validation_summary.json`。
当前验证目录是 `instruction_units_scoped_20260923`，固定选择文件为
`docs_assets/mask_pipeline/instruction_units/validation115_selection.json`。
`pipeline.log` 同时记录两阶段 tqdm。此前实验目录保留，不覆盖既有 mask。

注意：以下为历史复现参数。2026-09-23 人工剔除了 `color_00070.parquet:252`；旧选择文件和
115 条实验统计保留历史原样。使用当前生产 manifest 复跑时，应重选样本或从选择中去除此条，
不能继续期待它为 `SELECTED`。

```bash
CRISPEDIT_VLLM_PYTHON=/opt/tiger/tanyue/sam3-crispedit/.venv-scaleedit-vllm/bin/python \
CRISPEDIT_SAM_PYTHON=/opt/tiger/tanyue/sam3-crispedit/.venv-scaleedit-vllm/bin/python \
CRISPEDIT_BATCH_SIZE=4 bash scripts/run_crispedit_mask_pipeline.sh \
  /path/to/unused-run-dir docs_assets/mask_pipeline/instruction_units/validation115_selection.json
```

新增样本的可复现抽样入口是 `scripts/select_crispedit_mask_smoke.py --fresh-per-type 5
--seed 202609231`，指定当前 quality/difficulty manifest 目录和新 `--output`；使用三个
`--exclude-selection` 排除 `edit_units/validation90_selection.json`、`expanded/selection.json`、
`expanded/unseen/selection.json`（均位于 `docs_assets/mask_pipeline/`）。
画廊用 `scripts/review_crispedit_masks.py --input-dir ... --mask-dir ... --selection-file ...
--output-dir ...`；基线对照用 `scripts/compare_crispedit_masks.py`。

代码验证：当前 Python 环境执行 `python -m pytest tests -q`，308 项通过；
`bash -n scripts/run_crispedit_mask_pipeline.sh`、`git diff --check` 通过。

## 2026-09-23：当前坏例归因与修复优先级

本次只复核已有结果、处理一条用户确认的质量误保留，不改 mask 算法、不重跑推理。
依据是 `instruction_units_scoped_20260923/mask/*.parquet` 中的 `ground_json`、
`instance_masks[].candidate_audit_json`、原图和上面的逐例画廊。
成组审阅 50 条：23 条大体可用、21 条需修复、6 条编辑歧义/源图范围问题；不是全量准确率。
另复查用户重点案例。最新 prompt 没有稳定胜过上一轮，当前仍不适合全量生产。

### 已定点剔除的质量误保留

`color_00070.parquet:252` 前后仍可见花纹，原质量模型却描述为“花纹变为纯色”，人工确认是质量漏筛。
它不再作为 mask 失败例：质量有效结论改为 FAIL/drop，第二阶段有效记录移除；最终双 PASS
由 19,517 变为 **19,516**。保留原始数据和历史实验，原模型响应不改写为人工结论。
质量目录下 `review/manual_exclusion_color_00070_252/record.json` 保存前后校验和、原行及操作原因，
`backup/` 保存两阶段原 audit/manifest shard 和 summary，可恢复。主流程 doc 已同步当前规模。
已遍历核验全部 1,298 shard 的 audit/manifest 结论一致、行号和两阶段 PASS 对齐；原始 shard
SHA-256 未变，修改 shard 的其他行逐值未变，六个修改产物的最终校验和与记录一致。
实际 `prefilter_fields → apply_scene` 读取此行得到 `QUALITY_DROP`；筛选对齐测试 14 项通过。
比较页的 `mask IoU` 只是前后两版预测的重叠率，**不是与像素 GT 的 IoU**，不用于决定剔除。

### 具体原因：观察、拆分、定位、候选选择要分开判断

| 代表案例 | 本次核实的证据和根因 | 优先改进方向（尚未实施） |
| --- | --- | --- |
| `remove_01000.parquet:197` 水果碗 | 描述已说碗和水果均移除，但只有一个 `pink crocheted bowl with fruit` 单元；框基本覆盖，SAM 只留下碗壁。旧容器正则不支持 `pink crocheted`，未进入内容物补全。不是再扩框能解决。 | 把“描述提到的编辑对象”和 units 做一致性检查；碗、水果分别查询分割，最后并集，不靠形容词白名单。 |
| `remove_00893.parquet:85` 盘子和肥皂 | 同样合成 `white plate with soap`；SAM 主查询被简化为 `plate`。内容物补全确实执行，但 `soap` text-only 无候选，记录 `ATTACHMENT_UNSUPPORTED`、added_pixels=0。 | 肥皂独立单元、独立小框和 crop，避免拿整盘大框去约束小内容物；不能把所有洞直接填满。 |
| `add_00005.parquet:57` 勺子 | 两条 ref 均把勺子与配料混合，框扩到罐子。右侧 text-only 为空，joint 面积 10,598，但 `OBJECT_RECOVER_FRAGMENTED_SEMANTIC` 选择了 216,256 像素的 PVS 候选，最终标到整罐。 | 拆开新增物；“补完整”必须服从编辑语义。局部新增/部件不得因候选更连贯就回退到宿主整体。 |
| `color_00530.parquet:105` 儿童肤色 | 输出 `boy's legs`/`girl's legs` 而非左右分开；进入 `aggregate_region/dense`。男孩联合候选 55,953 像素、text 候选 15,651，取前者带入短裤；女孩双腿只给一个窄框，漏另一腿。 | 逐侧可见皮肤单元，框负责实例位置；局部语义候选优先。面积分歧作为复核触发，不用大框完整度替代语义。 |
| `motion change_00057.parquet:138` 击掌变合掌 | 两个人的手臂被合为 `raised arms and hands`，仅一个 `[518,295,600,645]` 框；猴子的手臂框也主要覆盖手。三个脸较完整，但多人手臂没被完整枚举/定位，crop 后 SAM 无法找回远处遗漏。 | 每个 owner、每条运动手臂分开；保留从肩/袖口到手的可见范围。先修单元和框，再讨论 SAM 边界。 |
| `remove_00573.parquet:242` 人和椅子；`replace_00494.parquet:19` 蛋糕托架 | 前者仅列人并错误描述“露出椅子”，实际右椅也消失；后者列三块蛋糕和丝带，没有共同移除/改变的托架。是第一轮观察漏项，不是 SAM 把已指定对象漏分。 | 对消失对象的承载物、内容物及相邻区域做有针对性的双图复核；未变的承载物仍须排除。 |
| `add_00376.parquet:87` 纸屑 | 新 ref 为 `red confetti cloud`，被判 `aggregate_region/dense`：`_SPARSE_REF_RE` 未覆盖 confetti；因此未启用 sparse 的大框分块和保粒子清理。text-only 为空，joint 只取少数团块，cleanup 从 43 个分量减到 11，最新仍自动 OK。 | 修正颗粒路由，按局部高分辨率 crop 查询并保留真实小分量；候选仅有团块时告警，禁止靠整框/凸包填实冒充完整。 |

这些审计值能定位失败发生的环节，但修复收益仍需消融实验验证。尤其纸屑的清理删除仅 983 像素，
**主因还包括候选本身漏检**，不能归咎于后处理一处。肥皂的原图外观类似点心，也存在视觉名词与
指令名词不匹配的可能；先独立框验证，不能把这个推测当作已证实原因。

代表图（Source / Target / mask overlay / binary）：

![局部新增被扩大到整个罐子](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_previous25/add_00005_57.jpg)

![局部皮肤过标与漏标并存](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_previous25/color_00530_105.jpg)

### 50 条审阅中全部 21 条待修案例

以下按主要现象分组，个别案例同时有多种原因；每条原图、mask 和短评见
[旧 25 条画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_previous25/index.html)与
[新 25 条画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/repo_cleanup_20260924/docs_assets/mask_pipeline/instruction_units/scoped_fresh25/index.html)。编号均为原 parquet 行号。

| 现象 | 案例（省略 `.parquet`） |
| --- | --- |
| 容器、内容物或附属物漏标 | `remove_01000:197` 水果；`remove_00893:85` 肥皂；`add_00514:166` 托盘/内容物 |
| 局部范围扩大、混入邻物 | `add_00005:57` 罐子；`add_00714:95` 桥旁植被；`motion change_00030:209` 右侧整人 |
| 人体部位范围或覆盖不足 | `color_00530:105` 短裤过标/女孩腿漏标；`color_00013:122` 警察实际重绘的脸；`motion change_00044:74` 手臂；`motion change_00036:126` 下半身共同变化 |
| 共同变化的物体/区域遗漏 | `remove_01100:34` 杯沿草莓；`replace_01134:104` 石槽；`add_01018:1` 蓝色冰区；`add_01192:243` 前臂；`color_01596:129` 下方鸟尾；`remove_00573:242` 椅子；`replace_00494:19` 托架 |
| 多实例、遮挡或远处组覆盖不足 | `color_00025:65` 后方摩托；`remove_00459:226` 左侧花组；`remove_00223:52` 远处花组；`remove_00953:78` 后方花束 |

其中 `color_00013:122` 说明不能按类型硬套“color 永远局部”：若身份/整体形状实际重绘，应按真实
变化定范围。6 条歧义/source-canvas 样本单列，不算成功，也不为“有 mask”而强行造区域；除用户
明确要求的 `color_00070:252` 外，本轮未删除其他数据。

### 下一轮建议顺序与验收

1. **先固定单元契约**：每个可独立分割实例/部位一条，描述与单元对应；结构合法不等于语义完整。
   复合名词、多 owner、左右肢体未拆时触发局部重拆，不简单按 `and/with` 字符串切词。
   prompt 保持简短，明确“实际变化、单实例/部位、保留宿主”；有问题才额外调用 Qwen3.8。
2. **修 SAM 路由和候选约束**：区分整物、部位、稀疏颗粒。局部部位不能自动用整宿主 PVS 补全；
   confetti 不进入 dense；容器缺的内容物用独立语义 mask 补，不泛化填洞。几何分数高不证明对象对。
3. **最后补遗漏与难框**：只对描述/单元冲突、候选严重分歧或疑似缺边的案例，追加上下文 crop
   grounding 或局部双图复核。不能在 crop 里找回第一轮完全没提到的远处实例；需保留全图定位上下文。
4. **逐项消融验收**：先固定 observation/boxes 比较候选策略，再单独比较观察/定位修复；
   同时守住水果碗、勺子、皮肤、手臂、乘骑、椅垫、鸟头等回归，并加新抽样，分别记录漏标和过标。
   自动 OK 仅作运行信号；记录额外 MLLM 调用数和耗时，不以连通域变少或 mask 变大作为成功。

本轮发现的根本矛盾是：**现有“完整性补偿”有时把错误语义分得更完整，而第一轮漏掉的编辑单元
又无法由后面的 SAM 自动补回。** 优先修这两个接口，比继续统一加长 prompt 或全局扩框更可控。

### 2026-09-23：新增下载数据的 64 条复核

从本日新增下载的 14 个 parquet shard 中，隔离运行质量与细粒度两阶段过滤，得到 100 条双 PASS；
在看 mask 结果前固定随机种子抽取 64 条（add 20 / color 6 / remove 23 / replace 15；本批没有新的
motion-change shard，color 只有 6 条可选）。使用当前 Qwen3.8-27B 双轮定位 + 8 卡 SAM3 打标，
64/64 完成，结构校验 OK 60、MASK_REVIEW 4、非空 mask 62，解析与运行错误均为 0。

[完整源图/目标图/mask 画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/mask_newdownload64_20260923/review/index.html)、
[逐例复核报告](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/mask_newdownload64_20260923/review_findings.md)、
[运行日志](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/mask_newdownload64_20260923/pipeline.log)。
图例中 `remove_00774:49`、`remove_00145:96` 等孤立或成组实体定位较好；但 `add_00248:46`
奶酪与 `add_00777:174` 纸屑为空 mask，`add_01105:91` 呼啦圈定位错误，`color_01635:254`
漏掉紫色大写 S，`color_01635:179` 漏掉右容器，`remove_00774:67` 漏掉第二棵树。
后四类说明自动 OK 不等于视觉正确，尤其第一轮漏掉的共同变化实例后续无法补回。
本次只做验证和问题归因，未修改打标策略，也不报告没有真值支持的 IoU 或准确率。

### 2026-09-23：已双 PASS、此前未做 mask 的 65 条（用户指定的“新数据”）

为避免将刚下载的源数据与用户指定样本混淆，另从既有两阶段过滤结果中抽取此前未做过 mask 的
65 条：add、color、motion change、remove、replace 各 13 条。使用相同当前流程打标，65/65 完成；
结构校验 OK 63、MASK_REVIEW 1、GROUND_FAIL 1，64 条非空，107 个实例 mask，解析与运行错误为 0。

[65 条完整画廊](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/mask_fresh65_doublepass_20260923/review/index.html)、
[目视复核及逐例原因](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/mask_fresh65_doublepass_20260923/review_findings.md)、
[运行日志](/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/iterations/experiments/mask_fresh65_doublepass_20260923/pipeline.log)。
全部 17 页已检查。`add_00758:10` 三只小鸭、`color_00035:227` 两组义肢、`remove_00869:28`
多人移除等基本合理；但自动 OK 中仍有 `add_00076:27` 树冠过标、`color_01425:123` 新娘裙漏标、
`replace_00186:214` 保留人物被误标等问题。此组才是用户所指的“未做过 mask 的双 PASS 样本”；
上节新增下载数据的 64 条是额外实验，不替代本组结论。
