# 50k 数据清理记录与剩余候选

## 已完成归档（2026-10-20，本页最新状态）

用户列出的115个根目录已处理：114个整目录迁入 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline`；50k仅保留正式release，其12项工作文件也归档。本轮删除2,870个可再生pyc，49,327,300逻辑字节；此前36.106GiB清理不重复计入本轮。其余源图、mask、核心清单、日志、代码及代表结果保留。431个软链接已重定向。

正式8份标注/统计哈希、134份冻结核心文件哈希、151,132项媒体与原文件身份、另53,438项协议媒体及15,994张生产源图检查全部通过。逐项旧/新路径、删除和移动回执见 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/00_admin/migration_20261020/`。本页下方是按时间保留的清理历史，不代表当前目录仍在datasets，也不能继续执行旧候选清理表。

## 115 个根目录的清理后复核（2026-10-20）

上轮删除的是26,687个可再生派生图片，逻辑大小36.11 GiB，目录和核心文件保留。本轮按用户列出的115个根目录再分类：正式生产/部署3个、共享输入6个、当前验证/基线6个均保留；`sa1b_edit_work`只能部分清理；历史实验2个仍有存活的跨根精确文件引用，97个可在保留方法记录、代表结果及唯一源数据后归档并退出工作目录。后97个剩余约8.99 GiB，为旧扫描减删除回执的估计值，并非重新实时测量。

正式保护：`sa1b_samtok_edit_50k_20261014`、`sa1b_samtok_group_train_20k_20261016`、`sa1b_samtok_twins_15k_27b_single_20261017`。历史但仍被精确引用的两个根目录是`sa1b_samtok_twins_blind_v2_compact12_20261015`和`sa1b_samtok_twins_blind_v2_holdout12_20261015`，共11个文件；`sa1b_edit_work`另有之前保留的引用和核心子目录。

完整逐项表：`/opt/tiger/tanyue/visualization_tmp/sa1b_directory_review_20261020/目录清理建议_完整115项.md`；可筛选CSV：同目录`全部115个目录分类.csv`。本轮没有新增删除，也没有生成整目录删除白名单：文件引用扫描不能替代对动态路径、唯一源数据和外部仓库的最终迁移检查。

## 2026-10-20 已授权执行：历史派生媒体清理

用户明确授权清理可清理文件，同时保护正式、核心、关键文件。本次重新检查53,621份文本中的文件引用，核验候选文件身份、单链接及无符号链接路径，仅逐文件unlink，不递归删除目录。实际删除 **26,687个文件，逻辑38,768,495,438字节（36.106 GiB）**，无删除报错。144份受保护关键清单、配置、入口前后SHA256全部一致；该结果不表示对全部正式媒体逐像素哈希验收，也不等于实际存储回收量。

此前39.167 GiB候选未全部删除：598个历史跨实验引用文件继续保留，复核新发现引用的18个文件继续保留，另保留2,022个交付可视化/对照图。正式50k、group_train生产进度、单机部署、共享输入池、当前基线、源图、mask/support、JSON/JSONL、日志、源码、PDF及原先三份旧基线均不在删除范围。

最大清理来自 `sa1b_edit_work/`（21.936 GiB），其次quality_repair（1.339 GiB）、multi_evidence_validation（1.151 GiB）、ranked_generation_20case（1.102 GiB）、speed_local（1.055 GiB）。旧实验元数据保留，部分派生图路径现已失效；不能再将旧清单视作当前文件存在性清单。

回执目录：`/opt/tiger/tanyue/visualization_tmp/qwen_tools_cleanup_20261020/`。`cleanup_receipt.jsonl`逐项记录删除，`cleanup_result.json`记录汇总，`protected_core_before.json`保留关键文件哈希，`cleanup_postcheck.json`记录删除路径消失及显式保留路径存在性检查，中文说明为 `清理执行记录.md`。

## 2026-10-20 补充：历史实验输出盘点，未执行删除

已枚举数据根目录全部115个 `sa1b_*` 目录，113个完成逐文件盘点，包含全部历史实验。正式50k与共享group_inputs整棵保护，不纳入本轮清理，容量未作完整重扫。正式50k的release标注/媒体清单、inputs和部署脚本另外完整流式检查，未检出指向其他现存SA1B根目录的绝对路径依赖。

提出29,325个历史派生图片、39.167 GiB的**分文件候选**：37.716 GiB未检出跨目录引用；1.451 GiB仍被其他历史实验引用，需一起归档。没有删除任何文件。排除正式任务、共享池、当前对照、递归引用的文件、源图/mask/support等，以及全部JSON/JSONL、日志、代码、归档和PDF。命名/绝对引用检查不能完全识别其他项目及运行时动态路径，执行前仍须核对。

最大的候选为 `sa1b_edit_work/` 内22.133 GiB，整目录实际29.205 GiB；保留其中full30、add_anchor_concise及instruction_concise三份基线。其次为quality_repair约1.422 GiB、speed_local约1.282 GiB、multi_evidence_validation约1.240 GiB、ranked_generation_20case约1.192 GiB。这些数均不是整目录删除许可或实际空间回收承诺。当前grounded_guard仍引用dual_aligned_opt，release_candidate仍引用grounding_ab，已保护。

完整中文报告、115目录CSV、逐文件候选JSONL、引用索引与只读复现脚本：`/opt/tiger/tanyue/visualization_tmp/specialist_routing_inventory_20261020/`，主要文件为 `数据目录清理盘点.md`、`directory_inventory.csv`、`cleanup_candidates_files.jsonl`、`inventory_validation.json`。以下保留2026-10-15的历史记录；旧B/C/D/E不能与本轮候选相加计算释放空间。

检查时间：2026-10-15。正式数据已经整理到 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014/release`，媒体由硬链接保存，50k 引用和与原文件同 inode 检查通过。用户已确认仅删除 A，现已完成：移除 `source_cache/` 的 4,004 个文件，逻辑大小 18,724,479,695 字节（17.439 GiB）。B/C/D/E 未删除。正式数据的标注哈希与媒体完整性复核见 `release/docs/post_cleanup_validation.json`；逐文件删除记录见 `release/docs/cleanup_A_receipt.json`。这里记录逻辑大小，不将它等同于存储服务报告的实际回收容量。

## 1. 已执行范围与剩余候选

下面 `RUN_ROOT` 为 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014`。

| 编号 | 候选路径 / 范围 | 已测逻辑大小 | 删除影响及前提 |
| --- | --- | ---: | --- |
| A（已删除） | `$RUN_ROOT/source_cache/` | 17.439 GiB，4,004 个文件 | 原始 TFRecord 供给缓存，正式训练不依赖它。删除后旧四机入口的缓存校验和未使用 reservoir 的重新物化不能直接进行；若要继续旧 run 补量，需要重新构建缓存。已按用户确认删除；正式训练不受影响。 |
| B | `$RUN_ROOT/nodes/node*/waves/*/planning_*/think_256/views/` 下仅 `annotated_crop.png`、`scene_locator.png` | 所属 36 个 planning 目录合计 255.187 GiB；views 子目录未单独统计，不能将该数当精确释放量 | 仅删除送入模型的定位全景和标注 crop；保留全部 `source_mask.png`（包括失败样本）及其他文件。正式训练和原始通过/失败样本的图与 mask 不受影响；旧规划图片输入需要从原图与 mask 重建。保留 planning 下 JSON/JSONL、journal、code_snapshot、run 记录，不能整目录删除 views。 |
| C | `$RUN_ROOT/nodes/node*/waves/*/generation/panels/` | 包含在 18 个 generation 目录合计 361.810 GiB 内；panels 未单独测量 | 自动审核拼图可重建，但有复核价值。建议独立人工验收前暂留，不纳入第一批删除。不要误删同级 units、results、dispatch 和 validation。 |
| D | `sa1b_edit_work/` 内已废弃实验的图片/中间裁剪（见下表） | 25 个旧实验目录总计 29.205 GiB；不是建议整目录删除的大小 | 正式训练不依赖这些旧媒体，但历史质量报告、失败对照和旧 JSONL 可能引用它们。先保存统计/配置/计划/审核 JSONL/已有 PDF，再仅删确认废弃的图片缓存。当前 30 图质量基线及 add/指令优化基线建议保留。 |
| E | `$RUN_ROOT/deployment/history/` | 0.000739 GiB（约 0.76 MiB） | 旧启动快照；体积很小且解释故障修复，建议保留，无需为了清理而删除。 |

**A 已完成，B/C/D/E 保留。** 本批旧四机入口需要源缓存，当前不应直接重启；如需恢复或扩量，先重建并校验缓存。规范 `release/` 的训练读取不依赖源缓存。

以上大小是逐文件 `st_size` 汇总。规范整理使用硬链接，删除旧链接不会释放仍被 `release/` 引用的数据块；某些规划 mask 会继续占用空间。目录逻辑大小、唯一 inode 的占用和最终释放量不同，不能相加宣称可释放多少磁盘。扫描期间规范硬链接正在建立，清单中的 `single_link_bytes` 仅是当时快照，不作为精确释放估计。

## 2. 旧实验目录明细

共同前缀：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/05_history/legacy/sa1b_edit_work/`。这些目录均不是正式 50k 训练入口。本次没有确认其他项目完全不依赖它们，因此不授权整棵树删除。

| 目录 | 逻辑 GiB | 文件数 | 建议 |
| --- | ---: | ---: | --- |
| `planning_iteration_20261012/` | 19.626 | 32,998 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `full30_concise2_20261013/` | 2.855 | 4,664 | 保留质量基线 |
| `quality_iteration_v1/` | 1.108 | 1,048 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `pilot_100/` | 0.771 | 480 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `disjoint_mllm_plan30/` | 0.649 | 296 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `per_mask_expanded_v2_2_20261012/` | 0.620 | 1,498 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `add_anchor_concise_20261013/` | 0.580 | 1,161 | 保留质量基线 |
| `quality_iteration_v2_vllm_omni_region_contract/` | 0.508 | 464 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `per_mask_paired_v2_3_20261012/` | 0.503 | 1,031 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `per_mask_paired_v2_4_20261012/` | 0.492 | 1,033 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `expanded_v3_20261012/` | 0.347 | 479 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `operation_v4_20261012/` | 0.254 | 339 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `natural_edits_v1_20261012/` | 0.203 | 292 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `protocol1_v5_20261012/` | 0.192 | 262 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `disjoint_pool/` | 0.130 | 3 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `specialists_v1_20261012/` | 0.115 | 149 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `per_mask_single_v2_20261012/` | 0.075 | 165 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `per_mask_v1_20261012/` | 0.061 | 128 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `instruction_concise2_20261013/` | 0.055 | 126 | 保留质量基线 |
| `specialists_v2_20261012/` | 0.047 | 66 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `visual_examples/` | 0.009 | 17 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `benchmark_v2_mllm/` | 0.002 | 5 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `disjoint_mllm_v3/` | 0.001 | 3 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `disjoint_mllm_v2/` | 0.000 | 1 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |
| `disjoint_mllm/` | 0.000 | 1 | 旧实验媒体候选；先保留 JSON/JSONL/配置/报告 |

其中 `planning_iteration_20261012/`（19.626 GiB）是旧实验中最大的候选；`pilot_100/`、`quality_iteration_v1/`、`quality_iteration_v2_vllm_omni_region_contract/`、`per_mask_*` 等也可以在保留结论与标注记录后清理媒体。实际执行前需要按文件级引用筛选，并记录删除清单，不能仅依据名字有旧日期就删除。

## 3. 目前应保留

- `release/` 全部内容：50k 正式数据、统计、可视化和使用文档。
- `nodes/node*/validation.jsonl`、`auto_pass.jsonl`、`done.json`，以及 `final/`：分别记录全部 79,821 条结果、节点通过清单及原始最终导出。
- 全部失败样本的目标图与审核结果：本次抽查已经发现自动审核漏检，建议先完成质量分析再考虑淘汰；不要整个删除 `generation/`。
- `config.json`、`inputs/`、当前 `deployment/code.tar.gz` 与校验和、规划原始回复和 journal：协议、来源和复现依据；`inputs/` 也用于后续分组数据隔离，不能作为纯缓存删除。
- `sources/`：原图和 dense 记录可能还被分组数据准备流程引用，虽然正式训练已使用 release 内图片，暂不删除。
- `verification/`（0.527 GiB）与 `logs/`（约 28 MiB）：保留 CUDA 兼容性和正式任务的诊断证据。
- `sa1b_samtok_group_*`、`sa1b_samtok_twins_*`、`sa1b_samtok_flashnext_*`：属于另行推进的新实验，本次没有将其判作废弃数据，也没有纳入删除候选。
- benchmark、模型权重、运行环境和其他数据集：不在本次清理范围。

## 4. 确认和执行方式

用户已明确批准 A；执行前检查了四节点完成状态、正式媒体无 source_cache 引用，并记录逐文件清单。仅删除源缓存，保护正式 release、原始图片、mask、全部审核记录和其他实验。剩余候选需要另行确认，本次没有得到删除授权。

测量证据：`$RUN_ROOT/control/storage_inventory_20261015.json`，规范目录副本为 `release/docs/storage_inventory_20261015.json`。共扫描 83 个互不包含的目标目录，扫描错误数 0。
