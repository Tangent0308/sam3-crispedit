# SA-1B SAMTok 编辑数据构造

基于 SA-1B 图像与 dense mask 构造局部编辑数据，覆盖 `add`、`remove`、`replace`、`attribute` 四种操作。指令协议 2.0（`benchmark-concise-3-blind-1`），mask 语义 2.0。

## 数据链路一览（2026-10-28）

| 链路 | 一条样本 | 规模 | 状态 | 主文档 |
|---|---|---|---|---|
| 孪生（twins） | 同一原图两个目标、同一操作，各自一张结果图 | 12,511 组 = 25,022 条，四种操作各约 3,128 组 | 已完成 | [TWINS_MULTI_ZH.md](docs/TWINS_MULTI_ZH.md) |
| 多目标（multi） | 一张原图上 2–4 个目标，全部编辑进一张结果图 | 100,117 个目标 / 39,814 张图，四种操作各约 2.5 万 | 已完成 | [TWINS_MULTI_ZH.md](docs/TWINS_MULTI_ZH.md) |
| 附带单目标（孪生/多目标生产的副产品） | 原图 → 结果图，一个目标 | 12,363 条（不均衡） | 已完成 | [TWINS_MULTI_ZH.md §5](docs/TWINS_MULTI_ZH.md#5-生产结果) |
| 单目标 v2.1 | 原图 → 结果图，一个目标 | 暂定 5 万，四种操作各 12,500 | 已打包，待提交 Arnold | [SINGLE_TARGET_ZH.md](docs/SINGLE_TARGET_ZH.md) |
| 单目标 50k（旧管线） | 原图 → 结果图，一个目标 | 50,000 条，四种操作各 12,500 | 已完成，正式 release | [DATASET_50K_ZH.md](docs/DATASET_50K_ZH.md) |

所有数据都是自动审核通过的版本，没有做独立人工验收。各文档写明了助手盲评和抽查结果，以及由谁判定。

## 各管线做什么

**证据管线（孪生、多目标、单目标 v2.1 共用）**

每台机器循环跑“波次”，每波 6 步：
1. **选样**：按规则挑图和目标，不调模型（见下节“源图怎么选”）。
2. **规划**：Qwen3.8-27B 逐 mask 规划，输出指代（Ref）、编辑内容和 no-ref 指令。
3. **证据门**：27B 复核规划能否成立，不过的目标不进入生成。复核六项：
   - 读者：只看 Ref 能否找回目标；
   - 身份：是否同一物体；
   - 内容：替换是否真的换了物体；
   - 光照、介质、视角。
4. **生成**：Qwen-Image-2.1，40 步，只写执行 mask 内的区域。之后做收尾：删除时用 SAM3 一并去掉投影和倒影，并补颗粒、匹配景深；含文字的编辑用 OCR 核验。
5. **审核**：27B 主审核，再由核验器按操作类型问几个缺陷问题。核验器只会拒绝，不会放行。
6. **导出**：按四种操作均衡导出。

三条链路只在生成和接受规则上不同：

| 链路 | 生成 | 接受 | 生产入口 |
|---|---|---|---|
| 孪生 | 两个成员各自从原图编辑，得到两张图 | 两个成员都通过，整组才接受 | `scripts/run_evidence_production_4node.py` |
| 多目标 | 独立目标并行编辑后逐像素拼接，有冲突的串行编辑；合成一张图 | 保留通过的目标，至少 2 个且组合合法 | 同上（与孪生同一个生产） |
| 单目标 v2.1 | 每个目标从原图单独编辑 | 每个目标单独审核；导出前逐像素证明支持区外与原图完全相同 | `scripts/run_single_production_4node.py` |

**旧管线（单目标 50k，已冻结）**
- Flash-Next 规划每个 mask，操作盲分配，指令协议 1.0；
- Qwen-Image-2.1 40 步编辑，27B 审核，没有证据门和核验器；
- 入口 `scripts/run_production_4node.py`。

## 源图怎么选

都是 SA-1B，规则是**先随机排序，再按规则筛选**，不是完全随机抽取。

**证据管线（孪生/多目标、单目标 v2.1）**
1. **分片顺序随机**：SA-1B 共 1,000 个分片，按固定种子随机打乱后轮流分给四台，各台的分片互不重叠（`production_pool.shard_order`）。每台按自己的顺序整片读入，用完一片再读下一片。
2. **排除**：
   - 两条链路共同排除 benchmark 200 张和评测预留 1,200 张；
   - 单目标 v2.1 另外排除 50k 正式数据的 1,132 张源图和单目标验证池，共 17,399 个来源；
   - 单目标避开孪生/多目标每台分片列表的前 48 片。孪生/多目标实际每台只用了 3–4 片，所以两次生产的源图不重叠；
   - 同一次生产内，图像哈希跨机器去重。
3. **规则筛选**：每张图算一份“菜单”，说明这张图能做哪些操作。规则有：
   - 物体：mask 质量分 ≥ 0.6，面积占全图 0.15%–30%（删除 ≤ 2%）；同类竞争物 ≤ 4 个；
   - 可编辑性：27B 判断该类别能否自然地新增、删除、替换、改属性；
   - 场景：不删穿戴物和覆盖层，不改天空、水面、地面的颜色，跳过黑白照片；
   - 孪生的两个成员：分离度至少 5%，面积比不超过 3。
4. **按缺口取图**：在打乱后的图像顺序里（`select_evidence_groups.py`，固定种子），每波从最缺的操作开始，依次取“菜单能满足该操作”的下一张图。
   - 结果：哪些图被选中是随机的，但只有满足规则的图才有机会，且会偏向能做稀缺操作的图，例如孪生的新增。
   - 每张图只用一次：孪生和多目标之间没有重叠的源图。

**旧 50k**
- 从 SA-1B 索引中按固定种子随机抽 16,000 张作为候选池，排除 benchmark（`production_sources.prepare_reservoir`）；
- 最终用到 1,132 张源图，每张平均 44 个 mask 样本。

**交叉核对**（2026-10-28）：孪生/多目标的 52,325 张源图里，有 8 张也是 50k 的源图，81 张在单目标验证池里，编辑的目标不同。原因是孪生/多目标生产当时没有排除这两部分。

## 结果路径（已核对）

| 数据 | 路径 | 行数 |
|---|---|---|
| 孪生 | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025/final/twins_balanced.jsonl` | 12,511 组（每行一组，2 条样本） |
| 多目标 | `…/sa1b_samtok_group_twins50k_multi100k_20261025/final/multi_balanced.jsonl` | 39,814 组（每行一张图，2–4 个目标） |
| 附带单目标 | `…/sa1b_samtok_group_twins50k_multi100k_20261025/final/singles.jsonl` | 12,363 |
| 孪生/多目标统计 | `…/sa1b_samtok_group_twins50k_multi100k_20261025/final/summary.json` | — |
| 单目标 v2.1（生产后） | `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_single50k_20261026/final/singles_balanced.jsonl` | 50,000 |
| 单目标 50k | `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014/release/annotations/train.jsonl` | 50,000 |

- 孪生/多目标的图片和 mask 没有复制，记录里是绝对路径，指向生产目录下的 `nodes/node<r>/waves/…`。不要删除或移动这些生产目录。
- 50k 的媒体用相对路径，根目录是 `release/`。
- 孪生/多目标生产目录名沿用最初计划（twins50k_multi100k），实际是孪生 2.5 万条、多目标 10 万个目标。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/TWINS_MULTI_ZH.md](docs/TWINS_MULTI_ZH.md) | 孪生 ＋ 多目标：设计、代码、生产结果、数据读取、质量、示例、已知问题 |
| [docs/SINGLE_TARGET_ZH.md](docs/SINGLE_TARGET_ZH.md) | 单目标 v2.1：设计、代码、四机入口、验证结果、示例、已知问题 |
| [docs/DATASET_50K_ZH.md](docs/DATASET_50K_ZH.md) | 50k 正式数据：统计、字段、读取、示例 |
| [docs/FOURNODE_RUN_GUIDE.md](docs/FOURNODE_RUN_GUIDE.md) | 各次四机运行的入口指令、进度、恢复；后半为历史部署 |
| [docs/PIPELINE.md](docs/PIPELINE.md) | 按时间记录的方法细节与对照实验（含早期分组数据方案） |
| [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md) | 迭代留痕；README 旧的“最新进展”段落也在这里 |
| [docs/CLEANUP_CANDIDATES_50K_ZH.md](docs/CLEANUP_CANDIDATES_50K_ZH.md) | 50k 相关的清理记录 |

四机入口：
- 孪生/多目标：[FOURNODE_RUN_GUIDE.md#入口指令](docs/FOURNODE_RUN_GUIDE.md#入口指令)
- 单目标 v2.1：[SINGLE_TARGET_ZH.md#入口指令](docs/SINGLE_TARGET_ZH.md#入口指令)（5 万，已打包）
- 50k：`scripts/run_production_4node.py`（见四机指南后半部分）

展示用的报告、拼图和 PDF 在 `/opt/tiger/tanyue/visualization_tmp/<实验名_日期>/`。实验目录的总索引在 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/README.md`：`01_shared_inputs`、`02_production_runs`、`04_current_validation`、`05_history`。

## 代码目录

```text
sa1b_pipeline/                         核心库
  evidence_selection.py, production_pool.py, shard_feeder.py    选样与源图池（证据管线）
  per_mask_planning.py, content_planning.py, group_planning.py  规划
  evidence_gates.py, text_edits.py, text_verification.py        证据门、显式文字
  twin_streaming.py, twin_contract.py, twin_compact_audit.py    孪生生成与审核
  multi_parallel.py, multi_transactional.py, multi_component_audit.py  多目标生成、拼接与审核
  single_parallel.py, single_planning.py, single_export.py      单目标 v2.1
  edit_finish.py, removal_effects.py, edit_verifier.py, audit_reconcile.py  收尾、核验器、Ref 核对
  inference_services.py                常驻模型服务池
  （其余为 50k 旧管线与早期分组实验所用模块）
scripts/
  run_evidence_production_4node.py, run_evidence_waves.py, run_dual_validation.py  孪生/多目标生产
  evidence_production/                 孪生/多目标 Arnold 启动与旁挂库
  run_single_production_4node.py, run_single_waves.py, run_single_validation.py, run_single_transactional.py  单目标 v2.1
  single_production/                   单目标 Arnold 启动
  select_evidence_groups.py, run_group_pipeline.py, run_twins_streaming.py, run_multi_transactional.py  各阶段单独入口
  run_production_4node.py, production/ 50k 旧管线四机入口
  group_production/                    共用的 CUDA 12.9 锁定环境安装
requirements/production/               锁定运行时
tests/                                 回归测试
docs/                                  文档，docs/assets/ 为文档内示例图
```

## 快速检查

```bash
python -m pytest -q tests                      # 全量（已知 1 个改动前就有的失败）
python -m pytest -q tests/test_evidence_waves.py tests/test_single_production.py tests/test_single_parallel.py
python3 scripts/run_evidence_production_4node.py --root <孪生多目标运行目录> status
python3 scripts/run_single_production_4node.py --root <单目标运行目录> status
```

## 已完成的 50k 数据

本批共生成并审核 79,821 条，自动通过 **50,000 条**，add/replace/remove/attribute 各 12,500 条，覆盖 1,132 张源图。推荐训练入口：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014/release/annotations/train.jsonl
```

图片、源区域 mask 和执行 support mask 都在同一 `release/` 内，所有媒体相对路径以 `release/` 为根。统计、字段协议、读取代码及每类三例的可视化见 [50k 数据说明](docs/DATASET_50K_ZH.md)；源记录缓存 A 已按确认删除，其他候选保留，见 [清理记录](docs/CLEANUP_CANDIDATES_50K_ZH.md)。数据为自动审核通过版本，尚未完成独立人工验收，文档保留了抽查发现的问题。

![50k 数据统计](docs/assets/production_50k/statistics.png)
