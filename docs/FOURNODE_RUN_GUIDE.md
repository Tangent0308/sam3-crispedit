# 运行指南（单机新入口与四机历史部署）

## 孪生约 2.5 万个样本 ＋ 多目标约 10 万个样本：四机生产（证据管线，2026-10-25）

当前推荐的大批量入口。配置与完整出图 5 相同（证据核验、文字核验、编辑收尾、审核核验、优先级调度，40 步），四台各自跑波次，最后合并。代码：`scripts/run_evidence_production_4node.py`，部署脚本在 `scripts/evidence_production/`。

- 实验目录（下文 `$ROOT`，目录名沿用最初的 50k/100k 计划）：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025`
- 已冻结（2026-10-25）：配置 `config.json`，代码包 `deployment/code.tar.gz`（sha256 `a1e49b9e…`）。还没有提交 Arnold。
- 配额（`config.json` 的 `quotas`，数量是大致目标）：
  - 孪生约 **25,000 个样本** = 12,500 组：每组两张目标图。每台每种操作 782 组，按自动通过的组数均衡；全部完成时 12,512 组（25,024 个样本）。
  - 多目标约 **100,000 个样本** = 100,000 个目标（每组 2–4 个目标）：每台每种操作 6,250 个目标。
- 停止规则（省时间）：每台每类剩余缺口不超过 2 个单位（孪生组 / 多目标目标）时不再开新波次，导出相应少最多 2 个。
- 每台的流程：
  1. **源图池**：按本 rank 固定的 SA-1B 分片顺序按需整片读入（`sa1b_pipeline/shard_feeder.py`，排除 benchmark 200 张和评测预留 1,200 张，图像哈希跨 rank 去重）。新类别由 27B 判断可编辑性（不思考，只判断能通过分数和面积预筛的物体类别）。然后计算每张图的“菜单”：每种操作能否组成孪生组，多目标能用哪些操作（`evidence_selection.menu`）。
  2. **每波**：
     - 按剩余需要和实际通过率选样。菜单先过滤，行在 32 个进程里构建，与串行选样结果一致。
     - 每波约 3 万 GPU·秒的工作量，孪生和多目标按剩余量分配。
     - 然后跑完整的规划、生成和审核（`run_dual_validation.py`），规划图 16 进程并行准备。
  3. **导出与合并**：结束后本 rank 均衡导出（多目标按组合类型精确求解），rank 0 等所有 rank 完成后合并到 `$ROOT/final/`。
- 资源：
  - 4 × 8 H100 80GB。
  - 每台本地盘至少 300 GB（模型副本和环境）。
  - 挂载 `/mnt/bn/strategy-mllm-train`。
  - SA-1B 的 HDFS 挂载 `/mnt/hdfs/sg_byte_ttlive_strategy_llm` 可选：没有时节点等待 `prefeed` 读入的分片。

### 入口指令

**Arnold 任务**：4 workers × 8 H100，挂载 `/mnt/bn/strategy-mllm-train`。四台的启动命令完全相同，平台注入 rank（`ARNOLD_ID`），不用 torchrun，不设 `CUDA_VISIBLE_DEVICES`。

```bash
#!/usr/bin/env bash
set -euo pipefail
exec bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025/deployment/arnold_entry.sh
```

`arnold_entry.sh` 的内容如下，也可以直接把这几行填进 Arnold：

```bash
#!/usr/bin/env bash
set -euo pipefail
export EVIDENCE_RUN_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025
export EVIDENCE_RUNTIME_ROOT="${EVIDENCE_RUNTIME_ROOT:-/opt/tiger/sa1b-evidence-production-runtime}"
exec bash "$EVIDENCE_RUN_ROOT/deployment/launch_4node.sh"
```

launcher 依次做这些事：
1. 检查 4×8 拓扑、代码包和配置校验和；
2. 解包到 `/opt/tiger/sa1b-evidence-production-code/<sha>`；
3. 安装两套锁定环境（`scripts/group_production/setup_runtime.sh`，CUDA 12.9）；
4. 安装两个旁挂库（`scripts/evidence_production/setup_extras.sh`：RapidOCR 用于文字核验，crc32c 和 tfd-utils 用于读分片）；
5. 检查字体；
6. 把三个模型复制到 `/opt/tiger/sa1b-evidence-model-cache`；
7. 启动本 rank。

**提交前（开发机，可选）**：若 Arnold 节点没有 HDFS 挂载，先在本机读入分片。每片约 10 分钟，按 rank 轮流；每台约需 4 片，读 5 片留余量，共约 1.5–2.5 小时、260 GB。可以和生产同时跑，节点会等待。

```bash
cd /opt/tiger/tanyue/sa1b-samtok-edit-data
ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_twins50k_multi100k_20261025
nohup python3 scripts/run_evidence_production_4node.py --root "$ROOT" prefeed --shards-per-rank 5 > "$ROOT/logs/prefeed.log" 2>&1 &
```

**本次冻结所用的命令**（已执行，不需要再跑）：

```bash
python3 scripts/run_evidence_production_4node.py --root "$ROOT" prepare --twin-samples 25000 --multi-samples 100000 --replace-unstarted
python3 scripts/run_evidence_production_4node.py --root "$ROOT" bundle
```

- `bundle` 打包当前工作区代码，记录代码、配置和模型清单的校验和。
- 改配额或代码：在任何节点启动前执行 `prepare ... --replace-unstarted`，再 `bundle`。旧配置和代码包留档到 `planning/replaced_<时间>/`。

### 进度

```bash
tail -F "$ROOT/logs/progress.log"          # rank 0 每 10 分钟一行：各 rank 波数、进度、ETA
python3 scripts/run_evidence_production_4node.py --root "$ROOT" status
tail -F "$ROOT/logs/node0.log"             # 每台完整日志
python3 -m json.tool "$ROOT/nodes/node0/waves/progress.json"   # 本 rank 的目标、可导出数、下一波计划
python3 scripts/run_evidence_production_4node.py --root "$ROOT" merge --partial   # 当前快照导出到 final_partial/
```

每个 rank 的波次在 `nodes/node<r>/waves/waveNN/`（选样、规划、生成、审核、计时），台账 `ledger.json`，分片池日志 `pool_log.jsonl`。四台完成后数据在 `$ROOT/final/`：`twins_balanced.jsonl`、`multi_balanced.jsonl`、`singles.jsonl`（单目标，另行使用），`summary.json`。

### 中断恢复

- 重新提交同一入口即可。波次从检查点续跑，分片、标签、菜单按标记续做。
- 波次协调器失败后自动重试 3 次，间隔 2 分钟。
- 上一个 Arnold trial 留下的锁会自动留档后接管。同一 trial 内跨主机残留的锁，确认旧进程已停止后执行 `recover-lock --rank <r> --previous-workers-stopped`。
- 不要同时提交两个写同一目录的任务。
- 某个组的数据错误（例如一张图上两个编辑区域在编辑后重叠）只记入该次运行目录下的 `failed_groups.jsonl` 并跳过该组。超过 5% 的组出错才停止这一波。
- **代码热修复**（已启动的运行，配置和输入不变）：
  ```bash
  python3 scripts/run_evidence_production_4node.py --root "$ROOT" bundle --hotfix "原因"
  ```
  - 旧代码包留档到 `planning/hotfix_<时间>/`。
  - 各节点重启后使用新代码。
  - 用旧代码开始、尚未完成的那一波会被改名为 `run.superseded.<时间>` 并从规划重新跑；已完成的波次不受影响。
  - 2026-10-25 第一次热修复：rank 3 第 1 波的多目标组在融合时两个编辑区域重叠，旧代码因此报错并关掉了共享服务。

### 冒烟与时间

- 单机冒烟：`$ROOT/planning/smoke2/`，经 `launch_4node.sh` 启动，`world_size=1`。
  - 第 1 波按生产规模：孪生 227 组、多目标 149 组，58 分钟，GPU 平均忙 90%。
  - 自动通过：孪生 89 组、多目标 83 组（213 个目标）。
- 时间（四台并行，`$ROOT/planning/eta.py`）：

  | 口径 | 每台选样（孪生组 / 多目标组） | 独占节点估计 | 本机实测直接换算 |
  |---|---:|---|---|
  | **当前：孪生约 2.5 万个样本 ＋ 多目标约 10 万个样本** | 7,350 / 17,953（79 波） | **约 2.0 天（1.8–2.1）** | 3.3 天 |
  | 孪生 5 万组 ＋ 多目标 10 万组（原计划） | 29,371 / 45,060 | 约 5.6 天 | 9.2 天 |

  - 本机每张 GPU 上有其他用户的进程，我们只分到约 55% 的算力，所以“独占节点估计”按这个比例折算。
  - 不含排队、节点重启，以及首次装环境和复制模型（每台约 0.5–1 小时）。
- 存储（当前计划）：约 3.6 TB、620 万个文件。
- 详细见报告 `/opt/tiger/tanyue/visualization_tmp/production_4node_20261025/四机生产实现与时间估计.md`。

## 迁移后的新任务入口与旧任务边界（2026-10-20）

实验和生产工作区统一归档到 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline`，正式结果仅发布到datasets。旧四机与单机冻结包保存原始内部路径和合同，**下文历史命令不应直接执行来恢复迁移后的旧任务**；仅改外层RUN_ROOT不足以恢复。旧任务进度、源输入、检查点及媒体均保留，但本次未验证迁移续跑。

当前工作仓库的新任务可使用下列入口，先做有上限的质量验证。两类分别创建独立新目录；此处是使用说明，本轮未启动生成。

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /opt/tiger/tanyue/sa1b-samtok-edit-data
export GROUP_KIND=twins  # 多目标改为multi
export GROUP_TARGET=15000  # 多目标改为5000
export GROUP_RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/new_${GROUP_KIND}_$(date -u +%Y%m%dT%H%M%SZ)"
export GROUP_SOURCE_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016"
export GROUP_ARCHIVE_PATH_MAP="/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/path_map.json"
exec bash scripts/groups_single_node_entry.sh --max-waves 2 --wave-size 128
```

需要单机8卡、不设置CUDA_VISIBLE_DEVICES、不用torchrun。入口使用27B规划/审核及Qwen/vLLM-Omni编辑，默认检查/安装运行环境；日志在 `$GROUP_RUN_ROOT/logs/single_node.log`，进度在 `progress.json`。当前质量仍需验证，不能据历史预算直接承诺15k/5k完成时间。两波128候选只是上限，不保证得到指定数量的合格组。

### 输入映射与续跑

`run_twins_single_node.py --archive-path-map`只映射源输入的绝对路径。原输入文件保持只读，原文件哈希及映射哈希一起写入**新任务**config；不会动态重写历史检查点。这避免重复复制大体积输入清单。共15,994个源图路径已在迁移后确认存在。

继续本入口创建的新任务时，必须把GROUP_RUN_ROOT设为首次实际目录，不要重新使用时间戳表达式；使用同一代码、输入、映射、目标及波次大小。配置变化会触发合同检查，不能直接覆盖。迁移前冻结任务另需恢复方案，不能套用此新任务续跑说明。

检查位置：

```bash
tail -n 30 "$GROUP_RUN_ROOT/logs/single_node.log"
python3 -m json.tool "$GROUP_RUN_ROOT/progress.json"
```

最终自动通过记录在 `accepted.jsonl`；未达配额输出partial，完整导出仍在工作区 `final/`。核验媒体、自包含路径、数量和质量后，才发布到datasets，不把运行目录本身放入datasets。

## 历史部署与流程说明

以下保留旧版本的结构、参数及故障处理供追溯。绝对目录已更新为归档位置，但冻结包内部不做改写；不能将这些示例视为已迁移验收的可执行续跑命令。

## 多目标并行融合与逐步保留

2026-10-17 起，按用户确认采用并行方案，当前源码新建的四机配置使用 `multi_commit_policy=parallel-components-v1`。多目标在规划结束后进入 `scripts/run_multi_transactional.py`：GPU 0–5 常驻编辑器、6–7 常驻审核器，独立写区域从原图并行，冲突分量内串行提交/回退；合格区域融合后联合审核。不同组也交错运行，审核请求合批。旧配置 `keep-passed-v1` 显式使用串行模式，缺少配置键继续旧流程。孪生流程不变。

下面列出的 `sa1b_samtok_group_train_20k_20261016` 是此前已冻结的部署，仅补入字体和共享配置读取竞争修复，没有升级生成流程；其运行中的检查点、模型、输出目录及配置均不改。新模式或提示词版本使用新的运行目录和代码快照，不能直接覆盖旧运行。

单机小批可复用已有、尚未生成的 `planned.jsonl`，使用新的输出目录。以下命令在仓库根目录运行，环境与模型路径按当前机器设置：

```bash
export MULTI_RUN_ROOT=/path/to/new_multi_run
python scripts/run_multi_transactional.py \
  --planned-manifest /path/to/planned.jsonl \
  --output-root "$MULTI_RUN_ROOT" \
  --editor-model /path/to/Qwen-Image-2.1 \
  --audit-model /path/to/Qwen3.8-27B \
  --editor-python /path/to/editor/bin/python \
  --audit-python /path/to/mllm/bin/python \
  --editor-devices 0,1,2,3,4,5 --audit-devices 6,7 \
  --groups-in-flight 16 --audit-batch-size 4 \
  --generation-mode parallel-components --local-generation-reference
```

正常运行每个目标至多尝试一次，不因质量失败追加生图：程序检查失败则不调用语义审核；语义审核通过才提交新图，失败则丢弃该结果并从同一分量上个合格图继续。异常中断时尚未持久化的工作可能重复，成本应包含失败启动。最后实际保留目标数、指令、mask、操作列表、顺序重新构建，再做一次整组审核；不重复逐步审核。最终至少保留两个合格编辑目标，不要求相邻。少于两个目标、不满足现有类型组合约束、或最终整组审核失败，都不会发布为多目标数据。可选 `--save-crop-evidence` 保存合成前后诊断图；串行对照使用 `--generation-mode serial`。

规划阶段使用 `run_group_pipeline.py --kind multi --plan-only --planner-model /path/to/Qwen3.8-27B --planner-tp 1`，其余输入/配额参数沿用原用法。局部指代选项和代码哈希均参与恢复合同，不能加到已经冻结的旧输出目录上续跑。本轮准确命令保存在 `sa1b_samtok_group_27b_quality_20261017/reproduction/` 及 `multi_current/*_command.json`，是实验复现记录。新建生产部署需重新准备代码快照；旧四机和旧单机共享包未被覆盖。

检查点与日志：

- `generated/GROUP/step_NN_UID/decision.json`：该次输入、成图、审核、通过/拒绝决定与校验和。
- `generated/GROUP/component_NN_progress.json`：并行模式每个分量的已尝试数、保留数和当前图；串行模式为 `progress.json`。
- `generated/GROUP/schedule.json` / `fusion/fusion_proof.json`：分量依赖和融合像素校验证据。
- `generated/GROUP/committed.json`：最终结果及图像校验和；`group.json` 同时保留不合格组。
- `transactional_progress.json` / `transactional_done.json`：已完成组数、自动通过数、最终目标数和类型分布。
- `services/{editor,audit}/SESSION/WORKER/worker.log`：模型加载、每个请求批次的数量和耗时。
- `audited.jsonl` 保存所有完成组；`accepted.jsonl` 为自动通过组，正式配额和发布校验由四机协调器继续执行。

已有串行实验续跑时显式使用 `--generation-mode serial`，不能使用新的并行默认。同样命令、目录和代码重启可续跑，已落盘的接受/拒绝决定不重试；已生图但审核未落盘的步骤复用图像，只补未完成审核。源码、计划、源图、mask、模型配置变化会拒绝复用旧检查点。机器回收留下 `transactional.lock` 时，须先确认旧进程停止；同主机确认 PID 消失后入口会自动恢复，跨主机锁须在确认后留档移走，不允许并发写同一 wave。

配额依据最终保留的操作列表和目标数重新计数。例如原四目标组保留三目标，只能计入三目标的对应组合；某组合已满时保留记录但不占其他组合名额，继续补缺额。因此不能用较高的“部分保留率”直接推算全量完成时间。

## 当前分组训练任务：15k 孪生＋5k 多目标（2026-10-16）

此节描述此前冻结的分组任务入口（仍采用整组保留策略）。后文的 50k 命令是已完成任务的历史说明，不能用来启动本次生成。四机曾于 2026-10-16 07:27 UTC 启动，但全部在模型加载前因 `missing audit font` 退出，尚无生成记录。已针对该启动故障准备字体补丁；重启后的实际进度以共享目录日志和 `progress.json` 为准。

- 资源：**4 台 × 8 张 H100 80GB**，CUDA 12.9；每台建议至少 64 CPU 核、512 GiB 主存和 500 GiB 本地可用磁盘（Flash-Next 使用 CPU offload）。
- 共享挂载：`/mnt/bn/strategy-mllm-train`。训练源图及 RLE 标注已复制到新目录，运行不依赖原始 HDFS 挂载。
- 新目录：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016`。
- 自动通过目标：孪生 **15,000 组**，四种编辑各 3,750；同类 4,500／跨类 10,500；双成员 12,750／三成员 2,250，成功组对应 32,250 张目标图。多目标 **5,000 组**，2／3／4／5 个目标分别 1,000／1,500／1,250／1,250，共 17,750 个成功编辑步骤。
- 最终孪生还限制每个 scope 至多 5%，实际源 selector 为部件的组不少于 40%；add 按解析后的附着部位统计。面积桶与同宿主部件比例会记录和监控，未作为额外硬配额，不能宣称它们自动精确达到 30/40/30 和 10%。
- 每个 rank 负责 3,750 个孪生、1,250 个多目标的最终名额。多目标按操作多重集分配名额，2–4 个目标不重复类型，5 个目标四类俱全且仅一类重复；最终各类编辑步骤约 4,438，整数差不超过 3。
- 每个 rank 先完成孪生再运行多目标；失败组不会减少成员数，低通过类别动态补样。达到尝试预算或候选耗尽后保留缺额，不将不足数发布为足额。孪生总尝试上限 150,000，多目标 100,000。
- 评测候选继续隔离，本入口不生成 200／100 组评测数据。自动通过不是人工验收，最终记录仍标记未发布。

### 提交到 Arnold 的完整入口

选择 4 workers、每 worker 8 H100，挂载上述共享卷。**四台使用完全相同的脚本**，平台注入 rank，不使用 torchrun，不手填 `ARNOLD_ID`：

```bash
#!/usr/bin/env bash
set -euo pipefail
exec bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016/deployment/arnold_entry_full.sh
```

该文件完整设置 `GROUP_RUN_ROOT`、`GROUP_RUNTIME_ROOT=/opt/tiger/sa1b-group-production-runtime`、`GROUP_SKIP_SETUP=0`，最后执行 launcher。Launcher 检查真实 4×8 拓扑、代码包与配置校验和、冻结模型文件清单，然后解包到节点本地、安装两套锁定环境、检查 CUDA 算子、缓存模型并启动本 rank。W&B 关闭。不要把入口最后的 `exec bash ...` 删掉。

冻结包来自本分支已提交代码。MLLM 环境固定已验证的 vLLM `0.31.1rc1.dev296+gfe4df3df0.cu129`；编辑器固定 vLLM `0.29.0+cu129` 和 vLLM-Omni commit `44ea27c8094095bbffd88fa3befdfaa55ba4bc50`。两者都使用 `torch 2.13.0+cu129`，不从默认索引重新选择 vLLM。依赖位于 `requirements/group_production/`。这次的新环境不覆盖旧 50k 的 runtime。

### 字体启动故障与恢复

`AssertionError: missing audit font` 表示旧入口要求节点安装 `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`。这是代码漏打包运行依赖；其后的 SIGTERM 是平台清理失败任务，并非 CUDA 或显存报错。不能只删除断言：规划标注和审核拼图也曾使用该绝对路径。

当前 `sa1b_pipeline/assets/fonts/` 随代码包含 DejaVu 普通／粗体字体、许可和 SHA-256 清单；`sa1b_pipeline/fonts.py` 统一提供字体。入口先用标准库核验字体，再在两套运行环境中实际绘制标签，无需 root、系统字体或额外下载。安装 Python 包时也会包含字体资源。

本次共享部署仅对原冻结包进行字体热修复，没有替换为最新生成管线。旧部署备份、逐文件变更清单及新旧包哈希保存在 `deployment/font_hotfix_20261016.json` 和其中记录的备份目录。配置、源输入、模型清单与数据目录不变；继续使用上面的同一入口重启四机。已有波次检查点的任务不可直接套用本次补丁：本任务修复时四个 rank 均未进入模型阶段，没有波次或已完成样本。

### 审核配置并发故障与原目录续跑（2026-10-17）

第二次启动已经完成规划和生图。rank 2／3 的部分审核进程读取 `audit_checkpoints/contract.json` 时出现 `FileNotFoundError`，随后其他 rank 收到 SIGTERM；rank 0 首轮 128 个候选中有 55 组已写入 `accepted.jsonl`。旧代码让多个审核进程同时检查／创建同一配置，在共享挂载上触发可见性竞争。

修复后只有协调器在启动审核进程前创建并确认配置，各审核进程通过 `production_state.require_json` 只读校验。读取仅对 ENOENT／ESTALE 最多尝试 8 次，总退避 4.5 秒；配置内容变化、JSON 损坏、权限错误立即失败，不覆盖配置或放宽质量审核。

本次冻结包仅修改审核配置交接及读取可靠性，保留原生成／规划方法。热修复需在四个 rank 停止后持有四个节点锁，备份旧部署和命令记录，仅迁移 `*_command.json` 中已核验的代码包路径；模型、参数、设备、数据路径及阶段退出标记全部保留。过程记录在 `deployment/audit_contract_hotfix_20261017.json`。同一入口重启后，rank 0 跳过首轮已汇总结果，其余节点复用已完成规划／生图，审核仅补缺失的组级检查点。不要删除 `nodes/`，也不要将当前最新管线整包覆盖到旧任务。

全局 `progress.json` 是定时快照，任务中止后可能仍显示 0 或旧阶段；以各 rank 的 `accepted.jsonl`、波次输出、`*_exit.json` 和 `failure.json` 判断恢复点。已通过数量是自动审核口径，不代表额外人工验收。

### 每台机器的运行方式

采用每波最多 128 组的分阶段调度，阶段之间释放模型，不让两个大模型争同一张卡：

1. CPU 从本 rank 冻结源图中按缺额提出新组合，精确面积检查；多目标用 RLE 检查源 mask 非空及冲突，不限制目标间距离。每张源图在每种数据集最多保留 2 组，同一个源图＋对象组合不重复规划。
2. GPU 0–3：Flash-Next FP8 TP4，一组一次规划。孪生随后单独用 GPU 0 上的 27B 做一次原图＋no-ref 盲测，保持本轮验证的模型组合。
3. GPU 0–7：每张卡一个编辑器，不同组并行，组内各成员／链步骤按现有方法生成；40 步、完整 Ref、区域贴回与 difference-first-3 不变。SAM3 只在需要时加载。
4. GPU 0–3：最多四个 27B 审核副本，每个副本按批次保存逐组审核检查点。孪生每组一次联合审核；多目标沿用每步审核＋最终联合审核。
5. 仅按完整自动通过组计数，下一波补缺额。整波结束后更新 `accepted.jsonl`，逐阶段日志与逐组检查点在波内实时落盘。

### 进度与数据位置

每 300 秒追加一行类似 tqdm 的进度到：

```bash
tail -F /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016/logs/progress_tqdm.log
```

它显示两种数据的最终通过数／目标、尝试数、各 rank 阶段和 ETA；至少通过 32 组后才基于当前实测吞吐给出 ETA。冷启动、补低通过类别、停机时间会影响该估计。rank 0 提前完成时仍保留监控，等其余 rank 结束后退出。聊天窗口不会自动推送日志。

```bash
ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016
# 在本仓库根目录可查询；只需 CPU 依赖。
python scripts/run_group_production_4node.py --root "$ROOT" status
# 每台的完整 stdout/stderr
 tail -F "$ROOT/logs/node0.log"
# 本 rank 的分阶段日志示例
 tail -F "$ROOT/nodes/node0/twins/waves/00001/planning.log"
```

每个 rank 的 `nodes/nodeN/{twins,multi}/progress.json` 包含分类缺额，`accepted.jsonl` 包含自动通过记录；`waves/NNNNN` 保存固定选择、规划、盲测、原始成图、审核、错误及每次启动成本。`pool/` 保存 CPU 候选池检查点。

四个 rank 结束后自动合并，数据入口为 `$ROOT/final/twins_train.jsonl` 和 `$ROOT/final/multi_train.jsonl`。所有图片路径相对于 **`$ROOT`**，不是 `final/`；源图、源 mask、执行区域、目标图和多目标中间图均在根目录内，不依赖外部图片链接。缺额时只写 `*.partial.jsonl`，`final/summary.json` 明确给出缺额。也可在运行中主动导出当前快照：

```bash
python scripts/run_group_production_4node.py --root "$ROOT" export
```

导出只读已落盘检查点，不重复规划、生图或审核。当前尚无生产通过记录；`verification/smoke/` 是入口连通性测试，永不混入生产配额。

### 中断恢复

正常失败会保存错误并释放 rank 锁；重新提交完全相同的入口和目录即可。已完成的阶段跳过，已有计划和目标图按原检查点恢复，审核也按组复用。若正好在模型返回但结果落盘前中断，仅未落盘的请求可能重新执行；不承诺跨进程崩溃的严格 exactly-once。

共享锁使用原子创建目录，兼容不支持 flock 的共享文件系统。若 worker 被 SIGKILL／机器直接回收，跨主机残留锁不会被擅自抢占。**确认上一次四机任务全部停止后**，只对遗留 rank 执行下面的恢复命令，然后仍使用原入口重启：

```bash
python scripts/run_group_production_4node.py --root "$ROOT" recover-lock \
  --rank 0 --previous-workers-stopped
```

锁会重命名留档，不删除数据。不要同时提交两个写入同一根目录的四机任务；不要修改冻结配置、代码包、模型权重或输入清单来绕过检查。候选库耗尽属于明确的产量缺口，需要另做可审计的候选扩充，单纯重启不会凭空补足。

### 初步时间预算

当前没有四机稳态实测。按已有小批的每张编辑约 19–20 秒、每组单次规划＋独立盲测、分阶段冷启动、审核并行及失败补样估算：孪生 15k 约 **2–4 天**，多目标 5k 约 **1–3 天**，全部自动生成暂按 **3–7 天** 排期，首次环境安装与模型复制另预留 **0.5–2 小时**。这是假设通过规划／前置检查后的完整组自动通过率，孪生约 45–60%、多目标约 20–35%；多目标在新版本下尚无足够样本证明这一范围。

本次入口验证的多目标 2 个候选没有自动通过组（1 个规划拒绝，1 个完成两步生成后审核拒绝），所以 20–35% 只是排期假设。

这不是保证完成时间。add／三成员／五目标组可能更慢；若实际通过率低于上述范围，时间可能超过一周，或先耗尽候选。原需求文档的 1–1.5 天估算采用旧版单条高通过率，不能直接沿用。四机启动后先用前两轮的分阶段计时、分类通过率和候选消耗更新预算；人工图像验收不计入上述自动流水线时间。


## 运行参数

- Arnold worker：4
- 每 worker：8 × H100
- 每 rank：2 个 TP=4 规划副本、6 个单卡编辑器、2 个单卡审核器
- 目标：50,000 条自动审核通过样本，`add/replace/remove/attribute` 各 12,500 条
- CUDA：12.9；PyTorch `2.13.0+cu129`
- 共享挂载：`/mnt/bn/strategy-mllm-train`
- 源记录：运行期间使用冻结的 `source_cache/`，无需 SG2 HDFS 挂载；本批完成后该缓存已按确认删除，恢复旧 run 前需重建
- W&B：关闭

## 入口

四个 Arnold worker 使用同一段脚本；平台负责注入 `ARNOLD_ID`：

```bash
#!/usr/bin/env bash
set -euo pipefail
export SA1B_RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014"
export SA1B_RUNTIME_ROOT="/opt/tiger/sa1b-production-runtime"
export SA1B_SKIP_SETUP=0
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=LAZY
exec bash "$SA1B_RUN_ROOT/deployment/launch_4node.sh"
```

也可以直接调用冻结入口：

```bash
bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014/deployment/arnold_entry_full.sh
```

脚本会拒绝非 4×8 配置。启动前应看到：

```text
Shared source cache verified: 16000 images, 32000 records; HDFS mount not required
Shared inputs and model configurations are accessible
```

## 安装和启动检查

`setup_runtime.sh` 为 planner/editor/audit 分别创建独立环境，并使用 `requirements/production/*.lock.txt`。编辑器固定 vLLM-Omni revision；所有角色使用 CUDA 12.9 wheel。`check_runtime.py` 在环境标记 ready 前执行无界面 OpenCV 检查、spawn 子进程导入、GPU matmul、vLLM RMSNorm、CUDA runtime 和编辑器 Omni 导入检查，避免 CUDA 13 wheel 混入 CUDA 12.9 任务。

## 查看进度

```bash
ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014
python -m scripts.run_production_4node --root "$ROOT" status
for n in 0 1 2 3; do tail -F "$ROOT/logs/node$n.log"; done
tail -F "$ROOT/logs/progress_monitor_tqdm.log"
```

`progress.json` 的阶段依次为 `source_materialization`、`planning`、`model_cache`、`generation_and_audit`、`complete`。planning 日志会记录模型 shard、JIT、每批 completed/accepted 和 batch_seconds；生成/审核阶段会记录队列、通过数和类别缺额。

## 恢复规则

1. 先确保旧 Arnold 任务的四个 rank 已退出，避免同一 rank 并行写共享检查点。
2. 使用完全相同的 `SA1B_RUN_ROOT`、代码包、模型和四机拓扑重新提交入口；共享 rank 锁会阻止重复 owner。
3. 已完成 wave 会跳过；已完成 planning checkpoint 会按输入 mask 数和 accepted 数校验后复用；生成 fingerprint 匹配的结果和已落盘审核结果直接恢复。 分组规划还会按 planner batch（默认 8）逐组原子落盘，rank 被回收时不重复已解析的 MLLM 结果。
4. 已提交但结果丢失的 planner request 会记录为 `interrupted_unknown`，不重复调用同一 mask，也不伪造拒绝结果。
5. 数据协议、输入清单、模型或生成参数改变时不混入旧运行。仅修复启动依赖、权重缓存和超时，可以在所有 rank 停止并持有共享锁时使用 `scripts/production/repair_deployment.py` 发布补丁：备份旧部署、逐项验证旧代码及 planner 快照，只更新允许的运行时文件及 `code_hashes`，保留原根目录和全部检查点。禁止手工删除 hash 检查。
6. 最后一个 rank 结束后自动尝试合并，成功后检查 `final/summary.json`。如果四个节点均已写出 `done.json` 和 `auto_pass.jsonl`，仅汇总失败（例如共享盘不支持 `flock`，报 `Errno 38`），无需重启四机或重新生成图片；在能访问共享目录的 CPU 机器上，使用更新后的仓库执行下述独立收尾入口。该入口保留冻结配置，验证原有导出及审核规则的代码哈希，使用原子建目录锁防止并发发布，并检查数量、类型配额、唯一性、审核标记及全部图片路径。图片校验包含存在性、非空及根目录内路径，不包含全量解码或人工质量复核。

```bash
python3 -u -m scripts.finalize_production_exports \
  --root /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014 \
  --workers 16
```

收尾进度写入 `final/progress.json`；最终标注为 `final/auto_pass.jsonl`，其中图片路径相对于数据根目录；`final/summary.json` 保存统计、输入导出哈希及最终标注哈希。目录锁 `final/finalize.lock.d` 不会自动抢占；异常退出遗留锁时，先核实 `owner.json` 对应进程已退出再处理。历史 `failure.json` 保留；本次 `Errno 38` 成功恢复后更新节点进度为完成。

## 本批最终数据入口（2026-10-15）

四机配额已完成，无需重启任务。CPU 收尾及规范整理后，训练入口为 `$ROOT/release/annotations/train.jsonl`，媒体路径相对于 `$ROOT/release`；`final/auto_pass.jsonl` 作为原始合并记录保留。全量统计、字段和可视化见 [50k 数据说明](DATASET_50K_ZH.md)。源记录缓存 A 已按用户确认删除；该已完成运行不应直接重新启动，恢复/扩量前需重建并校验缓存。其余候选保留，见 [清理记录](CLEANUP_CANDIDATES_50K_ZH.md)。

## 2026-10-14 启动修复与原地恢复

以下记录 2026-10-14 的故障恢复方式：当时使用同一 root `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014` 和原入口重新提交，保留 `SA1B_SKIP_SETUP=0`，无需重新 prepare。当前本批已经完成且源缓存已清理，不再直接套用该历史重启步骤。

- 审核 lock 曾同时安装 `opencv-python` 与 `opencv-python-headless`，二者覆盖同一 `cv2` 目录，子进程报 `libxcb.so.1`。修复为只保留 headless；旧环境 lock 变化时自动同步，并重新安装 headless 修复共享文件。
- 编辑器从共享盘并发读权重，触发 Omni 默认 600 秒启动超时。`model_cache.py` 在启动 GPU worker 前将编辑、审核权重各复制一份到节点本地盘，按文件 SHA-256 校验、断点复用；约需 83 GiB，建议预留 100 GiB。Omni 初始化及 stage 超时均设为 1800 秒。缓存阶段日志有字节进度，复制不会占用模型初始化的超时额度。
- 数据记录和生成 fingerprint 继续使用原模型路径，只有模型加载使用本地副本；步数、seed、指令和审核门槛保持原设置。
- 已完成的 planner 副本直接复用；未完成副本只处理未提交 mask。已提交但未落盘结果的 mask 标记 `interrupted_unknown`，不会重复调用；后续候选补足最终配额。已生成图按 fingerprint 复用，已审核结果从共享 `results/` 重建计数及本地队列。
- 失败时同步写入 `progress.json.status=failed`，避免界面继续显示 running。后台监控每 300 秒写日志；聊天窗口不具备后台日志自动推送能力。

验证记录位于运行根目录 `verification/startup_repair/`：229 项自动测试通过，实际 GPU 运行 4 条（四类各一条），全部完成生成和审核，其中 3 条自动通过、1 条由质量审核拒绝；4 张生成图的恢复验证未再次调用生图，图片哈希不变。编辑器加载 27.4 秒、4 条生成与保存 84.8 秒；审核权重加载 12.7 秒（不含初始化和预热）。这些结果用于验证启动与恢复，不计入正式 50k 配额，也不能用于估计长期通过率。

## 故障处理

- 看到 `/mnt/hdfs/...index`：确认使用的是更新后的冻结 `deployment/code.tar.gz`，并确认 `/mnt/bn/strategy-mllm-train` 已挂载；当前正式配置使用 shared source cache。
- 看到 `libcudart.so.13`：不要用默认 `vllm==...` 重新安装；清理对应 role 环境，保留 CUDA 12.9 lock，重新以 `SA1B_SKIP_SETUP=0` 启动。
- 只有旧的 `nodes/node*/failure.json` 而日志时间早于本次启动：先以 `owner.json`、最新 `progress.json` 和日志 mtime 判断，不能把旧 143 记录当作当前错误。
- 磁盘或共享目录异常：停止任务后保留 root、logs 和 checkpoints；不要删除 source_cache、dispatch、results 或队列数据库。

## 完成判定

```bash
cat "$ROOT/final/summary.json"
wc -l "$ROOT/final/auto_pass.jsonl"
```

完成必须同时满足总数 50,000、四类各 12,500、case ID 和 source mask 唯一、所有图片字段可解析、每条关键审核字段为真。自动完成不等于人工质量验收；训练发布前按 [PIPELINE.md](PIPELINE.md) 的质量项抽查和记录人工结论。

## 分组数据的独立运行

孪生/多目标任务不能使用 50k 的 `SA1B_RUN_ROOT`、部署包或队列数据库。先创建独立目录并预留评测源图：

```bash
GROUP_INPUT_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/01_shared_inputs/sa1b_samtok_group_inputs_20261014
SOURCE_ROOT=/mnt/hdfs/sg_byte_ttlive_strategy_llm/user/haobo.yuan/datasets
python scripts/prepare_group_inputs.py \
  --frozen-root /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014 \
  --benchmark-exclusion /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_edit_50k_20261014/inputs/benchmark_exclusion.jsonl \
  --source-root "$SOURCE_ROOT" --output-root "$GROUP_ROOT" \
  --eval-count 1200 --train-limit 16000 \
  --materialize-missing-train --materialize-workers 32
```

如果历史训练清单已知，应显式加入评测排除注册表（每个清单一个参数），例如：

```bash
  --skip-discovery \
  --existing-manifest /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTokEdit_Cleaned_20261009/manifest.jsonl \
  --existing-manifest /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined/manifest.jsonl
```

缺失的训练源会以 `--materialize-workers` 路并发读取；默认 8 路。评测预留只在新根目录首次创建，恢复时复用已有 `inputs/eval_reservation.json`。

孪生第一批使用四类配额各约 1,250 组；下面命令先只做规划，确认规划质量后再启动编辑器和 27B 审核：

```bash
python scripts/run_group_pipeline.py --kind twins \
  --manifest "$GROUP_INPUT_ROOT/inputs/train_candidates.jsonl" \
  --output-root "$GROUP_INPUT_ROOT/twins_train_wave1" --target 5000 --split train \
  --reservation "$GROUP_INPUT_ROOT/inputs/eval_reservation.json" \
  --planner-model /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-Flash-Next-FP8 \
  --planner-backend vllm --planner-tp 4 --planner-batch-size 8 \
  --candidate-pool-factor 12.0 --plan-only
```

Flash-Next 仍是待验证的规划/审核候选，不代表已经达到正式发布质量。当前适配根据本地 `config.json` 的 `Qwen4ExpForConditionalGeneration` 自动启用 Triton MoE、Engram CPU offload；四卡使用 TP4，八卡使用 TEP8。它需要支持该架构的运行时；`requirements/mllm-vllm-cu129.txt` 中的旧 27B 环境不能直接视作已验证兼容。本机实验复用 `/tmp/sa1b-next-runtime/bin/python`（vLLM 0.31.1rc1.dev296、CUDA 12.9）及已下载的节点缓存。审核切换模型时同时指定 `--audit-tp 4`，默认仍为 1。每种模型/输入变体必须使用新的输出根目录，避免复用另一模型的计划缓存。

正式 wave1 选择已经固定在独立根目录
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/05_history/20261015/sa1b_samtok_twins_flashnext_wave1_20261015/inputs/selected.jsonl`（5,000 组），四个分片位于同目录的 `shards/twins_train_wave1/`。这份清单已完成四类、scope、面积比和评测排除校验，不要重新选择或覆盖它：校验摘要和清单/分片哈希见 `$GROUP_ROOT/verification/selection_integrity.json`。全量训练/评测 source ID 与图像哈希隔离记录见 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/01_shared_inputs/sa1b_samtok_group_inputs_20261014/verification/source_isolation_full.json`。 重新生成该记录可运行 `scripts/verify_group_inputs.py`；它在模型启动前失败即停止，避免把泄漏或配额错误带入四机。

```bash
export GROUP_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/05_history/20261015/sa1b_samtok_twins_flashnext_wave1_20261015
export GROUP_INPUT_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/01_shared_inputs/sa1b_samtok_group_inputs_20261014
python scripts/shard_group_selection.py \
  --selection "$GROUP_ROOT/inputs/selected.jsonl" \
  --output-root "$GROUP_ROOT/shards/twins_train_wave1" --world-size 4
```

每个 Arnold worker（`ARNOLD_ID=0..3`）可以直接执行仓库内的统一入口
`scripts/group_4node_entry.sh`。它会校验 4×8 拓扑、原子复制本 rank shard、复用已有
checkpoint，并把日志写到 rank 独立目录。示例环境变量如下：

```bash
export GROUP_ROOT="$GROUP_ROOT"
export GROUP_KIND=twins GROUP_SPLIT=train
export GROUP_SHARD_ROOT="$GROUP_ROOT/shards/twins_train_wave1"
export GROUP_SELECTION="$GROUP_ROOT/inputs/selected.jsonl"
export GROUP_MANIFEST="$GROUP_INPUT_ROOT/inputs/train_candidates.jsonl"
export GROUP_RESERVATION="$GROUP_INPUT_ROOT/inputs/eval_reservation.json"
export GROUP_PLANNER_MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-Flash-Next-FP8
export GROUP_EDITOR_MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1
export GROUP_AUDIT_MODEL=/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B
export GROUP_PLANNER_TP=4 GROUP_MAX_NEW_TOKENS=1600 GROUP_AUDIT_TP=1
# The entry automatically enables clean binary/outline crops, full-scene localization,
# and the grounded group-audit prompt. Leave GROUP_PUBLISH unset until the PDF ledger is complete.
bash scripts/group_4node_entry.sh
```

四个 rank 都结束后合并并做硬校验：

```bash
python scripts/merge_group_shards.py --kind twins --split train \
  --input-root "$GROUP_ROOT/runs/twins_train" --target 5000 \
  --output "$GROUP_ROOT/final/twins_train.jsonl"
```

评测 200 组使用相同流程，但输入换为 `eval_twin_pairs_selected.jsonl`，`--split eval`，并把输出写到 `twins_eval/rank${RANK}`；评测组完成后必须生成图片内嵌 PDF，逐组填写 `human_review.jsonl`，全部确认后才可发布。训练发布至少需要孪生 200 组或多目标 100 组的显式人工通过记录；评测发布必须每组都人工通过。`GROUP_PUBLISH=1` 时必须设置 `GROUP_HUMAN_REVIEW`，入口会拒绝缺失或未签字的 ledger。多目标的 1,500 训练组和 100 评测组必须在孪生第一批人工确认后执行。当前选择文件已经完成并分片：

```text
$GROUP_ROOT/inputs/train_multi_selected.jsonl       # 1,500 groups
$GROUP_ROOT/inputs/eval_multi_selected.jsonl        # 100 groups
$GROUP_ROOT/shards/multi_train_wave1/               # 4 x 375
$GROUP_ROOT/shards/multi_eval/                     # 4 x 25
```

多目标选择文件在进入规划前必须完成操作配额检查。`scripts/rebalance_multi_operations.py` 会枚举每组可行序列并只更新 `operations` 字段；训练和评测的重排证据分别记录在 `$GROUP_ROOT/verification/multi_operation_balance.json`，随后必须重新运行 `scripts/shard_group_selection.py` 更新四机分片。不要用固定的目标位置顺序代替该步骤，也不要把不可行操作通过改标签强行填入配额。

选择阶段使用快速 bbox 容量池后再运行 `scripts/recheck_multi_selection.py` 做 RLE 非空及冲突复核（不要求相邻）；快速池本身不能进入规划或发布。四机入口将 `GROUP_KIND=multi`、`GROUP_SPLIT=train`、`GROUP_SHARD_ROOT=$GROUP_ROOT/shards/multi_train_wave1`、`GROUP_SELECTION=$GROUP_ROOT/inputs/train_multi_selected.jsonl`，目标设为 1,500；评测时改用 `multi_eval` 分片、`eval_multi_selected.jsonl` 和 `GROUP_SPLIT=eval`。多目标每组仍只做一次规划调用，随后按 `generation_order` 链式生图，组级生成/27B 审核/人工 ledger 全部通过后才可发布。

生图阶段追加 `--editor-model ... --audit-model ... --publish`，并将 `--split eval` 指向
`inputs/eval_candidates.jsonl` 生成 200 组评测；评测组必须全部通过人工 ledger 后才可发布。多目标第一批在孪生第一批训练侧确认后执行，命令把 `--kind twins` 改为 `--kind multi`，目标设为 1,500，并使用同一个 source-isolation reservation。每个组一条 planning call，生成和审核 checkpoint 可从同一独立根目录恢复；不要把失败组缩减成较少目标数。

分组规划实验可显式添加 `--outline-planning-crops`，在每个候选行追加 mask 外侧轮廓；仍只向模型传原图和一张候选拼图。该开关尚在验证，默认关闭，当前 50k 快照不受影响。


### 孪生小批量的分阶段并行入口（2026-10-15）

`group_4node_entry.sh` 的默认入口仍是每 rank 一个编辑器。新增的 `--editor-devices` 是显式选择的分阶段接口，已验证单机双卡；尚未完成四机性能验证。规划/审核与编辑运行环境版本不同，应分别使用各自的 Python。以下变量沿用本节的模型和独立运行根目录；`RUN_ROOT`、`MANIFEST`、`TARGET` 应指向本 rank 的根目录、候选清单和预定组数。

```bash
COMMON=(--kind twins --manifest "$MANIFEST" --output-root "$RUN_ROOT"   --target "$TARGET" --split train --reservation "$GROUP_RESERVATION"   --outline-planning-crops --locate-planning-scene)
"$PLANNER_PYTHON" scripts/run_group_pipeline.py "${COMMON[@]}"   --plan-only --planner-model "$GROUP_PLANNER_MODEL" --planner-tp "$GROUP_PLANNER_TP"   --max-new-tokens 1600
"$EDITOR_PYTHON" scripts/run_group_pipeline.py "${COMMON[@]}"   --generate-only --editor-model "$GROUP_EDITOR_MODEL" --editor-devices 0,1
"$AUDIT_PYTHON" scripts/run_group_pipeline.py "${COMMON[@]}"   --audit-only --audit-model "$GROUP_AUDIT_MODEL"
```

GPU 列表使用本机物理可见设备标识，每个 worker 独占列表中的一个编辑器实例；不要填写重复设备。每组只调度一次，恢复使用逐成员输入哈希与输出校验和。日志包含已生成/确定性通过/总组数，`release.json` 记录各类自动接受数量与缺口；缺口不会自动变成足额数据。新 prompt 实验必须用新根目录，恢复不会重复规划已拒绝组。

`--twin-audit-mode joint` 仅用于对照实验：现有回归表明它严重漏检，不能因为调用数少而直接替代默认审核。当前小批量结论与 PDF 在 PIPELINE.md 第 7.9 节；正式分组生产尚不建议启动。当前 50k 生产快照不受这些实验入口影响。

新分组运行已切换到 instruction/mask 2.0。必须新建运行根目录，不能用 2.0 代码直接续跑 1.0 规划检查点；`config.json` 会校验协议版本。源候选选择和 reservation 可以复用。生成仍使用 Ref、40 步及原执行区域；新 add site mask 在生成后解析。最终准入额外要求 `blind_protocol.pass=true`，定位失败不会退回宿主 mask 静默发布。


### 2.0 的 JSON 约束与 site 修正

在新的小批量运行中，可给上述 COMMON 同时追加 `--structured-planning-json` 和 `--site-model /mnt/bn/strategy-mllm-train/common/models/sam3`。规划运行时设置 `SA1B_FORCE_DIRECT_VLLM=1`，使用已验证支持 StructuredOutputsParams 的直接 vLLM 后端。三个阶段必须使用相同的配置参数，否则不可变 config 校验会拒绝继续。

`--site-model` 在每个编辑 worker 首次遇到 add/site 后懒加载 SAM3，在生成后、审核前修正源 mask。SAM3 与编辑器同卡常驻的峰值显存和四机性能尚未测试；当前已实测的是独立运行时对既有目标图的修正。无需再次调用编辑器或 MLLM：

```bash
# 在仓库根目录执行；OUT 必须是尚不存在的新目录。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=4 /tmp/sa1b-next-runtime/bin/python \
  scripts/refine_group_source_regions.py \
  --manifest /path/to/run/audited.jsonl \
  --output-root /path/to/run/site_refined \
  --site-model /mnt/bn/strategy-mllm-train/common/models/sam3
```

GPU 4 仅为例子，需选择本任务可用卡。该本机运行时为 Transformers 5.17；复制到新节点时应先验证 SAM3Model/SAM3VideoModel/SAM3Processor 可导入。新 `groups.jsonl` 只复用原目标图，变更源 mask 的组会撤销旧语义通过状态并标 pending，不会发布。它是诊断修正入口，不能把旧审核的通过率当成新 mask 的通过率；新版本须完成定位/轮廓复核再进入发布流程。


### 跨类别孪生配额与盲测

新孪生默认 `--twin-same-category-fraction 0.30`，其余 70% 为跨类别。四类编辑配额、两/三成员比例、面积比和源隔离不变。旧的纯同类 5,000 组清单不满足新配比；复用源候选库与 reservation，重新选择、冻结并分片，使用新运行根目录。不可变配置包含 `twin_contract=blind-cross-category-1`，旧计划不会静默当作新协议通过。

规划阶段现在包括每组一次规划和一次独立无定位盲测；`--plan-only` 的 planned.jsonl 只包含两关都通过的组，planned/ 中保留所有规划成功记录，rejected/*_blind.json 与 blind_probes/ 保留盲测拒绝/不确定结果。可读取 blind_probes/summary.json 查看额外调用和耗时。盲测失败组不会进入编辑器。

`group_4node_entry.sh` 自动加 `--quota-from-selection`，各 rank 按冻结 shard 的操作/同类跨类计数分配最终名额；这样不会因每 rank 各自四舍五入而破坏全局配额。选择清单中的 category_relation 必须来自新选择过程。`merge_group_shards.py` 对新协议复核盲测、粒度和同类约 30% 的全局比例（容差 max(1 组, 2% 总组数)）；任何 rank 缺额都不能发布为足额数据。本轮没有启动四机生成。

### 孪生动态补样入口（先做小批量验证）

`scripts/run_dynamic_twins.py` 按最终通过数补足编辑类型及同类/跨类配额，运行根目录需要独立于旧 50k。当前使用一个协调器串行推进波次，波内可以配置多张编辑卡；不等于已经验证了四机动态任务调度。先用以下入口验证，不要四个 rank 共用同一根目录启动：

```bash
python scripts/run_dynamic_twins.py \
  --manifest /path/to/train_candidates.jsonl \
  --reservation /path/to/eval_reservation.json \
  --output-root /path/to/new_dynamic_run \
  --target 12 --wave-size 12 --max-attempts 24 \
  --planner-python /path/to/planner/bin/python \
  --editor-python /path/to/editor/bin/python \
  --mllm-model /path/to/planner_model \
  --editor-model /path/to/Qwen-Image-2.1 \
  --site-model /path/to/sam3 \
  --mllm-devices 2 --editor-devices 2,3
```

支持独立的 `--audit-model`、`--audit-tp`、`--audit-devices`，用于经质量对照验证的审核模型。默认与规划同模型；当前 27B 联合审核仍存在明显漏判，不应直接用默认配置正式发布。`progress.json` 给出目标、已通过、各类缺额和停止原因；`waves/NNN/{planning,generation,audit}.log` 保存详细进度。中断后使用相同参数重启；模型/输入/配额变更应使用新根目录。`accepted.jsonl` 是自动通过的实验记录，不代表人工验收或发布。

审核模型配置在第一次进入审核阶段时单独冻结；可以先 `--stop-after planning` 或 `--stop-after generation` 检查阶段结果，再用原参数去掉停止选项继续。重复同一停止选项不会越过该阶段。需要扩充候选时用 `--candidate-extension /path/to/candidates.jsonl` 注册不可变增量清单，既有波次不变，已尝试源图仍不重复；增量也检查评测隔离。

建议沿用生产环境的已验证本地模型缓存，避免共享盘冷读拖慢每波启动。启动前设置 `SA1B_LOCAL_MODEL_MAP` 为 canonical 模型路径到本机缓存路径的 JSON（例如 `export SA1B_LOCAL_MODEL_MAP="$(cat /path/to/verified-model-map.json)"`）；缓存必须有 `stage_model` 产生的 `ready.json` 和 `files.json`，不能只随意写一个目录。加载使用本地路径，数据 provenance 仍记录 canonical 模型。`stage_attempts.jsonl` 保留失败/恢复启动耗时，不能只用最终成功退出的时间估算总成本。

## 单机 8 卡孪生入口（27B，2026-10-17）

新增 `scripts/twins_single_node_entry.sh`。GPU 0 运行 27B 规划，GPU 1–6 各运行一个 40 步 Qwen-Image-2.1 编辑器，GPU 7 运行 27B 盲测和联合审核。模型跨 wave 常驻、微批重叠执行。CUDA 12.9 环境由 `requirements/group_production/*.lock.txt` 锁定，首次启动安装并校验模型缓存及随代码打包的字体；无需 torchrun、rank 或 WandB。

**当前仅建议单批试跑**：原 32 候选测试自动通过 20 组，逐图 10 合格、8 失败、2 存疑；最新 16 候选定向修复自动通过 12 组，逐图 6 合格、3 失败、3 存疑，均未达到无人复核放量要求。下面共享包是先前冻结版本，不含最新修复；使用最新代码应先按下文 `prepare_twins_single_node.py` 新建部署目录。工程入口可运行不等于质量已验收。以下历史入口限一批 128 候选，最终目标仍冻结为 15,000，不会直接启动完整生成。

Arnold 选择 **1 worker × 8 H100 80GB**，挂载 `/mnt/bn/strategy-mllm-train`，填写完整入口：

```bash
#!/usr/bin/env bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export GROUP_RUNTIME_ROOT=/opt/tiger/sa1b-group-production-runtime
export TWINS_SOURCE_ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016
export TWINS_TARGET=15000
exec bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/03_deployments/sa1b_samtok_twins_15k_27b_single_20261017/deployment/launch_single_node.sh --max-waves 1
```

共享快照入口自动核验代码 SHA-256、解压到本地并设置 `TWINS_RUN_ROOT`。不依赖新机器预先存在 git checkout。源四机目录仅用作只读输入，新结果目录为 `sa1b_samtok_twins_15k_27b_single_20261017`；不会恢复、覆盖或停止旧四机任务。源目录中的图片和 masks 已收纳，不需要继续读取最早的 HDFS tar。

该包由以下命令生成，已有包拒绝覆盖；修改代码应选择新根目录：

```bash
python3 scripts/prepare_twins_single_node.py --root /path/to/new_run_root
```

质量确认后，去掉 `--max-waves 1` 可在同一目录按最终配额继续；当前没有启动完整生产，也不把此操作建议为已经达标。若核心代码/模型/协议更改，配置检查会拒绝混用旧结果，需新建根目录。相同版本中断后直接重复同一入口即可，完成规划和生成按内容指纹恢复，审核阶段中断可复用成图。

本机已有环境时可设 `GROUP_SKIP_SETUP=1`、`TWINS_MLLM_PYTHON=/tmp/sa1b-next-runtime/bin/python`、`TWINS_EDITOR_PYTHON=/tmp/sa1b-cu129-validation/editor/bin/python`。这两个 `/tmp` 路径仅适用于当前机器；新机器沿用上述默认安装。默认缓存目录是本地 `/opt/tiger/sa1b-group-model-cache`，不要将模型缓存放共享盘。

```bash
# 登录机器后查看持久日志：
tail -f /mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/03_deployments/sa1b_samtok_twins_15k_27b_single_20261017/logs/single_node.log
```

`progress.json` 是全局最终配额进度，`waves/00001/progress.json` 等记录逐组完成及自动通过类型；服务细节位于 `services/*/*/*/worker.log`。默认最终四类各 3,750 组，同类 30%、跨类 70%、三成员 15%，保留源复用、类别和部件比例约束。候选耗尽或预算到达明确写未完成状态，不降低审核来凑数。

只有最终配额齐备才写 `final/twins_train.jsonl`；否则写 `final/twins_train.partial.jsonl`。所有图片和 mask 收纳到运行目录，最终图片路径相对于运行根目录。`accepted.jsonl` 与 final 清单均是自动审核记录，不代表人工质量验收，`published=false`。
