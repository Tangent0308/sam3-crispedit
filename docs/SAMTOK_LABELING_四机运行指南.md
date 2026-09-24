# SAMTok 派生编辑数据：四机 32 卡运行指南

本入口只扩展并行调度，不重新设计打标方法。对应远端为
`https://github.com/Tangent0308/sam3-crispedit.git`，分支为
`samtok-derived-edit-labeling`，不是 MIRAGE 的 origin/main，也不是 SAMTokEdit 的 dev。

## 1. 本次部署的流程与范围

冻结最近一轮验证的 **remove 流程**：

```text
GRES/VER 正例与每个原始 region
  → Qwen3.8-27B / vLLM，relations-v16 thinking 规划
  → 邻居保留、附件连带区域解析（无 SAM 源 mask 审核）
  → Qwen-Image-2.1 / vllm-omni，40步、seed=0
  → guard-any-v1 + visible-v1 + adaptive-remove-v5 融合
  → Qwen3.8-27B / vLLM，completion-v5 / adaptive 双图 / pixel-veto
  → 汇总全部结果与 model_pass 候选，不将模型通过冒称人工验收
```

规划的 prompt、采样设置、既有修复调用、SAM 安全约束、编辑输入、KV cache、融合和审核规则不变。
没有新增强制改写调用：最近冻结流程也是生成加审核，历史的可选改写实验不因四机部署自动启用。
其他三类编辑没有被转换为 remove；输入含其他类型会明确报错，不能把当前入口称为四类均衡生产。

四台机器独立处理数据，各用本机 8 卡。**不使用训练的 DDP/Accelerate，也不跨机切分一个模型**。
参考训练指南的 Arnold 节点编号、统一 run ID、日志和跨节点完成门禁；不复用训练 rendezvous。
`ARNOLD_WORKER_HOSTS`、`MASTER_PORT`、worker-local `PORT` 均不参与推理协调。

按 source 分组做确定性负载分配；同一张图的全部 mask 留在同一节点，保留原本的邻居保护上下文。
image ID 不重编号，每个 region 只执行一次。4个本地8卡池即32个独立模型副本并行，SAM解析阶段各机使用本机第一卡。
分片改变批次组合，MLLM 有随机采样；“方法不变”不代表跨不同拓扑逐像素、逐字符复现。

## 2. 最短正式入口：四台 worker 都执行

先确保分支已包含本指南和 `scripts/labeling/`，再将
`scripts/labeling/bootstrap_arnold_4node.sh` 复制到四机可访问的共享路径，或将其完整内容粘贴为 Arnold entry。
本次维护的共享副本路径为：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/launchers/bootstrap_arnold_4node.sh
```

Arnold 配置 **4 workers × 8 GPUs**，在四个 worker 共同入口填写同一个新名称：

```bash
#!/usr/bin/env bash
set -euo pipefail
export SAMTOK_RUN_ID="samtok-remove-4n-20260924-001"  # 每次提交换新名字
export SAMTOK_LIMIT_SOURCES=0                      # 0=全部正例，不是0条
bash /mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/launchers/bootstrap_arnold_4node.sh
```

不需要 W&B key。不要仅在 worker0 执行，也不要用本机现存 repo 代替 clone。
默认会由 node0 从源 parquet 准备全部正例、每个原始 region 使用一次；不会为凑100k反复复制输入。
12,337条源记录中正例7,671条，排除4,666条No target；实际任务数等于正例所有mask之和。
历史试验是否重复不在这个全量入口额外筛选，避免擅自改变全量数据范围。

只跑小规模四机 smoke 时，设 `SAMTOK_LIMIT_SOURCES=8` 并换新 run ID。它限制源图数，不是 region 数。
若已经准备好了输入，在同一入口额外指定：

```bash
export SAMTOK_DATA_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/fresh_ownership_v15_20260924"
```

这里必须已有 `annotations.jsonl` 和 `sources/`；该20例目录只适合部署 smoke，不是正式全量输入。
不设置此变量时，默认数据准备位置为本次 `$SAMTOK_RUN_ROOT/data/source/`。

## 3. 不依赖预先复制共享脚本的入口

下面同样由四台 worker 执行。先下载小型 bootstrap，然后由 bootstrap 真正 clone 分支并安装；
不是尝试调用一个“还没 clone 的 repo”里的文件。下载与后续输出都保存到 experiments。

```bash
#!/usr/bin/env bash
set -euo pipefail
export SAMTOK_RUN_ID="samtok-remove-4n-20260924-002"
export SAMTOK_BRANCH="samtok-derived-edit-labeling"
export SAMTOK_RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID"
: "${ARNOLD_ID:?Arnold must set rank}"
mkdir -p "$SAMTOK_RUN_ROOT/logs"
exec > >(tee -a "$SAMTOK_RUN_ROOT/logs/entry.node$ARNOLD_ID.log") 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
entry_file="$(mktemp /tmp/samtok-label-bootstrap.XXXXXX.sh)"
curl --fail --location --retry 3 \
  "https://raw.githubusercontent.com/Tangent0308/sam3-crispedit/$SAMTOK_BRANCH/scripts/labeling/bootstrap_arnold_4node.sh" \
  --output "$entry_file"
bash "$entry_file"
```

如果网络要求代理，可改用第一种共享入口并设置 `SAMTOK_KEEP_PROXY=1`；不在脚本内硬编码代理或凭据。
bootstrap 默认直连。四机代码SHA、依赖版本和输入SHA必须一致，否则在规划前失败，不会混用移动中的分支。
更严格的发布可设置 `SAMTOK_EXPECTED_COMMIT`，要求 clone HEAD 等于指定完整 commit。

## 4. clone、安装和真实路径

每台机器执行以下操作，bootstrap 已自动串好，无需再手工重复：

```bash
git clone --branch samtok-derived-edit-labeling --single-branch \
  https://github.com/Tangent0308/sam3-crispedit.git \
  "/opt/tiger/tanyue/labeling_runs/$SAMTOK_RUN_ID/node$ARNOLD_ID/repo"
cd "/opt/tiger/tanyue/labeling_runs/$SAMTOK_RUN_ID/node$ARNOLD_ID/repo"
python3 -m pip install --user --index-url https://bytedpypi.byted.org/simple/ 'uv==0.11.32'
export UV_BIN="$(python3 -c 'import site; print(site.getuserbase())')/bin/uv"
bash scripts/labeling/setup_env.sh
```

本地仓库默认按run/node隔离，不共享可变venv，不在四机同时写同一repo。
与训练指南“node0安装共享单环境”不同，这里有三个依赖不兼容的已验证运行环境：

| 环境 | 核心版本 | 用途 |
|---|---|---|
| `.runtime/sam` | Python3.12.13、torch2.8.0/CUDA12.8、transformers4.57.6 | 源数据准备、SAM区域解析、CPU调度 |
| `.runtime/mllm` | torch2.13.0+cu129、vLLM0.28.0+cu129、transformers5.17.0 | 27B规划和审核 |
| `.runtime/editor` | torch2.13.0+cu129、vLLM0.29.0、diffusers0.40.0、transformers5.14.1 | Qwen-Image-2.1编辑 |

SAM源代码固定 `fff5ca124cf2551dd73c0de2af9c64bdadeea0b3`；Omni固定
`44ea27c8094095bbffd88fa3befdfaa55ba4bc50`，与当前跑通的官方Qwen2.1实现一致，不拉latest改变方法。
本仓库的regional扩展仍通过官方Omni接口加载。H100上仍使用当前已验证的 `TORCH_SDPA`，不换注意力实现。

`requirements/labeling-*.lock.txt` 保存已验证环境的完整版本快照，安装时使用 `--no-deps` 防止解析器
改写CUDA组合；不运行根目录历史 `pip install -r requirements.txt`。两个指定源均可信且所有包版本固定，
安装用 `unsafe-best-match` 解决PyTorch索引遮蔽普通包问题。SAM的无CUDA后缀版本使用严格 `===`，避免
`==2.8.0` 误选另一CUDA构建。环境会占用较多本地磁盘，预留至少100GB用于环境、依赖源码和缓存。
安装结束逐环境检查版本、CUDA、SAM/Omni API import，写 `.runtime/environment.json`；失败不进入生成。

随后默认将Qwen-Image-2.1逐文件复制、SHA256校验到每台机器本地的
`/opt/tiger/tanyue/labeling_model_cache/$SAMTOK_RUN_ID/qwen21`，再启动8个编辑进程。
这是字节相同的权重，不改dtype、模型内容或采样方法。源和目标的校验记录存于
`staging_manifest.json`，四机还会比对该清单。首次共享盘复制仍有成本，但只读一次，
避免8个进程在FUSE共享文件上同时mmap权重触发启动超时。
建议本地预留150GB（环境、下载缓存和31GB编辑模型副本）；可用`SAMTOK_MODEL_CACHE_ROOT`覆盖缓存父目录，
四机应使用相同的节点本地路径。已有高性能存储且明确无需复制时才设`SAMTOK_STAGE_EDITOR_MODEL=0`。

部署依据：[vLLM GPU安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)；
[本轮固定的Omni源码版本](https://github.com/vllm-project/vllm-omni/tree/44ea27c8094095bbffd88fa3befdfaa55ba4bc50)。
为避免ABI变化，优先复现本轮实测环境，不把文档的滚动latest版本当固定依赖。

四台机器都必须可读：

```text
源 parquet:
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet
27B:
/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B
编辑模型（共享路径，不再依赖旧机 /tmp）:
/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1
SAM:
/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt
```

训练用的2511不是这里的编辑模型。模型不联网补下载；不能读共享数据/权重时必须先修挂载。
Arnold镜像需已有git、curl、可运行的python3/pip、NVIDIA驱动；无需sudo apt大范围修改宿主。

## 5. 日志、输出与进度

默认正式运行根目录：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID
├── logs/bootstrap.node0.log ... node3.log    # clone、安装、准备与调度输出
├── logs/pipeline.node0.log ... node3.log     # 各节点规划/解析/编辑总入口
├── logs/audit.node0.log ... node3.log        # 审核总入口
├── reports/topology.json                    # 四节点一致性验收
├── reports/partition.json                   # 输入/各分片SHA、全部case数
├── reports/progress.node<N>.json            # 当前阶段/耗时
├── reports/final.json                       # 最终计数，成功时才有
├── control/                                # claim、done、failed、finalize
├── inputs/node<N>/                         # 按源图分组的manifest
├── nodes/node<N>/pipeline/                  # 原流程所有中间证据及内部日志
├── nodes/node<N>/audit/                     # 审核输入、prompt、完整回复
└── results/
    ├── all_cases.jsonl                     # 包含fail和no_output，不隐藏分母
    ├── audit.jsonl                         # 全部成图的审核
    └── model_pass.jsonl                    # 模型通过的候选；不是人工质量认证
```

这些日志路径全部位于experiments；不会散落到repo或旧 `.artifacts`。模型缓存和venv可在本地。
图片仍在各node的`pipeline/editing/context_grounded_v4_qwen21/edited`，汇总清单有绝对路径，避免复制几万张图。
不删失败图片、输入、规划拒绝或中间诊断图；磁盘容量应按全部尝试估计，不只按最终pass估计。

```bash
export SAMTOK_RUN_ID="samtok-remove-4n-20260924-001"
export RUN_ROOT="/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID"
tail -F "$RUN_ROOT/logs/bootstrap.node0.log"
tail -F "$RUN_ROOT/nodes/node0/pipeline/logs/plan_0.log"
tail -F "$RUN_ROOT/nodes/node0/pipeline/editing/logs/editor_0.log"  # 模型tqdm
tail -F "$RUN_ROOT/nodes/node0/audit/worker_0.log"                 # audited x/y
cat "$RUN_ROOT/reports/partition.json"
cat "$RUN_ROOT/reports/progress.node0.json"
cat "$RUN_ROOT/reports/final.json"
```

将node0换成node1--3可看其他机器。不同机器可以处于不同阶段，各自使用自己的卡，不需要全机阶段同步。
只有全部node完成、ID/审核覆盖检查通过，node0才写`control/finalize.ok.json`和最终清单。
`pass`是审核通过；`fail`是已成图但审核失败；`no_output`是规划或区域解析未放行，不是成功生成。

## 6. 防覆盖与故障

每个run ID只能提交一次；每个节点以独占claim文件认领，重复启动不会覆盖已有输出。
四节点比对源manifest SHA、代码SHA、完整冻结profile、包版本、GPU数、共享模型路径/配置。
默认要求4个不同hostname和每节点8个GPU槽位。`--local-test`只用于开发模拟，正式bootstrap不传。

任一节点发生异常会写`control/*.failed.json`，其他已进入调度的节点发现后终止自己的模型进程组，
不会单独把半批结果标为完成。被外部SIGKILL的进程无法写marker，因此仍依赖等待超时；当前没有心跳租约恢复。
环境准备期间其他节点可能仍在安装，安装是有限操作；安装后进入门禁会看到失败，不继续生成。

默认节点加入等待3小时，单阶段/最终等待7天。大任务按实际预算评估；入口不提供透明断点续跑、
自动换seed重试、失败case重采样或补齐到100k。故障后保留目录，换run ID；不要手动伪造done文件。
`SAMTOK_REPO_DIR`、`SAMTOK_RUN_ROOT`、模型位置和数据位置可覆盖，但四机共享数据/输出位置必须一致。

## 7. 本次验证边界

当前交互会话的真实配置为1台8卡，因此不能声称已完成真实四台主机32卡的网络/共享存储联调。
已覆盖的本地测试和真实模型链路结果记录在统一的 `docs/SAMTOK_QUALITY_ITERATION.md` 最新部署小节；
部署验证产物统一位于experiments下的`deployment_validation_20260924`。
单机4个逻辑worker、每个2卡的实测已完成：8源图9region，全部成图并完成审核汇总，四worker退出0、
`run_localweights/control/finalize.ok.json`存在。9条均为模型pass，本轮未新增逐图人工验收。
链路最长约9.1分钟（不含安装/数据准备/首次缓存复制），编辑本体平均15.43秒/条。
最终安装脚本也从空目录跑通三套环境，单元测试341 passed，另5项融合测试在新环境通过。
首次真实四机建议先用8张源图做smoke，确认四台bootstrap日志与finalize完成，再换run ID令limit=0正式运行。
小批次冷启动时间不能外推为100k稳定吞吐，也不保证四机刚好达到四倍加速。
