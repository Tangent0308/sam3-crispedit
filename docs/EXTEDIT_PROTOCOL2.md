# extedit：CrispEdit / ScaleEdit 协议 2.0 打标管线

状态：**已实现，验证与提速完成，待用户确认规模后量产**（2026-10-21）。计划与旧流程的逐段核查见 SAMTokEdit v3 仓库 `docs/07_SAMTokEdit_Qwen21_外部编辑数据打标管线.md`。

路径记号：
- `datasets/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/datasets/…`；
- `VAL/…` 指 `/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/04_current_validation/extedit_protocol2_20261020/…`。

## 0. 一页摘要
- **输入：**两个来源本地新下载的原始对：
  - CrispEdit `datasets/CrispEdit_Labeling/source/CrispEdit-2M/`；
  - ScaleEdit `datasets/ScaleEdit_Labeling/source/ScaleEdit-filtered-source/expand-dl300k20261020*`。
  - 旧流程用过的行按排除清单跳过。
- **输出：**与 D3 / D4 同格式的记录，外加 token 缓存、拒收清单、汇总和审阅图。
- **和旧流程的区别：**
  1. 去掉场景难度筛（用户 10-20 确认）。
  2. 27B 从每条 5–6 次合并成 2 次（规划、审核）。
  3. 像素预检排在所有 GPU 步骤之前。
  4. add 的 S 直接按锚点标，并切到落点。
  5. token 往返在分割后立即算。
- **验证结果：**360 个原始对收下 141 条（39.2%），看图抽查语义全部正确，见 §6。

## 1. 步骤
每一步写自己的 `records*.jsonl`，按 id 断点续跑；出错的行下次重跑；被丢的行都记下阶段和原因。

| 步骤 | 子命令 | 设备 | 做什么 |
|---|---|---|---|
| 1 取数 | `ingest` | CPU | 统一行格式，按类型配额抽样；排除旧流程用过的行、短边 < 512、同一源图 > 3 条、与 benchmark 2.6.0 源图相同或近重复（dHash 距离 ≤ 6）的行；图片按原字节写出 |
| 2 像素预检 | `precheck` | CPU | 尺寸对齐；结构不在原位时用 SIFT + RANSAC 仿射配准；Lab 色差滞回得到改动图并编号 |
| 3 规划 | `plan` | 27B | 一次调用：编辑是否干净、编辑单元、协议 2.0 槽位、S 的框 |
| 4 分割 | `segment` | GPU | SAM3 分割 S；add 先分割宿主，再切到新物体落点；SAMTok 编码再解码 |
| 5 把关 | `gate` | CPU | 每个单元的改动区域、门限、几何检查；整行贴回及贴回后检查 |
| 6 审核 | `audit` | 27B | 渲染 ref / no-ref 句子，两票审核，两票都过才收 |
| 7 导出 | `export`、`review` | CPU | D3 / D4 记录、token 缓存、拒收清单、汇总、审阅图 |
| 全流程 | `run` | 全部 | 单机上把 2–6 步作为并行的子进程一起启动，每步 `--follow` 跟随上游；27B 服务需先启动 |
| 合并 | `publish` | CPU | 把多个运行（多机的各节点）的 `final/` 合并成一个正式数据目录，图片一并复制 |

- **流式并行（`--follow`）：**
  - 每一步持续读取上游新写出的记录，直到上游写出 `DONE` 标记、且没有新行；
  - 这样 27B 服务同时接规划和审核的请求，SAM3 与 CPU 步骤同时在做，GPU 不会在步骤之间空等；
  - 记录按增量读取（只读文件新增的部分），10 万行规模也不会反复重读。

## 2. 规则

### 2.1 像素预检（`pixels.py`）
- **对齐：**
  - 先把编辑图缩放到源图尺寸，用纹理块互相关检查结构是否在原位（≤ 1 px 的块 ≥ 80% 即算对齐）。
  - 不在原位就配准：仿射，方法和参数同 `clean_pairs.register`。
  - 配准后对齐率要 ≥ 0.4 且比原来高 0.1，覆盖率 ≥ 85%；贴回区域还要求 ≥ 98% 落在配准后的画面内。
  - 仍不在原位的丢弃（`drop_misaligned`）。
- **改动图：**与 `edited_region` 相同的 Lab 色差滞回（阈值 6 / 12，随噪声升高），噪声取全图中位数的 3 倍。
- **丢弃：**

  | 原因 | 条件 |
  |---|---|
  | `drop_no_change` | 改动面积 < 0.05% |
  | `drop_global_change` | 改动面积 > 40%，或噪声 > 30（整张重画） |

- **编号：**相距 1% 对角线以内的改动块合并，最多编 8 个，给规划看。

### 2.2 规划（`plan.py`，Qwen3.8-27B，thinking low）
- **输入：**
  1. BEFORE（1 MP，框画在这张上）；
  2. BEFORE 上按编号描出改动块；
  3. 对齐后的 AFTER；
  4. 改动块的前后特写；
  5. 原指令只作参考。
- **输出：**
  - `quality`：`clean` 或 `reject`，拒收原因为 `no_change`、`global`、`partial`、`defect`、`unrelated_change`、`not_local`、`unclear`、`too_many` 之一。
  - `units`：1–4 个编辑单元。每个单元给：
    - 协议 2.0 槽位；
    - S 的框；
    - add 的宿主框，add / replace 的新物体框；
    - 所属改动块编号；
    - `comparable`：有没有可混淆的同类近邻，只记录，不用来筛。
- **槽位规则：**沿用生成 D3 的 protocol-adapt-4 槽位提示（add 的 S 分 object / part / surface 三种粒度），写作规则用 benchmark 2.6.0 的 `INSTRUCTION_STYLE`（去掉 add 选物策略）。
- **提示词布局：**固定的规则（约 3 千 token）放在图片前面，所有请求共用 vLLM 的前缀缓存；每对自己的原指令和改动块列表放在图片后面。
- **文字：**新写的字必须清晰、拼写正确，是真实的词、名称或数字（店名、品牌名可以）。乱码、错拼、粘连的拒收（`defect`），即使原指令就是这样要求的（ScaleEdit 有 "UNIVERSITYCHOLARS" 这类指令）。
- **自动纠错：**确定性修正（关系别名、冠词等）照搬 `normalize_slots`。槽位不合规时说明问题再问，最多 3 次，例如：
  - add 的关系是 near / beside；
  - 缺宿主框；
  - 新物体前写了 "the"。

### 2.3 分割与 token（`segment.py`）
- **SAM3：**设置与打标仓库相同（`segment_grounded_box`，框 < 65% 时留 25% 上下文裁图）。
- **非 add：**提示词是 ref 去掉冠词。
- **文字单元（改字、删字或 logo）：**
  - S 用文字所在的区域，而不是 SAM3 分出来的笔画：依次试凸包、闭运算填充后的区域、原笔画，取第一个 token 往返 ≥ 0.5 的；都不行就取往返最高的。
  - 旧 ScaleEdit 打标给文字用的也是"紧贴改动字符的凸多边形"。
  - 文字验证中，笔画 mask 往返达标率 62%，凸包 73%。
- **add：**
  1. 新物体位置 = 所属改动块（限制在新物体框内，取主块）。
  2. 分割宿主（object 粒度就是 S 本身）；part 粒度另分割部件，部件在宿主内 ≥ 70%、≤ 宿主 85%、≥ 新物体 0.3 倍时用部件。
  3. 用 `localize_anchor` 切到落点；有 site 时再用协议 2.0 的 `site_spot` 切。
- **token：**SAMTok（Qwen3-VL-8B-SAMTok 的 VQ-SAM2，经 v3 仓库 `samtok_edit21.models.codec` 加载），编码 S 再解码；IoU < 0.5 的单元不用。

### 2.4 把关与贴回（`gate.py`）
- **每个单元：**
  - 改动区域：`edited_region`（lab-diff-hysteresis-2）。
  - 支持区域：

    | 操作 | 支持区域 |
    |---|---|
    | attribute | S |
    | remove | S 外扩边长 18% |
    | replace | S 外扩边长 50% |
    | add | S + 新物体，外扩新物体边长 18% |

  - 贴回区域：S（add 为新物体）、支持区域内规划归给这个单元的改动块，以及与它们相距 1% 对角线以内的改动块（影子、倒影、露出的背景）。支持区域里孤立的重绘噪点不贴。改属性只贴 S 本身：S 没盖住的改动不贴，结果不完整时由审核拒收。
- **硬门限：**
  - S 几乎没变（< 10%）；
  - 替换的 S 改动 < 50%，改属性 < 30%（文字单元不查这一条，字与字之间的底色本来不变）。只挡大部分没改的；替换 < 85%、改属性 < 90% 打标记 `s_partly_changed`。车的轮胎和镀铬、标志的白底不会变色，完整的改色覆盖率也只有 0.33–0.72，所以是否改完整由审核的 result 一项判断。
  - add 锚点（除新物体处）改动 > 60%；
  - 几何：支持物不接触新物体、不在其下方等，规则同 `geometry_flags`；
  - token 往返 < 0.5。
- **整行：**
  - 所有贴回区域之外不得有成块改动（> 1% 图像，旧 `drop_C`）。
  - 色调：生成器常把整张图调亮或调暗。先在远离编辑处逐通道拟合增益和偏移，把 AFTER 的色调对齐到 BEFORE（差异下降 ≥ 1 个色阶才用），否则过渡带会出现一圈亮边或暗边。
  - 贴回：贴回区域外扩短边 2%（改属性 0.5%），高斯软边，再往外严格等于 BEFORE。
  - 贴回后复查，门限同旧 GATE：

    | 项 | 门限 |
    |---|---|
    | 区域外改动 | ≤ 2% |
    | 区域内改动 | ≥ 10% |
    | 边界被切 | ≤ 0.30 |
    | 接缝 | ≤ 25 |
    | 贴回面积 | ≤ 60% |

- 最后在贴回后的图对上重算每个单元的 `edited_region`，作为记录里的值。

### 2.5 审核（`audit.py`）
- **句子：**ref / no-ref 由 `render(spec, True/False)` 生成，no-ref 经 `validate_noref`；无法归类的限定语只留 ref（`noref_ineligible`）。
- **审核：**
  - 视图：BEFORE（S 绿、执行区橙）、贴回后的 AFTER、特写。
  - 7 项：edit、relation、target、outline、noref、wording、result。前 6 项同 protocol-adapt-4；result 是新加的，专门查半成品替换、整体没改完的改色、add 接触不对、伪影和接缝。
  - 两票都过才收。两票是同一请求里的两个采样（`n=2`），共用一次预填充；解析失败的那一票单独重问。
  - 多单元的行逐单元并行审核，提示 AFTER 里其他改动属于别的指令。
  - result 一项也查文字：必须清晰、拼写正确。

## 3. 环境
- **27B 服务：**`requirements/extedit-vllm.txt`（vLLM 0.31.0+cu129，与 benchmark 2.6.0 评审相同）。启动命令：

  ```bash
  python scripts/extedit_pipeline.py serve-command --vllm <env>/bin/vllm --gpus 8
  ```

  得到：

  ```bash
  vllm serve <Qwen3.8-27B> --served-model-name qwen38-27b --data-parallel-size 8 --max-model-len 20000 \
    --gpu-memory-utilization 0.9 --max-num-seqs 96 --enable-prefix-caching --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 6}' --host 127.0.0.1 --port 8000
  ```

- **其余步骤：**`requirements/extedit-worker.txt`（Python 3.11，torch 2.13.0+cu129）。
  - SAM3 与 `crispedit` 从本仓库导入；
  - SAMTok codec 从 SAMTokEdit v3 仓库导入（`--v3-repo`，默认 `/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3`）。
- 运行时设 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONPATH=<本仓库>`。

## 4. 运行

### 4.1 单机，分步（验证用）

```bash
RUN=<运行目录>; R=<本仓库>; PY="<worker env>/bin/python -B $R/scripts/extedit_pipeline.py"
$PY ingest   --run $RUN --crispedit add=30,remove=30,replace=30,color=30 \
             --scaleedit object_addition=24,object_removal=24,object_replacement=24,color_change=24,material_change=24
$PY precheck --run $RUN --workers 32
$PY plan     --run $RUN --url http://127.0.0.1:8000/v1/chat/completions --concurrency 64
$PY segment  --run $RUN --device cuda:0 --shard 0/1      # 多卡：每卡一个 --shard k/n
$PY gate     --run $RUN --workers 32
$PY audit    --run $RUN --url http://127.0.0.1:8000/v1/chat/completions --votes 2
$PY export   --run $RUN
$PY review   --run $RUN --accept 60 --reject 40
```

### 4.2 单机，流式并行（量产用）
先启动 27B 服务（§3），再用 `run` 把各步作为并行子进程一起跑，结束后自动导出：

```bash
$PY run --run $RUN --whole-shards --node 0/1 --crispedit add=0,remove=0,replace=0,color=0 \
        --url http://127.0.0.1:8000/v1/chat/completions --plan-concurrency 168 --audit-concurrency 112 \
        --segment-devices 7
```

- 配额写 0 表示该类型不限量；
- `--whole-shards` 按分片顺序整片读取，`--node k/n` 让第 k 台机器只取第 k、k+n、… 个分片；
- 各步日志在 `<RUN>/logs/`，中断后重跑同一条命令即可续上。

### 4.3 多机
- 每台机器运行 `scripts/extedit_node.sh <运行根目录> <k> <n> [run 的配额参数]`：
  - 脚本用 GPU 0–6 跑 27B 服务（可先把模型复制到本地盘），GPU 7 跑 SAM3 + SAMTok，再执行 `run --whole-shards --node k/n`；
  - 环境变量 `WORKER_PY`、`VLLM`、`MODEL`、`V3_REPO` 指定两个环境、模型和 v3 仓库。
- 全部结束后执行 `$PY publish --runs <根目录>/node0 … --out datasets/ExtEdit_Protocol2_<日期>`：
  - 把各节点 `final/` 合并成正式数据目录；
  - 收下行的原图和贴回后的 AFTER 复制到 `images/` 下。
- **存储：**每个原始对解出的两张原图约 3 MB，贴回后的 PNG 约 4 MB。运行目录里的原图可以在发布后删除，需要时能从源分片重新解出。

## 5. 输出（`<RUN>/final/`）

| 文件 | 内容 |
|---|---|
| `manifest_protocol_v2.jsonl` | 单目标记录，字段同 D3：`instruction_spec`、`instruction_ref`、`instruction_noref`、`selection`、`execution_region`、`edited_region`、`mask_semantics`、`flags`；另有 `provenance`、`observed`、`comparable`、`audit` |
| `manifest_multi_protocol_v2.jsonl` | 多目标记录，字段同 D4，`units` 逐单元同上 |
| `tokens.jsonl`、`multi_tokens.jsonl` | 字段同 round-2 的 `v1_tokens.jsonl` / `d4_tokens.jsonl`：`span`、`dec`、`roundtrip`、`size_ok`、`out_diff`、`out_share8` |
| `rejected.jsonl` | 每条被丢的行：阶段、原因、审核意见 |
| `summary.json` | 各来源类型的行数和收下数、各阶段拒收原因 |

- 图片路径相对运行目录：
  - 原图 `images/<id 末两位>/<id>.source|target.<原格式>`；
  - 贴回后的 AFTER 是 `pasted/<id 末两位>/<id>.target.png`。
- 审阅图在 `<RUN>/review/`。

## 6. 验证结果（2026-10-20）
- **验证数据：**两轮，都取自新下载的数据：
  - 第一轮 120 条：CrispEdit add / remove / replace / color 各 15，ScaleEdit 五类各 12；
  - 第二轮 240 条：数量翻倍。
- **运行环境：**单机，27B 用 6 卡，SAM3 用 2 卡。
- **结果位置：**`VAL/val_v1/`、`VAL/val_v2/`，迭代过程见 `VAL/README.md`。下面的数字是两轮都用最终版代码的结果。

### 6.1 产量
| 来源 / 类型 | 原始对 | 预检 | 规划 | 把关 | 审核（收下） | 产量 |
|---|---:|---:|---:|---:|---:|---:|
| CrispEdit add | 45 | 33 | 28 | 23 | 16 | 36% |
| CrispEdit color | 45 | 31 | 22 | 18 | 15 | 33% |
| CrispEdit remove | 45 | 27 | 25 | 22 | 17 | 38% |
| CrispEdit replace | 45 | 37 | 32 | 26 | 22 | 49% |
| ScaleEdit 改色 | 36 | 31 | 27 | 19 | 15 | 42% |
| ScaleEdit 改材质 | 36 | 29 | 29 | 22 | 20 | 56% |
| ScaleEdit 添加 | 36 | 32 | 28 | 11 | 6 | 17% |
| ScaleEdit 删除 | 36 | 28 | 25 | 16 | 13 | 36% |
| ScaleEdit 替换 | 36 | 29 | 27 | 23 | 17 | 47% |
| **合计** | **360** | **277** | **243** | **180** | **141** | **39.2%** |

- **收下的内容：**141 条，其中单目标 129 条、多目标 12 条，共 156 个单元：

  | 操作 | 单元数 |
  |---|---:|
  | 改属性 | 61 |
  | 删除 | 35 |
  | 替换 | 34 |
  | 添加 | 26 |

  - 只有 1 个单元没有 no-ref 句；
  - 收下单元的 token 往返 IoU 中位数 0.94（最低 0.52）；
  - 贴回后 AFTER 在编辑区域外与 BEFORE 逐像素相同（`out_diff` = 0）。
- **对比：**旧流程端到端只有 1.5%（CrispEdit）和 2.5%（ScaleEdit）；文档 07 的估计是 20%–30%。

### 6.2 拒收原因（219 条）
| 阶段 | 条数 | 主要原因 |
|---|---:|---|
| 预检 | 83 | 整张重画、换了构图，配准不上 50；没改动 18（CrispEdit 有编辑失败的对，配准后逐像素几乎相同）；全局变化 15 |
| 规划 | 34 | 无关改动 18；不是局部编辑 5；有缺陷 4；编辑过多 3 |
| 把关 | 63 | token 往返 < 0.5 有 32（主要是很小的物体和大片表面上的 add）；贴回边界切到编辑或有接缝 8；S 没变 7；区域外成块改动 6 |
| 审核 | 39 | 描边不对 24、目标不对 16、no-ref 不成立 15、结果不完整 13（半成品替换、删除留下污迹、两个圆顶只改了一个、改色漏掉 S 外的一片叶片）；一条可能有几项 |

- 两票一致率 90%（204 个单元里 183 个）。

### 6.3 看图抽查
- **样本：**两轮共看了约 30 条收下的样本，覆盖各操作和多目标。
- **语义：**全部正确，操作、目标、描边、ref / no-ref 都对得上。
- **措辞小问题：**跪着的人写成 "sitting"；"改成砖材质"实际更像红色面板。
- **贴回问题：**发现两处，已修正：
  - 整体偏色带来的亮边，现在先做色调匹配；
  - 替换的贴回区域里有孤立噪点，现在只贴与 S 相连的改动块，以及规划归给这个单元的改动块（否则一次添加的几朵新花里，小的几朵会被当成区域外的无关改动）。
- **门槛：**被硬门槛挡掉的样本也看过：
  - 覆盖率门槛最初偏严，误挡了完整的改色（车、标志、座椅），已放宽到只挡"大部分没改"；
  - 配准、无关改动、接缝几类拒收都是对的。
- 审阅图在各运行的 `review/` 下。

### 6.4 算力
分步跑时的估算已被 §6.6 的流式实测取代。

### 6.5 文字类（ScaleEdit 物体表面文字、建筑表面文字，2026-10-21）
- **结果：**60 个原始对；S 改用文字区域（凸包）后，收下 30 条（50%）；用笔画 mask 时只有 19 条（32%）。
- **看图：**改字正确，S 覆盖整行字。
- **措辞：**模板写成 "Change the text on the … lettering to "…""。
- **问题与修正：**发现原指令本身要求乱码（"UNIVERSITYCHOLARS"），生成器照写，审核也放过了。已在规划和审核里加上"新字必须拼写正确、清晰"。

### 6.6 提速（2026-10-21）
在同一台机器上（6 卡 27B + 2 卡 SAM3；本机另有一个进程长期占用约 40% GPU 算力），对 720 个原始对做了对比：

| 跑法 | 全流程用时 | 每分钟原始对 | 每个原始对的 GPU 秒（8 卡） |
|---|---:|---:|---:|
| 分步跑（§6.1 的验证） | 240 对约 26 分钟 | 9 | 约 52 |
| 流式并行，请求在主线程里准备 | 中途停止：客户端单核 100%，每个引擎只有 3 个请求在跑 | — | — |
| 流式并行 + 进程池准备请求 + 规则前缀缓存 + 审核 n=2 | 595 条规划、全流程 13 分钟 | 55 | 约 8.7 |

- **布局 A/B：**同一批 160 行，规则放在图片前与全部文字放在图片后：

  | 布局 | 重叠率中位数 | 严重错框 |
  |---|---:|---:|
  | 规则在图片前 | 0.976 | 6 |
  | 全部文字在图片后 | 0.993 | 10 |

  框和改动块的重叠率基本相同，前缀缓存不损失质量。
- **框校验：**120 行中有 2 行因框没碰到自己的改动块被重问，第二次都答对了。
- **产量：**这批 720 对收下 232 条（32.2%），文字两类 42%，比 §6.1 低。同一批行上换布局不影响框，差别来自样本（CrispEdit 不同分片的产量差异很大）。
- **量产估算：**
  - 按每个原始对 8.7 GPU 秒计（本机有占用，正式节点应更快），4 台 8 卡每小时约 1.3 万个原始对；
  - 10 万对约 7.5 小时；全部约 60 万对约 45 小时，预计产出约 19–23 万条。

## 7. 已知限制
- **大片表面上的 add：**天空、墙面、水面上的 add，锚点被切成一块没有边界的区域，SAMTok 难以还原，往返 IoU 常 < 0.5，这些单元不收。D3 有同样的问题。add 产量 17%–36%。
- **重新构图：**AFTER 被整张重画、换了构图的对配准不上，在预检丢弃。约占 11%–14%，主要是 ScaleEdit 删除类和 CrispEdit 部分分片。
- **多机脚本未在 ARNOLD 上实测：**`scripts/extedit_node.sh` 只在本机按单节点验证过各步命令。
