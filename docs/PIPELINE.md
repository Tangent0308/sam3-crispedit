# 当前 SA-1B 标注管线

## 1. 目标和当前状态

管线的输出是一条 mask 对应一条独立的局部编辑样本：原图、精简指令、源区域信息、目标图、执行区域和审核记录。当前协议支持四类操作：`add`、`replace`、`remove`、`attribute`。正式目标是 50,000 条自动审核通过样本，四类各 12,500 条；每个四机 rank 的目标是每类 3,125 条。

当前正式运行根目录为：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014
```

写本文时的运行快照是首个 wave 正在规划阶段：每个 rank 已加载完 131 个规划模型 shard，首个 wave 的单个规划副本已完成约 810 个 mask 调用，首轮候选约 64 张源图；自动通过数尚未增加，因为该 wave 还未进入生图审核。实时状态以共享目录中的 `nodes/node*/progress.json`、`nodes/node*/waves/*/planning_*.log` 和 `logs/progress_monitor.log` 为准。该快照会随任务继续变化。

## 2. 源数据

源数据是 SA-1B 图像及其 dense label 记录。准备阶段从 SG2 的随机访问索引冻结 16,000 个 source ID，并提取 32,000 条原始 TFRecord 记录（每张图的 image 和 annotation），缓存约 17.42 GiB。缓存保留原始 TFRecord 字节、记录长度、CRC32C、SHA-256 和 source ID，四机运行通过 `shared_record_cache_v1` 读取，不依赖任务内的 SG2 HDFS 挂载。

源选择固定随机种子，排除 benchmark source ID 和原图字节哈希；每个 rank 负责不重叠的 4,000 张源图。源图中的所有可解码、非空 mask 都进入供给池；尺寸不一致、空 mask 或无法解码的记录只记录为异常，不按面积、遮挡或主体美观进行隐式筛选。一个源图可以贡献多个独立 case，多个 case 不代表同一目标图联合执行多个编辑。

## 3. 数据要求和指令协议

每个 mask 必须绑定一个明确的源对象或部件。`ref` 是对该 mask 的唯一、无歧义指代；同类多个实例使用颜色、相对位置、宿主和部件层级区分。`box`、`point` 和 `mask` 是同一 source selector 的不同输入形式，不能改变目标含义。

- `remove`：删除 mask 所指的完整对象或明确部件；不适合删除手臂、人体关键部位或会破坏场景语义的区域时拒绝。
- `replace`：把 mask 所指对象替换为同一功能层级的新对象，指令只写对象和必要的属性/形态。
- `attribute`：只改变 mask 所指对象的指定可见属性，例如颜色、材质、纹理或图案；保持身份、轮廓和未请求属性。
- `add`：mask 是新增物体的直接宿主或直接附着区域，优先写“附着于/放置在”该宿主的自然表面；只有直接覆盖 mask 会突兀时，才退到相邻且距离很近的锚点。新增物体本身不作为已存在物体去描述。

指令采用单句、动词开头、只包含操作、目标和必要限定的形式。例如：`Change the color of the left wooden shutter of the middle window to deep forest green.`；`Add a small lantern attached to the lower rail of the selected shutter.`。不在 instruction 中堆叠审核规则、模型提示或多余位置解释；详细保护约束写入结构化字段和生成 prompt。

## 4. 代码实现

| 阶段 | 代码 | 作用 |
| --- | --- | --- |
| 源记录缓存 | `scripts/production/stage_source_cache.py`、`sa1b_pipeline/production_source_cache.py` | 按冻结 reservoir 提取原始 TFRecord；校验 CRC、长度、SHA 和 source ID；支持断点恢复。 |
| 源 materialize | `sa1b_pipeline/production_sources.py` | 从共享缓存还原原图和 dense mask，保持原始像素和所有有效 mask；写入 provenance。 |
| 规划 | `scripts/run_production_4node.py`、`scripts/run_balanced_planning.py`、`scripts/experiment_batched_planning.py`、`sa1b_pipeline/per_mask.py` | 每个 mask 一次 MLLM 请求；每个 wave 启动两个 TP=4 规划副本；输入为全景图、单 mask crop 和结构化 mask 信息；根据剩余类别缺额给定类型优先级，禁止把不合适的操作静默改成其他类型。 |
| 指令协议 | `sa1b_pipeline/instruction_protocol.py`、`instruction_style.py`、`instructions.py`、`planning_schema.py` | 解析 `ref`、`operation`、`scope`、`payload`、`attributes`、`placement`；检查精简句式、类型字段和 add 直接支撑原则。 |
| 生成 | `sa1b_pipeline/production_generation.py`、`scripts/run_generation_queue.py`、`sa1b_pipeline/editor.py`、`utils/qwen21_omni.py` | 每 rank 启动 6 个单卡编辑器；Qwen-Image-2.1/vLLM-Omni，40 steps、seed 0；区域保护和 compositor 记录 generation fingerprint。 |
| 审核 | `scripts/run_audit_queue.py`、`sa1b_pipeline/audit_backend.py`、`sa1b_pipeline/validate.py` | 每 rank 启动 2 个审核 worker；批量检查 target_done、non_target_preserved、natural、operation_correct、结构完整性和 mask/尺寸一致性。任一关键字段失败则拒绝。 |
| 配额与恢复 | `sa1b_pipeline/operation_balance.py`、`shared_operation_quota.py`、`production_state.py`、`work_queue.py` | 以自动审核通过数填充四类缺额；本地 SQLite 队列保存租约和提交；共享 JSON/JSONL checkpoint 支持中断恢复和重复提交隔离。 |
| 四机编排 | `scripts/run_production_4node.py`、`scripts/production/bootstrap_arnold_4node.sh` | 校验 rank、共享缓存、模型配置和代码 hash；按 source wave 规划、生成、审核、合并；最终验证数量、四类均衡、ID/mask 唯一性和图片可解码性。 |
| 运行时 | `scripts/production/setup_runtime.sh`、`scripts/production/check_runtime.py`、`requirements/production/*.lock.txt` | 三套独立 CUDA 12.9 环境；规划、编辑、审核均固定 PyTorch `2.13.0+cu129` 和明确的 cu129 vLLM wheel，ready 前执行真实 GPU matmul/RMSNorm 检查。 |

## 5. 小批量质量和速度

最新完整的 30 张源图、60 条独立编辑验证中，四类各 15 条：规划目视通过 52/60，目标图目视通过 46/60，自动审核通过 52/60，三者交集 40/60（66.7%）。该结果来自逐图检查，不能当成独立人工标注员统计；自动通过中仍有目视不合格样本，所以正式训练集不能只依赖自动审核。

主要缺陷是指令与 mask 错配、add 附着点不完整、attribute 边缘混色、replace 层级不准确、remove 对不适合目标的误用，以及自动审核漏检/误拒。当前正式管线保留所有原始审核记录，不覆盖自动判定；人工验收仍是训练集发布前的必要步骤。

速度证据分开记录：

- CUDA 12.9 新编辑环境的一条 40-step smoke：模型加载 28.3 秒，生成与保存 18.1 秒，输出尺寸 1500×2265；代表性图见下方。
- 已完成的 6 editor + 2 audit 小批量重放为 60 条原始结果约 293 秒，约 737 原始例/小时；这不含规划、冷启动和人工验收，也不是 50k 交付承诺。
- 当前四机首 wave 冷启动包含约 23–25 分钟规划模型加载/JIT；热态规划每约 54 个 mask 需要 67–75 秒。实际全量工期要以四机持续日志为准，不能用单条生成延迟线性替代规划、审核和类别补额成本。

![CUDA 12.9 smoke case](assets/production_cuda129_smoke_case.png)

该图展示同一 case 的源图、红色定位区域和 CUDA 12.9 生成目标图。原始 source、mask、target 和完整元数据保存在运行根目录 `verification/cuda129_fix/`，不复制进代码仓库。

## 6. 运行结果和查看方式

正式运行输出结构：

```text
nodes/node*/waves/*/planning_*/       规划输入、原始回复和 planned.jsonl
nodes/node*/waves/*/generation/       生成图、审核结果和队列 checkpoint
nodes/node*/auto_pass.jsonl           本 rank 自动审核通过样本
nodes/node*/validation.jsonl          本 rank 全部结果
final/auto_pass.jsonl                 四机合并的最终样本
final/summary.json                    数量、类型、唯一性和完整性检查
logs/node*.log                        Arnold rank 日志
logs/progress_monitor.log             后台进度监控快照
```

```bash
ROOT=/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014
python -m scripts.run_production_4node --root "$ROOT" status
tail -F "$ROOT/logs/node0.log" "$ROOT/logs/progress_monitor.log"
```

只有 `final/summary.json` 显示 `status=complete`、总数 50,000、四类各 12,500，才可以称为 pipeline 自动完成。人工发布前还要抽查并记录指令准确性、目标图完整性、非目标保持、自然性和 add 锚点是否正确。
