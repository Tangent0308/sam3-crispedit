# SA-1B SAMTok 编辑数据构造

本仓库实现基于 SA-1B 图像与 dense mask 的多类型局部编辑数据构造。当前生产版本逐个处理源图中的有效 mask，为每个 mask 规划一种固定编辑类型（`add`、`replace`、`remove`、`attribute`），用 Qwen3.8-Flash-Next 生成精简且无歧义的指令，用 Qwen-Image-2.1 生成目标图，再用 Qwen3.8-27B 做类型感知的自动审核。

当前正式入口是 `scripts/run_production_4node.py`，四机启动包装在 `scripts/production/bootstrap_arnold_4node.sh`。运行指南和冻结配置见 [docs/FOURNODE_RUN_GUIDE.md](docs/FOURNODE_RUN_GUIDE.md)。详细设计、代码路径、质量和实时进度见 [docs/PIPELINE.md](docs/PIPELINE.md)。历史方案已经压缩到 [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)。

## 已完成的 50k 数据

本批共生成并审核 79,821 条，自动通过 **50,000 条**，add/replace/remove/attribute 各 12,500 条，覆盖 1,132 张源图。推荐训练入口：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014/release/annotations/train.jsonl
```

图片、源区域 mask 和执行 support mask 都在同一 `release/` 内，所有媒体相对路径以 `release/` 为根。统计、字段协议、读取代码及每类三例的可视化见 [50k 数据说明](docs/DATASET_50K_ZH.md)；源记录缓存 A 已按确认删除，其他候选保留，见 [清理记录](docs/CLEANUP_CANDIDATES_50K_ZH.md)。数据为自动审核通过版本，尚未完成独立人工验收，文档保留了抽查发现的问题。

![50k 数据统计](docs/assets/production_50k/statistics.png)

## 目录

```text
sa1b_pipeline/                 核心数据、规划、生成、审核、队列和发布实现
scripts/run_production_4node.py  四机调度、配额、恢复和最终合并
scripts/production/            缓存源记录、启动、运行时检查和打包
scripts/run_balanced_planning.py  规划模型批处理入口
scripts/run_generation_queue.py   常驻编辑器 worker
scripts/run_audit_queue.py        批量审核 worker
scripts/shard_group_selection.py  分组数据四 rank 确定性分片
scripts/merge_group_shards.py     分片结果合并与硬校验
scripts/group_4node_entry.sh      分组数据 Arnold 四机 rank 入口
requirements/production/       CUDA 12.9 的三套锁定运行时
tests/                         协议、配额、恢复、缓存和生产回归测试
docs/PIPELINE.md               当前版本说明
docs/FOURNODE_RUN_GUIDE.md     四机运行指南
docs/DEVELOPMENT_HISTORY.md    迭代留痕
```

## 快速检查

```bash
python -m pytest -q tests/test_production_4node.py tests/test_production_source_cache.py tests/test_production_cuda_runtime.py
python -m scripts.run_production_4node --root /path/to/run status
```

正式四机入口需要 Arnold 注入 `ARNOLD_WORKER_NUM=4`、`ARNOLD_WORKER_GPU=8`、`ARNOLD_ID=0..3`，并挂载 `/mnt/bn/strategy-mllm-train`。入口会校验冻结代码包、共享源记录缓存和 CUDA 12.9 环境；不需要额外挂载原始 SG2 HDFS。完整命令、恢复和日志位置见四机指南。

## 分组数据（新增）

反事实孪生和紧邻多目标数据使用独立运行根目录，入口为
`scripts/prepare_group_inputs.py` 和 `scripts/run_group_pipeline.py`。准备阶段先写入
`eval_reservation.json`，再物化训练/评测候选；训练脚本会拒绝评测源图的 source ID
或原图字节哈希。`sa1b_pipeline/group_selection.py` 只做几何和配额预选，
`group_planning.py` 对每组只调用一次规划模型并校验实体/范围字段，`group_generation.py` 负责同 seed
孪生生成或 source→step 链式生成，`group_audit.py` 接入 27B 语义审核。

组级结果必须通过确定性检查、27B 审核和人工复核后才能发布；`--plan-only` 只产出
规划检查点，不会伪造目标图或人工结论。完整协议、字段和交付批次见
`docs/PIPELINE.md` 的“分组数据扩展”章节。新孪生允许跨类别：同类约 30%、跨类约 70%，保留同粒度、面积比 ≤3 和共享编辑自然性，生图前新增仅看干净原图＋no-ref 的独立盲测；见第 7.10 节。独立盲测仍是工程验证阶段：7 条已有规划因漏列/定位证据不足均为 uncertain，暂不建议用于正式生产；本轮未新生图。新运行采用 instruction/mask 2.0 盲指代，写作风格 `benchmark-concise-3-blind-1`；最新小批量见第 7.9 节，旧 grounding 对照见第 7.8 节：
新源图 12 个候选实际生成 9 组，自动通过 7 组，逐图确认合格 4 组；三阶段约 10.36 分钟。当前可产出人工确认合格组，但自动准入尚不足以支撑正式放量。add/site 可用 SAM3 在原审核前修正源 mask，原图与目标图保持不变。
生成阶段可用 `--generate-only --editor-devices 0,1` 启用常驻双卡编辑器；
单次联合审核 `--twin-audit-mode joint` 尚未通过质量对照，保持实验选项。

当前独立输入根目录为
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_group_inputs_20261014`：
训练候选 15,994、评测候选 1,199、评测 reservation 1,200；孪生第一批已固定为
`inputs/train_twin_selected.jsonl`（5,000 组）和
`inputs/eval_twin_selected.jsonl`（200 组）。四机运行先用
`shard_group_selection.py` 生成每个 rank 的 `selected_rank*.jsonl`，各 rank 使用独立
输出目录，完成后用 `merge_group_shards.py` 合并；快速 bbox 容量模式只用于估算，最终
数据必须经过严格 mask 几何、27B 审核和人工复核。
