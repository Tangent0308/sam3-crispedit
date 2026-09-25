# ScaleEdit 开发与验证记录

当前方法见 [SCALEEDIT_MASK.md](SCALEEDIT_MASK.md)，四机入口见 [SCALEEDIT_4NODE.md](SCALEEDIT_4NODE.md)。只保留通向当前方法的开发证据，不维护旧实现入口。
下表实验路径前缀：`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/ScaleEdit/`。

| 日期/实验 | 改动与验证 | 结果与路径 |
| --- | --- | --- |
| 2026-09-24 下载 | HF filtered manifest唯一指令匹配原始shard，排除全局类型、双键去重、并行下载/校验 | `download_200k_20260924/`：新增199,633条，总299,633条/1,155 shard；`validation.json`无新旧重复 |
| 2026-09-24 原生pipeline | 适配审核后字段、两轮PASS/DROP、逐单元路由、身份join和签名 | `local_edit_validation_20260924/`：512→380→66，66mask技术OK；1张旧坏图安全DROP，无最终MLLM解析失败 |
| 2026-09-24 文字/重复实例 | 文字输出紧凑多边形；规则排列对象分别列单元 | `local_edit_refined_regions_20260924/`：重跑66条，65 OK/1解析拒绝；斜字减少背景覆盖，试管盖从1组拆为15单元 |
| 2026-09-24 多边形解析 | 从只收四角改为4–8个合法凸顶点，保留退化/越界/自交检查；重解析保存回答 | `local_edit_current_20260924/`：66grounding OK，无额外MLLM调用；SAM3实跑66条，结构校验无错误 |
| 2026-09-25 人工检查 | 全66条scene PASS和mask；另查20质量PASS、15语义DROP、32scene DROP | 当前`review/index.html`展示代表好坏例；`review/validation.json`保留全部512行/66mask校验 |
| 2026-09-25 分支整理/四机 | 仅保留ScaleEdit包、脚本/测试、SAM3依赖；阶段负载均衡、失败传播、恢复与合并校验 | `labeling_4node_smoke_20260925/`：20→15→10，10非空mask/26实例，无执行错误；retry1复用84个Parquet且SHA/mtime不变；92测试通过 |

## 改善与限制

- `expand-20260924-00071.parquet:84`：TOKYO文字mask 48,800→15,450像素，贴合斜向字符带。
- `expand-20260924-00071.parquet:116`：AD文字14,577→9,718像素，减少相邻文字覆盖。
- `part-00307.parquet:435`：前排盖子1→15实例，347,090→304,537像素，减少管身覆盖。面积变化不是准确率。
- scene可疑PASS：`part-00206.parquet:103`补写指令里不存在的chef selector；`expand-20260924-00519.parquet:209`相对单个明显道路新增长椅，与相似DROP不一致。
- mask明确失败：`expand-20260924-00519.parquet:27`被观察轮当作尖塔顶部替换，source mask漏掉灯笼范围；`expand-20260924-00526.parquet:31`寺庙碎裂/外溢。
- 边界：`part-00306.parquet:130`大范围树冠；`expand-20260924-00526.parquet:66`材质变化伴随新增袖子。下一步先检查观察单元是否遗漏/范围错误，再处理候选和边缘，避免按SAM分数或面积全局扩张。

代表画廊包含失败例。此次整理与分布式开发保持当前mask方法；尚未启动ScaleEdit全量打标。

## 清理与恢复

旧数据集代码、旧ScaleEdit实现和冗余图片已从本分支移除，仅保留三张当前联系图。SAM3为推理依赖，保留上游源码和许可证。
整理前代码/文档/图片备份：`/opt/tiger/tanyue/scaleedit-pre-cleanup-20260925.tar.gz`，不含环境和Git；旧已提交内容也能从Git历史恢复。源数据与外部实验结果未删除。
