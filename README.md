# CrispEdit, ScaleEdit, and RefEdit mask labeling

本分支包含三条已经完成生产验证的图像编辑 mask 打标流程：

- **CrispEdit-2M**：fact prefilter → Qwen3.5/vLLM grounding → SAM3 mask。
- **ScaleEdit**：Qwen3.5/vLLM planner → bbox locator → SAM3 mask。
- **RefEdit**：Qwen3.8 编辑对质量预筛 → PASS-only Qwen3.5/vLLM grounding → SAM3 mask
  → strict final dataset。

详细方法、代码入口、安装、完整运行命令、生产路径和可视化样例分别见：

- [CrispEdit-2M 打标文档](docs/CRISPEDIT_MASK.md)
- [ScaleEdit 打标文档](docs/SCALEEDIT_MASK.md)
- [RefEdit 打标文档](docs/REFEDIT_MASK.md)

## 环境安装

```bash
cd /opt/tiger/tanyue/sam3-crispedit

# CrispEdit：一次创建 prefilter/SAM3 环境和 Qwen3.5/vLLM grounding 环境
bash scripts/setup_crispedit_envs.sh

# ScaleEdit：一次创建同时支持 vLLM grounding 和 SAM3 mask 的环境
bash scripts/setup_scaleedit_vllm_env.sh

# RefEdit：从零创建独立的 vLLM + SAM3 环境
bash scripts/setup_refedit_vllm_env.sh
```

三套安装默认使用不同的 virtualenv，避免互相覆盖。模型权重和数据不会由安装脚本
下载或修改。

## 生产入口

```text
CrispEdit
  crispedit_mllm_prefilter.py
  crispedit_mllm_grounding.py
  crispedit_grounded_mask_runner.py

ScaleEdit
  scaleedit_mllm_grounding.py
  scaleedit_grounded_mask_runner.py

RefEdit
  refedit_quality_prefilter.py
  refedit_mllm_grounding.py
  refedit_grounded_mask_runner.py
  scripts/build_refedit_final_mask_dataset.py
```

这些入口的参数和 pipeline 代码保持各自版本。请勿跨数据集混用 source、grounding
或 manifest。

RefEdit 推荐直接运行完整的 8 卡、可续跑流程：

```bash
cd /opt/tiger/tanyue/sam3-crispedit
bash scripts/run_refedit_full.sh
```

也可以先运行 `scripts/run_refedit_quality_prefilter_full.sh`，检查 PASS manifest 后再运行
`scripts/run_refedit_filtered_mask_full.sh`。最终 7,804 条严格通过样本位于：

```text
/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-mask-prefiltered-qwen38/final
```

## 验证

```bash
.venv-scaleedit-vllm/bin/python -m pytest -q
```

最终 ScaleEdit 与 CrispEdit mask 的严格 QC、统一 schema 和自包含训练集导出见
[UNIFIED_MASK_DATASET.md](docs/UNIFIED_MASK_DATASET.md)。

## 同类多实例指代难度筛选

对统一数据集做“多个同类实例中只编辑一个/真子集”的训练难度筛选：

```bash
.venv-scaleedit-vllm/bin/python scripts/filter_referential_edits.py --help
```

流程采用确定性预筛、单次 Qwen3.5 语义判断、SAM3 同类计数、已有 edit mask 覆盖核验，输出
严格 `keep` 清单和单独的 `review` 清单。完整方法、参数、小批量结果与可视化路径见
[同类多实例细粒度指代编辑筛选文档](docs/REFERENTIAL_EDIT_DIFFICULTY_FILTER.md)。

全量 8 卡入口：

```bash
.venv-scaleedit-vllm/bin/python -u scripts/run_referential_filter_full.py supervise
```

它使用 4 个 vLLM TP=2 worker，完成后自动切换为 8 个单卡 SAM3 worker，并按源 shard
保存可恢复的中间证据。

运行中可另开终端查看带 elapsed/ETA 的 tqdm 进度条；监视器只读取 checkpoint，不占 GPU：

```bash
.venv-scaleedit-vllm/bin/python scripts/watch_referential_filter_progress.py
```

2026-09-14 全量筛选已完成：输入 167,345 条，严格 `keep` 17,095 条，`review`
23,289 条，宽松集合 `keep + review` 共 40,384 条。生产结果位于
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-CrispEdit-mask-referential-filter`；
完整统计与固定随机抽样看板见
[筛选文档的全量结果章节](docs/REFERENTIAL_EDIT_DIFFICULTY_FILTER.md#8-全量结果与随机抽样2026-09-14)。
