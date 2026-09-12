# CrispEdit and ScaleEdit mask labeling

本分支汇总两条已经完成全量运行的图像编辑 mask 打标流程，不共享 prompt、
路由策略或后处理逻辑：

- **CrispEdit-2M**：fact prefilter → Qwen3.5/vLLM grounding → SAM3 mask。
- **ScaleEdit**：Qwen3.5/vLLM planner → bbox locator → SAM3 mask。

详细方法、代码入口、安装、完整运行命令、生产路径和可视化样例分别见：

- [CrispEdit-2M 打标文档](docs/CRISPEDIT_MASK.md)
- [ScaleEdit 打标文档](docs/SCALEEDIT_MASK.md)

## 环境安装

```bash
cd /opt/tiger/tanyue/sam3-crispedit

# CrispEdit：一次创建 prefilter/SAM3 环境和 Qwen3.5/vLLM grounding 环境
bash scripts/setup_crispedit_envs.sh

# ScaleEdit：一次创建同时支持 vLLM grounding 和 SAM3 mask 的环境
bash scripts/setup_scaleedit_vllm_env.sh
```

两套安装默认使用不同的 virtualenv，避免互相覆盖。模型权重和数据不会由安装脚本
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
```

这些入口的参数和 pipeline 代码保持各自生产版本。请勿把 CrispEdit 的 grounding/
manifest 与 ScaleEdit 的输入混用。

## 验证

```bash
.venv-scaleedit-vllm/bin/python -m pytest -q
```

最终 ScaleEdit 与 CrispEdit mask 的严格 QC、统一 schema 和自包含训练集导出见
[UNIFIED_MASK_DATASET.md](docs/UNIFIED_MASK_DATASET.md)。
