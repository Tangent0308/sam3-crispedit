# CrispEdit labeling

分支 `crispedit-labeling` 只维护 CrispEdit：Qwen3.8-27B/vLLM 两阶段筛选 → 编辑单元观察与定位 → SAM3 局部 mask。支持单机 8 卡和四机各 8 卡，保留原 shard / row_idx。仅处理 add、color、motion、remove、replace。

本地仓库：`/opt/tiger/tanyue/sam3-crispedit-crispedit-labeling`。共享部署副本与运行日志路径见四机指南。

- [当前方法、运行命令、数据与结果、可视化](docs/CRISPEDIT_MASK.md)
- [mask 迭代记录与已知问题](docs/CRISPEDIT_MASK_ITERATION.md)
- [四机完整流水线入口与日志](docs/CRISPEDIT_4NODE.md)

```text
crispedit/
  common.py, inference.py   数据工具、Qwen3.8/vLLM 推理
  prefilter/               图像对质量筛选
  difficulty/              难定位局部编辑筛选
  mask/                    编辑单元、grounding、SAM3、结果契约
  distributed.py           四节点调度、交接、合并与恢复
scripts/                   安装、运行、下载、验证、可视化
tests/                     当前实现的回归测试
sam3/                      实际运行依赖的上游 SAM3 代码与资源
```

根目录四个 `crispedit_*.py` 是命令行入口，实现位于 `crispedit/`。其他数据集和已撤回实验代码不在本分支维护。历史实验数据未删除；旧图归档位置见迭代文档。

安装与测试（CUDA 12.9，8 卡）：

```bash
bash scripts/setup_crispedit_env.sh
.venv-crispedit/bin/python -m pytest -q
```

所有 MLLM 使用 Qwen3.8-27B。mask 的 `OK` 是自动结构/QC 状态，不代表语义准确率；全量 mask 质量尚未验收。
