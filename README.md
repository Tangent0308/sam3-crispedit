# CrispEdit / ScaleEdit → 协议 2.0 训练数据（extedit）

分支 `extedit-protocol2-labeling`：把 CrispEdit-2M、ScaleEdit 的原始编辑对打标成 SAMTokEdit 训练数据，格式与 D3（单目标）、D4（多目标）相同。数据标准包括：
- 指令协议 2.0；
- mask 语义 2.0；
- 区域外与 BEFORE 逐像素相同；
- SAMTok token 能还原 S。

本分支基于 `crispedit-labeling`，复用其中的 `sam3/` 和 `crispedit/mask/`（SAM3 分割）；其余 CrispEdit 两阶段筛选代码保留但不再使用。

本地仓库：`/opt/tiger/tanyue/sam3-extedit-protocol2-labeling`。

- [方法、运行命令、输出格式、验证结果](docs/EXTEDIT_PROTOCOL2.md)

```text
extedit/
  sources.py      取数：两个来源统一成一种行格式，按类型配额，排除旧流程用过的行和 benchmark 图
  pixels.py       像素预检（尺寸、配准、改动图）、贴回及其检查
  plan.py         27B 规划（质量 + 编辑单元 + 协议槽位 + 框），一次调用
  segment.py      SAM3 分割 S，add 锚点切到落点，SAMTok 往返
  gate.py         改动区域、门限、几何检查、贴回
  audit.py        渲染 ref / no-ref 句子，27B 审核
  export.py       D3 / D4 记录、token 缓存、拒收清单、汇总、审阅图
  masks.py, slots.py, views.py, llm.py
  instruction_protocol.py, instruction_style.py   benchmark 2.6.0 原文件，逐字节复制（测试固定哈希）
scripts/extedit_pipeline.py   命令行，每步一个子命令，可断点续跑
tests/test_extedit_pipeline.py
requirements/extedit-worker.txt, extedit-vllm.txt
```

测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <worker env>/bin/python -m pytest -q -p no:cacheprovider tests/test_extedit_pipeline.py
```
