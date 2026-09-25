# ScaleEdit labeling

Current pipeline: filtered HF data download → Qwen3.8-27B quality filter → difficult local-scene filter → edit-unit grounding → SAM3 masks. Branch: `scaleedit-labeling`.

- [Method, download, commands, paths and visual results](docs/SCALEEDIT_MASK.md)
- [Development and validation record](docs/SCALEEDIT_DEVELOPMENT.md)
- [Four-node / 32-GPU launch and recovery](docs/SCALEEDIT_4NODE.md)

```bash
bash scripts/setup_scaleedit_env.sh
bash scripts/run_scaleedit_pipeline.sh /absolute/new/output-directory
```

`scaleedit/` contains the dataset pipeline; `scripts/` contains supported commands; `tests/` covers data contracts and orchestration. `sam3/` is the vendored SAM3 dependency with its original license.

Native `sample_id`, `final_task`, `final_instruction` and shard-local `row_idx` are preserved. Filters output PASS/DROP. Mask QC flags are structural diagnostics, not a guarantee of semantic accuracy.
