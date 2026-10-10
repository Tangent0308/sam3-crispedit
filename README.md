# SA-1B SAMTok edit-data construction

This branch is the SA-1B adaptation of the derived-edit workflow. It selects
benchmark-v2-like difficult scenes with Qwen3.8-27B through vLLM, constructs
four-type local edit contracts, generates Qwen-Image-2.1 targets, and audits
the resulting pairs. See [the pipeline design](docs/SA1B_EDIT_DATA_PIPELINE.md)
and the `sa1b-edit` command line entry point.

New SA-1B work belongs under `sa1b_pipeline/`; the inherited
`synthesis_pipeline/` remains available for comparison with the earlier
CrispEdit-derived data.

This branch derives localized image-editing pairs from the existing SAMTok
GRES-8k and VER-4k training data. It reuses MIRAGE's regional composition
implementation and its Git history. The completed formal four-type dataset is
documented in [SAMTOK_FINAL_FOUR_TYPE_DATASET.md](docs/SAMTOK_FINAL_FOUR_TYPE_DATASET.md).

The data flow is:

1. Read the canonical SAMTok parquet and build a positive-only index. Rows whose
   answer is `No target` are excluded.
2. Decode selected source images from the embedded parquet bytes and retain the
   original COCO RLE instance masks.
3. Reuse each source image for every annotated mask and generate one independent
   regional edit case per mask.
4. Plan remove versus add/replace/attribute with task-specific Qwen3.8-27B
   prompts, then edit with Qwen-Image-2.1 through vLLM-Omni (40 steps).
5. Audit source/edited pairs with Qwen3.8-27B through vLLM and deliver the
   model-pass cases with their masks and images.

The final combined dataset is at
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/combined/`.
It contains all four types in one manifest, regular hard-linked source/edited
PNGs, COCO RLE masks, model audit fields, and a standalone HTML gallery.
The two canonical run roots remain under
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/runs/four_node/`.

## Environment

The formal Arnold entry installs its pinned cu129 SAM, Qwen3.8 vLLM, and
Qwen-Image-2.1 vLLM-Omni environments on each worker. See the
[four-node guide](docs/SAMTOK_LABELING_四机运行指南.md) for the exact launch and
resume commands. The combined-data materializer needs Python 3.12; its gallery
builder also uses Pillow and pycocotools.

See [`synthesis_pipeline/README_SAMTOK.md`](synthesis_pipeline/README_SAMTOK.md)
for historical pilot reproduction.

The full design rationale, task taxonomy, quality rubric, 100-case pilot
commands, and measured results are documented in
[`docs/SAMTOK_DERIVED_EDIT_PIPELINE.md`](docs/SAMTOK_DERIVED_EDIT_PIPELINE.md).

## Lineage

The branch starts from MIRAGE commit `50a5df5` and retains its history. MIRAGE
is described in:

> Ziqian Liu and Stephan Alaniz, *MIRAGE: Benchmarking and Aligning
> Multi-Instance Image Editing*, 2026.
