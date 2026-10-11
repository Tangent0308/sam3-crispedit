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

## SA-1B benchmark-v2 pilot status

The reproducible benchmark-v2 pilot is on branch `sa1b-benchmark-v2`. A
Qwen3.8-27B selection/planning pass accepted 14 hard scenes from the local
100-image SA-1B candidate pool, producing 70 units (five per scene) with the
seven benchmark interaction combinations. Qwen-Image-2.1 generated all 70
targets at 40 steps; the Qwen3.8-27B audit passed 35 pairs (attribute 16,
replace 10, add 8, remove 1). Only those 35 pairs were encoded with the
official SAMTok tokenizer and materialized into 140 protocol rows.

The resumable artifacts are under
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work/pilot_100/selection_20/`:
`selected_cases_v4.jsonl`, `generated_v4_40step/validation.jsonl`,
`generated_v4_40step/mask_codes.jsonl`, and
`generated_v4_40step/training_rows.jsonl`. The 13 failed remove retries remain
in separate `generated_remove_retry_*` directories and are excluded from
training. Remove generations remain subject to the same semantic audit; no
remove pair is exported unless it passes.

A disjoint expansion run then used 100 non-benchmark SA-1B candidates selected
by Qwen3.8-27B, planned 22 cases, and generated a bounded ten-case/50-unit
manifest with Qwen-Image-2.1. The audit passed 18 pairs (9 attribute, 7
replace, 2 add); all ten remove units failed and were excluded. The 18 passing
pairs were encoded with the official Qwen3-VL-8B-SAMTok tokenizer and expanded
to 72 training rows. Artifacts are under
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work/disjoint_mllm_plan30/generated_10_40step/`.
The source-pool manifest excludes every frozen benchmark source ID and image
byte hash, and `training.py` repeats that guard at materialization time.

## Lineage

The branch starts from MIRAGE commit `50a5df5` and retains its history. MIRAGE
is described in:

> Ziqian Liu and Stephan Alaniz, *MIRAGE: Benchmarking and Aligning
> Multi-Instance Image Editing*, 2026.
