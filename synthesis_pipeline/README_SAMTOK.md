# SAMTok-derived edit pipeline

For the detailed Chinese design document, including the task taxonomy,
instruction-generation safeguards, per-case manual review policy, and the
100-case pilot, see
[`../docs/SAMTOK_DERIVED_EDIT_PIPELINE.md`](../docs/SAMTOK_DERIVED_EDIT_PIPELINE.md).

## Data policy

- Source: embedded image bytes and COCO RLE masks from
  `mix_gres8k_ver4k_train.parquet`.
- Eligibility: no ranking filter yet. A row is retained when it is not
  `No target` and has at least one non-empty mask record.
- One image, many edits: each parquet source image is materialized once. Every
  source mask creates an independent single-region edit case referencing that
  shared source through `source_image`.
- Coverage is strict: a plan must contain every `mask_index` from `0` through
  `num_masks - 1` exactly once. Missing or duplicate masks fail immediately.
- Mask policy: source RLE masks are reused exactly; SAM/SAM2 is not run.
- Editing: Qwen-Image-Edit-2511 with MIRAGE regional branches.
- Audit: Qwen3-VL-8B through vLLM continuous batching by default, plus
  deterministic pixel-locality metrics.

The checked-in `pilot_plan.jsonl` contains eight representative source rows,
all with two masks. It therefore produces sixteen independent edit cases from
eight unique source images. Add/remove/replace/attribute each have four cases.

The newer stratified 100-case pilot is generated with
`generate_samtok_plan.py`: 50 two-mask sources yield 100 single-region cases,
GRES/VER and the four edit types are balanced. Its strict 100/100 manual review
found 72 pass, 7 review, and 21 fail; see the detailed design document for the
per-type breakdown, failure analysis, and artifact paths.

## Prepare the positive index and pilot

```bash
python3 synthesis_pipeline/prepare_samtok_data.py \
  --parquet /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions \
  --plan-jsonl synthesis_pipeline/pilot_plan.jsonl \
  --force-index
```

`positive_rows.jsonl` contains 7,671 positive rows with provenance and all
source masks, but does not duplicate the embedded images. Selected images are
decoded on demand and resized using the same one-megapixel canvas calculation
as Qwen-Image-Edit. Masks use nearest-neighbor resize before COCO RLE re-encode.

The relevant manifest fields are:

```json
{
  "image": "unique_edit_case_and_output.png",
  "source_image": "shared_source_gres_r31.png",
  "mask_index": 0,
  "mask": [{"size": [896, 1184], "counts": "..."}],
  "editing_instruction": "..."
}
```

## Edit on eight GPUs

```bash
/opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  synthesis_pipeline/run_qwen_edit_pool.py \
  --image-root /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/sources \
  --instruction-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/annotations.jsonl \
  --crop-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/crops \
  --results-full-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/edited \
  --model-id Qwen/Qwen-Image-Edit-2511 \
  --gpus 0,1,2,3,4,5,6,7 \
  --python /opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  --dtype bf16 --cpu-offload none \
  --patch-ratio 0.2 --num-steps 40 \
  --true-cfg-scale 4.0 --guidance-scale 1.0 --seed 0
```

The output name is the edit case id, while the input is resolved from
`source_image`. The pool retains one model per GPU and dynamically claims cases.

## Audit with vLLM

The vLLM backend supports the same three-image audit prompt and deterministic
temperature-zero decoding as the Hugging Face backend.

```bash
CUDA_VISIBLE_DEVICES=0 \
  /opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  synthesis_pipeline/audit_edit_pairs.py \
  --annotations-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/annotations.jsonl \
  --source-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/sources \
  --edited-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/edited \
  --out-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/audit_vllm \
  --batch-size 16 --vlm qwen8b-vllm --vlm-device cuda:0
```

`qwen8b-vllm` is now the default. Use `--vlm qwen8b` only for an HF baseline.
The script reports backend loading and inference time separately and explicitly
shuts down the vLLM engine process.

## Visualize

```bash
python3 synthesis_pipeline/build_pilot_gallery.py \
  --annotations-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/annotations.jsonl \
  --source-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/sources \
  --overlay-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/overlays \
  --edited-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/edited \
  --audit-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/audit_vllm/edit_audit.jsonl \
  --out-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/gallery
```

## Measured v2 speed

Measured on eight H100 80GB GPUs for editing and one H100 for audit:

| Stage | Wall time | Throughput |
|---|---:|---:|
| Positive-only index | 0.69 s | 12,337 rows scanned |
| Prepare 8 sources / 16 masks | 24.90 s | 0.643 edit case/s |
| Qwen/MIRAGE edit pool | 181.80 s | 5.28 case/min including startup |
| Qwen3-VL vLLM model load | 41.62 s | one-time per process |
| Qwen3-VL vLLM inference | 8.69 s | 110.53 case/min |
| Full vLLM audit | 55.24 s | 17.38 case/min including startup |

With Qwen-Image-Edit already resident, the second case on each worker took
64.70--69.00 seconds (mean 66.50), corresponding to about 7.22 case/min over
eight GPUs.

### vLLM versus Hugging Face audit

The same sixteen cases and the same Qwen3-VL-8B model were audited through both
backends:

| Backend | Model load | Inference | Inference throughput |
|---|---:|---:|---:|
| Hugging Face, batch 2 | 8.11 s | 40.92 s | 23.46 case/min |
| vLLM, continuous batch 16 | 41.62 s | 8.69 s | 110.53 case/min |

vLLM makes the inference portion 4.71x faster. Its larger cold-start cost means
very small one-off audits do not benefit, but it is strongly preferable for a
long-running or large labeling job. All sixteen structured verdicts matched the
HF backend exactly in this comparison.
