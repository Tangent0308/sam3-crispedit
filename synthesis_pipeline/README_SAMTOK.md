# SAMTok-derived edit pipeline

## Retained formal run

The only retained generated dataset is the completed four-node run. Use the
Arnold entry and resume instructions in
[`../docs/SAMTOK_LABELING_四机运行指南.md`](../docs/SAMTOK_LABELING_四机运行指南.md).
Its delivered manifests and gallery are at
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/remove/final/`;
the complete source/planning/editing/audit evidence is exposed through
`.../SAMTok_Derived_Edit_Labeling/remove/intermediate/`, with the canonical immutable
run root under
`/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/samtok-derived-4n-20260925/`.

The pilot commands below are historical development templates. Their old
external pilot directories were intentionally removed after the formal run and
must not be treated as current inputs or resume paths; use a new explicit
output directory for any future experiment.

The non-remove four-node runner is independent of that frozen result:
`scripts/labeling/bootstrap_arnold_4node_multitype.sh` expands every positive
dataset mask into one `add`, one `replace`, and one `attribute` case, then uses
the generic mask planner and the same Qwen-Image-2.1 editor. Its run root is
`.../experiments/SAMTok_Derived_Edit_Labeling/four_node/<run-id>/` and its input
manifest is under `data/add_replace_attribute/`; it never writes `remove/`.

For the detailed Chinese design document, including the task taxonomy,
instruction-generation safeguards, per-case manual review policy, and the
100-case pilot, see
[`../docs/SAMTOK_DERIVED_EDIT_PIPELINE.md`](../docs/SAMTOK_DERIVED_EDIT_PIPELINE.md).

## Data policy

- Source: embedded image bytes and COCO RLE masks from
  `mix_gres8k_ver4k_train.parquet`.
- Eligibility: the positive index retains every non-`No target` row; instruction
  planning then over-samples and keeps only whole sources whose masks are compatible
  with their assigned edit types.
- One image, many edits: each parquet source image is materialized once. Every
  source mask creates an independent single-region edit case referencing that
  shared source through `source_image`.
- Coverage is strict: a plan must contain every `mask_index` from `0` through
  `num_masks - 1` exactly once. Missing or duplicate masks fail immediately.
- Mask policy: source RLE masks are reused exactly; SAM/SAM2 is not run.
- Editing: Qwen-Image-Edit-2511 with MIRAGE regional branches.
- Planning localization: exactly two images, the clean full source and an enlarged
  photographic context crop with a thin external black/white mask contour and
  margin label. No target pixels are painted or cut out; colored overlays and
  multi-mask SAMTok question/answer text are never shown to the instruction VLM.
- Instruction localization: `refer_object` must include a full-image locator,
  and the full edit instruction must retain its distinguishing position and
  landmark (allowing minor grammatical omissions). Each edit type receives
  only its own task-specific guidance.
- Regional writeback: remove/replace/attribute use a feathered exact mask; add
  retains bbox guidance because its mask is a placement anchor.
- Audit: one Qwen3-VL-8B vLLM call per case, binary pass/fail output, an exact-mask
  edge on aligned before/after crops, an aligned source/mask/edit/difference panel,
  and narrow deterministic guards for contradictory remove verdicts.

The checked-in `pilot_plan.jsonl` contains eight representative source rows,
all with two masks. It therefore produces sixteen independent edit cases from
eight unique source images. Add/remove/replace/attribute each have four cases.

The newer stratified 100-case pilot is generated with
`generate_samtok_plan.py`: 50 two-mask sources yield 100 single-region cases,
GRES/VER and the four edit types are balanced. Its two-round 100/100 manual review
found 66 pass, 7 review, and 27 fail; see the detailed design document for the
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

The vLLM backend receives three annotation-safe views in one call: aligned tight
before/after crops with the exact-mask edge, an aligned source/mask/edit/difference
panel, and a full source/edit comparison. The
prompt applies distinct add/remove/replace/attribute criteria and asks for
explicit failure evidence before a verdict. No red-mask overlay is shown.

```bash
CUDA_VISIBLE_DEVICES=0 \
  /opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  synthesis_pipeline/audit_edit_pairs.py \
  --annotations-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/annotations.jsonl \
  --source-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/sources \
  --edited-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/edited \
  --out-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v2_all_regions/audit_vllm \
  --batch-size 16 --vlm qwen8b-vllm --vlm-device cuda:0 \
  --max-new-tokens 384
```

`qwen8b-vllm` is now the default. Use `--vlm qwen8b` only for an HF baseline.
The script reports backend loading and inference time separately and explicitly
shuts down the vLLM engine process.

An experimental two-image audit is also available in
`synthesis_pipeline/audit_edit_pairs_v2.py`. It uses an outlined source and
clean edited view in one VLM request, separates visual quality from original
instruction compliance, and emits human-review-only rewrite candidates for
visually sound edits with a mismatched result. It supports both
`qwen8b-vllm` and `qwen38-vllm`; the latter requires the separate
`/opt/tiger/tanyue/.venvs/qwen38_audit` environment. See section 9 of
`docs/SAMTOK_DERIVED_EDIT_PIPELINE.md` sections 9–13 for the exact command,
prompt policy, 52-case, 20-case, and source-disjoint 100-case comparisons,
and limitations. Its
`--prompt-variant auto` selects the human-style rubric for 8B and the more
precise original rubric for 27B, plus a conservative no-op veto for
remove/replace when fewer than 30% of source-mask pixels change. The
`legacy_tolerant` 27B variant improves recall on one pilot but admits more
bad images on another; it is available explicitly, not selected by default.
This experiment does not automatically change the existing production audit
or training labels.

The fresh 100-case pilot is at
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_100_audit27_seed20260922/`.
Its `audit_comparison_gallery/index.html` shows every source/edited pair
beside both 27B decisions and the independent visual review. The more
conservative prompt improved the 60-case development split but increased
false acceptance on the 40-case source-disjoint holdout, so it is **not**
the default; see section 13 before using model verdicts for admission.

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
