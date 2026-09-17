# SAMTok-derived edit pipeline

## Scope of the first pilot

- Source: embedded image bytes and COCO RLE masks from
  `mix_gres8k_ver4k_train.parquet`.
- Eligibility: no candidate-ranking filter yet. A row is retained when it is
  not `No target` and has at least one non-empty mask record.
- Edit granularity: one or two regions per case, never more than two.
- Pilot balance: two cases each of `add`, `remove`, `replace`, and `attribute`;
  six single-region cases and two dual-region cases.
- Mask policy: reuse source RLE masks exactly. No SAM/SAM2 mask regeneration.
- Editing: Qwen-Image-Edit-2511 with MIRAGE regional branches.

The pilot instructions are checked into `pilot_plan.jsonl`. Keeping this first
plan fixed makes the source/mask/edit smoke test deterministic and separates
Qwen/MIRAGE behavior from instruction-MLLM variance. Automatic instruction
generation can write the same plan schema in the next iteration.

## Prepare a positive index and the pilot

```bash
python3 synthesis_pipeline/prepare_samtok_data.py \
  --parquet /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1 \
  --plan-jsonl synthesis_pipeline/pilot_plan.jsonl \
  --force-index
```

`positive_rows.jsonl` is a 7,671-row index containing provenance and all source
RLE masks but not duplicated image bytes. It contains no `No target` rows.
Selected embedded images are decoded on demand and resized to the same
one-megapixel canvas calculation used by Qwen-Image-Edit; masks are resized with
nearest-neighbor interpolation and re-encoded as COCO RLE.

## Edit on eight GPUs

```bash
/opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  synthesis_pipeline/run_qwen_edit_pool.py \
  --image-root /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/sources \
  --instruction-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/annotations.jsonl \
  --crop-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/crops \
  --results-full-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/edited \
  --model-id Qwen/Qwen-Image-Edit-2511 \
  --gpus 0,1,2,3,4,5,6,7 \
  --python /opt/tiger/tanyue/.venvs/mirage_official/bin/python \
  --dtype bf16 --cpu-offload none \
  --patch-ratio 0.2 --num-steps 40 \
  --true-cfg-scale 4.0 --guidance-scale 1.0 --seed 0
```

The pool uses one persistent model process per GPU and dynamic claims. A fresh
seeded generator is created for every case, so scheduling order does not change
the generated image.

## Audit and visualize

The audit code supports both the original polygon annotations and SAMTok COCO
RLE annotations.

```bash
CUDA_VISIBLE_DEVICES=0 python3 synthesis_pipeline/audit_edit_pairs.py \
  --annotations-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/annotations.jsonl \
  --source-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/sources \
  --edited-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/edited \
  --out-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/audit \
  --batch-size 2 --vlm qwen8b --vlm-device cuda:0

python3 synthesis_pipeline/build_pilot_gallery.py \
  --annotations-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/annotations.jsonl \
  --source-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/sources \
  --overlay-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/overlays \
  --edited-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/edited \
  --audit-jsonl /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/audit/edit_audit.jsonl \
  --out-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Derived_Edit_Labeling/pilot_v1/gallery
```

## Measured pilot speed

Measured on eight H100 80GB GPUs with eight cases:

| Stage | Wall time | Observed speed |
|---|---:|---:|
| Positive index | 0.67 s | 12,337 rows scanned |
| Read embedded image column | 6.56 s | entire 5.1GB parquet image column |
| Prepare 8 canvases + 10 RLE regions | 20.42 s total | 0.392 case/s |
| Qwen/MIRAGE edit pool | 151.17 s | 3.175 case/min |
| Qwen3-VL audit | 57.50 s | 8.35 case/min |
| Gallery | 3.2 s | 8 cases |

Per-case Qwen/MIRAGE inference, excluding model startup, ranged from 68.39 to
98.21 seconds (median 75.50 seconds). Dual-region cases were the slowest.

