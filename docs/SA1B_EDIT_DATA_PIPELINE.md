# SA-1B → SAMTok fine-grained edit pairs

This branch adapts the derived-edit workflow to SA-1B and the current
benchmark-v2 contract. The source selector is an MLLM pass, while masks remain
immutable dense SA-1B visible-region annotations. A selected scene has 2–5
different physical targets and records evidence for same-class instances,
occlusion, touching boundaries, parts, thin regions, holes, small regions and
disconnected visible pieces.

The four operations are exactly `add`, `replace`, `remove`, and `attribute`.
Every unit carries its source mask, tight box, deterministic interior point,
ref and no-ref instructions, preservation contract, and model evidence. Add
targets the selected host/attachment site; remove and replace require a
self-contained visible target; attribute keeps identity and geometry.

## Data flow

```text
SA-1B dense TFRecord / JSONL
  → normalize.py (candidate manifest + provenance)
  → select.py (Qwen3.8-27B vLLM scene selection + edit planning)
  → editor.py (Qwen-Image-2.1 checkpointed target generation)
  → validate.py (mask-aware deterministic checks + optional Qwen3.8 audit)
```

The pipeline can resume each stage from JSONL. It never treats a pixel diff as
the semantic mask ground truth: the diff is a veto/diagnostic, while the
semantic audit checks target completion, non-target preservation and natural
quality.

## Running against the documented SA-1B layout

```bash
SA1B=/mnt/hdfs/byte_ttlive_strategy_llm/user/haobo.yuan/datasets
python -m sa1b_pipeline.normalize \
  --image-records-dir "$SA1B/sa1b_tfd_1k" \
  --dense-records-dir "$SA1B/sa1b_dense_label/annotation" \
  --id-file ids.txt \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work/candidates
```

For a pilot, put a small set of SA IDs in `ids.txt`; this avoids scanning the
full 11M-record corpus. `--input exported.jsonl` is also supported.

An exported JSONL row contains `source_id`, `image_path`, and `objects`; an
object contains `object_id`, `category`, and either `mask_path` or COCO `rle`.
Original shard/index and offset are retained under `provenance`.

```bash
python -m sa1b_pipeline.select \
  --manifest .../candidates/candidates.jsonl \
  --output .../selection/selected_cases.jsonl \
  --model-id /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B \
  --tensor-parallel-size 8

python -m sa1b_pipeline.editor \
  --manifest .../selection/selected_cases.jsonl \
  --output-dir .../generated --model-id /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1 --steps 40

python -m sa1b_pipeline.validate \
  --manifest .../generated/generated.jsonl \
  --output .../validation.jsonl \
  --model-id /mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B
```
