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
  → training.py (SAMTok code binding + edit_ntp/edit_umt/edit rows)
```

The pipeline can resume each stage from JSONL. It never treats a pixel diff as
the semantic mask ground truth: the diff is a veto/diagnostic, while the
semantic audit checks target completion, non-target preservation and natural
quality.

The selector requires the pinned vLLM CUDA runtime in
`requirements/mllm-vllm-cu129.txt`. A generic older vLLM may install
successfully but reject the local checkpoint architecture
`Qwen3_5ForConditionalGeneration` at startup.

Run `python -m sa1b_pipeline.doctor` before loading a model. It verifies the
two SA-1B roots, CUDA visibility, vLLM version, and registry support for the
checkpoint architecture.

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

When the SG2 FUSE volume is unavailable, stream a pilot directly from HDFS
without copying the corpus:

```bash
python -m sa1b_pipeline.normalize \
  --hdfs-image-uri hdfs://harunasg/home/byte_ttlive_strategy_llm/user/haobo.yuan/datasets/sa1b_tfd_1k/sa_000000.tfrecord \
  --hdfs-dense-uri hdfs://harunasg/home/byte_ttlive_strategy_llm/user/haobo.yuan/datasets/sa1b_dense_label/annotation/00000_of_00999.tfrecord \
  --scan-records 5000 --minimum-matches 100 \
  --output-dir /mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work/candidates
```

This remote mode is for bounded pilots; production should use the handoff
indexes with `tfd-utils` on a node where the SG2 volume is mounted.

`scripts/run_sa1b_pipeline.sh` performs the same four stages and exits before
model loading when either handoff root is not mounted. For a remote pilot, set
`HDFS_IMAGE_URI`, `HDFS_DENSE_URI`, and optionally `SCAN_RECORDS` and
`MINIMUM_MATCHES`; the script then uses the bounded HDFS stream mode above.

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

python -m sa1b_pipeline.training \
  --generated .../generated/generated.jsonl \
  --mask-codes .../sam_tok_codes.jsonl \
  --output .../training/train.jsonl
```

Encode the codes with the official Qwen3-VL-8B-SAMTok VQ-SAM2 checkpoint:

```bash
scripts/setup_sa1b_samtok.sh
PYTHONPATH="$PWD:$SAMTOK_REPO_ROOT" "$SA1B_SAMTOK_RUNTIME/bin/python" \
  -m sa1b_pipeline.samtok \
  --generated .../generated/generated.jsonl \
  --output .../sam_tok_codes.jsonl \
  --model-dir /mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok \
  --samtok-root "$SAMTOK_REPO_ROOT"
```

The encoder uses the official SAM2 image path, source mask, normalized tight
box, and two residual codebooks. It records the tokenizer checkpoint hash and
Sa2VA revision beside every code. Coordinate, bbox, or hand-written token
substitutes are rejected by the training materializer.

`sam_tok_codes.jsonl` is intentionally an explicit input: codes must be
produced by the Qwen3-VL-8B-SAMTok mask tokenizer that will be used for
training. The materializer refuses missing or malformed codes instead of
silently writing fake mask tokens. Each validated pair expands to one
`edit_ntp`, two `edit_umt` (`ref`/`noref`), and one plain `edit` row, matching
the latest Qwen-Image-2.1 protocol.

### Isolated MLLM environment

Run `UV_BIN=$HOME/.local/bin/uv scripts/setup_sa1b_mllm.sh` to install the inherited
full lock in a new Python 3.12.13 environment. CUDA-specific Torch wheels come
from the PyTorch cu129 index; installing the generic PyPI Torch wheel is not
an equivalent runtime. The environment was imported successfully on the pilot
node with Torch 2.13.0+cu129, vLLM 0.28.0 and Transformers 5.17.0.

The run script accepts `PREP_PYTHON`, `MLLM_PYTHON`, and `EDITOR_PYTHON` so the
MLLM and editor dependency stacks can remain isolated. `doctor --handoff`
accepts either a mounted directory or an `hdfs://` root. This checks imports,
model registration and data access only; it does not prove generation quality.
Selection saves a `.decisions.jsonl` audit beside its output. Set
`SA1B_MLLM_LOG` to a JSONL path to capture direct-vLLM raw output before parsing.
RLE masks are exported to PNG only for selected targets.

The validated pilot artifacts are under
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work/pilot_100`:
`candidates/` contains 100 joined SA1B records, `selection/` contains the
Qwen3.8 decisions and selected case, and `generated_smoke/` contains a real
Qwen-Image-2.1 PNG from a one-step interface smoke. Use the normal 40-step
setting before accepting generated pairs for training.

The completed 40-step pilot is in `generated_40step/`: its Qwen-Image-2.1
output passed deterministic checks and a Qwen3.8-27B semantic audit, and
`training_rows.jsonl` contains the four contract rows. The sample is an
attribute edit of the selected metal fence; its official SAMTok code is
`<|mt_start|><|mt_0223|><|mt_0356|><|mt_end|>`.

For the Qwen-Image-2.1 Omni editor, run with
`DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`. The editor lock currently resolves a
FlashAttention3 wheel built for CUDA 13 while the tested Arnold H100 node uses
a CUDA 12.9 driver; SDPA is the compatible attention backend. The first smoke
request produced a PNG successfully. Use the default 40 steps for data, and
judge the 1-step smoke only as an interface check.

Install the editor stack with `scripts/setup_sa1b_editor.sh`. It installs the
pinned editor lock, checks out the tested vLLM-Omni revision, and installs this
repo so Omni worker subprocesses can import the regional Qwen-Image-2.1
pipeline. Export the printed `EDITOR_PYTHON` and `EDITOR_RUNTIME_ROOT` before
using `scripts/run_sa1b_pipeline.sh`.
