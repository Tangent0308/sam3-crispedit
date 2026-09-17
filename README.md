# SAMTok-derived fine-grained edit labeling

This branch derives localized image-editing pairs from the existing SAMTok
GRES-8k and VER-4k training data. It reuses MIRAGE's regional Qwen-Image-Edit
implementation and its Git history, while removing MIRAGE benchmark-generation
and evaluation code that is unrelated to SAMTok labeling.

The data flow is:

1. Read the canonical SAMTok parquet and build a positive-only index. Rows whose
   answer is `No target` are excluded.
2. Decode selected source images from the embedded parquet bytes and retain the
   original COCO RLE instance masks.
3. Reuse each source image for every annotated mask and generate one independent
   regional edit case per mask.
4. Run Qwen-Image-Edit-2511 with MIRAGE regional latent composition.
5. Audit localization/background preservation with batched Qwen3-VL vLLM and
   export a comparison gallery.

Pilot data and full-run artifacts are intentionally stored outside Git under
`/mnt/bn/strategy-mllm-train/user/tanyue/datasets/`.

## Environment

The existing environments used for MIRAGE are supported:

- data preparation: `/usr/bin/python3`
- instruction VLM: `/opt/tiger/tanyue/.venvs/vllm_mirage/bin/python`
- Qwen image editing: `/opt/tiger/tanyue/.venvs/mirage_official/bin/python`

See [`synthesis_pipeline/README_SAMTOK.md`](synthesis_pipeline/README_SAMTOK.md)
for the implemented pilot, exact reproduction commands, output schema, and
measured runtime.

## Lineage

The branch starts from MIRAGE commit `50a5df5` and retains its history. MIRAGE
is described in:

> Ziqian Liu and Stephan Alaniz, *MIRAGE: Benchmarking and Aligning
> Multi-Instance Image Editing*, 2026.
