"""Materialize one add/replace/attribute case for every positive source mask.

The remove run has a deliberately separate manifest and must never be mutated by
this script.  Source pixels and dataset RLE masks are materialized once, then
each region is expanded into the three non-remove task types.  Instructions are
left empty because ``plan_dataset_regions`` writes a grounded instruction after
seeing the clean source and the exact mask crop.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from synthesis_pipeline.prepare_samtok_data import (
    build_positive_index,
    decode_rle,
    encode_rle,
    qwen_canvas_size,
    resize_mask,
    write_jsonl,
)
from synthesis_pipeline.reference_binding import bind_reference


TASK_TYPES = ("add", "replace", "attribute")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--limit-sources",
        type=int,
        default=0,
        help="0=all positive source rows; positive values bound the smoke run",
    )
    parser.add_argument("--force-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_sources < 0:
        raise SystemExit("--limit-sources must be nonnegative")
    args.out_root.mkdir(parents=True, exist_ok=True)
    positive, index_summary = build_positive_index(
        args.parquet, args.out_root / "positive_rows.jsonl", args.force_index
    )
    if args.limit_sources:
        positive = positive[: args.limit_sources]

    import pyarrow.parquet as pq

    selected = {int(row["parquet_row_index"]): row for row in positive}
    source_dir = args.out_root / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    source_rows = 0
    offset = 0
    with tqdm(total=len(positive), desc="materialize multitype source images") as bar:
        for batch in pq.ParquetFile(args.parquet).iter_batches(
            batch_size=64, columns=["images"]
        ):
            for local, cell in enumerate(batch.column(0)):
                row_index = offset + local
                if row_index not in selected:
                    continue
                indexed = selected[row_index]
                payload = cell.as_py()[0]["bytes"]
                with Image.open(io.BytesIO(payload)) as handle:
                    original = handle.convert("RGB")
                canvas_size = qwen_canvas_size(*original.size)
                source_name = f"source_{indexed['source_subset']}_r{row_index}.png"
                original.resize(canvas_size, Image.Resampling.LANCZOS).save(
                    source_dir / source_name, format="PNG", optimize=True
                )
                for mask_index, raw_mask in enumerate(indexed["masks"]):
                    resized = resize_mask(decode_rle(raw_mask), canvas_size)
                    rle = encode_rle(resized)
                    for task_type in TASK_TYPES:
                        case_id = len(records)
                        image_name = (
                            f"{case_id:06d}_{indexed['source_subset']}_r{row_index}_"
                            f"m{mask_index}_{task_type}.png"
                        )
                        records.append(
                            {
                                "image": image_name,
                                "source_image": source_name,
                                "source_subset": indexed["source_subset"],
                                "parquet_row_index": row_index,
                                "mask_index": mask_index,
                                "num_masks": indexed["num_masks"],
                                "mask": [rle],
                                "task_type": task_type,
                                "editing_instruction": "",
                                "new_instruction": "",
                                "problem": indexed["problem"],
                                "answer": indexed["answer"],
                                "reference_binding": bind_reference(
                                    indexed["answer"], mask_index, indexed["num_masks"]
                                ),
                                "derived_from": "positive_dataset_mask",
                            }
                        )
                source_rows += 1
                bar.update(1)
            offset += batch.num_rows

    if source_rows != len(positive):
        raise ValueError(
            f"Incomplete source materialization: {source_rows} != {len(positive)}"
        )
    write_jsonl(args.out_root / "annotations.jsonl", records)
    summary = {
        "source_rows": source_rows,
        "source_images": source_rows,
        "cases": len(records),
        "task_types": list(TASK_TYPES),
        "task_type_counts": {task: sum(r["task_type"] == task for r in records) for task in TASK_TYPES},
        "regions_per_source": "every dataset mask x three task types",
        "limit_sources": args.limit_sources,
        "index_summary": index_summary,
    }
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
