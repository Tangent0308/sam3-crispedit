"""Expand validated unit pairs into Qwen-Image-2.1 SAMTok training rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


def mask_cot(code: str, label: str) -> str:
    return "```json\n" + json.dumps([{"mask_2d": code, "label": label}], ensure_ascii=False) + "\n```"


def inline_mask(instruction: str, phrase: str, code: str) -> str:
    index = instruction.lower().find(phrase.lower())
    if index < 0:
        raise ValueError(f"cannot bind mask label to instruction: {phrase!r}")
    end = index + len(phrase)
    return instruction[:end] + " " + code + instruction[end:]


def build_training_rows(pair: dict[str, Any], mask_code: str) -> list[dict[str, Any]]:
    if not mask_code.startswith("<|mt_start|>") or "<|mt_end|>" not in mask_code:
        raise ValueError("mask_code must be a SAMTok four-token span")
    ref = pair["instruction"]
    phrase = pair.get("ref_phrase") or _infer_phrase(pair)
    no_ref = pair.get("instruction_noref") or ref
    ref_inline = inline_mask(ref, phrase, mask_code)
    no_ref_phrase = _infer_noref_phrase(no_ref)
    no_ref_inline = inline_mask(no_ref, no_ref_phrase, mask_code)
    common = {
        "edit_type": pair["operation"], "edit_image": pair["source_image"],
        "source_id": pair.get("source_id"), "case_id": pair.get("case_id"),
        "unit_id": pair.get("unit_id"), "mask_path": pair.get("mask"),
        "mask_code": mask_code, "provenance": pair.get("source_provenance", {}),
    }
    return [
        {**common, "sample_type": "edit_ntp", "prompt": ref, "mt_cot": mask_cot(mask_code, phrase)},
        {**common, "sample_type": "edit_umt", "prompt": ref_inline, "image": pair["edited_image"], "instr_variant": "ref"},
        {**common, "sample_type": "edit_umt", "prompt": no_ref_inline, "image": pair["edited_image"], "instr_variant": "noref"},
        {**common, "sample_type": "edit", "prompt": ref, "image": pair["edited_image"]},
    ]


def _infer_phrase(instruction: dict[str, Any]) -> str:
    text = str(instruction.get("instruction", ""))
    for marker in (" to ", " with ", " from "):
        if marker in text:
            return text.split(marker, 1)[0].split(" ", 1)[-1]
    return text.rstrip(".")


def _infer_noref_phrase(instruction: str) -> str:
    # The instruction builder emits "the selected <scope>" for every no-ref
    # unit; keeping this explicit avoids inserting a span after an article.
    lower = instruction.lower()
    marker = "the selected "
    start = lower.find(marker)
    if start < 0:
        raise ValueError("no-ref instruction has no selected-object anchor")
    end = instruction.find(" ", start + len(marker))
    # Include the scope words until the operation's final payload preposition.
    words = instruction[start:].rstrip(".").split()
    if len(words) < 3:
        raise ValueError("invalid no-ref instruction")
    stop = next((i for i, word in enumerate(words[2:], 2) if word.lower() in {"to", "with"}), len(words))
    return " ".join(words[:stop])


def materialize(generated_manifest: Path, codes_jsonl: Path, output: Path) -> list[dict[str, Any]]:
    codes = {(str(row["case_id"]), str(row["unit_id"])): row["mask_code"] for row in read_jsonl(codes_jsonl)}
    rows = []
    for pair in read_jsonl(generated_manifest):
        key = (str(pair["case_id"]), str(pair["unit_id"]))
        if key not in codes:
            raise ValueError(f"missing SAMTok code for {key}")
        rows.extend(build_training_rows(pair, codes[key]))
    write_jsonl(output, rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--mask-codes", type=Path, required=True, help="JSONL with case_id, unit_id, mask_code")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    rows = materialize(a.generated, a.mask_codes, a.output)
    print(json.dumps({"rows": len(rows), "output": str(a.output)}))


if __name__ == "__main__":
    main()
