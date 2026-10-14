"""Expand validated unit pairs into Qwen-Image-2.1 SAMTok training rows."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


MASK_SPAN = re.compile(
    r"^<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>$"
)


def mask_cot(code: str, label: str) -> str:
    return "```json\n" + json.dumps([{"mask_2d": code, "label": label}], ensure_ascii=False) + "\n```"


def inline_mask(instruction: str, phrase: str, code: str) -> str:
    matches = list(re.finditer(re.escape(phrase), instruction, flags=re.IGNORECASE))
    if len(matches) != 1:
        raise ValueError(f"cannot bind mask label to instruction: {phrase!r}")
    end = matches[0].end()
    return instruction[:end] + " " + code + instruction[end:]


def build_training_rows(pair: dict[str, Any], mask_code: str) -> list[dict[str, Any]]:
    match = MASK_SPAN.fullmatch(mask_code)
    if not match or not (0 <= int(match.group(1)) <= 255 and 256 <= int(match.group(2)) <= 511):
        raise ValueError("mask_code must contain one SAMTok span with codes 0000-0255 and 0256-0511")
    spec = pair.get('instruction_spec') or {}
    if spec:
        from .instruction_protocol import validate_pair
        validate_pair(spec, pair['instruction'], pair['instruction_noref'])
    ref = pair["instruction"]
    phrase = pair.get("ref_phrase") or _infer_phrase(pair)
    no_ref = pair.get("instruction_noref") or ref
    if spec:
        from .instruction_protocol import render
        # Bind the semantic selector slot, not a substring that may also occur
        # inside an exact requested text value.
        ref_inline = render({**spec, 'ref': spec['ref'] + ' ' + mask_code})
        no_ref_inline = render({**spec, 'scope': spec['scope'] + ' ' + mask_code}, False)
    else:
        ref_inline = inline_mask(ref, phrase, mask_code)
        no_ref_inline = inline_mask(no_ref, _infer_noref_phrase(no_ref), mask_code)
    common = {
        "edit_type": pair["operation"],
        "edit_image": pair["source_image"],
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
    # Keep the mask after a complete type-specific region phrase.
    lower = instruction.lower()
    selected = re.search(r"the selected\s+.+?(?=\s+to\s|\s+with\s|\.?$)", lower)
    if selected:
        return instruction[selected.start():selected.end()].rstrip(" .")
    for phrase in ("the object in this region", "this region", "in this region"):
        if phrase in lower:
            return instruction[lower.find(phrase):lower.find(phrase) + len(phrase)]
    # Accept manifests written before the benchmark-v2 region wording was
    # enforced; newly built rows always use one of the phrases above.
    marker = "the selected "
    start = lower.find(marker)
    if start >= 0:
        words = instruction[start:].rstrip(".").split()
        stop = next((i for i, word in enumerate(words[2:], 2) if word.lower() in {"to", "with"}), len(words))
        return " ".join(words[:stop])
    raise ValueError("no-ref instruction has no benchmark region anchor")


def materialize(generated_manifest: Path, codes_jsonl: Path, output: Path,
                exclude_benchmark: Path | None = None) -> list[dict[str, Any]]:
    from .geometry import sha256
    from .source_pool import benchmark_keys
    forbidden_ids, forbidden_hashes = benchmark_keys(exclude_benchmark) if exclude_benchmark else (set(), set())
    image_hashes = {}
    codes = {(str(row["case_id"]), str(row["unit_id"])): row["mask_code"] for row in read_jsonl(codes_jsonl)}
    rows = []
    for pair in read_jsonl(generated_manifest):
        provenance = pair.get("source_provenance", {})
        if str(provenance.get("selection_status", "")).startswith("benchmark_shortlist"):
            raise ValueError("benchmark calibration pairs cannot be materialized for training")
        if exclude_benchmark:
            source_id = str(pair.get("source_id") or pair["case_id"]).removeprefix("sa1b-")
            image_path = pair["source_image"]
            if image_path not in image_hashes:
                image_hashes[image_path] = sha256(image_path)
            if source_id in forbidden_ids or image_hashes[image_path] in forbidden_hashes:
                raise ValueError(f"benchmark source leakage: {source_id}")
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
    p.add_argument("--exclude-benchmark", type=Path, required=True,
                   help="frozen benchmark cases JSONL; checks source IDs and image bytes before export")
    a = p.parse_args(argv)
    rows = materialize(a.generated, a.mask_codes, a.output, a.exclude_benchmark)
    print(json.dumps({"rows": len(rows), "output": str(a.output)}))


if __name__ == "__main__":
    main()
