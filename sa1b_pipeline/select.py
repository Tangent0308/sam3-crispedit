"""Scene selection and MLLM-assisted unit planning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .geometry import difficulty_features, load_mask, sha256
from .instructions import build_unit
from .io import load_candidates, write_jsonl
from .mllm import VLLMSelector, planning_prompt, selection_prompt
from .schema import Candidate, SelectionCase, validate_case


def _candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return candidate.to_dict()


def _review_priority(candidate: Candidate) -> tuple[int, float, int, str]:
    """Cheap pre-MLLM priority for bounded pilots.

    It deliberately uses only annotation metadata. Mask decoding and visual
    difficulty features remain downstream of the MLLM decision, so this never
    replaces visual review in a full run.
    """
    categories = [o.category.strip().lower() for o in candidate.objects if o.category.strip()]
    duplicates = len(categories) - len(set(categories))
    count = len(candidate.objects)
    # Very crowded annotations make a bounded pilot overlay and prompt
    # needlessly large. Keep them eligible, but prefer scenes that fit a
    # human-scale numbered review before the full run is scheduled.
    crowding_penalty = max(0, count - 96) * 1.5
    return (int(count <= 96), min(count, 96) + 2.0 * duplicates - crowding_penalty, -count, candidate.source_id)


def _overlay(candidate: Candidate) -> str:
    """Make a temporary numbered dense-mask view for reliable MLLM grounding."""
    cache = Path(candidate.image_path).parent / ".sa1b_selector_overlays"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (candidate.source_id + ".jpg")
    if path.exists():
        return str(path)
    image = Image.open(candidate.image_path).convert("RGB")
    colors = [(235, 40, 75), (0, 150, 240), (220, 165, 0), (145, 75, 210), (0, 175, 120)]
    for index, obj in enumerate(candidate.objects):
        try:
            mask = load_mask(obj.__dict__, image.size)
        except (ValueError, RuntimeError):
            continue
        color = colors[index % len(colors)]
        layer = Image.new("RGB", image.size, color)
        alpha = Image.fromarray((mask.astype("uint8") * 80), mode="L")
        image = Image.composite(layer, image, alpha)
        x, y = np.where(mask)[1][0], np.where(mask)[0][0]
        ImageDraw.Draw(image).text((int(x), int(y)), str(index + 1), fill=color)
    image.save(path, quality=94)
    return str(path)


def select_and_plan(candidates: list[Candidate], selector: VLLMSelector, *, min_targets: int = 2,
                    max_targets: int = 5, max_cases: int | None = None) -> list[SelectionCase]:
    selection_jobs = [(_overlay(c), selection_prompt(_candidate_dict(c), min_targets=min_targets, max_targets=max_targets) + " The image has numbered translucent mask overlays matching candidate order; inspect the clean scene beneath them.") for c in candidates]
    selected = selector.generate(selection_jobs)
    cases: list[SelectionCase] = []
    for candidate, decision in zip(candidates, selected):
        ids = [str(x) for x in decision.get("selected_object_ids", [])]
        if not decision.get("accept") or not min_targets <= len(ids) <= max_targets:
            continue
        lookup = {o.object_id: o for o in candidate.objects}
        chosen = [lookup[i] for i in ids if i in lookup]
        if len(chosen) != len(ids):
            continue
        with Image.open(candidate.image_path) as image:
            size = image.size
        candidate.provenance.setdefault("image_size", [size[0], size[1]])
        candidate.image_sha256 = candidate.image_sha256 or sha256(candidate.image_path)
        obj_dicts = [o.__dict__ for o in chosen]
        difficulty = difficulty_features(obj_dicts, size)
        geom_by_id = {x["object_id"]: x for x in difficulty["target_features"]}
        plan = selector.generate([(_overlay(candidate), planning_prompt(_candidate_dict(candidate), ids, difficulty))])[0]
        units = []
        protection = [str(x) for x in decision.get("protection_notes", [])]
        for index, raw in enumerate(plan.get("units", []), 1):
            raw = dict(raw)
            raw["unit_index"] = index
            obj = lookup.get(str(raw.get("object_id")))
            if obj is None or not obj.mask_path:
                continue
            unit = build_unit(raw, obj.__dict__, obj.mask_path, geom_by_id[obj.object_id], protection)
            unit.locator = "mask" if index % 3 == 0 else "point" if index % 3 == 1 else "box"
            unit.has_ref = index % 4 != 0
            units.append(unit)
        if len(units) != len(chosen):
            continue
        case = SelectionCase(
            case_id="sa1b-" + candidate.source_id, source=candidate, units=units,
            difficulty={**difficulty, "scene_summary": decision.get("scene_summary", ""), "evidence": decision.get("evidence", "")},
            selector={"model_decision": decision, "model_plan": plan},
        )
        try:
            validate_case(case, min_targets, max_targets)
        except ValueError:
            # A strict contract is a filter, not a reason to abort a long
            # MLLM batch. The full model decision remains in selector metadata
            # for later diagnosis/replanning.
            continue
        cases.append(case)
        if max_cases and len(cases) >= max_cases:
            break
    return cases


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="normalized SA-1B JSON/JSONL")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-cases", type=int)
    p.add_argument("--max-candidates", type=int, help="bounded pilot input; omit for the full manifest")
    p.add_argument("--min-targets", type=int, default=2)
    p.add_argument("--max-targets", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=1200, help="pilot generation budget")
    args = p.parse_args(argv)
    candidates = load_candidates(args.manifest)
    if args.max_candidates:
        candidates = sorted(candidates, key=_review_priority, reverse=True)[:args.max_candidates]
    selector = VLLMSelector(args.model_id, tensor_parallel_size=args.tensor_parallel_size,
                            max_new_tokens=args.max_new_tokens)
    cases = select_and_plan(candidates, selector, min_targets=args.min_targets, max_targets=args.max_targets, max_cases=args.max_cases)
    write_jsonl(args.output, [c.to_dict() for c in cases])
    print(json.dumps({"status": "ok", "cases": len(cases), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
