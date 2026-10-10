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
from .schema import Candidate, CandidateObject, SelectionCase, validate_case


# The frozen v2 release exercises every valid locator/ref combination except
# no-ref+none. Rotate these across units so a training pilot does not collapse
# to one interaction form while keeping no-ref locators spatially grounded.
BENCHMARK_V2_INTERACTIONS = (
    (True, "none"),
    (True, "point"),
    (True, "box"),
    (True, "mask"),
    (False, "point"),
    (False, "box"),
    (False, "mask"),
)


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
    path = cache / (candidate.source_id + ".v2.jpg")
    if path.exists():
        return str(path)
    image = Image.open(candidate.image_path).convert("RGB")
    colors = [(235, 40, 75), (0, 150, 240), (220, 165, 0), (145, 75, 210), (0, 175, 120)]
    for index, obj in enumerate(candidate.objects):
        try:
            mask = load_mask(obj.__dict__, image.size)
        except (ValueError, RuntimeError, TypeError):
            continue
        if not mask.any():
            continue
        color = colors[index % len(colors)]
        layer = Image.new("RGB", image.size, color)
        alpha = Image.fromarray((mask.astype("uint8") * 80), mode="L")
        image = Image.composite(layer, image, alpha)
        x, y = np.where(mask)[1][0], np.where(mask)[0][0]
        # The prompt enumerates object_id values, so the visual marker must
        # carry that same identifier rather than a position-only index.
        ImageDraw.Draw(image).text((int(x), int(y)), str(obj.object_id), fill=color)
    # Keep labels legible while reducing vision-token cost for large SA-1B
    # sources. The original source and masks remain full resolution.
    max_side = 1280
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(32, round(image.width * scale)),
                              max(32, round(image.height * scale))), Image.Resampling.LANCZOS)
    image.save(path, quality=92)
    return str(path)


def select_and_plan(candidates: list[Candidate], selector: VLLMSelector, *, min_targets: int = 2,
                    max_targets: int = 5, max_cases: int | None = None,
                    audit_path: Path | None = None,
                    selection_decisions: dict[str, dict[str, Any]] | None = None,
                    planning_decisions: dict[str, dict[str, Any]] | None = None) -> list[SelectionCase]:
    def record(stage: str, source_id: str, value: dict) -> None:
        if audit_path:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'stage': stage, 'source_id': source_id, **value}, ensure_ascii=False) + '\n')
    selection_jobs = [(_overlay(c), selection_prompt(_candidate_dict(c), min_targets=min_targets, max_targets=max_targets) + " The image has numbered translucent mask overlays matching candidate order; inspect the clean scene beneath them.") for c in candidates]
    if selection_decisions is None:
        selected = selector.generate(selection_jobs)
        for candidate, decision in zip(candidates, selected):
            record('selection', candidate.source_id, {'decision': decision})
    else:
        selected = [selection_decisions.get(c.source_id, {'accept': False, '_parse_error': 'missing cached selection'}) for c in candidates]
    cases: list[SelectionCase] = []
    planning_context: list[tuple[Candidate, dict[str, Any], list[str], dict[str, Any], dict[str, CandidateObject], tuple[int, int], dict[str, Any], list[str]]] = []
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
        protection = [str(x) for x in decision.get("protection_notes", [])]
        planning_context.append((candidate, decision, ids, difficulty, lookup, size, geom_by_id, protection))
    planning_jobs = [(_overlay(candidate), planning_prompt(_candidate_dict(candidate), ids, difficulty))
                     for candidate, _decision, ids, difficulty, _lookup, _size, _geom, _protection in planning_context]
    if planning_decisions is None:
        plans = selector.generate(planning_jobs) if planning_jobs else []
    else:
        plans = [planning_decisions.get(candidate.source_id, {'_parse_error': 'missing cached planning'})
                 for candidate, *_rest in planning_context]
    for case_index, ((candidate, decision, ids, difficulty, lookup, size, geom_by_id, protection), plan) in enumerate(zip(planning_context, plans)):
        record('planning', candidate.source_id, {'plan': plan})
        if plan.get('_parse_error') or not isinstance(plan.get('units'), list):
            record('rejection', candidate.source_id, {'error': plan.get('_parse_error', 'planning output has no units list')})
            continue
        units = []
        for index, raw in enumerate(plan.get("units", []), 1):
            raw = dict(raw)
            raw["unit_index"] = index
            obj = lookup.get(str(raw.get("object_id")))
            if obj is None or obj.object_id not in geom_by_id:
                continue
            if not obj.mask_path:
                mask = load_mask(obj.__dict__, size)
                mask_dir = Path(candidate.image_path).parent / 'selected_masks'
                mask_dir.mkdir(parents=True, exist_ok=True)
                import hashlib
                key = hashlib.sha256((candidate.source_id + ':' + obj.object_id).encode()).hexdigest()
                mask_file = mask_dir / (key + '.png')
                Image.fromarray(mask.astype('uint8') * 255).save(mask_file)
                obj.mask_path = str(mask_file.resolve())
            try:
                unit = build_unit(raw, obj.__dict__, obj.mask_path, geom_by_id[obj.object_id], protection)
            except (TypeError, ValueError) as exc:
                record('rejection', candidate.source_id, {'error': f'unit {raw.get("object_id")}: {exc}'})
                units = []
                break
            unit.has_ref, unit.locator = BENCHMARK_V2_INTERACTIONS[(case_index + index - 1) % len(BENCHMARK_V2_INTERACTIONS)]
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
        except ValueError as exc:
            # A strict contract is a filter, not a reason to abort a long
            # MLLM batch. The full model decision remains in selector metadata
            # for later diagnosis/replanning.
            record('rejection', candidate.source_id, {'error': str(exc)})
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
    p.add_argument("--batch-size", type=int, default=8, help="MLLM request chunk size for resumable progress")
    p.add_argument("--selection-decisions", type=Path, help="reuse stage=selection decisions from an audit JSONL")
    args = p.parse_args(argv)
    candidates = load_candidates(args.manifest)
    if args.max_candidates:
        candidates = sorted(candidates, key=_review_priority, reverse=True)[:args.max_candidates]
    cached = None
    cached_plans = None
    if args.selection_decisions:
        cached = {}
        cached_plans = {}
        planning_rows = 0
        for line in args.selection_decisions.open(encoding='utf-8'):
            row = json.loads(line)
            if row.get('stage') == 'selection':
                cached[row['source_id']] = row.get('decision', {})
            elif row.get('stage') == 'planning':
                cached_plans[row['source_id']] = row.get('plan', {})
                planning_rows += 1
        if planning_rows == 0:
            cached_plans = None
    needs_model = cached is None or cached_plans is None
    selector = VLLMSelector(args.model_id, tensor_parallel_size=args.tensor_parallel_size,
                            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size) if needs_model else None
    try:
        cases = select_and_plan(candidates, selector, min_targets=args.min_targets, max_targets=args.max_targets, max_cases=args.max_cases,
                                audit_path=args.output.with_suffix('.decisions.jsonl'), selection_decisions=cached,
                                planning_decisions=cached_plans)
        write_jsonl(args.output, [c.to_dict() for c in cases])
        print(json.dumps({"status": "ok", "cases": len(cases), "output": str(args.output)}, ensure_ascii=False))
    finally:
        if selector:
            selector.close()


if __name__ == "__main__":
    main()
