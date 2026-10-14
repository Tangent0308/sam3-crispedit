"""Scene selection and MLLM-assisted unit planning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import difficulty_features, load_mask, mask_geometry, operation_quality_gate, sha256
from .grounding import validate_spatial_grounding
from .instructions import build_unit
from .io import load_candidates, write_jsonl
from .mllm import VLLMSelector, planning_prompt, selection_prompt
from .schema import Candidate, CandidateObject, SelectionCase, validate_case, OPERATIONS
from .operation_policy import assign_operations, operation_affinity


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


_PART_HINTS = ("strap", "sleeve", "handle", "trim", "pole", "pipe", "rail", "button", "logo", "wheel", "tire", "window", "door", "armband", "necklace", "text", "sign")
_BACKGROUND_HINTS = ("background", "sky", "ground", "wall", "floor", "water", "road", "scene")


def _review_objects(candidate: Candidate, limit: int = 40, operation: str | None = None) -> list[CandidateObject]:
    """Bound overlay/multimodal context while retaining difficult targets.

    Dense SA-1B scenes can contain hundreds of masks. Decode only a compact
    visual shortlist for MLLM review; the original candidate retains every mask
    and a selected object must still resolve against that immutable map.
    """
    if len(candidate.objects) <= limit:
        return list(candidate.objects)
    image = Image.open(candidate.image_path)
    width, height = image.size
    categories = [o.category.lower() for o in candidate.objects]
    frequency = {category: categories.count(category) for category in set(categories)}
    ranked = []
    for index, obj in enumerate(candidate.objects):
        category = obj.category.lower()
        bbox = obj.bbox or []
        area = obj.area_fraction
        if area is None and len(bbox) == 4:
            area = abs(float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1])) / max(1, width * height)
        area = max(float(area or 0.0), 1e-8)
        score = (3.0 if any(token in category for token in _PART_HINTS) else 0.0)
        score += 8.0 if operation and operation_affinity(category, operation) else 0.0
        score += 1.5 if frequency.get(category, 0) > 1 else 0.0
        score += 1.0 if not any(token in category for token in _BACKGROUND_HINTS) else -2.0
        score += min(2.0, 0.02 / area)
        ranked.append((score, -area, index, obj))
    ranked.sort(reverse=True, key=lambda value: (value[0], value[1], -value[2]))
    chosen = [value[3] for value in ranked[: max(1, limit - 4)]]
    # Keep a few large objects so the review is not biased to tiny fragments.
    chosen_ids = {obj.object_id for obj in chosen}
    largest = sorted(candidate.objects, key=lambda obj: -(obj.area_fraction or 0.0))
    for obj in largest:
        if obj.object_id not in chosen_ids:
            chosen.append(obj)
            chosen_ids.add(obj.object_id)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def _candidate_view(candidate: Candidate, operation: str | None = None) -> dict[str, Any]:
    value = _candidate_dict(candidate)
    reviewed = {obj.object_id for obj in _review_objects(candidate, operation=operation)}
    value["objects"] = [obj for obj in value["objects"] if str(obj.get("object_id")) in reviewed]
    return value


def _overlay(candidate: Candidate, objects: list[CandidateObject] | None = None) -> str:
    """Make a temporary numbered dense-mask view for reliable MLLM grounding."""
    cache = Path(candidate.image_path).parent / ".sa1b_selector_overlays"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (candidate.source_id + ".v2.jpg")
    if path.exists():
        return str(path)
    image = Image.open(candidate.image_path).convert("RGB")
    colors = [(235, 40, 75), (0, 150, 240), (220, 165, 0), (145, 75, 210), (0, 175, 120)]
    for index, obj in enumerate(objects or _review_objects(candidate)):
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


def planning_view(candidate: Candidate, selected_ids: list[str]) -> str:
    """Clean appearance at left, aligned mask-only identity map at right.

    Never paint locator colors over the appearance image. Key this cache by
    both source pixels and actual masks so a different shortlist is not stale.
    """
    import hashlib
    source = Image.open(candidate.image_path).convert('RGB')
    objects = [obj for obj in candidate.objects if obj.object_id in set(selected_ids)]
    masks = [(obj, load_mask(obj.__dict__, source.size)) for obj in objects]
    digest = hashlib.sha256(source.tobytes())
    for obj, mask in masks:
        digest.update(obj.object_id.encode()); digest.update(mask.tobytes())
    directory = Path(candidate.image_path).parent / '.sa1b_planning_views'
    directory.mkdir(exist_ok=True)
    path = directory / (digest.hexdigest()[:24]+'.png')
    if path.exists():
        return str(path)
    scale = min(1., 1280/max(source.size))
    size = (max(1,round(source.width*scale)), max(1,round(source.height*scale)))
    clean = source.resize(size, Image.Resampling.LANCZOS)
    guide = Image.new('RGB', size, 'black')
    labels = []
    for obj, mask in masks:
        guide.paste((190,190,190), mask=Image.fromarray(mask.astype('uint8')*255).resize(size,Image.Resampling.NEAREST))
        ys,xs = np.nonzero(mask)
        if len(xs):
            # Select a point on this mask, not the center of its bounding box.
            i = len(xs)//2
            labels.append((round(xs[i]*scale),round(ys[i]*scale),obj.object_id))
    draw = ImageDraw.Draw(guide)
    font = ImageFont.load_default(size=20)
    for x,y,label in labels:
        draw.text((x,y),label,font=font,fill='white',stroke_width=2,stroke_fill='black')
    panel = Image.new('RGB',(size[0]*2,size[1]))
    panel.paste(clean,(0,0));panel.paste(guide,(size[0],0));panel.save(path)
    return str(path)


def select_and_plan(candidates: list[Candidate], selector: VLLMSelector, *, min_targets: int = 2,
                    max_targets: int = 5, max_cases: int | None = None,
                    audit_path: Path | None = None,
                    selection_decisions: dict[str, dict[str, Any]] | None = None,
                    planning_decisions: dict[str, dict[str, Any]] | None = None,
                    enforce_operation_mix: bool = False,
                    operation_assignments: dict[str, str] | None = None) -> list[SelectionCase]:
    if operation_assignments is not None:
        for candidate in candidates:
            if operation_assignments.get(candidate.source_id) not in OPERATIONS:
                raise ValueError(f'missing/invalid operation assignment for {candidate.source_id}')
    def assigned(candidate):
        return operation_assignments.get(candidate.source_id) if operation_assignments is not None else None
    def record(stage: str, source_id: str, value: dict) -> None:
        if audit_path:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'stage': stage, 'source_id': source_id, **value}, ensure_ascii=False) + '\n')
    if selection_decisions is None:
        selection_jobs = []
        job_indices = []
        selected = [{'accept': False, 'evidence': 'fewer_than_min_operation_eligible_candidates'} for _ in candidates]
        for candidate_index, c in enumerate(candidates):
            reviewed = _review_objects(c, operation=assigned(c))
            if assigned(c):
                with Image.open(c.image_path) as source_image:
                    size = source_image.size
                eligible = []
                for obj in reviewed:
                    try:
                        feature = mask_geometry(load_mask(obj.__dict__, size))
                        gate = operation_quality_gate(feature, assigned(c))
                    except (ValueError, TypeError) as exc:
                        record('candidate_mask_rejection', c.source_id, {'object_id': obj.object_id,
                               'required_operation': assigned(c), 'reasons': ['invalid_mask'], 'detail': str(exc)})
                        continue
                    if gate['eligible']:
                        eligible.append(obj)
                    else:
                        record('candidate_mask_rejection', c.source_id, {'object_id': obj.object_id,
                               'required_operation': assigned(c), 'reasons': gate['reasons']})
                reviewed = eligible
            if len(reviewed) < min_targets:
                continue
            view = _candidate_dict(c)
            reviewed_ids = {obj.object_id for obj in reviewed}
            view['objects'] = [obj for obj in view['objects'] if str(obj.get('object_id')) in reviewed_ids]
            selection_jobs.append((planning_view(c, list(reviewed_ids)),
                                   selection_prompt(view, min_targets=min_targets, max_targets=max_targets,
                                                    required_operation=assigned(c))))
            job_indices.append(candidate_index)
        for job_index, result in zip(job_indices, selector.generate(selection_jobs) if selection_jobs else []):
            selected[job_index] = result
        for candidate, decision in zip(candidates, selected):
            record('selection', candidate.source_id, {'decision': decision})
    else:
        selected = [selection_decisions.get(c.source_id, {'accept': False, '_parse_error': 'missing cached selection'}) for c in candidates]
    cases: list[SelectionCase] = []
    planning_context: list[tuple[Candidate, dict[str, Any], list[str], dict[str, Any], dict[str, CandidateObject], tuple[int, int], dict[str, Any], list[str]]] = []
    for candidate, decision in zip(candidates, selected):
        ids = [str(x) for x in decision.get("selected_object_ids", [])]
        if assigned(candidate):
            record('operation_assignment', candidate.source_id, {'required_operation': assigned(candidate)})
        if not decision.get("accept") or not min_targets <= len(ids) <= max_targets:
            record('rejection', candidate.source_id, {'error': 'selection_rejected', 'required_operation': assigned(candidate), 'evidence': decision.get('evidence', '')})
            continue
        # The source-pool importer deliberately marks every candidate as
        # unreviewed.  Once this MLLM decision accepts a scene, carry an
        # auditable status into the immutable case provenance so downstream
        # manifests do not look like unreviewed calibration data.
        candidate.provenance["selection_status"] = "mllm_selected"
        model_id = getattr(selector, "model_id", None)
        if model_id:
            candidate.provenance["selection_model"] = str(model_id)
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
        # Keep multiple targets per image, but do not spend a generation call
        # on masks that cannot support any quality-first operation.
        quality_ids = {
            str(feature["object_id"])
            for feature in difficulty["target_features"]
            if any(feature["operation_quality"][op]["eligible"]
                   for op in ([assigned(candidate)] if assigned(candidate) else OPERATIONS))
        }
        ids = [object_id for object_id in ids if object_id in quality_ids]
        if len(ids) < min_targets:
            record('rejection', candidate.source_id, {
                'error': 'fewer_than_min_quality_targets', 'selected_ids': ids,
            })
            continue
        chosen = [lookup[i] for i in ids]
        # Keep the planning evidence synchronized with the targets that can
        # actually be generated; rejected masks must not influence operation
        # assignment or make the prompt claim more units than will survive.
        difficulty["target_features"] = [
            feature for feature in difficulty["target_features"]
            if str(feature["object_id"]) in set(ids)
        ]
        difficulty["target_count"] = len(difficulty["target_features"])
        difficulty["quality_target_ids"] = ids
        geom_by_id = {x["object_id"]: x for x in difficulty["target_features"]}
        protection = [str(x) for x in decision.get("protection_notes", [])]
        planning_context.append((candidate, decision, ids, difficulty, lookup, size, geom_by_id, protection))
    planning_jobs = [(planning_view(candidate, ids), planning_prompt(_candidate_dict(candidate), ids, difficulty, required_operation=assigned(candidate)))
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
        for skipped in plan.get("skipped_objects", []):
            if isinstance(skipped, dict):
                record('rejection', candidate.source_id, {
                    'error': 'no_natural_local_edit',
                    'object_id': skipped.get('object_id'),
                    'reason': skipped.get('reason', ''),
                })
        for index, raw in enumerate(plan.get("units", []), 1):
            raw = dict(raw)
            raw["unit_index"] = index
            obj = lookup.get(str(raw.get("object_id")))
            if obj is None or obj.object_id not in geom_by_id:
                continue
            feature = geom_by_id[obj.object_id]
            operation = str(raw.get("operation", "")).lower()
            if operation not in {"attribute", "replace", "add", "remove"}:
                record('rejection', candidate.source_id, {
                    'error': 'unknown_operation', 'object_id': obj.object_id,
                    'operation': operation,
                })
                continue
            if assigned(candidate) and operation != assigned(candidate):
                record('rejection', candidate.source_id, {'error': 'assigned_operation_mismatch',
                       'object_id': obj.object_id, 'required_operation': assigned(candidate), 'operation': operation})
                continue
            gate = operation_quality_gate(feature, operation)
            if not gate["eligible"]:
                record('rejection', candidate.source_id, {
                    'error': 'operation_quality_gate', 'object_id': obj.object_id,
                    'operation': operation, 'reasons': gate['reasons'],
                })
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
                continue
            unit.has_ref, unit.locator = BENCHMARK_V2_INTERACTIONS[(case_index + index - 1) % len(BENCHMARK_V2_INTERACTIONS)]
            units.append(unit)
        if len(units) < min_targets:
            record('rejection', candidate.source_id, {
                'error': 'fewer_than_min_quality_units', 'planned': len(units),
                'eligible_targets': len(ids),
            })
            continue
        # Different annotation IDs may still describe the same/nested pixels.
        # Reject the scene rather than claim independent multi-target supervision.
        masks = [(u, load_mask({'mask_path': u.mask_path}, size)) for u in units]
        conflicts = []
        for i, (u, first) in enumerate(masks):
            for v, second in masks[i + 1:]:
                overlap = float((first & second).sum()) / max(1, min(first.sum(), second.sum()))
                if overlap > 0.20:
                    conflicts.append({'units': [u.unit_id, v.unit_id], 'overlap_smaller_fraction': overlap})
        if conflicts:
            record('rejection', candidate.source_id, {'error': 'overlapping_edit_targets', 'conflicts': conflicts})
            continue
        grounding_errors = validate_spatial_grounding(units, candidate.objects)
        if grounding_errors:
            record('rejection', candidate.source_id, {
                'error': 'mask_instruction_grounding',
                'reasons': grounding_errors,
                'unit_ids': [unit.unit_id for unit in units],
            })
            continue
        case = SelectionCase(
            case_id="sa1b-" + candidate.source_id, source=candidate, units=units,
            difficulty={**difficulty, "scene_summary": decision.get("scene_summary", ""), "evidence": decision.get("evidence", "")},
            selector={"model_decision": decision, "model_plan": plan, "required_operation": assigned(candidate)},
        )
        try:
            validate_case(case, min_targets, max_targets, enforce_operation_mix=enforce_operation_mix)
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
    p.add_argument("--planning-mode", choices=["per-mask", "legacy-scene"], default="per-mask",
                   help="default: independently design and audit EVERY mask; legacy-scene reproduces old cohorts")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--manifest", type=Path, required=True, help="normalized SA-1B JSON/JSONL")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default="/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--operation-policy", choices=["balanced", *OPERATIONS], default="balanced",
                   help="assign an operation BEFORE selecting targets; reject rather than switch types")
    p.add_argument("--max-cases", type=int)
    p.add_argument("--max-candidates", type=int, help="bounded pilot input; omit for the full manifest")
    p.add_argument("--min-targets", type=int, default=2)
    p.add_argument("--max-targets", type=int, default=5)
    p.add_argument("--enforce-operation-mix", action="store_true",
                   help="compatibility mode: require benchmark-v2 operation diversity per scene")
    p.add_argument("--max-new-tokens", type=int, default=1000, help="pilot generation budget")
    p.add_argument("--batch-size", type=int, default=8, help="MLLM request chunk size for resumable progress")
    p.add_argument("--selection-decisions", type=Path, help="reuse stage=selection decisions from an audit JSONL")
    args = p.parse_args(argv)
    if args.planning_mode == "per-mask":
        if args.selection_decisions or args.max_cases or args.enforce_operation_mix:
            p.error("per-mask does not accept scene-selection caches, max-cases, or scene operation-mix constraints")
        from .per_mask import main as run
        forwarded = ["--manifest", str(args.manifest), "--output", str(args.output),
                     "--model-id", args.model_id, "--backend", args.backend,
                     "--operation-policy", args.operation_policy,
                     "--tensor-parallel-size", str(args.tensor_parallel_size),
                     "--max-new-tokens", str(args.max_new_tokens)]
        if args.max_candidates:
            forwarded += ["--max-candidates", str(args.max_candidates)]
        run(forwarded)
        return
    candidates = load_candidates(args.manifest)
    if args.max_candidates:
        candidates = sorted(candidates, key=_review_priority, reverse=True)[:args.max_candidates]
    assignments = assign_operations([c.source_id for c in candidates], policy=args.operation_policy)
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
                                planning_decisions=cached_plans,
                                enforce_operation_mix=args.enforce_operation_mix, operation_assignments=assignments)
        write_jsonl(args.output, [c.to_dict() for c in cases])
        print(json.dumps({"status": "ok", "cases": len(cases), "output": str(args.output)}, ensure_ascii=False))
    finally:
        if selector:
            selector.close()


if __name__ == "__main__":
    main()
