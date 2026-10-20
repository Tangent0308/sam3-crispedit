"""Segment: SAM3 for the source object S of every unit, the add anchor cut to the spot, and the SAMTok round trip.

SAM3 runs exactly as the labeling repos segment a grounded object (and as samtok_edit
scripts/data_cleaning/protocol_remask.py re-masks S): the same checkpoint and processor threshold, a context crop for
boxes under 65 % of the image, and crispedit.mask.pipeline.segment_grounded_box with the model's box and a short noun
phrase.  The SAMTok codec is the released VQ-SAM2 tokenizer loaded through samtok_edit21.models.codec of the
SAMTokEdit v3 checkout (as the token caches of round-2 training were built); the round trip is the IoU of the
decoded token with S, and training uses a unit only when it is at least MIN_ROUNDTRIP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import masks as mk
from . import slots as sl

SAM3_CHECKPOINT = "/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt"
SAMTOK_DIR = "/mnt/bn/strategy-mllm-train/user/tanyue/models/SAMTok/Qwen3-VL-8B-SAMTok"
V3_REPO = "/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3"
MIN_ROUNDTRIP = 0.5
# A part of the host is used as S when SAM3's mask for it is mostly inside the host (70 %), at most 85 % of it (else
# it is the host), and at least 0.3x the new item and 0.02 % of the image (else a speck) - the part rules of
# protocol-adapt-4 (PART_PARAMS).
PART_RULES = {"inside": 0.7, "max_host_frac": 0.85, "min_item_k": 0.3, "min_area_frac": 0.0002}


class Segmenter:
    def __init__(self, device="cuda", checkpoint=SAM3_CHECKPOINT, samtok_dir=SAMTOK_DIR, v3_repo=V3_REPO):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        if device.startswith("cuda:"):  # the SAM3 builder moves the model only for the exact string "cuda"
            torch.cuda.set_device(int(device.split(":")[1]))
            device = "cuda"
        model = build_sam3_image_model(device=device, checkpoint_path=checkpoint, load_from_HF=False,
                                       enable_inst_interactivity=True)
        self.processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
        for sub in ("src", "third_party"):
            path = str(Path(v3_repo) / sub)
            if path not in sys.path:
                sys.path.insert(0, path)
        from samtok_edit21.models.codec import load_codec
        self.codec = load_codec(samtok_dir, device)
        self._image_key, self._state = None, None

    def segment(self, image, box, prompt, key=None):
        """SAM3 mask of the object in `box` (0-1000) named by `prompt`, at full resolution, and its metadata."""
        from crispedit.mask.pipeline import _box_area, normalized_box_to_pixels, segment_grounded_box
        from crispedit.mask.regions import context_crop
        shape = (image.height, image.width)
        raw = normalized_box_to_pixels(box, shape)
        crop = None
        if _box_area(raw) / (shape[0] * shape[1]) < 0.65:  # as crispedit.mask.pipeline._segment_items
            crop = context_crop(raw, shape)
            x1, y1, x2, y2 = crop
            local_shape = (y2 - y1, x2 - x1)
            local_box = [(raw[0] - x1) * 1000 / (x2 - x1), (raw[1] - y1) * 1000 / (y2 - y1),
                         (raw[2] - x1) * 1000 / (x2 - x1), (raw[3] - y1) * 1000 / (y2 - y1)]
            state = self.processor.set_image(image.crop(crop))
        else:
            local_shape, local_box = shape, box
            if key is None or key != self._image_key:
                self._state, self._image_key = self.processor.set_image(image), key
            state = self._state
        mask, meta = segment_grounded_box(self.processor, state, prompt, local_box, local_shape, edit_type=None)
        mask = np.asarray(mask) > 0
        if crop is not None:
            full = np.zeros(shape, bool)
            full[crop[1]:crop[3], crop[0]:crop[2]] = mask
            mask = full
        keep = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool)) or v is None}
        return mask, {"prompt": prompt, "box_2d": box, "crop_xyxy": list(crop) if crop else [], **keep}

    def roundtrip(self, image, masks):
        """(span, decoded mask, IoU) per mask; masks of one image are batched."""
        if not masks:
            return []
        spans = self.codec.encode_single_batch([(image, m) for m in masks])
        decoded = [d > 0 for d in self.codec.decode_single_batch([(image, s) for s in spans])]
        out = []
        for mask, span, dec in zip(masks, spans, decoded):
            union = float((mask | dec).sum())
            out.append((span, dec, round(float((mask & dec).sum()) / union, 4) if union else 0.0))
        return out


def footprint(unit, areas, change, shape):
    """Where the new item of an add is: the unit's changed areas, kept to its AFTER box grown by 10 % when there is
    one, and only the main piece (protocol_v2_adapt.main_item)."""
    pieces = np.zeros(shape, bool)
    for a in areas:
        if a["n"] in unit["areas"]:
            pieces |= mk.decode(a)
    box = mk.box_mask(unit["item_box"], shape, 0.1) if unit.get("item_box") else None
    if box is not None:
        inside = pieces & box if pieces.any() else change & box
        if inside.sum() >= 0.05 * box.sum() or (inside.any() and not pieces.any()):
            pieces = inside
        elif not pieces.any():
            return None, "no_change_in_item_box"
    if not pieces.any():
        return None, "no_area"
    return mk.main_item(pieces), "areas" + ("_in_item_box" if box is not None else "")


def segment_unit(seg, image, key, unit, areas, change):
    """S of one unit as a mask, with how it was found; (None, reason) when it cannot be found."""
    spec, slots = unit["spec"], unit["slots"]
    shape = (image.height, image.width)
    info = {}
    if spec["operation"] != "add":
        mask, meta = seg.segment(image, unit["s_box"], sl.sam_prompt(spec, part="part_as_s" in unit["fixes"]), key)
        if not mask.any():
            return None, {"reason": "sam3_empty", "sam3": meta}
        return mask, {"source": "sam3", "sam3": meta, "flags": mk.sam3_flags(mask, unit["s_box"])}
    item, how = footprint(unit, areas, change, shape)
    if item is None:
        return None, {"reason": "add_" + how}
    info["footprint"] = {**mk.encode(item), "how": how}
    gran = slots.get("granularity")
    host_box = unit.get("host_box") or unit["s_box"]
    if gran == "object":
        host, meta = seg.segment(image, unit["s_box"], sl.sam_prompt(spec, slots.get("host_ref")), key)
    else:
        host, meta = seg.segment(image, host_box, sl.sam_prompt(spec, slots.get("host_ref") or slots.get("host_scope")), key)
    info["sam3_host"] = meta
    if not host.any():
        return None, {**info, "reason": "sam3_empty"}
    s_mask, info["source"] = host, "sam3_host"
    if gran == "part":
        part, pmeta = seg.segment(image, unit["s_box"], spec["scope"], key)
        info["sam3_part"] = pmeta
        p = PART_RULES
        h, w = shape
        if (part.any() and (part & host).sum() >= p["inside"] * part.sum() and part.sum() <= p["max_host_frac"] * host.sum()
                and part.sum() >= max(p["min_item_k"] * item.sum(), p["min_area_frac"] * h * w)):
            s_mask, info["source"] = part, "sam3_part"
        else:
            info["part_rejected"] = True
    local, cut = mk.localize_anchor(s_mask, item)
    if local is None:
        return None, {**info, "reason": "anchor_not_at_item"}
    info["localized"] = cut
    s_mask = local
    if spec["placement"].get("site"):
        spot, how = mk.site_spot(s_mask, item)
        if spot is None:
            return None, {**info, "reason": "site_spot_" + str(how)}
        info["site_spot"] = how
        s_mask = spot
    return s_mask, info


def segment_row(seg, image, row_id, plan, areas, change):
    """Masks of every unit of a planned row, their tokens and round trips."""
    out = []
    masks, where = [], []
    for k, unit in enumerate(plan["units"]):
        try:
            mask, info = segment_unit(seg, image, row_id, unit, areas, change)
        except Exception as exc:  # noqa: BLE001 - one unit failing is recorded, not fatal
            mask, info = None, {"reason": f"error:{type(exc).__name__}: {exc}"[:300]}
        entry = {"unit": k, **{key: v for key, v in info.items() if key != "footprint"}}
        if "footprint" in info:
            entry["footprint"] = info["footprint"]
        if mask is not None:
            entry["selection"] = mk.encode(mask)
            masks.append(mask)
            where.append(len(out))
        out.append(entry)
    for position, (span, dec, iou) in zip(where, seg.roundtrip(image, masks)):
        out[position].update(span=span, dec=mk.encode(dec), roundtrip=iou)
    return out
