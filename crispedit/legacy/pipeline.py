"""Legacy CrispEdit pixel-difference-assisted mask pipeline.

This module is retained only for regression comparison.  The production path
lives under :mod:`crispedit.mask` and does not use pixel differences.

目标: 对每条 (input_img, output_img, instruction, type) 生成“被编辑区域/对象”的 mask，
      追求 不错(按类型分治+局部消歧) / 不漏(diff 兜底) / 不冗余(拒绝局部编辑全图化)。

设计原则:
    1) 先把数据集 raw type 归一化成内部 type
    2) input/output 先归一到同一坐标系，避免 mask/diff shape 不一致
    3) instruction 只走本地确定性解析，不依赖 LLM
    4) SAM3 只做语义候选，diff 主要用于“挑对实例”，不是直接定义边界
    5) 对 background/style/motion 做专门分支，不强行套局部物体编辑逻辑

依赖:
    pip install -e .            # sam3
    pip install opencv-python numpy pillow
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# 0. 配置
# ----------------------------------------------------------------------------


@dataclass
class Cfg:
    sam_conf: float = 0.5
    diff_lowfreq_ksize: int = 51
    diff_min_area_frac: float = 5e-4
    global_diff_min_area_frac: float = 1.5e-3
    disambig_min_cover: float = 0.03
    select_touch_px: int = 12
    diff_expand_px: int = 10
    motion_expand_px: int = 16
    dilate_inpaint_px: int = 6
    local_max_area_frac: float = 0.82
    global_style_cover: float = 0.18
    global_mask_warn_frac: float = 0.35
    foreground_seed_min_frac: float = 0.01
    foreground_seed_max_frac: float = 0.65
    background_box_margin_px: int = 18
    background_full_image_frac: float = 0.95
    multi_box_margin_px: int = 18
    max_multi_instances: int = 6


CFG = Cfg()

TYPE_ALIASES = {
    "background change": "background",
    "background": "background",
    "motion change": "motion",
    "motion": "motion",
    "add": "add",
    "remove": "remove",
    "replace": "replace",
    "color": "color",
    "style": "style",
}

LOCAL_EDIT_TYPES = {"add", "remove", "replace", "color"}
GLOBAL_EDIT_TYPES = {"background", "style"}
HUMAN_WORDS = [
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "character",
    "subject",
    "figure",
    "face",
]
PERSON_LIKE_TOKENS = {
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "character",
    "subject",
    "figure",
    "soldier",
    "samurai",
    "warrior",
    "model",
    "dancer",
    "nurse",
    "queen",
    "king",
}
BODY_PARTS = [
    "mouth",
    "hand",
    "arm",
    "leg",
    "head",
    "face",
    "eye",
    "eyes",
    "nose",
    "ear",
    "ears",
    "shoulder",
    "posture",
]
NOOP_PATTERNS = [
    r"\bno change\b",
    r"\bno changes\b",
    r"\bunchanged\b",
    r"\bremain(?:s|ed)?\b.*\bno change\b",
    r"\bleave (?:it|them|the subject|the object) as is\b",
    r"\bkeep (?:it|them|the subject|the object) the same\b",
    r"\bwithout changing\b",
]
TOXIC_PROMPT_WORDS = {
    "background",
    "foreground",
    "image",
    "scene",
    "area",
    "center",
    "middle",
    "corner",
    "quadrant",
    "region",
}
BACKGROUND_GLOBAL_TOKENS = {
    "background",
    "sky",
    "backdrop",
    "scene",
    "environment",
    "surroundings",
    "horizon",
    "landscape",
}

# ----------------------------------------------------------------------------
# 1. 通用小工具
# ----------------------------------------------------------------------------


def canonicalize_type(raw_type: str) -> str:
    s = re.sub(r"\s+", " ", (raw_type or "").strip().lower())
    if s not in TYPE_ALIASES:
        raise ValueError(f"unknown type {raw_type}")
    return TYPE_ALIASES[s]


def _normalize_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_phrase(phrase: Optional[str]) -> Optional[str]:
    if not phrase:
        return None
    s = phrase.strip(" ,.;:")
    s = re.sub(r"^(?:please|kindly|can you|could you)\s+", "", s)
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ,.;:")
    return s or None


def _simplify_prompt_phrase(phrase: Optional[str]) -> Optional[str]:
    if not phrase:
        return None
    s = _clean_phrase(phrase)
    if not s:
        return None
    s = re.sub(
        r"^(?:one|two|three|four|five|six|seven|eight|nine|ten|several|multiple|many|pair of|group of|\d+)\s+",
        "",
        s,
    )
    s = re.sub(r"\b(?:observing the scene|in the scene|from the scene|in the image|in the picture)\b.*$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")
    return s or None


def _head_noun(phrase: Optional[str]) -> Optional[str]:
    if not phrase:
        return None
    s = _clean_phrase(phrase)
    if not s:
        return None
    tokens = re.findall(r"[a-z0-9']+", s.lower())
    if not tokens:
        return None
    stop = {
        "with",
        "without",
        "featuring",
        "containing",
        "holding",
        "wearing",
        "visible",
        "located",
        "occupying",
        "spanning",
        "covering",
        "parked",
        "positioned",
        "facing",
        "across",
        "beneath",
        "under",
        "over",
        "near",
        "behind",
        "beside",
        "next",
        "throughout",
        "across",
        "through",
        "from",
        "to",
        "of",
        "in",
        "on",
        "at",
        "and",
        "the",
        "a",
        "an",
        "entire",
        "whole",
        "majority",
        "central",
        "upper",
        "lower",
        "left",
        "right",
        "background",
        "foreground",
        "image",
        "scene",
        "part",
        "area",
        "corner",
        "quadrant",
        "section",
    }
    head = None
    for tok in tokens:
        if tok in stop:
            break
        head = tok
    return head or tokens[-1]


def _looks_plural(text: Optional[str]) -> bool:
    if not text:
        return False
    s = f" {text.lower()} "
    if " and " in s:
        return True
    if re.search(r"\b(?:two|three|four|five|six|several|multiple|many|pair of|group of)\b", s):
        return True
    if re.search(r"\b\d+\b", s):
        return True
    head = _head_noun(text)
    if head and re.search(r"s$", head) and head not in {"glass", "dress", "bass", "grass"}:
        return True
    return False


def _ellipse_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if mask is None:
        return None
    if px <= 0:
        return mask.astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), _ellipse_kernel(px))


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(np.uint8)
    return cv2.erode(mask.astype(np.uint8), _ellipse_kernel(px))


def _union(masks: List[np.ndarray], shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    masks = [m.astype(np.uint8) for m in masks if m is not None and m.sum() > 0]
    if not masks:
        if shape is None:
            return None
        return np.zeros(shape, np.uint8)
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        out |= mask.astype(np.uint8)
    return out


def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _ensure_mask_shape(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = mask.astype(np.uint8)
    if mask.shape == shape:
        return mask
    h, w = shape
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def _cover(mask: np.ndarray, ref: np.ndarray) -> float:
    area = int(mask.sum())
    if area <= 0:
        return 0.0
    return float((mask & ref).sum()) / float(area)


def _recall(mask: np.ndarray, ref: np.ndarray) -> float:
    area = int(ref.sum())
    if area <= 0:
        return 0.0
    return float((mask & ref).sum()) / float(area)


def _bbox_center(box: np.ndarray, shape: Tuple[int, int]) -> Tuple[float, float]:
    h, w = shape
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1) / max(w, 1)
    cy = 0.5 * (y0 + y1) / max(h, 1)
    return float(cx), float(cy)


def _location_score(box: Optional[np.ndarray], shape: Tuple[int, int], hint: Optional[Dict]) -> float:
    if box is None or not hint or hint.get("x") is None or hint.get("y") is None:
        return 0.0
    cx, cy = _bbox_center(box, shape)
    radius = max(float(hint.get("radius", 0.35)), 1e-3)
    dx = (cx - float(hint["x"])) / radius
    dy = (cy - float(hint["y"])) / radius
    dist = float(np.sqrt(dx * dx + dy * dy))
    return max(0.0, 1.0 - dist)


def _clip_box_xyxy(box: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = box.astype(np.float32)
    x0 = float(np.clip(x0, 0, w - 1))
    y0 = float(np.clip(y0, 0, h - 1))
    x1 = float(np.clip(x1, x0 + 1, w))
    y1 = float(np.clip(y1, y0 + 1, h))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _box_xyxy_to_norm_cxcywh(box: np.ndarray, shape: Tuple[int, int]) -> List[float]:
    h, w = shape
    x0, y0, x1, y1 = _clip_box_xyxy(box, shape)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = x0 + 0.5 * bw
    cy = y0 + 0.5 * bh
    return [
        float(np.clip(cx / max(w, 1), 0.0, 1.0)),
        float(np.clip(cy / max(h, 1), 0.0, 1.0)),
        float(np.clip(bw / max(w, 1), 1e-4, 1.0)),
        float(np.clip(bh / max(h, 1), 1e-4, 1.0)),
    ]


def _component_dicts(binm: np.ndarray) -> List[Dict]:
    binm = (binm > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm)
    h, w = binm.shape
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        mask = (lab == i).astype(np.uint8)
        comps.append(
            {
                "mask": mask,
                "area": area,
                "area_frac": float(area) / float(h * w),
                "box": np.array([x, y, x + bw, y + bh], dtype=np.float32),
            }
        )
    return comps


def _select_component_by_location(binm: np.ndarray, hint: Optional[Dict]) -> Optional[np.ndarray]:
    comps = _component_dicts(binm)
    if not comps:
        return None
    shape = binm.shape
    scored = []
    for comp in comps:
        loc = _location_score(comp["box"], shape, hint)
        score = loc + 0.2 * min(comp["area_frac"] / 0.05, 1.0)
        scored.append((score, comp))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]["mask"].astype(np.uint8)


# ----------------------------------------------------------------------------
# 2. Diff 图
# ----------------------------------------------------------------------------


def _global_register(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """ECC 全局仿射配准；失败则原样返回。"""
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        gi = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        gm = cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-4)
        cv2.findTransformECC(gi, gm, warp, cv2.MOTION_AFFINE, criteria)
        h, w = ref.shape[:2]
        return cv2.warpAffine(
            mov,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT101,
        )
    except cv2.error:
        return mov


def _drop_small_cc(binm: np.ndarray, min_frac: float) -> np.ndarray:
    h, w = binm.shape
    out = np.zeros_like(binm, dtype=np.uint8)
    for comp in _component_dicts(binm):
        if comp["area"] >= min_frac * h * w:
            out |= comp["mask"].astype(np.uint8)
    return out


def robust_diff(
    img_in: np.ndarray,
    img_out: np.ndarray,
    register: bool = True,
    remove_lowfreq: bool = True,
    min_area_frac: Optional[float] = None,
) -> np.ndarray:
    """返回局部编辑友好的二值变化图。"""
    h, w = img_in.shape[:2]
    mov = cv2.resize(img_out, (w, h), interpolation=cv2.INTER_LINEAR)
    if register:
        mov = _global_register(img_in, mov)

    lab_i = cv2.cvtColor(img_in, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_o = cv2.cvtColor(mov, cv2.COLOR_RGB2LAB).astype(np.float32)
    d_e = np.sqrt(((lab_i - lab_o) ** 2).sum(axis=2))

    if remove_lowfreq:
        k = max(3, int(CFG.diff_lowfreq_ksize) | 1)
        low = cv2.GaussianBlur(d_e, (k, k), 0)
        d_e = np.clip(d_e - low, 0, None)

    gray_i = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
    gray_o = cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY)
    grad_i = cv2.Sobel(gray_i, cv2.CV_32F, 1, 1)
    grad_o = cv2.Sobel(gray_o, cv2.CV_32F, 1, 1)
    struct = np.abs(grad_i - grad_o)

    score = d_e + 0.5 * struct
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binm = cv2.threshold(score, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    area_thr = CFG.diff_min_area_frac if min_area_frac is None else min_area_frac
    return _drop_small_cc(binm, area_thr)


# ----------------------------------------------------------------------------
# 3. instruction 解析（纯本地确定性）
# ----------------------------------------------------------------------------


def _parse_location_from_text(loc_text: str) -> Optional[Dict]:
    s = _normalize_text(loc_text)
    x_terms = []
    y_terms = []
    if "left" in s:
        x_terms.append(0.24)
    if "right" in s:
        x_terms.append(0.76)
    if any(tok in s for tok in ("upper", "top")):
        y_terms.append(0.24)
    if any(tok in s for tok in ("lower", "bottom")):
        y_terms.append(0.76)
    if any(tok in s for tok in ("center", "central", "middle")):
        if not x_terms:
            x_terms.append(0.5)
        if not y_terms:
            y_terms.append(0.5)

    hint = {}
    if x_terms or y_terms:
        hint["x"] = float(np.mean(x_terms)) if x_terms else 0.5
        hint["y"] = float(np.mean(y_terms)) if y_terms else 0.5
        hint["radius"] = 0.32 if ("quadrant" in s or "side" in s) else 0.38
    if "foreground" in s:
        hint["region"] = "foreground"
    if "background" in s:
        hint["region"] = "background"
    return hint or None


def _extract_location_hint(phrase: Optional[str]) -> Tuple[Optional[str], Optional[Dict]]:
    if not phrase:
        return None, None

    s = phrase
    hints = []
    patterns = [
        r"\b(?:positioned|located|occupying)\s+in\s+the\s+([a-z\s]+?(?:area|quadrant))\b",
        r"\bin\s+the\s+([a-z\s]+?(?:area|quadrant|background|foreground|center|middle))\b",
        r"\bon\s+the\s+([a-z\s]+?side)\b",
    ]
    for pat in patterns:
        for match in re.finditer(pat, s):
            hint = _parse_location_from_text(match.group(1))
            if hint:
                hints.append(hint)
            s = s.replace(match.group(0), " ")

    if not hints:
        fallback = _parse_location_from_text(s)
        if fallback and ("x" in fallback or fallback.get("region")):
            hints.append(fallback)

    merged = None
    if hints:
        merged = {}
        xs = [h["x"] for h in hints if "x" in h]
        ys = [h["y"] for h in hints if "y" in h]
        if xs:
            merged["x"] = float(np.mean(xs))
        if ys:
            merged["y"] = float(np.mean(ys))
        merged["radius"] = min(h.get("radius", 0.38) for h in hints)
        for region in ("foreground", "background"):
            if any(h.get("region") == region for h in hints):
                merged["region"] = region
                break

    return _clean_phrase(s), merged


def _strip_attachment_tail(phrase: Optional[str], etype: str) -> Optional[str]:
    if not phrase:
        return None
    s = phrase
    if etype == "add":
        s = re.sub(r"\bback\s+(?:onto|into|on top of|behind|beside|beneath|under|next to)\b.*$", "", s)
        s = re.sub(r"\b(?:onto|into|on top of|behind|beside|beneath|under|next to)\b.*$", "", s)
    elif etype == "remove":
        s = re.sub(r"\bfrom\b.*$", "", s)
    return _clean_phrase(s)


def _detect_noop(instruction: str, etype: str) -> bool:
    if etype != "motion":
        return False
    s = _normalize_text(instruction)
    if re.search(r"\b(?:except|but|while|however|instead)\b", s):
        return False
    return any(re.search(pat, s) for pat in NOOP_PATTERNS)


def _extract_motion_focus(instruction: str) -> Optional[str]:
    s = _normalize_text(instruction)
    for part in BODY_PARTS:
        if re.search(rf"\b{re.escape(part)}s?\b", s):
            return part
    for word in HUMAN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", s):
            return word
    return None


def parse_instruction(instruction: str, etype: str) -> Dict:
    s = _normalize_text(instruction)
    source = None
    target = None
    location_hint = None
    is_global = etype in GLOBAL_EDIT_TYPES
    is_noop = _detect_noop(s, etype)
    parse_ok = True

    if etype == "add":
        m = re.search(r"\b(?:add|insert|place|put|restore|bring back)\b\s+(.+)", s)
        target = m.group(1) if m else s
        target, target_loc = _extract_location_hint(target)
        target = _strip_attachment_tail(target, etype)
        target = _simplify_prompt_phrase(target)
        location_hint = target_loc
    elif etype == "remove":
        m = re.search(r"\b(?:remove|erase|delete|get rid of|take away)\b\s+(.+)", s)
        source = m.group(1) if m else s
        source, source_loc = _extract_location_hint(source)
        source = _strip_attachment_tail(source, etype)
        source = _simplify_prompt_phrase(source)
        head = _head_noun(source)
        if head:
            source = head
        location_hint = source_loc
    elif etype == "replace":
        m = re.search(r"\breplace\s+(.+?)\s+with\s+(.+)", s)
        if not m:
            m = re.search(r"\bswap\s+(.+?)\s+for\s+(.+)", s)
        if not m:
            m = re.search(r"\bchange\s+(.+?)\s+(?:into|to)\s+(.+)", s)
        if m:
            source, target = m.group(1), m.group(2)
            source, source_loc = _extract_location_hint(source)
            target, target_loc = _extract_location_hint(target)
            source = _simplify_prompt_phrase(source)
            target = _simplify_prompt_phrase(target)
            source_head = _head_noun(source)
            target_head = _head_noun(target)
            if source_head:
                source = source_head
            if target_head:
                target = target_head
            location_hint = source_loc or target_loc
        else:
            parse_ok = False
    elif etype == "color":
        patterns = [
            r"\bchange the color of\s+(.+?)\s+to\s+.+",
            r"\brecolor\s+(.+?)(?:\s+to\s+.+)?$",
            r"\bturn\s+(.+?)\s+into\s+.+",
            r"\bmake\s+(.+?)\s+(?:more\s+)?(?:red|green|blue|yellow|purple|orange|pink|black|white|brown|gold|silver|gray|grey).+",
        ]
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                source = m.group(1)
                break
        if source is None:
            parse_ok = False
    elif etype == "motion":
        if not is_noop:
            source = _extract_motion_focus(s)
            target = source
        parse_ok = source is not None or is_noop
    elif etype == "background":
        source = "__background__"
        is_global = True
    elif etype == "style":
        source = None
        target = None
        is_global = True
    else:
        raise ValueError(f"unknown type {etype}")

    if location_hint is None:
        source, source_loc = _extract_location_hint(source)
        target, target_loc = _extract_location_hint(target)
        location_hint = source_loc or target_loc
    else:
        source = _clean_phrase(source)
        target = _clean_phrase(target)

    if etype == "background":
        lowered = s
        if not any(tok in lowered for tok in BACKGROUND_GLOBAL_TOKENS):
            is_global = False
            parse_ok = False

    if etype == "style":
        # CrispEdit style 基本都是全图风格迁移；只有极少数可能是局部 style。
        is_global = True

    phrase_basis = source or target or s
    allow_multiple = _looks_plural(phrase_basis)

    return {
        "source": _clean_phrase(source),
        "target": _clean_phrase(target),
        "location_hint": location_hint,
        "is_global": is_global,
        "is_noop": is_noop,
        "allow_multiple": allow_multiple,
        "parse_ok": parse_ok,
    }


# ----------------------------------------------------------------------------
# 4. 归一化工作区 + SAM 查询
# ----------------------------------------------------------------------------


def _prepare_workspace(processor, pil_in: Image.Image, pil_out: Image.Image) -> Dict:
    pil_in = pil_in.convert("RGB")
    pil_out = pil_out.convert("RGB")
    target_size = pil_in.size
    if pil_out.size != target_size:
        pil_out = pil_out.resize(target_size, Image.BILINEAR)

    np_in = np.array(pil_in)
    np_out = np.array(pil_out)
    shape = np_in.shape[:2]

    local_diff = robust_diff(np_in, np_out, register=True, remove_lowfreq=True)
    motion_diff = robust_diff(np_in, np_out, register=False, remove_lowfreq=False)
    global_diff = robust_diff(
        np_in,
        np_out,
        register=True,
        remove_lowfreq=False,
        min_area_frac=CFG.global_diff_min_area_frac,
    )

    state_in = processor.set_image(pil_in)
    state_out = processor.set_image(pil_out)

    return {
        "pil_in": pil_in,
        "pil_out": pil_out,
        "np_in": np_in,
        "np_out": np_out,
        "shape": shape,
        "local_diff": local_diff.astype(np.uint8),
        "motion_diff": motion_diff.astype(np.uint8),
        "global_diff": global_diff.astype(np.uint8),
        "state_in": state_in,
        "state_out": state_out,
    }


def _query_sam(processor, state: Dict, shape: Tuple[int, int], phrase: Optional[str] = None, box_hint: Optional[np.ndarray] = None) -> List[Dict]:
    processor.reset_all_prompts(state)
    out = state
    if phrase:
        out = processor.set_text_prompt(prompt=phrase, state=state)
    if box_hint is not None:
        norm_box = _box_xyxy_to_norm_cxcywh(box_hint, shape)
        out = processor.add_geometric_prompt(norm_box, True, state=state)

    if "masks" not in out or out["masks"].shape[0] == 0:
        return []

    masks = out["masks"]
    boxes = out["boxes"]
    scores = out["scores"]
    instances = []
    for i in range(masks.shape[0]):
        mask = masks[i, 0].detach().cpu().numpy().astype(np.uint8)
        mask = _ensure_mask_shape(mask, shape)
        box = boxes[i].detach().cpu().numpy().astype(np.float32)
        instances.append(
            {
                "mask": mask,
                "score": float(scores[i]),
                "box": _clip_box_xyxy(box, shape),
                "area_frac": float(mask.mean()),
            }
        )
    return instances


def _best_diff_box(diff_map: np.ndarray, location_hint: Optional[Dict]) -> Optional[np.ndarray]:
    comps = _component_dicts(diff_map)
    if not comps:
        return None
    shape = diff_map.shape
    scored = []
    for comp in comps:
        loc = _location_score(comp["box"], shape, location_hint)
        score = 1.2 * loc + 0.3 * min(comp["area_frac"] / 0.03, 1.0)
        scored.append((score, comp["box"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _invert_components(diff_map: np.ndarray, min_area_frac: float = 0.01) -> np.ndarray:
    stable = (1 - (diff_map > 0).astype(np.uint8)).astype(np.uint8)
    stable = cv2.morphologyEx(stable, cv2.MORPH_OPEN, _ellipse_kernel(8))
    stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, _ellipse_kernel(12))
    stable = _erode(stable, 4)
    return _drop_small_cc(stable, min_area_frac)


def _expand_box(box: np.ndarray, shape: Tuple[int, int], margin_px: int) -> np.ndarray:
    x0, y0, x1, y1 = box.astype(np.float32)
    expanded = np.array([x0 - margin_px, y0 - margin_px, x1 + margin_px, y1 + margin_px], dtype=np.float32)
    return _clip_box_xyxy(expanded, shape)


def _merge_boxes(boxes: List[np.ndarray], shape: Tuple[int, int], margin_px: int = 0) -> Optional[np.ndarray]:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    stacked = np.stack(boxes, axis=0)
    merged = np.array(
        [stacked[:, 0].min(), stacked[:, 1].min(), stacked[:, 2].max(), stacked[:, 3].max()],
        dtype=np.float32,
    )
    if margin_px > 0:
        merged = _expand_box(merged, shape, margin_px)
    return _clip_box_xyxy(merged, shape)


def _mask_from_box(box: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    if box is None:
        return canvas
    x0, y0, x1, y1 = _clip_box_xyxy(box, shape).astype(int)
    canvas[y0:y1, x0:x1] = 1
    return canvas


def _top_diff_boxes(diff_map: np.ndarray, location_hint: Optional[Dict], limit: int = 3) -> List[np.ndarray]:
    comps = _component_dicts(diff_map)
    if not comps:
        return []
    shape = diff_map.shape
    scored = []
    for comp in comps:
        loc = _location_score(comp["box"], shape, location_hint)
        score = 1.2 * loc + 0.4 * min(comp["area_frac"] / 0.03, 1.0)
        scored.append((score, comp["box"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [box for _, box in scored[:limit]]


def _foreground_text_candidates(np_img: np.ndarray) -> List[str]:
    h, w = np_img.shape[:2]
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(edges.mean() / 255.0)
    sat = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)[..., 1].mean() / 255.0
    candidates = ["person", "animal", "object", "foreground object"]
    if edge_density < 0.08 and sat > 0.18:
        candidates = ["object", "product", "toy", "foreground object", "person", "animal"]
    elif h >= w:
        candidates = ["person", "foreground object", "object", "animal"]

    # background change 里常见“建筑/产品/玩具主体”的情况，统一补一组确定性通用前景词。
    candidates.extend(["building", "house", "architecture", "structure", "vehicle", "food"])

    dedup = []
    seen = set()
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            dedup.append(cand)
    return dedup


def _prompt_variants(phrase: Optional[str]) -> List[str]:
    if not phrase:
        return []
    base = _clean_phrase(phrase)
    if not base:
        return []
    variants = [base]
    lowered = base.lower()

    head = _head_noun(lowered) or lowered.split()[-1]
    if head not in variants:
        variants.append(head)
    singular = re.sub(r"s$", "", head)
    if singular and singular not in variants:
        variants.append(singular)

    # 对 remove / replace 的异常生物类目标补一组更贴近 SAM3 语义空间的别名。
    if any(tok in lowered for tok in ("alien", "extraterrestrial", "monster")):
        variants.extend(["humanoid", "figure", "creature"])
    if any(tok in lowered for tok in ("person", "human", "subject", "character")):
        variants.extend(["person", "human", "figure"])
    if any(tok in lowered for tok in ("building", "house", "temple", "pagoda")):
        variants.extend(["building", "house", "architecture", "structure"])

    dedup = []
    seen = set()
    for cand in variants:
        cand = _clean_phrase(cand)
        if not cand or cand in seen:
            continue
        if cand in TOXIC_PROMPT_WORDS:
            continue
        seen.add(cand)
        dedup.append(cand)
    return dedup


def _select_foreground_candidate(instances: List[Dict], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not instances:
        return None
    center_hint = {"x": 0.5, "y": 0.5, "radius": 0.45}
    h, w = shape
    edge_band = np.zeros(shape, np.uint8)
    edge = max(24, int(0.08 * min(h, w)))
    edge_band[:edge, :] = 1
    edge_band[-edge:, :] = 1
    edge_band[:, :edge] = 1
    edge_band[:, -edge:] = 1

    scored = []
    for inst in instances:
        area_frac = float(inst["area_frac"])
        if area_frac <= 0.005 or area_frac >= 0.75:
            continue
        loc = _location_score(inst["box"], shape, center_hint)
        edge_touch = _cover(inst["mask"], edge_band)
        area_pref = 1.0 - min(abs(area_frac - 0.18) / 0.18, 1.0)
        score = 1.3 * loc + 0.55 * area_pref + 0.2 * float(inst["score"]) - 0.45 * edge_touch
        scored.append((score, inst))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]["mask"].astype(np.uint8)


def _query_best_foreground_mask(processor, state: Dict, shape: Tuple[int, int], prompts: List[str]) -> Optional[np.ndarray]:
    instances = []
    for prompt in prompts:
        instances.extend(_query_sam(processor, state, shape, phrase=prompt))
    return _select_foreground_candidate(instances, shape)


def _is_nonhuman_humanoid_phrase(phrase: Optional[str]) -> bool:
    if not phrase:
        return False
    lowered = phrase.lower()
    return any(tok in lowered for tok in ("alien", "extraterrestrial", "monster", "creature"))


def _is_person_like_phrase(phrase: Optional[str]) -> bool:
    if not phrase:
        return False
    lowered = phrase.lower()
    return any(tok in lowered for tok in PERSON_LIKE_TOKENS)


def _select_instances(
    instances: List[Dict],
    ref_map: np.ndarray,
    shape: Tuple[int, int],
    location_hint: Optional[Dict],
    etype: str,
    allow_multiple: bool = False,
) -> Tuple[Optional[np.ndarray], List[Dict]]:
    if not instances:
        return None, []

    ref_touch = _dilate(ref_map, CFG.select_touch_px) if ref_map is not None else None
    scored = []
    for inst in instances:
        mask = inst["mask"]
        box = inst["box"]
        cover = _cover(mask, ref_touch) if ref_touch is not None else 0.0
        recall = _recall(mask, ref_touch) if ref_touch is not None else 0.0
        loc = _location_score(box, shape, location_hint)
        area_frac = float(mask.mean())

        score = 1.6 * cover + 0.7 * recall + 0.35 * loc + 0.25 * float(inst["score"])
        if etype in LOCAL_EDIT_TYPES and area_frac > CFG.local_max_area_frac:
            score -= 2.0
        scored.append(
            {
                **inst,
                "cover": cover,
                "recall": recall,
                "loc": loc,
                "rank_score": score,
            }
        )

    scored.sort(key=lambda x: x["rank_score"], reverse=True)
    if not scored:
        return None, []

    if allow_multiple:
        keep = [
            cand
            for cand in scored
            if cand["cover"] >= CFG.disambig_min_cover and cand["area_frac"] <= CFG.local_max_area_frac
        ]
        if not keep:
            keep = [scored[0]] if scored[0]["rank_score"] > 0 else []
        keep = keep[: CFG.max_multi_instances]
    else:
        best = scored[0]
        keep = [best] if best["rank_score"] > -0.25 else []

    if not keep:
        return None, scored
    return _union([cand["mask"] for cand in keep], shape=shape), scored


# ----------------------------------------------------------------------------
# 5. 融合分支
# ----------------------------------------------------------------------------


def _fallback_local(diff_map: np.ndarray, location_hint: Optional[Dict], shape: Tuple[int, int], expand_px: int = 6) -> np.ndarray:
    if diff_map is None or diff_map.sum() == 0:
        return np.zeros(shape, np.uint8)
    focus = _select_component_by_location(diff_map, location_hint) if location_hint else diff_map
    if focus is None or focus.sum() == 0:
        focus = diff_map
    return _dilate(focus, expand_px)


def _touching_components(diff_map: np.ndarray, anchor_mask: np.ndarray, expand_px: int) -> np.ndarray:
    if diff_map is None or diff_map.sum() == 0:
        return np.zeros_like(anchor_mask, dtype=np.uint8)
    if anchor_mask is None or anchor_mask.sum() == 0:
        return diff_map.astype(np.uint8)

    keep = []
    touch = _dilate(anchor_mask, expand_px)
    for comp in _component_dicts(diff_map):
        if (comp["mask"] & touch).sum() > 0:
            keep.append(comp["mask"].astype(np.uint8))
    if keep:
        return _union(keep, shape=diff_map.shape)
    return np.zeros_like(diff_map, dtype=np.uint8)


def _estimate_background_mask(processor, workspace: Dict, phrases: Dict) -> Tuple[np.ndarray, str]:
    shape = workspace["shape"]
    global_diff = workspace["global_diff"]

    h, w = shape
    center_hint = {"x": 0.5, "y": 0.5, "radius": 0.45}
    edge_band = np.zeros(shape, np.uint8)
    edge = max(24, int(0.08 * min(h, w)))
    edge_band[:edge, :] = 1
    edge_band[-edge:, :] = 1
    edge_band[:, :edge] = 1
    edge_band[:, -edge:] = 1

    # 1) 先尝试从 diff 里找到“主体较稳定、背景变化更大”的场景。
    stable = _invert_components(global_diff, min_area_frac=CFG.foreground_seed_min_frac)
    comps = _component_dicts(stable)
    seed_mask = None
    seed_box = None
    if comps:
        candidates = []
        for comp in comps:
            area_frac = comp["area_frac"]
            if area_frac < CFG.foreground_seed_min_frac or area_frac > CFG.foreground_seed_max_frac:
                continue
            loc = _location_score(comp["box"], shape, center_hint)
            edge_touch = _cover(comp["mask"], edge_band)
            score = 1.5 * loc + 0.2 * min(area_frac / 0.12, 1.0) - 0.3 * edge_touch
            candidates.append((score, comp))
        if not candidates:
            candidates = [(_location_score(comp["box"], shape, center_hint), comp) for comp in comps]
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_comps = [comp for _, comp in candidates[:2]]
        seed_mask = _union([comp["mask"] for comp in top_comps], shape=shape)
        seed_box = _merge_boxes([comp["box"] for comp in top_comps], shape, margin_px=CFG.background_box_margin_px)

    # 2) 从 input 端显式找稳定前景主体；对产品/玩具/建筑等场景更稳。
    fg_candidates = []
    prompt_list = _foreground_text_candidates(workspace["np_in"])
    for phrase in prompt_list:
        fg_candidates.extend(_query_sam(processor, workspace["state_in"], shape, phrase=phrase))
    if seed_box is not None:
        for phrase in prompt_list:
            fg_candidates.extend(_query_sam(processor, workspace["state_in"], shape, phrase=phrase, box_hint=seed_box))
        fg_candidates.extend(_query_sam(processor, workspace["state_in"], shape, phrase=None, box_hint=seed_box))

    ref_map = seed_mask if seed_mask is not None and seed_mask.sum() > 0 else stable if comps else None
    fg_mask = None
    generic_fg = _select_foreground_candidate(fg_candidates, shape) if fg_candidates else None
    if fg_candidates:
        fg_mask, _ = _select_instances(
            fg_candidates,
            ref_map if ref_map is not None and ref_map.sum() > 0 else np.zeros(shape, np.uint8),
            shape,
            center_hint,
            etype="background",
            allow_multiple=False,
        )

    # 若 instruction 并不是真背景编辑（例如改屋顶/结构材质），直接回退成局部对象编辑。
    if not phrases.get("is_global", True):
        object_prompts = ["building", "house", "architecture", "structure", "roof"]
        local_candidates = []
        for prompt in object_prompts:
            local_candidates.extend(_query_sam(processor, workspace["state_in"], shape, phrase=prompt))
        local_mask, _ = _select_instances(local_candidates, global_diff, shape, center_hint, etype="color", allow_multiple=False)
        if local_mask is not None and local_mask.sum() > 0:
            return local_mask.astype(np.uint8), "BACKGROUND_LOCAL_OBJECT"

    if generic_fg is not None and (fg_mask is None or fg_mask.mean() < 0.05 or fg_mask.mean() > 0.65):
        fg_mask = generic_fg

    status = "OK"
    if fg_mask is None or fg_mask.mean() < 0.015:
        if generic_fg is not None and generic_fg.sum() > 0:
            fg_mask = generic_fg
            status = "LOW_CONF_BG"
        elif seed_mask is not None and seed_mask.sum() > 0:
            fg_mask = seed_mask
            status = "LOW_CONF_BG"
        else:
            # 彻底找不到稳定前景时，用“高变化区域扩张”近似背景。
            coarse_bg = postprocess(_dilate(global_diff, 18), fill_holes=False)
            return coarse_bg.astype(np.uint8), "LOW_CONF_BG"

    fg_mask = postprocess(fg_mask, fill_holes=False)
    bg = (1 - fg_mask).astype(np.uint8)

    # 只保留与高变化区域同连通/相邻的背景，避免“全图减主体”过度扩张。
    bg_candidates = _touching_components(bg, _dilate(global_diff, 10), CFG.diff_expand_px)
    if bg_candidates is not None and bg_candidates.sum() > 0:
        bg = bg_candidates.astype(np.uint8)

    # 若主体太小，说明几乎整图都变；允许大背景，但避免空 mask。
    if bg.mean() < CFG.global_mask_warn_frac:
        coarse_bg = postprocess(_dilate(global_diff, 18), fill_holes=False)
        if coarse_bg.mean() > bg.mean():
            bg = coarse_bg
        status = "LOW_CONF_BG"

    if bg.mean() > CFG.background_full_image_frac:
        status = "LOW_CONF_BG_FULL"

    return bg.astype(np.uint8), status


def _query_and_select_local(
    processor,
    state: Dict,
    shape: Tuple[int, int],
    phrase: Optional[str],
    diff_map: np.ndarray,
    location_hint: Optional[Dict],
    etype: str,
    allow_multiple: bool,
    max_area_frac: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], str]:
    if not phrase:
        return None, "NO_PHRASE"

    area_cap = CFG.local_max_area_frac if max_area_frac is None else max_area_frac

    def _accept(mask: Optional[np.ndarray]) -> bool:
        return mask is not None and mask.sum() > 0 and float(mask.mean()) <= area_cap

    prompt_variants = _prompt_variants(phrase)
    ref_touch = _dilate(diff_map, CFG.select_touch_px) if diff_map is not None else None

    def _format_tag(base_tag: str, variant_idx: int, prompt: str) -> str:
        if variant_idx == 0:
            return base_tag
        if base_tag == "SAM_TEXT":
            return f"SAM_TEXT_ALIAS:{prompt}"
        if base_tag == "SAM_TEXT_BOX":
            return f"SAM_TEXT_BOX_ALIAS:{prompt}"
        if base_tag == "SAM_TEXT_MULTI_BOX":
            return f"SAM_TEXT_MULTI_BOX_ALIAS:{prompt}"
        return base_tag

    def _score_mask(mask: Optional[np.ndarray]) -> float:
        if mask is None or mask.sum() == 0:
            return float("-inf")
        box = _mask_to_box(mask)
        loc = _location_score(box, shape, location_hint) if box is not None else 0.0
        cover = _cover(mask, ref_touch) if ref_touch is not None else 0.0
        recall = _recall(mask, ref_touch) if ref_touch is not None else 0.0
        area_frac = float(mask.mean())
        score = 1.2 * recall + 0.9 * cover + 0.25 * loc + 0.05 * min(area_frac / 0.08, 1.0)
        if etype in LOCAL_EDIT_TYPES and area_frac > area_cap:
            score -= 2.0
        return score

    def _best_prompt_candidate(base_tag: str, box_hint: Optional[np.ndarray] = None):
        best = None
        for variant_idx, prompt in enumerate(prompt_variants):
            instances = _query_sam(processor, state, shape, phrase=prompt, box_hint=box_hint)
            selected, _ = _select_instances(instances, diff_map, shape, location_hint, etype, allow_multiple=allow_multiple)
            if not _accept(selected):
                continue
            score = _score_mask(selected)
            tag = _format_tag(base_tag, variant_idx, prompt)
            if best is None or score > best[0]:
                best = (score, selected, tag)
        return best

    diff_boxes = _top_diff_boxes(diff_map, location_hint, limit=3 if allow_multiple else 1)
    if allow_multiple and etype == "add":
        best_text = _best_prompt_candidate("SAM_TEXT")
        best_merged = None
        if diff_boxes:
            merged_box = _merge_boxes(diff_boxes, shape, margin_px=CFG.multi_box_margin_px)
            if merged_box is not None:
                best_merged = _best_prompt_candidate("SAM_TEXT_MULTI_BOX", box_hint=merged_box)
        if best_text is not None and best_merged is not None:
            if best_merged[0] > best_text[0] + 0.02:
                return best_merged[1], best_merged[2]
            return best_text[1], best_text[2]
        if best_text is not None:
            return best_text[1], best_text[2]
        if best_merged is not None:
            return best_merged[1], best_merged[2]
    else:
        for variant_idx, prompt in enumerate(prompt_variants):
            instances = _query_sam(processor, state, shape, phrase=prompt)
            selected, _ = _select_instances(instances, diff_map, shape, location_hint, etype, allow_multiple=allow_multiple)
            if _accept(selected):
                return selected, "SAM_TEXT" if variant_idx == 0 else f"SAM_TEXT_ALIAS:{prompt}"

    if diff_boxes:
        for variant_idx, prompt in enumerate(prompt_variants):
            box_instances = []
            for diff_box in diff_boxes:
                box_instances.extend(_query_sam(processor, state, shape, phrase=prompt, box_hint=diff_box))
            selected, _ = _select_instances(box_instances, diff_map, shape, location_hint, etype, allow_multiple=allow_multiple)
            if _accept(selected):
                tag = "SAM_TEXT_BOX" if variant_idx == 0 else f"SAM_TEXT_BOX_ALIAS:{prompt}"
                return selected, tag

        visual_instances = []
        for diff_box in diff_boxes:
            visual_instances.extend(_query_sam(processor, state, shape, phrase=None, box_hint=diff_box))
        selected, _ = _select_instances(visual_instances, diff_map, shape, location_hint, etype, allow_multiple=allow_multiple)
        if _accept(selected):
            return selected, "SAM_BOX"

        if allow_multiple:
            merged_box = _merge_boxes(diff_boxes, shape, margin_px=CFG.multi_box_margin_px)
            if merged_box is not None:
                for variant_idx, prompt in enumerate(prompt_variants):
                    instances = _query_sam(processor, state, shape, phrase=prompt, box_hint=merged_box)
                    selected, _ = _select_instances(instances, diff_map, shape, location_hint, etype, allow_multiple=allow_multiple)
                    if _accept(selected):
                        tag = "SAM_TEXT_MULTI_BOX" if variant_idx == 0 else f"SAM_TEXT_MULTI_BOX_ALIAS:{prompt}"
                        return selected, tag

    return None, "SAM_EMPTY"


def fuse_mask(etype: str, processor, workspace: Dict, phrases: Dict) -> Tuple[np.ndarray, str]:
    shape = workspace["shape"]
    src = phrases["source"]
    tgt = phrases["target"]
    location_hint = phrases["location_hint"]
    allow_multiple = phrases["allow_multiple"]
    local_diff = workspace["local_diff"]
    motion_diff = workspace["motion_diff"]
    global_diff = workspace["global_diff"]

    if etype == "style":
        if phrases["is_global"] or global_diff.mean() >= CFG.global_style_cover:
            return np.ones(shape, np.uint8), "GLOBAL_STYLE"
        if src:
            mask, status = _query_and_select_local(
                processor,
                workspace["state_in"],
                shape,
                src,
                local_diff,
                location_hint,
                etype="color",
                allow_multiple=allow_multiple,
            )
            if mask is not None:
                return mask, status
        return _fallback_local(global_diff, location_hint, shape, expand_px=8), "LOCAL_STYLE_DIFF_ONLY"

    if etype == "background":
        mask, status = _estimate_background_mask(processor, workspace, phrases)
        return mask.astype(np.uint8), status

    if etype == "add":
        mask, status = _query_and_select_local(
            processor,
            workspace["state_out"],
            shape,
            tgt,
            local_diff,
            location_hint,
            etype,
            allow_multiple=allow_multiple,
        )
        if mask is not None:
            if allow_multiple:
                return mask.astype(np.uint8), status
            touched = _touching_components(local_diff, mask, CFG.diff_expand_px)
            merged = _union([mask, touched], shape=shape)
            if merged is not None and merged.mean() <= CFG.local_max_area_frac:
                return merged, status
        fallback_diff = local_diff
        if fallback_diff.sum() == 0:
            fallback_diff = _union([local_diff, global_diff], shape=shape)
        return _fallback_local(fallback_diff, location_hint, shape, expand_px=6), "DIFF_ONLY_ADD"

    if etype == "remove":
        mask, status = _query_and_select_local(
            processor,
            workspace["state_in"],
            shape,
            src,
            local_diff,
            location_hint,
            etype,
            allow_multiple=allow_multiple,
            max_area_frac=0.60,
        )
        if mask is not None:
            if allow_multiple:
                # 非人类人形多实例删除时，减去 output 里仍然存在的主体，可明显抑制把保留人物一起卷进去。
                if _is_nonhuman_humanoid_phrase(src):
                    survivor = _query_best_foreground_mask(
                        processor,
                        workspace["state_out"],
                        shape,
                        ["person", "human", "man", "subject"],
                    )
                    if survivor is not None:
                        mask = (mask & (1 - _dilate(survivor, 10))).astype(np.uint8)
                return _dilate(mask, CFG.dilate_inpaint_px), status
            touched = _touching_components(local_diff, mask, CFG.diff_expand_px)
            out = _union([mask, touched], shape=shape)
            return _dilate(out, CFG.dilate_inpaint_px), status
        return _dilate(_fallback_local(local_diff, location_hint, shape, expand_px=6), CFG.dilate_inpaint_px), "DIFF_ONLY_REMOVE"

    if etype == "replace":
        person_like = _is_person_like_phrase(src) and _is_person_like_phrase(tgt)
        replace_area_cap = 0.45 if person_like else 0.60
        src_mask, src_status = _query_and_select_local(
            processor,
            workspace["state_in"],
            shape,
            src,
            local_diff,
            location_hint,
            etype,
            allow_multiple=allow_multiple,
            max_area_frac=replace_area_cap,
        )
        tgt_mask, tgt_status = _query_and_select_local(
            processor,
            workspace["state_out"],
            shape,
            tgt,
            local_diff,
            location_hint,
            etype,
            allow_multiple=allow_multiple,
            max_area_frac=replace_area_cap,
        )
        merged = _union([src_mask, tgt_mask], shape=shape)
        if merged is not None and merged.sum() > 0:
            if person_like:
                person_box = _best_diff_box(local_diff, location_hint)
                if person_box is not None:
                    merged = (merged & _mask_from_box(person_box, shape)).astype(np.uint8)
                return _dilate(merged, CFG.dilate_inpaint_px), f"{src_status}+{tgt_status}"
            touched = _touching_components(local_diff, merged, CFG.diff_expand_px)
            merged = _union([merged, touched], shape=shape)
            if merged is not None and merged.mean() <= replace_area_cap:
                return _dilate(merged, CFG.dilate_inpaint_px), f"{src_status}+{tgt_status}"
        return _dilate(_fallback_local(local_diff, location_hint, shape, expand_px=7), CFG.dilate_inpaint_px), "DIFF_ONLY_REPLACE"

    if etype == "color":
        if not phrases["parse_ok"]:
            return _fallback_local(local_diff, location_hint, shape, expand_px=6), "PARSE_FAIL_DIFF_ONLY"
        mask, status = _query_and_select_local(
            processor,
            workspace["state_in"],
            shape,
            src,
            local_diff,
            location_hint,
            etype,
            allow_multiple=allow_multiple,
            max_area_frac=0.55,
        )
        if mask is not None and mask.mean() <= 0.55:
            return mask, status
        return _fallback_local(local_diff, location_hint, shape, expand_px=5), "DIFF_ONLY_COLOR"

    if etype == "motion":
        if phrases["is_noop"]:
            return np.zeros(shape, np.uint8), "NOOP"

        focus_phrase = src
        in_mask, in_status = _query_and_select_local(
            processor,
            workspace["state_in"],
            shape,
            focus_phrase,
            motion_diff,
            location_hint,
            etype,
            allow_multiple=False,
        )
        out_mask, out_status = _query_and_select_local(
            processor,
            workspace["state_out"],
            shape,
            focus_phrase,
            motion_diff,
            location_hint,
            etype,
            allow_multiple=False,
        )
        anchor = _union([in_mask, out_mask], shape=shape)
        if anchor is not None and anchor.sum() > 0:
            trail = _touching_components(motion_diff, anchor, CFG.motion_expand_px)
            return _union([anchor, trail], shape=shape), f"{in_status}+{out_status}"
        return _fallback_local(motion_diff, location_hint, shape, expand_px=6), "DIFF_ONLY_MOTION"

    raise ValueError(f"unknown type {etype}")


# ----------------------------------------------------------------------------
# 6. 后处理 + QC
# ----------------------------------------------------------------------------


def postprocess(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if fill_holes:
        ff = mask.copy()
        h, w = mask.shape
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
        mask = mask | (1 - ff)
    return mask.astype(np.uint8)


def quality_score(mask: np.ndarray, workspace: Dict, etype: str, phrases: Dict, status: str) -> Dict:
    ref = workspace["motion_diff"] if etype == "motion" else workspace["local_diff"]
    inter = int((mask & ref).sum())
    union = int((mask | ref).sum())
    mask_area = float(mask.mean())
    diff_iou = float(inter / union) if union else 0.0
    diff_precision = float(inter / max(int(mask.sum()), 1))
    diff_recall = float(inter / max(int(ref.sum()), 1)) if ref.sum() > 0 else 0.0

    flag = "OK"
    if phrases.get("is_noop") and mask.sum() > 0:
        flag = "NOOP_NONEMPTY"
    elif etype in LOCAL_EDIT_TYPES and mask_area > CFG.local_max_area_frac:
        flag = "TOO_LARGE_LOCAL"
    elif etype == "background" and status != "OK":
        flag = status
    elif etype in GLOBAL_EDIT_TYPES and phrases.get("is_global") and mask_area < CFG.global_mask_warn_frac:
        flag = "TOO_SMALL_GLOBAL"
    elif not phrases.get("is_noop") and mask_area < 1e-4:
        flag = "EMPTY_EDIT"
    elif etype == "color" and status.startswith("DIFF_ONLY"):
        flag = "LOW_CONF_COLOR"

    return {
        "diff_iou": round(diff_iou, 4),
        "diff_precision": round(diff_precision, 4),
        "diff_recall": round(diff_recall, 4),
        "area_frac": round(mask_area, 4),
        "flag": flag,
    }


# ----------------------------------------------------------------------------
# 7. 单样本入口
# ----------------------------------------------------------------------------


def annotate_one(processor, sample: Dict) -> Dict:
    if hasattr(processor, "set_confidence_threshold"):
        processor.set_confidence_threshold(CFG.sam_conf)

    etype = canonicalize_type(sample["type"])
    workspace = _prepare_workspace(processor, sample["input_img"], sample["output_img"])
    phrases = parse_instruction(sample["instruction"], etype)

    mask, status = fuse_mask(etype, processor, workspace, phrases)
    mask = _ensure_mask_shape(mask, workspace["shape"])

    fill_holes = etype not in {"background", "style", "add", "color"}
    mask = postprocess(mask, fill_holes=fill_holes)

    if phrases.get("is_noop"):
        mask = np.zeros(workspace["shape"], np.uint8)

    qc = quality_score(mask, workspace, etype, phrases, status)
    qc["status"] = status
    qc["etype"] = etype
    return {
        "mask": mask.astype(np.uint8),
        "phrases": phrases,
        "qc": qc,
        "debug": {
            "local_diff": workspace["local_diff"],
            "motion_diff": workspace["motion_diff"],
            "global_diff": workspace["global_diff"],
        },
    }
