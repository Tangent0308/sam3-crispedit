"""Editor wrapper: OCR-verified text edits with bounded retries and a glyph fallback.

Runs inside the editor worker (editor Python environment). RapidOCR is loaded
from an isolated directory (``SA1B_OCR_LIB``, default /tmp/sa1b-ocr-lib) so the
frozen editor environment is not modified. Only units with a text requirement
(see ``text_edits.requirement``) are checked; all others pass through untouched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import os
import sys
import time

import numpy as np
from PIL import Image

from . import text_edits as te

DEFAULT_OCR_LIB = '/tmp/sa1b-ocr-lib'


class OCR:
    """Thin RapidOCR wrapper returning dicts {text, conf, box} in crop pixels."""
    def __init__(self, threads=4):
        lib = os.environ.get('SA1B_OCR_LIB', DEFAULT_OCR_LIB)
        if lib not in sys.path:
            sys.path.append(lib)
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1)
        self.config = {'engine': 'rapidocr_onnxruntime', 'threads': threads}

    def __call__(self, rgb):
        result, _ = self.engine(np.ascontiguousarray(rgb))
        return [{'text': str(text), 'conf': float(conf), 'box': [[float(x), float(y)] for x, y in box]}
                for box, text, conf in (result or [])]


def _crop_box(mask, size, pad=.35, min_pad=24):
    ys, xs = np.nonzero(mask)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    p = int(max(min_pad, pad * max(w, h)))
    return (max(0, int(xs.min()) - p), max(0, int(ys.min()) - p),
            min(size[0], int(xs.max()) + p + 1), min(size[1], int(ys.max()) + p + 1))


def _ocr_view(image, box, max_side=1280, min_side=480):
    crop = image.crop(box)
    s = 1.0
    if max(crop.size) > max_side:
        s = max_side / max(crop.size)
    elif max(crop.size) < min_side:
        s = min(3.0, min_side / max(crop.size))
    if s != 1.0:
        crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
    return np.asarray(crop.convert('RGB')), s


def _scaled_mask(mask, box, shape):
    local = mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255
    return np.asarray(Image.fromarray(local).resize((shape[1], shape[0]), Image.NEAREST)) > 0


def check(source, result, region_mask, req, ocr):
    """OCR both images in the same crop around the writable region.

    Skips the second OCR call when one side already settles the decision:
    no new glyphs for ``none``, no legible source text for ``preserve``.
    """
    import cv2
    box = _crop_box(region_mask, source.size)
    src_view, s = _ocr_view(source, box)
    res_view, _ = _ocr_view(result, box)
    changed = np.max(np.abs(src_view.astype(np.int16) - res_view.astype(np.int16)), axis=2) > 30
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    target = _scaled_mask(region_mask, box, changed.shape)
    t = time.monotonic()
    src_tokens = res_tokens = None
    if req['mode'] in ('none', 'numerals'):
        res_tokens = ocr(res_view)
        if not any(x['conf'] >= .6 and len(te.normalize(x['text'])) >= 2 and
                   (te._overlap(x['box'], changed) >= .3 or te._overlap(x['box'], target) >= .5) for x in res_tokens):
            src_tokens = []
    elif req['mode'] == 'erase':
        src_tokens = ocr(src_view)
        if not any(x['conf'] >= .7 and len(te.normalize(x['text'])) >= 2 and te._overlap(x['box'], target) >= .3
                   for x in src_tokens):
            res_tokens = []  # nothing written there before; a remove rarely invents lettering
    if src_tokens is None:
        src_tokens = ocr(src_view)
    if res_tokens is None:
        res_tokens = ocr(res_view)
    verdict = te.compare(src_tokens, res_tokens, changed, target, req)
    verdict.update(ocr_seconds=round(time.monotonic() - t, 3), crop_box=list(box), view_scale=round(s, 4),
                   source_tokens=[(x['text'], round(x['conf'], 3)) for x in src_tokens],
                   result_tokens=[(x['text'], round(x['conf'], 3)) for x in res_tokens])
    return verdict, (src_view, res_view, src_tokens, res_tokens, changed, target, box, s)


class TextVerifiedEditor:
    """Wrap any editor exposing ``generate(image_path, unit, seed)``.

    Text units: generate, OCR-verify, retry with a new seed up to
    ``max_attempts``; then, for exact wording, try one glyph-render repair of
    the best attempt. Failing units are returned with an explicit failed
    verification so the deterministic gate drops them. Other units: one pass.
    """
    def __init__(self, editor, *, max_attempts=3, glyph_repair=False, ocr=None):
        self.editor = editor
        self.max_attempts, self.glyph_repair = int(max_attempts), bool(glyph_repair)
        self.ocr = ocr or OCR()
        self.config = {**getattr(editor, 'config', {}), 'text_verification': te.VERSION,
                       'text_max_attempts': self.max_attempts, 'text_glyph_repair': self.glyph_repair,
                       'ocr': self.ocr.config}
        self.last_generation_context = {}

    @property
    def pipe(self):
        return self.editor.pipe

    def __getattr__(self, name):
        # Delegate support-mask helpers and similar attributes to the wrapped editor.
        editor = self.__dict__.get('editor')
        if editor is None:
            raise AttributeError(name)
        return getattr(editor, name)

    def generate(self, image_path, unit, seed=0):
        spec = unit.get('instruction_spec') or {}
        try:
            req = te.requirement(spec)
        except ValueError as exc:
            req = {'mode': 'invalid', 'error': str(exc)}
        if not req:
            out = self.editor.generate(image_path, unit, seed=seed)
            self.last_generation_context = dict(getattr(self.editor, 'last_generation_context', {}))
            return out
        if req['mode'] == 'invalid':
            # Wording that cannot be rendered legibly is never generated.
            out = self.editor.generate(image_path, unit, seed=seed)
            self.last_generation_context = {**getattr(self.editor, 'last_generation_context', {}),
                                            'text_verification': {'version': te.VERSION, 'requirement': req,
                                                                  'decision': 'fail', 'reasons': ['invalid_text_requirement']}}
            return out
        unit = {**unit, 'generator_text_rule': te.generator_rule(req, unit['operation'])}
        source = Image.open(image_path).convert('RGB')
        from .specialist_editors import writable_mask, read_binary_mask
        region = writable_mask(unit, source.size, require_add_placement=False)
        if unit['operation'] == 'attribute':
            region = read_binary_mask(unit['mask_path'], source.size)
        attempts, best = [], None
        for k in range(self.max_attempts):
            s = seed + 7919 * k
            t = time.monotonic()
            attempt_unit = unit
            if k and unit.get('diagnostic_dir'):
                # Each attempt keeps its own checksummed crop evidence; a retry must
                # never overwrite the files that a kept attempt's context points to.
                attempt_unit = {**unit, 'diagnostic_dir': str(Path(unit['diagnostic_dir']) / f'text_retry{k}')}
            result = self.editor.generate(image_path, attempt_unit, seed=s)
            context = dict(getattr(self.editor, 'last_generation_context', {}))
            gen_seconds = time.monotonic() - t
            verdict, views = check(source, result, region, req, self.ocr)
            attempts.append({'seed': s, 'generation_seconds': round(gen_seconds, 3),
                             **{key: value for key, value in verdict.items() if key != 'source_tokens'}})
            if best is None or len(verdict['reasons']) < len(best[1]['reasons']):
                best = (result, verdict, context, views)
            if verdict['decision'] == 'pass':
                break
        result, verdict, context, views = best
        repair, final_source = None, 'generator'
        if verdict['decision'] != 'pass' and self.glyph_repair and req['mode'] == 'exact':
            repair = self._glyph_repair(source, result, region, req, views)
            image = repair.pop('image', None)
            if image is not None and repair['verdict']['decision'] == 'pass':
                result, verdict, final_source = image, repair['verdict'], 'glyph_repair'
        context['text_verification'] = {
            'version': te.VERSION, 'requirement': req, 'decision': verdict['decision'], 'reasons': verdict['reasons'],
            'attempts': attempts, 'glyph_repair': repair, 'final_source': final_source,
            'final_pixels_sha256': hashlib.sha256(result.tobytes()).hexdigest()}
        self.last_generation_context = context
        return result

    def _glyph_repair(self, source, result, region, req, views):
        src_view, res_view, src_tokens, res_tokens, changed, target, box, s = views
        # Prefer the wrong wording the generator wrote; else the old wording to replace.
        tokens = [t for t in res_tokens if te._overlap(t['box'], changed) >= .3 and t['conf'] >= .5]
        base = res_view
        if not tokens and req.get('replaces_source'):
            tokens = [t for t in src_tokens if te._overlap(t['box'], target) >= .5 and t['conf'] >= .5]
        if not tokens:
            return {'applied': False, 'reason': 'no text box to rewrite', 'verdict': {'decision': 'fail', 'reasons': ['no_text_box']}}
        try:
            quad = te.text_quad(tokens)
            drawn, evidence = te.render_glyphs(base, quad, req['text'])
        except (ValueError, OSError) as exc:
            return {'applied': False, 'reason': str(exc), 'verdict': {'decision': 'fail', 'reasons': ['glyph_error']}}
        # Map the edited OCR view back into the full-resolution result.
        patch = Image.fromarray(drawn).resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS)
        changed_patch = np.max(np.abs(drawn.astype(np.int16) - base.astype(np.int16)), axis=2) > 0
        alpha = Image.fromarray((changed_patch * 255).astype(np.uint8)).resize(patch.size, Image.BILINEAR)
        repaired = result.copy()
        repaired.paste(patch, box[:2], alpha)
        verdict, _ = check(source, repaired, region, req, self.ocr)
        return {'applied': True, 'evidence': evidence, 'verdict': verdict, 'image': repaired}
