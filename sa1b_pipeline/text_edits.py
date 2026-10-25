"""Exact-text edits: planning slots, generator wording and OCR verification (opt-in).

Text is the edit content that a diffusion editor most often gets wrong. Three
rules keep it checkable:

1. Wording is explicit. An added/replacing item that carries writing names its
   exact wording ``reading "OPEN"``; a text attribute edit already quotes the new
   value. A text-bearing item without explicit wording is not planned.
2. Generation is told the wording, or told that no lettering may appear.
3. An external OCR engine (RapidOCR, CPU) reads the edited region of the
   source and the result. Requested wording must appear exactly (case- and
   spacing-insensitive), replaced wording must be gone, and no unrequested
   letters may appear in newly edited pixels.

OCR is evidence for a bounded retry and a deterministic veto, never semantic
approval; the normal image audit still runs afterwards.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

VERSION = 'text-edits-5-dial-numerals'
# Items whose function implies writing: without wording they invite gibberish.
TEXT_REQUIRED = ('signboard', 'shop sign', 'street sign', 'name sign', 'sign', 'nameplate', 'name plate',
                 'name tag', 'label', 'price tag', 'price sign', 'tag', 'banner', 'placard', 'poster', 'notice',
                 'menu', 'license plate', 'number plate', 'billboard', 'scoreboard', 'ticket', 'chalkboard sign',
                 'plaque with an inscription', 'inscription', 'lettering', 'caption', 'headline', 'slogan', 'text')
# Items that are fine either way; without wording they must stay plain.
TEXT_OPTIONAL = ('sticker', 'decal', 'badge', 'emblem', 'patch', 'logo', 'plaque', 'card', 'note', 'flag',
                 'pennant', 'sash', 'ribbon', 'stamp', 'seal', 'coin', 'medal', 'button', 'pin')
MAX_WORDS, MAX_CHARS = 4, 24
ALLOWED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 &'.,!?:#$%/+-]*$")
QUOTED = re.compile(r'"([^"]{1,60})"')
# New items whose ordinary face carries numerals (a clock dial, a gauge): digits
# belong to the object, words and logos still do not. A wall clock was rejected
# three times for its 10/11/12 under the plain no-glyph rule (2026-10-23 run).
DIAL_ITEMS = {'clock', 'watch', 'wristwatch', 'stopwatch', 'timer', 'thermometer', 'gauge', 'dial', 'speedometer',
              'barometer', 'compass', 'calculator', 'keypad', 'calendar', 'scoreboard', 'dice', 'die', 'ruler'}


def _has_term(payload, terms):
    text = ' ' + re.sub(r'[^a-z ]+', ' ', str(payload).lower()) + ' '
    return any(re.search(r'\b' + re.escape(t) + r's?\b', text) for t in terms)


def requires_text(payload):
    return _has_term(payload, TEXT_REQUIRED)


def text_bearing(payload):
    return _has_term(payload, TEXT_REQUIRED + TEXT_OPTIONAL)


def clean_wording(text):
    """Validated short wording, or ValueError (never silently rewritten)."""
    text = re.sub(r'\s+', ' ', str(text or '')).strip().strip('"“”\'')
    if not text:
        return ''
    if '"' in text or not ALLOWED.match(text):
        raise ValueError('text must use plain English letters, digits and basic punctuation')
    if len(text.split()) > MAX_WORDS or len(text) > MAX_CHARS:
        raise ValueError(f'text is limited to {MAX_WORDS} words / {MAX_CHARS} characters for legible rendering')
    if len(normalize(text)) < 2:
        raise ValueError('text needs at least two letters or digits')
    return text


def payload_with_text(payload, text):
    """Bind the wording into the payload, so Ref and no-ref both keep it."""
    payload = re.sub(r'\s+', ' ', str(payload or '')).strip()
    if QUOTED.search(payload):
        raise ValueError('wording belongs in the text slot, not inside the payload')
    text = clean_wording(text)
    if text:
        if not text_bearing(payload):
            raise ValueError('wording requires an item that can carry writing')
        return f'{payload} reading "{text}"'
    if requires_text(payload):
        raise ValueError('a text-bearing item needs explicit wording')
    return payload


def requirement(spec):
    """What the generated pixels must show, derived only from the protocol spec.

    exact    - the quoted wording must be written (add/replace payload, text attribute)
    none     - new content must carry no glyphs (add/replace without wording)
    numerals - like none, but digits of a dial-type item (DIAL_ITEMS) are allowed
    erase    - removed content leaves no legible text and no new glyphs (remove)
    preserve - existing legible text on the edited surface survives (other attributes)
    """
    op = spec.get('operation')
    if op == 'attribute':
        for a in spec.get('attributes') or []:
            if a.get('property') == 'text':
                return {'mode': 'exact', 'text': clean_wording(a.get('value')), 'replaces_source': True}
        return {'mode': 'preserve'}
    if op in ('add', 'replace'):
        quoted = QUOTED.findall(spec.get('payload') or '')
        if quoted:
            return {'mode': 'exact', 'text': clean_wording(quoted[-1]), 'replaces_source': op == 'replace'}
        if dial_item(spec.get('payload') or ''):
            return {'mode': 'numerals'}
        return {'mode': 'none'}
    if op == 'remove':
        return {'mode': 'erase'}
    return None


def dial_item(payload: str) -> bool:
    """Is the payload's head noun an item whose face normally shows numerals?"""
    words = re.findall(r'[a-z]+', str(payload).lower().split(' with ')[0])
    if not words:
        return False
    head = words[-1]
    return bool({head, head[:-1] if head.endswith('s') else head, head[:-2] if head.endswith('es') else head}
                & DIAL_ITEMS)


def generator_rule(req, operation):
    """Short generator wording; empty for edits without a text requirement."""
    if not req:
        return ''
    if req['mode'] == 'exact':
        rule = (f' Write exactly the text "{req["text"]}" on it: the same letters, digits and spelling, '
                'clearly legible, in one consistent typeface that follows the surface perspective and lighting. '
                'Do not add any other letters, words, numbers or logos.')
        if req.get('replaces_source'):
            rule += ' The previous wording must be completely gone.'
        return rule
    if req['mode'] == 'none':
        return (' The new content must not show any letters, words, numbers or logos; keep it plain. '
                'Existing writing outside the new content stays unchanged.')
    if req['mode'] == 'numerals':
        return (' The new content must not show any words, letters or logos; numerals and tick marks that '
                'belong to its face are fine. Existing writing outside the new content stays unchanged.')
    if req['mode'] == 'preserve' and operation == 'attribute':
        return ' Keep every existing letter and number exactly as it is: same characters, positions and legibility.'
    if req['mode'] == 'erase':
        return ' Do not write any new letters, words or numbers where the content was removed.'
    return ''


# ----------------------------------------------------------------------------
# OCR comparison (pure functions; the engine itself lives in text_verification)

CONFUSABLE = str.maketrans({'0': 'O', '1': 'I', '5': 'S', '8': 'B', '|': 'I'})


def normalize(text):
    return re.sub(r'[^A-Z0-9一-鿿]', '', str(text).upper())


def loose(text):
    """OCR-confusion-tolerant key; used only to compare same-length strings."""
    return normalize(text).translate(CONFUSABLE)


def similar(a, b):
    a, b = normalize(a), normalize(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _poly_mask(box, shape):
    import cv2
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(box, dtype=np.int32).reshape(-1, 2)], 1)
    return mask.astype(bool)


def _overlap(box, region):
    poly = _poly_mask(box, region.shape)
    n = int(poly.sum())
    return float((poly & region).sum() / n) if n else 0.0


def _reading_order(tokens):
    if not tokens:
        return []
    heights = [max(p[1] for p in t['box']) - min(p[1] for p in t['box']) for t in tokens]
    line = max(4.0, float(np.median(heights)) * .6)
    keyed = sorted(tokens, key=lambda t: (round(np.mean([p[1] for p in t['box']]) / line),
                                          np.mean([p[0] for p in t['box']])))
    return keyed


def _height(token):
    """Glyph height of an oriented OCR quad (not its axis-aligned extent)."""
    q = _order_quad(token['box'])
    return float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def _partial(token_text, wanted):
    """Best similarity of a token against any same-length window of the wanted text."""
    t, w = normalize(token_text), normalize(wanted)
    if not t or not w:
        return 0.0
    if t in w:
        return 1.0
    if len(t) >= len(w):
        return SequenceMatcher(None, t, w).ratio()
    return max(SequenceMatcher(None, t, w[i:i + len(t)]).ratio() for i in range(len(w) - len(t) + 1))


def _same_place(a, b, shape):
    return _overlap(a['box'], _poly_mask(b['box'], shape)) > .3 or _overlap(b['box'], _poly_mask(a['box'], shape)) > .3


def compare(source_tokens, result_tokens, changed, target, req, *, min_conf=.5):
    """Decide whether the result shows exactly the required text (OCR tokens, crop coordinates).

    exact:   the requested wording is read in the edited/target area; no NEW glyphs
             beyond it (gibberish); when replacing, the old headline (largest source
             text in the target) is gone. Smaller secondary source lines may stay.
    none:    no NEW glyphs appear in changed pixels.
    preserve: legible source text in the target survives.
    ``changed``/``target`` are boolean maps; tokens are {text, conf, box}.
    """
    shape = changed.shape
    out = {'mode': req['mode'], 'requested': req.get('text', ''), 'decision': 'pass', 'reasons': []}
    src = [t for t in source_tokens if t['conf'] >= min_conf and normalize(t['text'])]
    res = [t for t in result_tokens if t['conf'] >= min_conf and normalize(t['text'])]

    def pre_existing(t):
        return any(similar(t['text'], s['text']) >= .75 and _same_place(t, s, shape) for s in src)

    in_edit = [t for t in res if _overlap(t['box'], changed) >= .3 or _overlap(t['box'], target) >= .5]
    fresh = [t for t in res if _overlap(t['box'], changed) >= .3 and not pre_existing(t)]
    src_area = np.zeros(shape, bool)
    for s in src:
        src_area |= _poly_mask(s['box'], shape)
    if req['mode'] == 'exact':
        wanted = normalize(req['text'])
        got = ''.join(normalize(t['text']) for t in _reading_order(in_edit))
        hit = wanted in got or (len(got) >= len(wanted) and any(
            loose(got[i:i + len(wanted)]) == loose(wanted) for i in range(len(got) - len(wanted) + 1)))
        out['read'] = [t['text'] for t in _reading_order(in_edit)]
        if not hit:
            out['reasons'].append('requested_text_not_found')
        matched = [t for t in in_edit if _partial(t['text'], req['text']) >= .75]
        word_h = max((_height(t) for t in matched), default=0.0)
        # Gibberish: new glyphs that are not the requested wording, either where the
        # source had no text, or as prominent as the requested wording. Small
        # re-rendered secondary lines on dense signs are tolerated (OCR noise).
        gibberish = [t['text'] for t in fresh if len(normalize(t['text'])) >= 2 and t['conf'] >= .6
                     and _partial(t['text'], req['text']) < .75
                     and (_overlap(t['box'], src_area) < .3 or (word_h and _height(t) >= .6 * word_h))]
        if gibberish:
            out['reasons'].append('unrequested_text')
            out['unrequested_text'] = gibberish
        if req.get('replaces_source'):
            old = [s for s in src if _overlap(s['box'], target) >= .5 and len(normalize(s['text'])) >= 2]
            if old:
                top = max(_height(s) for s in old)
                headline = [s for s in old if _height(s) >= .85 * top and s['conf'] >= .7]
                # Old wording replaced in place by the new wording must be gone too.
                replaced = [s for s in old if s['conf'] >= .7 and any(_same_place(s, m, shape) for m in matched)]
                must_go = {id(s): s for s in headline + replaced}.values()
                if headline and similar(''.join(normalize(h['text']) for h in headline), wanted) >= .8:
                    out['reasons'].append('no_text_change')
                kept = [h['text'] for h in must_go if normalize(h['text']) not in wanted and
                        any(similar(h['text'], t['text']) >= .8 and _same_place(h, t, shape) for t in res)]
                if kept:
                    out['reasons'].append('old_text_remains')
                    out['old_text_remaining'] = kept
    elif req['mode'] in ('none', 'erase', 'numerals'):
        garbled = [t['text'] for t in fresh if t['conf'] >= .6 and len(normalize(t['text'])) >= 2
                   and (req['mode'] != 'numerals' or len(re.sub(r'[^A-Z]', '', normalize(t['text']))) >= 2)]
        if garbled:
            out['reasons'].append('unrequested_text')
            out['unrequested_text'] = garbled
        if req['mode'] == 'erase':
            old = [s for s in src if _overlap(s['box'], target) >= .5 and s['conf'] >= .7 and len(normalize(s['text'])) >= 2]
            kept = [s['text'] for s in old if any(similar(s['text'], t['text']) >= .8 and _same_place(s, t, shape) for t in res)]
            if kept:
                out['reasons'].append('removed_text_remains')
                out['old_text_remaining'] = kept
    elif req['mode'] == 'preserve':
        keep = [s for s in src if _overlap(s['box'], target) >= .5 and s['conf'] >= .8 and len(normalize(s['text'])) >= 3]
        if keep:
            ok = [s['text'] for s in keep if any(similar(s['text'], t['text']) >= .75 for t in res)]
            out['source_text'] = [s['text'] for s in keep]
            if len(ok) < .8 * len(keep):
                out['reasons'].append('source_text_damaged')
        # A color/material edit must not invent lettering on the recolored surface.
        garbled = [t['text'] for t in fresh if t['conf'] >= .6 and len(normalize(t['text'])) >= 2]
        if garbled:
            out['reasons'].append('unrequested_text')
            out['unrequested_text'] = garbled
    if out['reasons']:
        out['decision'] = 'fail'
    return out


# ----------------------------------------------------------------------------
# Glyph rendering: the external "photo-editing" fallback for exact wording

def _order_quad(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def text_quad(tokens):
    """One oriented quad around all given OCR boxes (a single text block)."""
    import cv2
    pts = np.concatenate([np.asarray(t['box'], dtype=np.float32).reshape(-1, 2) for t in tokens])
    rect = cv2.minAreaRect(pts)
    return _order_quad(cv2.boxPoints(rect))


def _split_colors(pixels):
    """Text vs background colour inside a text box (text is the minority class)."""
    import cv2
    lum = (0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]).astype(np.uint8)
    thresh, _ = cv2.threshold(lum.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = lum <= thresh
    text_is_dark = dark.mean() < .5
    text = pixels[dark] if text_is_dark else pixels[~dark]
    background = pixels[~dark] if text_is_dark else pixels[dark]
    return np.median(text, 0), np.median(background, 0), float(thresh), text_is_dark


def render_glyphs(image, quad, text, *, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
    """Erase the glyphs inside ``quad`` and draw ``text`` along its perspective.

    ``image`` is an RGB uint8 array (a crop). Returns (new_image, evidence).
    The replacement keeps the old text's colour and fills the box; it is a
    proposal that OCR and the image audit must still accept.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    h_img, w_img = image.shape[:2]
    quad = _order_quad(quad)
    width = float(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    height = float(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
    if width < 8 or height < 6:
        raise ValueError('text box too small for legible glyphs')
    region = _poly_mask(quad, (h_img, w_img))
    pixels = image[region].reshape(-1, 3).astype(np.float32)
    text_color, bg_color, thresh, text_is_dark = _split_colors(pixels)
    lum = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])
    glyphs = region & ((lum <= thresh) if text_is_dark else (lum > thresh))
    erase = cv2.dilate(glyphs.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & cv2.dilate(
        region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    base = cv2.inpaint(image, erase.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    # Supersampled rectified canvas; choose one or two lines, whichever is larger.
    scale = 4
    W, H = max(8, int(round(width * scale))), max(8, int(round(height * scale)))
    words = text.split()
    layouts = [[text]] + ([[' '.join(words[:len(words) // 2 + len(words) % 2]), ' '.join(words[len(words) // 2 + len(words) % 2:])]]
                          if len(words) >= 2 else [])
    best = None
    for lines in layouts:
        lo, hi = 4, H
        while lo < hi:
            size = (lo + hi + 1) // 2
            font = ImageFont.truetype(font_path, size)
            boxes = [font.getbbox(line) for line in lines]
            tw = max(b[2] - b[0] for b in boxes)
            th = sum(b[3] - b[1] for b in boxes) + int(.25 * size) * (len(lines) - 1)
            if tw <= .9 * W and th <= .78 * H:
                lo = size
            else:
                hi = size - 1
        if best is None or lo > best[0]:
            best = (lo, lines)
    size, lines = best
    font = ImageFont.truetype(font_path, size)
    canvas = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(canvas)
    boxes = [font.getbbox(line) for line in lines]
    gap = int(.25 * size)
    total = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
    y = (H - total) / 2
    for line, b in zip(lines, boxes):
        draw.text(((W - (b[2] - b[0])) / 2 - b[0], y - b[1]), line, fill=255, font=font)
        y += (b[3] - b[1]) + gap
    small = np.asarray(canvas.resize((max(2, int(round(width))), max(2, int(round(height)))), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    src = np.array([[0, 0], [small.shape[1], 0], [small.shape[1], small.shape[0]], [0, small.shape[0]]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, quad)
    alpha = cv2.warpPerspective(small, M, (w_img, h_img), flags=cv2.INTER_LINEAR)
    sigma = .45 if height < 30 else .7
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)[..., None]
    out = base.astype(np.float32) * (1 - alpha) + np.asarray(text_color, np.float32)[None, None] * alpha
    evidence = {'tool': 'glyph-render-1', 'font': font_path.rsplit('/', 1)[-1], 'font_px': round(size / scale, 1),
                'lines': lines, 'text_rgb': [int(v) for v in text_color], 'background_rgb': [int(v) for v in bg_color],
                'quad': quad.round(1).tolist()}
    return np.clip(out, 0, 255).astype(np.uint8), evidence
