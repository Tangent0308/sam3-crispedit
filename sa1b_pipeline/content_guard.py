"""Conservative checks for model-owned slots, before protocol rendering.

Only whitespace, redundant site connectors and literally repeated hosts are
repaired. Missing semantic content is rejected, never guessed. These checks do
not verify visual grounding.
"""
import re


def phrase(value, field):
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a string')
    return ' '.join(value.split())


def complete_phrase(value, field):
    value = phrase(value, field)
    # A quoted visible inscription may be non-English; the surrounding noun
    # phrase must remain English. Do not apply this to literal text-edit values.
    prose = re.sub(r'"[^"]*"|(?<!\w)\x27[^\x27]*\x27(?!\w)', '', value)
    if re.search(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]', prose):
        raise ValueError(f'{field} contains non-English prose outside a quoted inscription')
    if value.count('(') != value.count(')') or value.count('[') != value.count(']'):
        raise ValueError(f'{field} has an unfinished bracketed phrase')
    # Do not strip quoted words: a sign reading "on" is a complete reference.
    # Direction words can be complete adverbs ("the window below"), and
    # terminal prepositions can belong to a relative clause. Do not ban them.
    unfinished = re.search(
        r'\b(?:a|an|the|of|with|without|and|or|whose|such|un)\s*[,;:]?$', value, re.I)
    # "beneath the letter A" is complete: A names a visible character, not
    # a dangling article. Keep the exception lexical and require its noun.
    named_letter = re.search(r'\b(?:letter|character)\s+[a-z]$', value, re.I)
    if not value or re.search(r'(?:\.{2,}|…)', value) or (unfinished and not named_letter):
        raise ValueError(f'{field} is empty, a placeholder, or ends in an unfinished phrase')
    return value


def reference(value):
    value = complete_phrase(phrase(value, 'ref').rstrip(',;').rstrip(), 'ref')
    if not value.startswith('the ') or value.count('{') or value.count('}'):
        raise ValueError('ref must be a complete noun phrase starting with the')
    return value


def attribute_value(property_name, value):
    """Keep literal text; reject an instruction accidentally put in a value.

    Do not extract a color from an imperative: doing so could silently broaden
    a subpart edit to the whole mask. This is syntax validation, not grounding.
    """
    if property_name == 'text':
        return value
    value = complete_phrase(value, 'value')
    if re.search(r'\bchanged\s+from\b.+\bto\b', value, re.I) or re.match(
        r'^(?:please\s+)?(?:change|recolor|recolour|paint|make|turn|replace|remove|add)'
        r'\s+(?:the|this|that|these|those|its|it|them|a|an)\b',
        value, re.I,
    ):
        raise ValueError('attribute_value_is_instruction: supply the desired value, not an edit command')
    return value


def unchanged_declared_text(ref, edit):
    """Detect an exact no-op declared by the model, not text read from pixels.

    Only explicit reading/saying clauses qualify. Case, spelling and punctuation
    remain significant; a color/style change or a substring is never rejected.
    This cannot establish that the model read the source correctly.
    """
    if (not isinstance(ref, str) or not isinstance(edit, dict)
            or edit.get('property') != 'text' or not isinstance(edit.get('value'), str)):
        return None
    value = edit['value'].strip()
    if not value:
        return None
    clauses = re.finditer(r'\b(?:reading|reads|saying|says)\s+', ref, re.I)
    for clause in clauses:
        prefix = ref[:clause.start()]
        if re.search(r'\bnot\s*$', prefix, re.I):
            continue
        # Text on a locating landmark is not necessarily text on the target.
        # Be conservative even if this misses a valid target with a long ref.
        if re.search(r'\b(?:of|on|in|at|above|below|beside|behind|under|near|'
                     r'with|next|front|held|carried)\b', prefix, re.I):
            continue
        tail = ref[clause.end():]
        # Quoted source wording has an unambiguous boundary.
        pairs = {'"': '"', "'": "'", '“': '”', '‘': '’'}
        if tail[:1] in pairs:
            closing = tail.find(pairs[tail[0]], 1)
            same = (closing >= 0 and tail[1:closing] == value
                    and bool(re.match(
                        r'^(?:\s*$|\s+(?:mounted|hanging|positioned|located|'
                        r'attached|standing|above|below|beside|behind|near|on|in)\b)',
                        tail[closing+1:], re.I)))
        else:
            # For unquoted text, require an end or a clear placement verb.
            # "on/with/above" can be part of the inscription, so do not split
            # there. Literal comparison deliberately preserves capitalization.
            same = bool(re.match(
                re.escape(value) +
                r'(?=$|\s+(?i:mounted|hanging|positioned|located|attached|standing)\b)',
                tail))
        if same:
            return 'unchanged_declared_text: desired text exactly repeats the source wording stated in ref.'
    return None


def site(value, *, ref=None):
    value = phrase(value, 'site')
    if not value:
        return value
    # The renderer owns the connector between site and ref. No host is added.
    value = re.sub(r'\s+of$', '', value, flags=re.I)
    value = re.sub(r'^its\s+', '', value, flags=re.I)
    if ref:
        # Only remove a literally repeated anchor, never infer a body part or
        # collapse a generic site such as "a branch" to the whole tree.
        head = re.split(r'\s+(?:of|on|in|at|behind|beside|near|with|under|above|below|worn|standing|located)\b',
                        ref, maxsplit=1, flags=re.I)[0]
        for anchor in sorted({ref, head}, key=len, reverse=True):
            bare = re.sub(r'^the\s+', '', anchor, flags=re.I)
            if re.sub(r'^the\s+', '', value, flags=re.I).casefold() == bare.casefold():
                return ''
            suffix = re.search(r'\s+of\s+'+re.escape(anchor)+r'$', value, re.I)
            if suffix:
                value = value[:suffix.start()].strip()
            value = re.sub(r'^(?:the\s+)?'+re.escape(bare)+r"['’]s\s+", '', value, flags=re.I)
    return complete_phrase(value, 'site')


def appearance_only_replacement(ref, replacement):
    """Catch explicit same-design recolor/material substitutions, not synonyms.

    A deliberately narrow lexical check: unknown modifiers and different
    shapes/subtypes remain model/auditor decisions. It never changes the
    assigned operation or guesses a replacement.
    """
    if not isinstance(ref, str) or not isinstance(replacement, str):
        return None
    appearance = set(('white black red green blue yellow orange purple pink gray grey brown '
                      'beige cream crimson navy teal turquoise cyan magenta violet maroon '
                      'burgundy ivory tan reddish greenish bluish yellowish pinkish brownish grayish greyish '
                      'gold golden silver bronze copper wooden '
                      'wood metal metallic steel leather plastic glass marble stone ceramic '
                      'fabric cotton satin velvet linen silk wool woolen '
                      'matte glossy frosted transparent translucent opaque plain').split())
    locating = {'the', 'a', 'an', 'left', 'right', 'leftmost', 'rightmost',
                'nearest', 'farthest', 'foreground', 'background'}
    source_head = re.split(r'\s+(?:of|on|in|at|behind|beside|near|with|under|above|below|worn|standing|located|that)\b',
                           ref.casefold(), maxsplit=1)[0]
    # Avoid treating an object's new contents/accessories or geometry as color.
    new_words = re.findall(r"[a-z]+(?:-[a-z]+)*", replacement.casefold())
    old_words = re.findall(r"[a-z]+(?:-[a-z]+)*", source_head)
    old_core = [w for w in old_words if w not in appearance | locating]
    new_core = [w for w in new_words if w not in appearance | {'a', 'an', 'the'}]
    if old_core and old_core == new_core and set(new_words) & appearance:
        return 'replacement_only_changes_appearance: the named design is unchanged; color/material/finish edits belong to attribute.'
    return None


_SHORT_LOWER = re.compile(r'\b(?:shorts|trunks|briefs|miniskirts?|mini[- ]skirts?|skorts?|hot pants|bikini bottoms?|skirts?)\b')
_LONG_WORDS = re.compile(r'\b(?:long|maxi|midi|ankle[- ]length|floor[- ]length|full[- ]length|knee[- ]length)\b')
_LONG_LOWER = re.compile(r'\b(?:trousers|pants|jeans|slacks|leggings|tights|overalls|dungarees|jumpsuits?|sweatpants|'
                         r'joggers|chinos|breeches)\b|\b(?:long|maxi|midi|floor[- ]length|ankle[- ]length)\s+'
                         r'(?:[\w-]+\s+){0,3}(?:skirt|dress)\b|\b(?:gown|robe)s?\b')


def garment_extent(ref, replacement):
    """A short lower garment replaced by a long one (opt-in planning check).

    The editor may only write the selected garment and a narrow collar, so trouser legs or a
    long hem below shorts or a short skirt cannot be drawn. Among the assistant-labelled edits
    of full runs 1-4 such replacements were bad 4 of 6 (knee-length shorts, legs cut at the old
    hem) and minor 1; in the paired A/B outputs 3 of 4 targets still came out good, so this is a
    deliberate precision/recall trade (opt-in). Length is read from the words only; a skirt
    named long/maxi/midi is not short.
    """
    if not isinstance(ref, str) or not isinstance(replacement, str):
        return None
    head = re.split(r'\s+(?:of|on|in|at|behind|beside|near|with|under|above|below|worn|standing|located|that)\b',
                    ref.casefold(), maxsplit=1)[0]
    new = replacement.casefold()
    if not _SHORT_LOWER.search(head) or _LONG_WORDS.search(head):
        return None
    if re.search(r'\b(?:shorts|short\s+(?:pants|trousers)|cropped|capri)\b', new) or not _LONG_LOWER.search(new):
        return None
    return ('garment_extent: a longer garment would need pixels below the selected short garment, outside the '
            'editable region; keep a garment of similar length.')


def rejection(value):
    value = phrase(value, 'reject')
    if value.lower().strip(' .!') in {'', 'true', 'false', 'null', 'none', 'n/a', 'reject', 'unknown', 'pass', 'fail', 'error', '...'}:
        raise ValueError('reject must state a concrete identification or edit incompatibility reason')
    return value


def incompatible_edit(ref, operation):
    """Reject a narrow class of impossible *declared requests*, not source masks.

    This is not an entity recognizer. Explicit depictions/prostheses remain
    model/auditor decisions; a wrong ref still needs visual grounding checks.
    """
    if not isinstance(ref, str) or operation not in {'remove', 'replace'}:
        return None
    text = ref.casefold()
    if re.search(r'\b(?:statue|sculpture|doll|mannequin|poster|drawing|painting|'
                 r'printed|illustration|costume|prosthetic|artificial|toy|robot)\b', text):
        return None
    human = re.search(r"\b(?:man|woman|person|boy|girl|child|baby|weaver|human)"
                      r"(?:'s)?\b|\b(?:his|her|their)\b", text)
    text = re.sub(r'^the (?:patch|area|section) of (?:bare )?skin\b',
                  'the skin patch', text)
    # Match the named target, not a locating landmark ("the hat above his nose").
    # Participial location clauses describe the target's host; their last noun
    # is not the target ("bag held in the hand of the man"). Conversely a
    # "hand holding a bag" still names a hand.
    head = re.split(
        r'\b(?:of|on|in|above|below|beside|behind|with|worn|extending|belonging|'
        r'held|carried|holding|carrying)\b', text, 1)[0]
    head = head.strip()
    if not head or head.split()[-1] not in {
        'forehead','nose','cheek','cheeks','skin','eyelid','eyelids','eye','eyes',
        'arm','arms','forearm','forearms','leg','legs','hand','hands','foot','feet',
        'finger','fingers','toe','toes','ear','ears','head','torso','knee','knees','patch','surface',
    }:
        return None
    if not human:
        roles = (r'paddleboarder|surfer|cyclist|rider|driver|passenger|'
                 r'pedestrian|worker|athlete|officer|soldier|runner|player')
        # Bind new role words to the anatomical target's owner, not any person
        # elsewhere in the ref: a tripod leg near a cyclist is not anatomy.
        possessive = re.search(r'\b(?:' + roles + r")[’']s\b", head)
        owner = re.split(r'\bof\b', text, maxsplit=1)
        owner_head = re.split(
            r'\b(?:of|on|in|at|above|below|beside|behind|near|with|held|carried|'
            r'holding|carrying|who|that|wearing|standing|walking|riding)\b',
            owner[1], maxsplit=1)[0].strip() if len(owner) == 2 else ''
        human = possessive or re.search(r'\b(?:' + roles + r')$', owner_head)
    tissue = re.search(r'\b(?:forehead|nose|cheek|cheeks|skin|eyelid|eyelids|eye|eyes)\b', head)
    removable_part = re.search(r'\b(?:arms?|forearms?|legs?|hands?|foot|feet|fingers?|toes?|ears?|head|torso|knees?)\b', head)
    if human and (tissue or removable_part):
        return 'The declared edit removes or substitutes attached human anatomy, not a natural photo edit.'
    return None


def incompatible_attribute(ref, edit):
    """Only an explicit flat-color request on an entire named real person.

    Never infer a clothing/skin submask to make this malformed scope fit.
    Monochrome/tonal effects, skin/hair/garments and depictions are not vetoed.
    """
    if not isinstance(ref, str) or not isinstance(edit, dict) or edit.get('property') != 'color':
        return None
    text = ref.casefold()
    if re.search(r'\b(?:statue|sculpture|doll|mannequin|poster|drawing|painting|'
                 r'printed|illustration|costume|toy|robot|photo|picture)\b', text):
        return None
    head = re.split(r'\b(?:of|on|in|at|above|below|beside|behind|with|wearing|worn|'
                    r'walking|standing|sitting|holding|held|carried|carrying|between|'
                    r'near|under|next|who|that|which)\b', text, 1)[0].strip()
    # An omitted relative pronoun can leave a second noun phrase in the head:
    # "the chair the man ..." names a chair, never the man sitting on it.
    if any(w in {'the', 'a', 'an'} for w in head.split()[1:]):
        return None
    if not re.search(r'\b(?:man|woman|person|boy|girl|child|baby|runner|player|'
                     r'athlete|paddleboarder|surfer|cyclist|pedestrian)$', head):
        return None
    value = str(edit.get('value', '')).casefold().strip()
    if re.fullmatch(r'(?:(?:light|dark|deep|royal|pale|bright|warm|cool|neon|pastel)\s+)?'
                    r'(?:red|blue|green|yellow|orange|purple|pink|gray|grey|white|black|brown)', value):
        return 'whole_person_flat_color: a flat recolor of the entire person has no natural attribute scope.'
    return None
