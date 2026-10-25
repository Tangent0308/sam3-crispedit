"""Shared, single-call visual planning requirements for the 27B pipelines."""

VERSION = 'scene-grounded-no-examples-2'


def appearance_only_replacement(scope, payload):
    """Catch explicit same-entity appearance rewrites, not certify visual fit.

    Unknown words remain structural candidates for the visual judge. Keeping
    this gate narrow avoids rejecting actual redesigns through an ontology.
    """
    import re
    appearance = set(('a an the pair of new different modern polished matte glossy shiny smooth rough frosted '
        'wood wooden metal metallic steel stainless plastic ceramic glass marble stone concrete '
        'brass copper gold golden silver aluminum aluminium alloy leather fabric cotton satin foam '
        'black white red blue green yellow orange purple pink gray grey brown beige bright dark light '
        'navy teal turquoise crimson maroon burgundy ivory cream violet cyan '
        'solid chrome finish texture coating paint surface').split())
    def words(text):
        # Intensity describes the color, not a new object's construction.
        # Do not remove standalone "deep": a deep drawer can change form.
        text = re.sub(r'\b(?:deep|pale|vivid|muted|saturated)\s*[- ]\s*(?='
                      r'black|white|red|blue|green|yellow|orange|purple|pink|gray|grey|brown|beige|navy|teal|crimson)',
                      '', str(text).casefold())
        return {w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith(('ss', 'us')) else w
                for w in re.findall(r'[a-z]+', text)}
    nouns, values = words(scope), words(payload)
    # These trouser names alone imply no visible construction change.
    synonyms = {'jean', 'trouser', 'pant'}
    if nouns & synonyms:
        nouns |= synonyms
    return bool(values) and not (values - nouns - words(' '.join(appearance)))

RULES = """Account for ALL selected pixels, including disconnected visible pieces. Name their actual coherent extent, not an easier subpart or the larger host. Context explains identity but does not grant ownership of excluded pixels. If no truthful coherent entity can be identified, record the uncertainty rather than guessing.
Check the public reference against the ORIGINAL full scene and its nearest matching instance. Use the shortest verified distinguishing phrase; specify the counting set if an ordinal is needed. The reference must identify the source before editing and cover exactly the selected extent. Do not describe the source using its desired result or an unseen component.
Respect the assigned operation and source extent. A replacement needs a recognizable physical form/construction change; an attribute edit preserves structure. An addition needs a visible direct contact site within the selected material and adequate free space for its complete shape. Preserve existing content, functional openings and excluded neighbors. A removal must leave the remaining scene physically coherent.
"""

from .instruction_diversity import PREFERENCE_RULE as NATURALNESS_RULES
