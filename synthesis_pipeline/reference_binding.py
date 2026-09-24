"""Bind ordered canonical SAMTok spans to raw masks without a model call.

The dataset contract specifies that masks follow span order. A shared group
label is retained as shared, never converted into an invented instance label.
"""
import json
import re

SPAN = re.compile(r'<\|mt_start\|>(?:<\|mt_\d+\|>)+<\|mt_end\|>')


def bind_reference(answer, mask_index, mask_count):
    text=str(answer or '').strip()
    spans=list(SPAN.finditer(text))
    result=dict(version=1,mask_index=int(mask_index),mask_count=int(mask_count),
                status='unresolved',label=None,canonical_span=None)
    if len(spans)!=mask_count or not 0<=mask_index<mask_count:
        return {**result,'reason':'span_count_or_mask_index_mismatch'}
    result['canonical_span']=spans[mask_index].group()
    if len({m.group() for m in spans})!=len(spans):
        return {**result,'reason':'duplicate_canonical_spans'}
    try:
        values=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text))
    except ValueError:
        values=None
    if isinstance(values,list):
        if len(values)!=mask_count or any(not isinstance(v,dict) for v in values):
            return {**result,'reason':'json_region_count_mismatch'}
        if any(v.get('mask_2d')!=m.group() for v,m in zip(values,spans)):
            return {**result,'reason':'json_span_order_mismatch'}
        label=values[mask_index].get('label')
        if not isinstance(label,str) or not label.strip():
            return {**result,'reason':'missing_region_label'}
        shared=sum(v.get('label')==label for v in values)>1 or label.lower().startswith('one of ')
        return {**result,'status':'bound_shared_group' if shared else 'bound_label',
                'label':label.strip(),'method':'json_entry_and_canonical_span_order'}
    # Multiple spans in ONE pair of parentheses share one group expression.
    left=text.rfind('(',0,spans[mask_index].start())
    right=text.find(')',spans[mask_index].end())
    if left>=0 and right>=0 and len(SPAN.findall(text[left:right+1]))>1:
        earlier=[m for m in spans if m.end()<left]
        prefix=text[earlier[-1].end() if earlier else 0:left].strip()
        prefix=re.sub(r'^\)\s*(?:,\s*)?(?:and\s+|or\s+)?','',prefix).strip()
        if prefix and len(prefix.split())<=35 and not re.search(r'[.!?;]|<\|',prefix):
            return {**result,'status':'bound_shared_group','label':prefix,
                    'method':'shared_parentheses_and_canonical_span_order'}
    # VER places a canonical span immediately after its referring noun phrase.
    # Keep that local phrase, not a list of every object's names in the answer.
    start=spans[mask_index-1].end() if mask_index else 0
    prefix=text[start:spans[mask_index].start()]
    if not re.search(r'\(\s*$',prefix):
        return {**result,'reason':'no_unambiguous_parenthesized_mention'}
    prefix=re.sub(r'\(\s*$','',prefix).strip()
    if mask_index:
        if not re.match(r'^\)\s*(?:,\s*)?(?:and\s+|or\s+)?',prefix):
            return {**result,'reason':'complex_inter_span_relation'}
        prefix=re.sub(r'^\)\s*(?:,\s*)?(?:and\s+|or\s+)?','',prefix).strip()
    # Complex clauses remain usable context, but not a confident object binding.
    if not prefix or len(prefix.split())>35 or re.search(r'[.!?;]|<\|',prefix):
        return {**result,'reason':'ambiguous_local_mention'}
    return {**result,'status':'bound_mention','label':prefix,
            'method':'parenthesized_mention_and_canonical_span_order'}


def binding_prompt(binding):
    if not binding or binding.get('status')=='unresolved':
        return ''
    shared=binding['status']=='bound_shared_group'
    return (
        f"Dataset reference bound to THIS mask (region {binding['mask_index']+1} of {binding['mask_count']}): "
        + json.dumps(binding['label'],ensure_ascii=True)
        + ('\nThis is a shared GROUP expression, not an individual instance locator. '
           'Use the outlined pixels and the full image to identify this one member; do not edit the whole group.' if shared else
           '\nThis phrase belongs to this mask, not another region mentioned elsewhere in the source answer. '
           'Use it to cross-check identity; derive the precise instance locator from the clean full photograph.')
        + '\nIf the object identity and outlined photographic content conflict, mark incompatible; do not silently rename the object. '
          'A broad owner/group phrase may cover more than this mask: name the actual selected part or member, not the whole owner/group.'
    )
