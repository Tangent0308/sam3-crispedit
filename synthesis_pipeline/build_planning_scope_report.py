"""Self-contained planning regression report, including every rejected input."""
import argparse
import base64
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
from synthesis_pipeline.merge_verified_edits import comparison_sheet, preview_data_uri


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def escaped(value):
    return html.escape(str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--plan-root', type=Path, required=True)
    parser.add_argument('--baseline-plan', type=Path)
    parser.add_argument('--parent-plan', type=Path, help='Include initial planner attempts and rejections before scope preflight')
    parser.add_argument('--regions-root', type=Path)
    parser.add_argument('--edited-root', type=Path, help='Optionally include actual outputs; never interpret their presence as passing QA')
    parser.add_argument('--raw-root', type=Path, help='Show rejected raw outputs when final composition failed, explicitly marked diagnostic')
    parser.add_argument('--reviews', type=Path, action='append', default=[], help='Per-case reviews; later files add/override fields for the same exact image')
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=False)
    original = {r['image']: r for r in rows(args.data_root/'annotations.jsonl')}
    baseline = {r['image']: r for r in rows(args.baseline_plan/'annotations.jsonl')} if args.baseline_plan else {}
    planned = {r['image']: r for r in rows(args.plan_root/'annotations.jsonl')}
    refined = {r['image']: r for r in rows(args.regions_root/'annotations.jsonl')} if args.regions_root else {}
    reviews = {}
    for review_path in args.reviews:
        for review in rows(review_path):
            reviews[review['image']] = {**reviews.get(review['image'],{}), **review}
    history = {}
    if args.parent_plan:
        for response in rows(args.parent_plan/'responses.jsonl'):
            history.setdefault(response['image'], []).append({**response, 'stage':'visual grounding'})
    for response in rows(args.plan_root/'responses.jsonl'):
        history.setdefault(response['image'], []).append({**response, 'stage':'grounded plan'})
    cards = []
    fields = ['masked_content', 'edit_unit_status', 'outside_dependencies', 'refer_object',
              'mask_compatibility', 'compatibility_reason', 'segmentation_target',
              'mask_refinement', 'protected_objects', 'structural_parts']
    for name, attempts in sorted(history.items()):
        row = original[name]
        source = Image.open(args.data_root/'sources'/row['source_image']).convert('RGB')
        input_path = args.plan_root/'inputs'/name
        if not input_path.exists() and args.parent_plan:
            input_path = args.parent_plan/'inputs'/name
        crop = Image.open(input_path).convert('RGB')
        panels = [('SOURCE', source), ('PLANNER INPUT: ORIGINAL MASK', crop)]
        region = refined.get(name)
        dataset_mask = bool(region and region['region_contract'].get('provenance')=='dataset_mask_no_source_sam')
        if region and not dataset_mask:
            panels.append(('EXECUTION MASK: '+region['region_contract']['status'],
                           instruction_target_crop(source, mask_array(source.size, region['mask']))))
        canvas = Image.new('RGB', (640*len(panels), 700), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (title, picture) in enumerate(panels):
            draw.text((i*640+12, 10), title, fill='black')
            thumb = ImageOps.contain(picture, (628, 658))
            canvas.paste(thumb, (i*640+(640-thumb.width)//2, 35+(658-thumb.height)//2))
        filename = Path(name).stem+'.jpg'
        canvas.save(args.out_root/filename, quality=92)
        data = base64.b64encode((args.out_root/filename).read_bytes()).decode('ascii')
        current = planned.get(name, attempts[-1].get('parsed') or {})
        verdict = 'PLANNED' if name in planned else 'NOT ACCEPTED'
        if region and region['region_contract']['status'] == 'unresolved':
            verdict += ' / MASK UNRESOLVED'
        review = reviews.get(name, {'status': 'not manually reviewed'})
        output_panel = ''
        if args.edited_root or args.raw_root:
            edited = args.edited_root/'edited'/name if args.edited_root else Path('/nonexistent')
            diagnostic = False
            if not edited.exists() and args.raw_root:
                edited=args.raw_root/'edited'/name
                diagnostic=True
            if edited.exists():
                sheet = comparison_sheet(args.data_root/'sources'/row['source_image'], edited,
                                         (region or row)['mask'])
                sheet.save(args.out_root/(Path(name).stem+'_result.jpg'), quality=92)
                label='原始出图诊断：后处理未形成final，不能作为已通过样本' if diagnostic else '实际出图：左原图、右结果（不等同质量通过）'
                output_panel = '<h3>'+label+'</h3><img src="'+preview_data_uri(sheet)+'">'
        details = ''.join('<details><summary>'+escaped(a['stage'])+' attempt '+str(a['attempt'])+
            '</summary><pre>'+escaped(a['prompt'])+'</pre><h4>Full model response</h4><pre>'+
            escaped(a['raw_response'])+'</pre></details>' for a in attempts)
        latest = attempts[-1].get('parsed') or {}
        model_stage=('流程MLLM grounded plan' if attempts[-1]['stage']=='grounded plan'
                     else '流程MLLM visual grounding（未进入指令规划）')
        model_note = '<p>'+model_stage+'：'+escaped(latest.get('decision',latest.get('mask_compatibility','')))+' — '+escaped(latest.get('reason',latest.get('compatibility_reason','')))+'；代码状态：'+escaped(attempts[-1].get('status',''))+'</p>'
        cards.append('<article id="'+escaped(Path(name).stem)+'"><h2>'+escaped(name)+' — '+verdict+'</h2><img src="data:image/jpeg;base64,'+
            data+'"><p>Previous: '+escaped(baseline.get(name, row).get('editing_instruction', 'No previous plan'))+
            '</p><p>Current: '+escaped(current.get('editing_instruction', ''))+
            '</p>'+('<p>执行mask：直接复用数据集原始mask；未调用出图前SAM。</p>' if dataset_mask else '')+model_note+
            '<h3>Codex逐图复核（本次分析记录，非pipeline模型输出）</h3><pre>'+escaped(json.dumps(review, ensure_ascii=False, indent=2))+
            '</pre>'+output_panel+'<details><summary>Planning and execution contracts</summary><pre>'+
            escaped(json.dumps({'planning':{k:current.get(k) for k in fields},
                                'execution':{k:v for k,v in (region or {}).get('region_contract',{}).items() if k!='protected_mask'}}, ensure_ascii=False, indent=2))+
            '</pre></details>'+details+'</article>')
    summary = json.loads((args.plan_root/'summary.json').read_text())
    summary['full_input_cases_shown'] = len(cards)
    summary['executable_regions'] = sum(r['region_contract']['status']!='unresolved' for r in refined.values())
    summary['actual_outputs_shown'] = sum((args.edited_root/'edited'/name).exists() for name in history) if args.edited_root else 0
    summary['raw_diagnostics_shown'] = sum((args.raw_root/'edited'/name).exists() and not (args.edited_root and (args.edited_root/'edited'/name).exists()) for name in history) if args.raw_root else 0
    summary['codex_review_not_pipeline_audit'] = {
        'reviewed_inputs': sum(name in reviews for name in history),
        'visually_reviewed_outputs_including_raw': sum(reviews.get(name,{}).get('output_quality') in ('pass','fail') for name in history),
        'plan_pass': sum(reviews.get(name,{}).get('plan_status')=='pass' for name in history),
        'plan_risk': sum(reviews.get(name,{}).get('plan_status')=='risk' for name in history),
        'plan_fail': sum(reviews.get(name,{}).get('plan_status')=='fail' for name in history),
        'quality_and_instruction_pass_including_raw': sum(
            reviews.get(name,{}).get('output_quality')=='pass' and reviews.get(name,{}).get('instruction_match')=='pass'
            for name in history),
        'strict_usable_final': sum(
            reviews.get(name,{}).get('plan_status')=='pass' and reviews.get(name,{}).get('output_quality')=='pass'
            and reviews.get(name,{}).get('instruction_match')=='pass'
            and bool(args.edited_root and (args.edited_root/'edited'/name).exists()) for name in history),
    }
    (args.out_root/'report_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    document = '<!doctype html><meta charset="utf-8"><title>Planning scope iteration</title><style>' \
        'body{font:16px system-ui;margin:24px;background:#eee}article{background:white;padding:20px;margin:24px 0}' \
        'img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style>' \
        '<h1>指令范围修复：完整输入、结果与逐条复核</h1><p>先看每条Current指令，再看源图及轮廓crop；如有实际出图，下方左原图、右结果，上排全图、下排局部。PLANNED表示计划接收，不等于画质通过。流程MLLM理由和Codex逐图复核分别显示；拒绝和失败样本全部保留。展开visual grounding/grounded plan可看两个阶段的完整prompt及每次原始回复。</p><pre>' \
        +escaped(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre>'+''.join(cards)
    (args.out_root/'index.html').write_text(document)
    print(json.dumps({'cases': len(cards), 'planned': len(planned), 'path': str(args.out_root/'index.html')}))


if __name__ == '__main__':
    main()
