"""Reproducible examples: three hash-selected, visible-region cases per operation."""
import argparse,hashlib,json,textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def font(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
def fit(canvas,img,box):
    x,y,w,h=box;im=img.copy();im.thumbnail((w,h));canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
def build(root,repo):
    media_root = root/'release' if (root/'release/annotations/audit.jsonl').exists() else root
    manifest = media_root/'annotations/audit.jsonl' if media_root != root else root/'final/auto_pass.jsonl'
    rows=[json.loads(l) for l in manifest.open() if l.strip()]
    selected=[]
    for op in ['add','replace','remove','attribute']:
        pool=[r for r in rows if r['operation']==op and r['difficulty']['target_features'][0]['area_fraction']>=.01]
        selected.extend(sorted(pool,key=lambda r:hashlib.sha256(('release-examples-20261015:'+r['job_id']).encode()).hexdigest())[:3])
    dest=root/'release/visualizations';dest.mkdir(parents=True,exist_ok=True)
    docassets=repo/'docs/assets/production_50k';docassets.mkdir(parents=True,exist_ok=True)
    previous = dest/'cases.json'
    previous_reviews = {c['job_id']:c for c in json.loads(previous.read_text())} if previous.exists() else {}
    pages=[];index=[]
    for i,r in enumerate(selected):
        src=Image.open(media_root/r['source_image']).convert('RGB');out=Image.open(media_root/r['edited_image']).convert('RGB')
        mask=Image.open(media_root/r['mask']).convert('L');support=Image.open(media_root/r['support_mask']).convert('L')
        assert src.size==out.size==mask.size==support.size
        binary=mask.point(lambda x:255 if x>127 else 0)
        overlay=Image.new('RGB',src.size,(224,38,144));marked=Image.composite(Image.blend(src,overlay,.38),src,binary)
        # A bright outline remains legible on both light and dark objects.
        edge=binary.filter(ImageFilter.MaxFilter(9));inner=binary.filter(ImageFilter.MinFilter(5))
        import PIL.ImageChops
        edge=PIL.ImageChops.subtract(edge,inner)
        marked.paste((255,30,145),mask=edge)
        bbox=support.getbbox() or mask.getbbox();x0,y0,x1,y1=bbox
        w,h=x1-x0,y1-y0;pad=max(48,int(max(w,h)*.35))
        crop=(max(0,x0-pad),max(0,y0-pad),min(src.width,x1+pad),min(src.height,y1+pad))
        canvas=Image.new('RGB',(1800,1300),'#f2f5fa');d=ImageDraw.Draw(canvas)
        d.rectangle((0,0,1800,155),fill='#16243b')
        d.text((32,18),f"{r['operation'].upper()}   /   {r['source_id']}   /   mask {r['object_id']}",font=font(30,True),fill='white')
        lines=textwrap.wrap(r['instruction'],width=104)
        for j,line in enumerate(lines):d.text((32,66+j*31),line,font=font(25),fill='#e5edf8')
        for j,title in enumerate(['Source image','Source region / '+r['mask_role'],'Edited image']):
            d.text((32+j*590,176),title,font=font(25,True),fill='#213a5c')
            fit(canvas,[src,marked,out][j],(25+j*590,220,570,460))
            fit(canvas,[src,marked,out][j].crop(crop),(25+j*590,745,570,460))
        d.text((32,704),'Aligned detail crop (same coordinates in all three columns)',font=font(23),fill='#526477')
        d.text((32,1230),'Magenta = source mask. Automatic pass; display sample, not independent human certification.',font=font(22),fill='#526477')
        verdict = previous_reviews.get(r['job_id'], {}).get('assistant_visual_review', 'pending')
        label = {'acceptable': 'ACCEPTABLE', 'needs_review': 'NEEDS REVIEW', 'issue': 'ISSUE FOUND', 'pending': 'PENDING'}[verdict]
        detail = {4: 'Replace / attribute boundary is ambiguous.', 6: 'Replacement wall does not preserve the blue welding light.',
                  9: 'Floating ropes remain after the boat is removed.'}.get(i+1, '')
        d.text((32,1268),f'Assistant visual check: {label}. {detail}',font=font(19,True),
               fill='#aa3f33' if verdict in {'needs_review','issue'} else '#31596a')
        name=f'{i+1:02d}_{r["operation"]}_{r["source_id"]}_{r["object_id"]}.jpg'
        canvas.save(dest/name,quality=90)
        canvas.save(docassets/name,quality=85)
        pages.append(canvas)
        index.append({'page':i+1,'file':name,'job_id':r['job_id'],'operation':r['operation'],
                      'instruction':r['instruction'],'source_id':r['source_id'],'object_id':r['object_id'],
                      'selection':'first 3 per type by seeded SHA-256 ordering, source region >= 1% image area',
                      'assistant_visual_review':'pending'})
    for case in index:
        old = previous_reviews.get(case['job_id'], {})
        for key in ('sample_id', 'assistant_visual_review', 'review_note_zh'):
            if key in old: case[key] = old[key]
    pages[0].save(dest/'examples_12.pdf',save_all=True,append_images=pages[1:],resolution=150)
    for p in [dest/'cases.json',docassets/'cases.json']:p.write_text(json.dumps(index,ensure_ascii=False,indent=2))
    print(json.dumps(index,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();build(a.root,a.repo)
