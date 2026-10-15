"""Optional local SAM3 site constraint, loaded once per generation worker."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
from PIL import Image


class Sam3SiteResolver:
    def __init__(self, model_path, device='cuda'):
        import torch
        from transformers import Sam3Model, Sam3VideoModel, Sam3Processor
        started = time.perf_counter()
        config = json.loads((Path(model_path)/'config.json').read_text())
        cls = Sam3VideoModel if 'Sam3VideoModel' in config.get('architectures', []) else Sam3Model
        loaded = cls.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16)
        self.model = getattr(loaded, 'detector_model', loaded).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
        self.device, self.model_path = device, str(model_path)
        self.load_seconds = time.perf_counter()-started

    def __call__(self, source, host, spec, difference_site):
        import torch
        ys,xs = np.where(host)
        margin = max(32, round(max(xs.max()-xs.min(),ys.max()-ys.min())*.15))
        box = (max(0,int(xs.min())-margin),max(0,int(ys.min())-margin),
               min(source.width,int(xs.max())+margin+1),min(source.height,int(ys.max())+margin+1))
        crop = source.crop(box)
        prompt = spec['scope']+' '+spec['placement']['site']
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.processor(images=crop,text=prompt,return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,threshold=.3,target_sizes=[crop.size[::-1]])[0]
        candidates=[]
        for index,(mask,score) in enumerate(zip(result['masks'],result['scores'])):
            full=np.zeros_like(host)
            full[box[1]:box[3],box[0]:box[2]]=mask.cpu().numpy().astype(bool)
            region=full & host
            fraction=float(region.sum()/max(1,host.sum()))
            overlap=float((region & difference_site).sum()/max(1,region.sum())) if difference_site is not None else 1.
            # SAM sometimes returns both the named part and its entire host.
            # Never choose the latter merely because it has a higher score.
            valid=region.sum()>=8 and fraction<.9 and overlap>=.05
            candidates.append({'index':index,'score':float(score),'host_fraction':fraction,
                               'contact_overlap':overlap,'eligible':bool(valid),'mask':region})
        eligible=[c for c in candidates if c['eligible']]
        best=max(eligible,key=lambda c:c['score']*c['contact_overlap']) if eligible else None
        region=best['mask'] if best is not None else None
        if region is not None and difference_site is not None:
            region=region & difference_site
        info={'method':'sam3_site_constrained_difference','model':self.model_path,'prompt':prompt,
              'model_load_seconds':self.load_seconds,'seconds':time.perf_counter()-started,
              'candidates':[{k:v for k,v in c.items() if k!='mask'} for c in candidates],
              'selected_index':best['index'] if best else None}
        return region,info
