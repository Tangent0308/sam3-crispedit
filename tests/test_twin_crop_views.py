from copy import deepcopy
from unittest.mock import patch
import json
import numpy as np
import pytest
from PIL import Image
from sa1b_pipeline.twin_crop_views import twin_crop_views, INTRO
from sa1b_pipeline.full_contour_views import COLORS
from sa1b_pipeline import content_planning as content


def test_three_views_untouched_source_exact_alignment_holes_and_components():
    y,x=np.indices((360,480));rgb=np.stack([x%256,y%256,(x+y)%256],axis=-1).astype('uint8')
    source=Image.fromarray(rgb)
    a=np.zeros((360,480),bool);a[70:125,90:145]=True;a[90:110,105:125]=False;a[150:163,160:175]=True
    b=np.zeros_like(a);b[280:360,400:480]=True
    views,meta=twin_crop_views(source,[a,b])
    assert len(views)==3 and np.array_equal(np.array(views[0]),rgb)
    for i,(mask,row,view) in enumerate(zip([a,b],meta,views[1:])):
        assert row['unit_id']==f'U{i+1}' and row['selected_pixels_under_label']==0
        l,t,r,bottom=row['crop_box_xyxy'];assert mask[t:bottom,l:r].sum()==mask.sum()
        expected=source.crop((l,t,r,bottom)).resize(row['resized_size'],Image.Resampling.LANCZOS)
        selected=np.array(Image.fromarray(mask[t:bottom,l:r].astype('uint8')*255).resize(row['resized_size'],Image.Resampling.NEAREST))>0
        dx,dy=row['body_offset'];w,h=row['resized_size']
        body=np.array(view)[dy:dy+h,dx:dx+w]
        assert np.array_equal(body[selected],np.array(expected)[selected])
        assert (np.array(view)==COLORS[i]).all(axis=-1).any()
    with pytest.raises(ValueError,match='exactly two'):
        twin_crop_views(source,[a])


def test_view_cache_separates_styles_and_preserves_source_order(tmp_path):
    from sa1b_pipeline.group_planning import group_views
    source=tmp_path/'source.png';Image.new('RGB',(480,360),'orange').save(source)
    a=np.zeros((360,480),bool);a[20:70,20:60]=True
    b=np.zeros_like(a);b[200:240,350:390]=True
    objects=[{'object_id':188},{'object_id':81}]
    with patch('sa1b_pipeline.group_planning.load_mask',side_effect=[a,b]*3):
        args=({'image_path':str(source)},objects,tmp_path/'views')
        new=group_views(*args,twin_crops=True)
        assert new==group_views(*args,twin_crops=True)
        old=group_views(*args,clean_contours=True)
    assert len(new[0])==3 and new[2]!=old[2]
    from pathlib import Path
    metadata=json.loads((Path(new[0][0]).parent/'crop_metadata.json').read_text())
    assert [m['object_id'] for m in metadata]==['188','81']
    assert np.array_equal(np.array(Image.open(new[1][0]))>0,a)


def test_twins_retry_keeps_three_images_hint_binding_and_resumes(tmp_path):
    from sa1b_pipeline.lean_retry import plan_batch
    assignments={'U1':'remove','U2':'remove'};hints={'U1':'hat','U2':'shirt'}
    prompt=content.prompt('twins',assignments,view_style='twin-crops',source_hints=hints)
    assert prompt.startswith(INTRO)
    assert prompt.split('\n',1)[1]==content.prompt('twins',assignments,view_style='clean-contours',source_hints=hints).split('\n',1)[1]
    with pytest.raises(ValueError,match='two-member'):
        content.prompt('multi',assignments,view_style='twin-crops')
    with pytest.raises(ValueError,match='two-member'):
        content.prompt('twins',{**assignments,'U3':'remove'},view_style='twin-crops')
    first={'reject':'Removing a worn shirt is unsuitable.'}
    second={'members':[{'unit_id':'U1','ref':'the left hat'},{'unit_id':'U2','ref':'the right shirt'}],
            'edit':{'property':'color','value':'blue'}}
    class Planner:
        calls=0
        def generate_multimage(self,jobs,schemas=None):
            self.calls+=1
            assert jobs[0][0]==['original.png','U1.png','U2.png']
            assert jobs[0][1].startswith(INTRO) and '"U1": "hat", "U2": "shirt"' in jobs[0][1]
            return [deepcopy(first if self.calls==1 else second)]
    planner=Planner()
    item=dict(gid='crops',views=['original.png','U1.png','U2.png'],objects=[{'object_id':188},{'object_id':81}],
              selection={'operation':'remove'},prompt=prompt,planning_view_style='twin-crops',planning_source_hints=hints)
    opts=dict(kind='twins',root=tmp_path,retry=1,minimal=True,content=True,accepted_counts={'add':20,'replace':20,'attribute':0})
    result=plan_batch(planner,[deepcopy(item)],**opts)[0]
    assert result['operation']=='attribute'
    assert result==plan_batch(planner,[deepcopy(item)],**opts)[0] and planner.calls==2
