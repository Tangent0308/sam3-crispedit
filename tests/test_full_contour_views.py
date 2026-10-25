"""Protect clean inputs, true mask topology, and the retry input contract."""
import json
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from PIL import Image
from sa1b_pipeline.full_contour_views import contour_pair, INTRO, COLORS
from sa1b_pipeline import content_planning as content


def test_clean_pixels_holes_components_and_label_endpoints():
    source = Image.new('RGB', (640, 480), (71, 102, 133))
    a = np.zeros((480,640),dtype=bool);a[70:210,80:220]=True;a[110:160,120:170]=False
    a[280:310,190:240]=True
    b = np.zeros_like(a);b[330:420,400:500]=True
    clean, marked, labels = contour_pair(source,[a,b])
    assert np.array_equal(np.array(source),np.array(clean))
    assert marked.size == source.size
    for i,(mask,label) in enumerate(zip([a,b],labels)):
        x,y=label['leader_end'];assert mask[y,x]
        assert label['unit_id']==f'U{i+1}' and label['color']==list(COLORS[i])
        assert label['selected_pixels_under_label']==0
    pixels=np.array(marked)
    # Untinted interior, untouched hole centre, both boundary components present.
    assert tuple(pixels[185,180])==(71,102,133)
    assert tuple(pixels[135,145])==(71,102,133)
    assert tuple(pixels[110,140])==COLORS[0]
    assert tuple(pixels[310,215])==COLORS[0]
    clean2,marked2,labels2=contour_pair(source,[a,b])
    assert labels==labels2 and np.array_equal(np.array(marked),np.array(marked2))


def test_group_view_cache_isolated_and_original_resolution(tmp_path):
    from sa1b_pipeline.group_planning import group_views
    source=tmp_path/'source.png';Image.new('RGB',(1600,400),'orange').save(source)
    a=np.zeros((400,1600),dtype=bool);a[50:150,80:200]=True
    args=({'image_path':str(source)},[{'object_id':'abc'}],tmp_path/'views')
    with patch('sa1b_pipeline.group_planning.load_mask',return_value=a):
        old=group_views(*args,aligned_outline=True,save_debug_views=False)
        new=group_views(*args,aligned_outline=True,clean_contours=True,save_debug_views=False)
        assert new==group_views(*args,clean_contours=True,save_debug_views=False)
    assert new[2]!=old[2] and len(new[0])==2
    assert np.array_equal(np.array(Image.open(source)),np.array(Image.open(new[0][0])))
    assert Image.open(new[0][1]).size==(1600,400)
    assert np.array_equal(np.array(Image.open(new[1][0]))>0,a)


def test_retry_retains_two_full_frame_description_and_resumes(tmp_path):
    from sa1b_pipeline.lean_retry import plan_batch
    assignments={'U1':'remove','U2':'remove'}
    original=content.prompt('multi',assignments)
    prompt=content.prompt('multi',assignments,view_style='clean-contours')
    assert prompt==INTRO+original.split('\n',1)[1]
    first={'targets':[{'unit_id':'U1','ref':'the left chair','edit':{}},{'unit_id':'U2','reject':'would sever an arm'}]}
    second={'targets':[{'unit_id':'U2','ref':'the right sleeve','edit':{'property':'color','value':'blue'}}]}
    class Planner:
        calls=0
        def generate_multimage(self,jobs,schemas=None):
            self.calls+=1
            assert jobs[0][0]==['original.png','contours.png']
            assert jobs[0][1].startswith(INTRO) and 'ORIGINAL CROP' not in jobs[0][1]
            return [deepcopy(first if self.calls==1 else second)]
    planner=Planner()
    item=dict(gid='contour',views=['original.png','contours.png'],objects=[{'object_id':'a'},{'object_id':'b'}],selection={'operations':['remove','remove']},prompt=prompt,planning_view_style='clean-contours')
    opts=dict(kind='multi',root=tmp_path,retry=1,minimal=True,content=True,accepted_counts={'add':20,'replace':20,'attribute':0})
    out=plan_batch(planner,[deepcopy(item)],**opts)[0]
    assert out['targets'][1]['operation']=='attribute'
    assert plan_batch(planner,[deepcopy(item)],**opts)[0]==out and planner.calls==2
