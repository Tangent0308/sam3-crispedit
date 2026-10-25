from copy import deepcopy
import pytest
from sa1b_pipeline import minimal_planning as m
from sa1b_pipeline.instruction_protocol import render

@pytest.mark.parametrize('op,ref,ins',[
 ('remove','the left chair','Remove the left chair.'),
 ('replace','the left chair','Replace the left chair with a folding stool.'),
 ('add','the left chair','Add a cushion on the left chair.'),
 ('add','the left chair','Add a cushion resting on the seat of the left chair.'),
 ('add','the tree','Add a bird on a branch of the tree.'),
 ('attribute','the left chair','Change the color of the left chair to blue.'),
 ('attribute','the left chair','Change the surface texture of the left chair to ribbed.'),
 ('attribute','the label','Change the text on the label to "Open".'),
 ('attribute','the left window','Tint the left window green.'),
 ('attribute','the left window','Make the left window translucent.'),
])
def test_protocol_roundtrip_and_blind(op,ref,ins):
 s=m.parse_instruction(ref,ins,op)
 assert render(s)==ins
 assert 'the selected region' in render(s,False) and ref not in render(s,False)
 if 'seat of' in ins:assert s['placement']['site']=='seat' and 'seat' not in render(s,False)

@pytest.mark.parametrize('op,ref,ins',[
 ('remove','the left chair','Remove the right chair.'),
 ('remove','the left chair','Remove the left chair and the table.'),
 ('attribute','the left chair','Make the left chair blue.'),
 ('add','the left chair','Add a cushion beside the left chair.'),
 ('add','the left chair','Add a cushion under the left chair.'),
 ('remove','the selected region','Remove the selected region.'),
])
def test_do_not_silently_reinterpret_invalid_instructions(op,ref,ins):
 with pytest.raises(ValueError):m.parse_instruction(ref,ins,op)


def test_rejection_is_minimal_and_shared_sentence_checked():
 raw={'reject':'would remove anatomy'}
 decoded=m.decode(raw,'twins',{'U1':'remove','U2':'remove'})
 assert not decoded['feasible'] and not decoded.get('_minimal_error')
 raw={'members':[{'unit_id':'U1','ref':'the left chair'}, {'unit_id':'U2','ref':'the right chair'}],
      'instruction':'Change the color of {ref} to blue.'}
 decoded=m.decode(raw,'twins',{'U1':'attribute','U2':'attribute'})
 assert decoded['feasible'] and decoded['attributes']==[{'property':'color','value':'blue'}]
 raw['instruction']='Change the color of the left chair to blue.'
 assert m.decode(raw,'twins',{'U1':'attribute','U2':'attribute'}).get('_minimal_error')


def test_missing_visual_observations_are_not_invented():
 raw={'targets':[{'unit_id':'U1','ref':'the left chair','instruction':'Add a cushion on the left chair.'},
                 {'unit_id':'U2','ref':'the right chair','instruction':'Change the color of the right chair to blue.'}]}
 a={'U1':'add','U2':'attribute'}
 out=m.normalize(m.decode(raw,'multi',a),'multi',a,[{'object_id':'a'},{'object_id':'b'}])
 for t in out['targets']:
  assert 'contact_xy' not in t['grounding'] and 'current_value' not in t['grounding']
  m.validate_edit(t)
 assert out['_minimal_response']==raw


def test_minimal_retry_preserves_good_target_and_recovers(tmp_path):
 from sa1b_pipeline.lean_retry import plan_batch
 raw1={'targets':[{'unit_id':'U1','ref':'the left chair','instruction':'Remove the left chair.'}, {'unit_id':'U2','reject':'anatomy'}]}
 raw2={'targets':[{'unit_id':'U2','ref':'the right sleeve','instruction':'Change the color of the right sleeve to blue.'}]}
 class P:
  calls=0
  def generate_multimage(self,jobs,*,schemas=None):
   self.calls+=1
   return [deepcopy(raw1 if self.calls==1 else raw2)]
 def item():return dict(gid='m',views=[],objects=[{'object_id':'a'},{'object_id':'b'}],selection={'operations':['remove','remove']},
                       prompt=m.prompt('multi',{'U1':'remove','U2':'remove'},locations=[{'unit_id':'U1','bbox':[0,0,10,10]}]))
 p=P();opts=dict(kind='multi',root=tmp_path,retry=1,minimal=True,accepted_counts={'add':20,'replace':20,'attribute':0})
 a=plan_batch(p,[item()],**opts)[0];b=plan_batch(p,[item()],**opts)[0]
 assert a==b and p.calls==2
 assert a['targets'][0]['_minimal_instruction']==raw1['targets'][0]['instruction']
 assert a['targets'][1]['operation']=='attribute'


def test_minimal_both_builders(tmp_path):
 import numpy as np
 from PIL import Image
 from sa1b_pipeline.group_planning import build_twin_group,build_multi_group
 source=tmp_path/'source.png';Image.new('RGB',(100,60),'white').save(source)
 objects=[];paths=[];geoms=[]
 for i,x in enumerate((10,60)):
  mask=np.zeros((60,100),dtype='uint8');mask[10:30,x:x+20]=255
  p=tmp_path/f'{i}.png';Image.fromarray(mask).save(p);paths.append(str(p))
  objects.append({'object_id':str(i),'category':'chair'})
  geoms.append(dict(bbox=[x,10,x+20,30],point=[x+10,20],area_fraction=400/6000))
 candidate=dict(source_id='test',image_path=str(source));a={'U1':'add','U2':'add'}
 rows=[dict(unit_id='U'+str(i+1),ref=f'the {side} chair',instruction=f'Add a cushion on the {side} chair.') for i,side in enumerate(('left','right'))]
 for kind,key in [('twins','members'),('multi','targets')]:
  response={key:rows} if kind=='multi' else {'members':[{k:v for k,v in t.items() if k!='instruction'} for t in rows],'instruction':'Add a cushion on {ref}.'}
  raw=m.normalize(m.decode(response,kind,a),kind,a,objects)
  args=dict(split='train',group_id=kind,geometry=geoms,require_grounding=True)
  if kind=='twins':g=build_twin_group(candidate,objects,raw,paths,scope_level='object',assigned_operation='add',**args)
  else:g=build_multi_group(candidate,objects,raw,paths,assignments={'0':'add','1':'add'},**args)
  for u in g.get('members',g.get('units',[])):
   assert 'generation_contact' not in u['unit']
   assert u['instruction_noref']=='Add a cushion on the selected region.'


def test_safe_terminal_period_normalization_and_anatomy_reference():
 assert render(m.parse_instruction('the left chair','Remove the left chair','remove'))=='Remove the left chair.'
 item=dict(operation='remove',ref_phrase='the arm of the man',scope='arm of the man')
 with pytest.raises(ValueError):m.validate_edit(item)
 item.update(ref_phrase='the arm of the chair',scope='arm of the chair')
 m.validate_edit(item)
 m.parse_instruction('the black face mask','Remove the black face mask.','remove')


def test_named_site_contact_is_bounded_and_does_not_guess(tmp_path):
 import numpy as np
 from PIL import Image
 a=np.zeros((50,50),dtype='uint8');a[10:40,20:30]=255
 p=tmp_path/'mask.png';Image.fromarray(a).save(p)
 assert m.named_site_contact(p,'top edge')['point'][1]==10
 assert m.named_site_contact(p,'lower edge')['point'][1]==39
 assert m.named_site_contact(p,'front') is None
 assert m.named_site_contact(p,'') is None
 assert m.named_site_contact(p,'top edge')['verified'] is False


def test_rejected_target_without_entity_does_not_poison_valid_siblings():
 from sa1b_pipeline.group_planning import normalize_multi_grounding
 raw={'grounding_version':7,'targets':[
  {'scope':'chair','grounding':{'entity':'chair','edit_feasible':True}},
  {'scope':'','grounding':{'entity':'','edit_feasible':False,'reason':'unsupported'}}]}
 assert normalize_multi_grounding(raw,[])==raw


def test_mechanical_head_is_not_human_anatomy():
 item=dict(operation='replace',ref_phrase='the traffic light head',scope='traffic light head',payload='a surveillance camera')
 m.validate_edit(item)
 item.update(ref_phrase='the head of the man',scope='head of the man')
 with pytest.raises(ValueError):m.validate_edit(item)
