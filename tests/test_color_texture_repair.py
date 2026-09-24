import numpy as np
from PIL import Image
from synthesis_pipeline.repair_color_texture import restore_color_shading


def test_restore_shading_without_touching_surroundings():
    gray=np.tile(np.linspace(40,220,48,dtype=np.uint8),(48,1))
    source=Image.fromarray(np.stack([gray,gray,gray],axis=2))
    raw=np.asarray(source).copy();mask=np.zeros((48,48),bool);mask[5:43,5:43]=True
    raw[mask]=[210,30,30]
    out,meta=restore_color_shading(source,Image.fromarray(raw),mask,
        {'task_type':'attribute','editing_instruction':'Change the left shirt to bright red.'})
    assert meta['applied']
    assert np.array_equal(np.asarray(out)[~mask],raw[~mask])
    assert np.std(np.asarray(out)[mask,0])>10


def test_not_for_other_operations_or_new_material():
    im=Image.new('RGB',(48,48),'gray');mask=np.ones((48,48),bool)
    for task,text in [('replace','Replace the left cup with a red bowl.'),
                      ('attribute','Change the shirt to metallic red.')]:
        out,meta=restore_color_shading(im,im,mask,{'task_type':task,'editing_instruction':text})
        assert not meta['eligible'] and not meta['applied']


def test_no_change_when_source_itself_flat():
    im=Image.new('RGB',(48,48),'gray')
    out,meta=restore_color_shading(im,Image.new('RGB',(48,48),'red'),np.ones((48,48),bool),
        {'task_type':'attribute','editing_instruction':'Change the wall to red.'})
    assert not meta['applied']
