import unittest
from PIL import Image
import numpy as np
from synthesis_pipeline.audit_quality_v4 import admission, parse_verification, parse_candidate_verification, parse_selective_verification, clean_overview, paired_overview, native_full_pair, corrected_annotation
from synthesis_pipeline.audit_policy_photographic import critical_verification_prompt, grounded_verification_prompt, observed_verification_prompt, observed_reconstruction_prompt, GROUNDED_RECONSTRUCTION
from synthesis_pipeline.audit_pixel_evidence import changed_surface_evidence, evidence_prompt


class AuditV4Tests(unittest.TestCase):
    def test_observation_is_fallible_evidence_not_a_prior_verdict(self):
        prompt=observed_verification_prompt('add','Add hands to the laptop.','The foreground laptop was removed.')
        self.assertIn('foreground laptop was removed',prompt)
        self.assertIn('This reading may itself be wrong',prompt)
        self.assertLess(prompt.index('foreground laptop was removed'),prompt.index('Now test the proposed label'))
        self.assertNotIn('Original label',prompt)
        self.assertEqual(observed_verification_prompt('add','Add a pin.',None),
                         grounded_verification_prompt('add','Add a pin.'))
        reconstruction=observed_reconstruction_prompt('The laptop was removed.')
        self.assertIn('WITHOUT the requested instruction',reconstruction)
        self.assertIn('not a requested edit and may be inaccurate',reconstruction)
        self.assertIn('25-word limit',reconstruction)
        self.assertEqual(observed_reconstruction_prompt(None),GROUNDED_RECONSTRUCTION)

    def test_native_full_pair_preserves_every_after_pixel(self):
        pixels=np.arange(96*64*3,dtype=np.uint8).reshape(64,96,3)
        source=Image.fromarray(pixels);edited=Image.fromarray(pixels[::-1].copy())
        mask=np.zeros((64,96),bool);mask[20:40,30:60]=True
        before,after=native_full_pair(source,edited,mask)
        self.assertEqual(before.size,(96,104))
        self.assertEqual(after.crop((0,40,96,104)).tobytes(),edited.tobytes())
        self.assertTrue(np.array_equal(np.asarray(before)[40:][mask],pixels[mask]))
        with self.assertRaises(ValueError):
            native_full_pair(source,edited.resize((48,32)),mask)

    def test_paired_layout_only_rearranges_same_pixels(self):
        src=Image.new('RGB',(96,64),(34,56,78));dst=Image.new('RGB',(96,64),(67,89,12))
        mask=np.zeros((64,96),bool);mask[20:40,30:60]=True
        vertical=clean_overview(src,dst,mask);paired=paired_overview(src,dst,mask)
        self.assertEqual([p.size for p in paired],[(2048,672),(2048,672)])
        for i,panel in enumerate(paired):
            for j,original in enumerate(vertical):
                self.assertEqual(panel.crop((j*1024,0,(j+1)*1024,672)).tobytes(),
                    original.crop((0,i*672,1024,(i+1)*672)).tobytes())

    def test_selective_original_must_also_obey_mask_scope(self):
        import json
        q={'visual_quality':'pass'}
        result=parse_selective_verification(json.dumps(dict(reason='both photos agree',quality='pass',label_choice='original',chosen_scope='same')))
        self.assertEqual(admission(q,None,result),'keep_original')
        self.assertEqual(admission(q,None,{**result,'candidate_scope':'wrong'}),'reject_scope')
        self.assertEqual(admission(q,None,{**result,'candidate_scope':'subset'}),'needs_mask_realignment')
        self.assertEqual(admission({'visual_quality':'fail'},None,result),'reject_quality')
        self.assertIsNone(parse_selective_verification('{"label_choice":"maybe"}'))

    def test_grounded_label_policy_checks_inventory_before_hypothesis(self):
        prompt=grounded_verification_prompt('attribute','Recolor the left boat blue.')
        self.assertLess(prompt.index('visible AFTER state'),prompt.index('Recolor the left boat blue.'))
        self.assertIn('unchanged extremities',prompt)
        self.assertIn('including separate panels in a collage',GROUNDED_RECONSTRUCTION)
        self.assertNotIn('Removal:',prompt)

    def test_pixel_evidence_is_local_and_not_an_automatic_verdict(self):
        a=np.zeros((32,32,3),np.uint8)
        a[:]=np.linspace(20,220,32,dtype=np.uint8)[None,:,None]
        b=np.full_like(a,(230,20,20));mask=np.zeros((32,32),bool);mask[4:28,4:28]=True
        evidence=changed_surface_evidence(Image.fromarray(a),Image.fromarray(b),mask,'attribute')
        self.assertGreater(evidence['before_luminance_range'],20)
        self.assertLess(evidence['after_luminance_range'],.01)
        self.assertIn('NOT a verdict',evidence_prompt(evidence))
        self.assertNotIn('quality',evidence)
        self.assertIsNone(changed_surface_evidence(Image.fromarray(a),Image.fromarray(b),mask,'replace'))
        self.assertIsNone(changed_surface_evidence(Image.fromarray(a),Image.fromarray(a),mask,'attribute'))

    def test_rewrite_updates_both_active_instruction_fields(self):
        row={'task_type':'replace','editing_instruction':'Old global command.',
             'new_instruction':'Old local command.','mask':{'counts':'unchanged'}}
        result=corrected_annotation(row,{'task_type':'add','editing_instruction':'Add a pin.'},'accept_rewrite')
        self.assertEqual(result['new_instruction'],result['editing_instruction'])
        self.assertEqual(result['task_type'],'add')
        self.assertEqual(result['audit_v4']['original_new_instruction'],'Old local command.')
        self.assertEqual(row['new_instruction'],'Old local command.')
        self.assertEqual(result['mask'],row['mask'])

    def test_verification_is_specific_to_candidate_operation(self):
        add=critical_verification_prompt('add','Add a pin to the left jacket.')
        remove=critical_verification_prompt('remove','Remove the left jacket.')
        self.assertIn('Addition:',add)
        self.assertNotIn('Removal:',add)
        self.assertIn('Removal:',remove)
        self.assertNotIn('Addition:',remove)
        self.assertIn('at most 100 words',remove)
        self.assertIn('candidate_match must fail',critical_verification_prompt(None,None))

    def setUp(self):
        self.q={'visual_quality':'pass'}
        self.c={'task_type':'replace','editing_instruction':'Replace the left toy with a bear toy.'}
        self.v={'quality':'pass','original_match':'fail','candidate_match':'pass','candidate_scope':'same','reason':'visible toy'}

    def test_quality_failure_cannot_be_rewritten_away(self):
        self.assertEqual(admission({'visual_quality':'fail'},self.c,self.v),'reject_quality')

    def test_independent_verifier_required(self):
        self.assertEqual(admission(self.q,self.c,None),'reject_verification_parse')
        self.assertEqual(admission(self.q,self.c,{**self.v,'quality':'fail'}),'reject_verified_quality')

    def test_scope_requires_realign_not_silent_old_mask(self):
        self.assertEqual(admission(self.q,self.c,{**self.v,'candidate_scope':'subset'}),'needs_mask_realignment')
        self.assertEqual(admission(self.q,self.c,{**self.v,'candidate_scope':'wrong'}),'reject_scope')

    def test_rewrite_or_keep(self):
        self.assertEqual(admission(self.q,self.c,self.v),'accept_rewrite')
        self.assertEqual(admission(self.q,None,{**self.v,'original_match':'pass'}),'keep_original')

    def test_invalid_response(self):
        self.assertIsNone(parse_verification('<think>unfinished'))
        self.assertIsNone(parse_verification('{"quality":"pass"}'))

    def test_candidate_only_never_claims_old_instruction_verified(self):
        import json
        result=parse_candidate_verification(json.dumps(self.v))
        self.assertEqual(result['original_match'],'not_checked')
        self.assertEqual(admission(self.q,self.c,result),'accept_rewrite')

    def test_no_silent_resizing(self):
        with self.assertRaises(ValueError):
            clean_overview(Image.new('RGB',(32,32)),Image.new('RGB',(31,32)),np.ones((32,32),bool))

if __name__=='__main__':unittest.main()
