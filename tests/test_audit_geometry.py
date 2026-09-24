import unittest
import numpy as np
from synthesis_pipeline.audit_geometry import addition_contact, apply_contact_gate


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.host=np.zeros((100,100),bool);self.host[40:60,40:60]=True

    def test_contact_not_majority_containment(self):
        added=np.zeros_like(self.host);added[20:45,30:70]=True
        evidence=addition_contact(self.host,added)
        self.assertEqual(evidence['status'],'pass')
        self.assertLess(evidence['overlap_fraction'],.5)

    def test_small_segmentation_gap_tolerated(self):
        added=np.zeros_like(self.host);added[62:66,45:55]=True
        self.assertEqual(addition_contact(self.host,added)['status'],'pass')

    def test_disjoint_host_rejected(self):
        added=np.zeros_like(self.host);added[75:80,45:55]=True
        self.assertEqual(addition_contact(self.host,added)['status'],'fail')

    def test_empty_not_silently_passed(self):
        self.assertEqual(addition_contact(self.host,np.zeros_like(self.host))['status'],'unmeasured')
        with self.assertRaises(ValueError):addition_contact(self.host,np.zeros((90,90)))

    def test_prior_quality_evidence_and_rejections_preserved(self):
        record={'decision':'accept_rewrite','quality':'pass','audit':{'visual_quality':'pass'}}
        result=apply_contact_gate(record,{'status':'fail'})
        self.assertEqual(result['decision'],'reject_geometric_anchor')
        self.assertEqual(result['audit']['visual_quality'],'pass')
        self.assertEqual(record['decision'],'accept_rewrite')
        bad={**record,'decision':'reject_quality','quality':'fail'}
        self.assertEqual(apply_contact_gate(bad,{'status':'fail'})['decision'],'reject_quality')


if __name__=='__main__':unittest.main()
