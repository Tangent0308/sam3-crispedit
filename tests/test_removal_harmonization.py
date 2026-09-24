"""Run with the Omni interpreter for scipy-backed numerical regression coverage."""
import unittest
import numpy as np
from PIL import Image

try:
    from utils.removal_harmonization import compose_harmonized_removal
    HAS_SCIPY=True
except ModuleNotFoundError as exc:
    if exc.name!='scipy':raise
    HAS_SCIPY=False


@unittest.skipUnless(HAS_SCIPY,'Optional removal harmonization requires scipy in the editor environment')
class HarmonizationTests(unittest.TestCase):
    def scene(self, offset=6):
        source=np.full((96,96,3),80,np.uint8)
        raw=source+offset
        alpha=np.zeros((96,96),np.uint8);alpha[24:72,24:72]=255
        return source,raw,alpha

    def test_constant_offset_removed_without_outside_writes(self):
        source,raw,alpha=self.scene()
        result,_=compose_harmonized_removal(*map(Image.fromarray,(source,raw,alpha)))
        result=np.asarray(result)
        self.assertTrue(np.array_equal(result[alpha==0],source[alpha==0]))
        self.assertLessEqual(np.abs(result.astype(float)[alpha>0]-source[alpha>0]).max(),1)

    def test_excluded_hole_is_exactly_preserved(self):
        source,raw,alpha=self.scene();alpha[40:56,40:56]=0
        source[40:56,40:56]=[15,80,160]
        result,_=compose_harmonized_removal(*map(Image.fromarray,(source,raw,alpha)))
        self.assertTrue(np.array_equal(np.asarray(result)[alpha==0],source[alpha==0]))

    def test_large_offset_is_bounded(self):
        source,raw,alpha=self.scene(30)
        result,_=compose_harmonized_removal(*map(Image.fromarray,(source,raw,alpha)))
        correction=raw.astype(float)[alpha>0]-np.asarray(result)[alpha>0]
        self.assertLessEqual(correction.max(),17)
        self.assertGreater(correction.mean(),10)

    def test_native_boundary_band_corrects_seam_without_core_or_exterior_writes(self):
        from utils.removal_harmonization import correct_boundary_band
        source,raw,alpha=self.scene(8)
        corrected=np.asarray(correct_boundary_band(*map(Image.fromarray,(source,raw,alpha))))
        self.assertTrue(np.array_equal(corrected[alpha==0],raw[alpha==0]))
        self.assertTrue(np.array_equal(corrected[40:56,40:56],raw[40:56,40:56]))
        self.assertLess(np.abs(corrected[24,32:64].astype(float)-source[24,32:64]).mean(),3)

    def test_boundary_band_does_not_import_large_neighbor_color_difference(self):
        from utils.removal_harmonization import correct_boundary_band
        source,raw,alpha=self.scene(0);source[alpha==0]=255
        corrected=np.asarray(correct_boundary_band(*map(Image.fromarray,(source,raw,alpha))))
        self.assertTrue(np.array_equal(corrected,raw))


if __name__=='__main__':unittest.main()
