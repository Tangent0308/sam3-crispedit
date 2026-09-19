import numpy as np
import torch
from importlib.util import find_spec
from unittest import skipIf


@skipIf(find_spec("diffusers") is None, "Run in the image-generation environment")
def test_opt_in_dilation_frees_old_contour_without_freeing_distant_pixels():
    from utils.geometry import region_write_weight

    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[7, 7] = 1
    kwargs = dict(
        latent_hw=(15, 15),
        latent_bboxes=[(7, 8, 7, 8)],
        margin=2,
        device="cpu",
        dtype=torch.float32,
        region_masks=[mask],
    )
    baseline = region_write_weight(**kwargs)
    explicit_zero = region_write_weight(**kwargs, core_dilation=0)
    wider = region_write_weight(**kwargs, core_dilation=2)
    assert torch.equal(baseline, explicit_zero)
    assert baseline[0, 0, 7, 9] < 1
    assert wider[0, 0, 7, 9] == 1
    assert 0 < wider[0, 0, 7, 10] < 1
    assert wider[0, 0, 0, 0] == 0
    assert torch.all(wider >= baseline)
