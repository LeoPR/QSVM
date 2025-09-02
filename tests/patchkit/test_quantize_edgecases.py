import pytest
import torch
import math
from patchkit.quantize import ImageQuantizer

def test_invalid_levels():
    img = torch.rand(1,8,8)
    with pytest.raises(Exception):
        ImageQuantizer.quantize(img, levels=1, method='uniform')

def test_nan_inf_input():
    img = torch.rand(1,8,8)
    img[0,0,0] = float('nan')
    with pytest.raises(Exception):
        ImageQuantizer.quantize(img, levels=2, method='uniform')
    img = torch.rand(1,8,8)
    img[0,0,0] = float('inf')
    with pytest.raises(Exception):
        ImageQuantizer.quantize(img, levels=2, method='uniform')

def test_dithering_option():
    img = torch.rand(1,8,8)
    q1 = ImageQuantizer.quantize(img, levels=4, method='uniform', dithering=False)
    q2 = ImageQuantizer.quantize(img, levels=4, method='uniform', dithering=True)
    # ambos são válidos e no range [0,1]
    assert float(q1.min()) >= 0.0 and float(q1.max()) <= 1.0
    assert float(q2.min()) >= 0.0 and float(q2.max()) <= 1.0