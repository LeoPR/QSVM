import numpy as np
from patchkit import patches

def test_generate_patches_basic():
    img = np.ones((8, 8))
    result = patches.generate_patches(img, patch_size=(2, 2), stride=2)
    assert result.shape[1:] == (2, 2)
    assert result.shape[0] > 0

def test_generate_patches_stride_edge_case():
    img = np.zeros((4, 4))
    result = patches.generate_patches(img, patch_size=(2, 2), stride=4)
    assert result.shape[0] == 1

def test_generate_patches_invalid_input():
    img = np.ones((2, 2))
    try:
        patches.generate_patches(img, patch_size=(4, 4), stride=1)
    except ValueError:
        assert True
    else:
        assert False