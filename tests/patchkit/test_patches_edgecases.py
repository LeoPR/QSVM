import pytest
import numpy as np
from PIL import Image
from patchkit import OptimizedPatchExtractor

def make_image(h,w):
    grid = np.tile(np.linspace(0,255,w,dtype=np.uint8), (h,1))
    return Image.fromarray(grid, mode="L")

def test_stride_larger_than_image(cache_dir):
    img = make_image(8,8)
    extractor = OptimizedPatchExtractor(patch_size=(4,4), stride=16, cache_dir=cache_dir, image_size=(8,8))
    patches = extractor.process(img, index=0)
    # stride maior que imagem deve produzir pelo menos um patch (centro ou padded)
    assert patches.shape[0] >= 1

def test_patch_size_larger_than_image_raises_or_handles(cache_dir):
    img = make_image(4,4)
    extractor = OptimizedPatchExtractor(patch_size=(8,8), stride=1, cache_dir=cache_dir, image_size=(4,4))
    # comportamento aceito: levantar ValueError ou retornar patches com padding.
    try:
        patches = extractor.process(img, index=0)
        assert patches.shape[1:] == (8,8)
    except ValueError:
        assert True

def test_invalid_input_type(cache_dir):
    with pytest.raises(Exception):
        extractor = OptimizedPatchExtractor(patch_size=(4,4), stride=2, cache_dir=cache_dir, image_size=(8,8))
        # passar None como imagem deve falhar
        extractor.process(None, index=0)