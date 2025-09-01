import pytest
import torch
from PIL import Image
import numpy as np

from patchkit import OptimizedPatchExtractor

def test_reconstruct_identity(cache_dir):
    # imagem sintética 28x28 grayscale
    H = W = 28
    grid = np.tile(np.linspace(0,255,W,dtype=np.uint8), (H,1))
    img = Image.fromarray(grid, mode="L")

    extractor = OptimizedPatchExtractor(
        patch_size=(4,4), stride=2,
        cache_dir=cache_dir,  # agora usa cache temporário
        image_size=(H,W), max_memory_cache=2
    )
    patches = extractor.process(img, index=0)
    recon = extractor.reconstruct_image(patches)  # [H,W] uint8

    diff = (recon.numpy().astype(np.int32) - grid.astype(np.int32))
    mae = np.abs(diff).mean()
    assert mae <= 2.0