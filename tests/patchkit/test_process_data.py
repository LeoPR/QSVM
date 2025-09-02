# Substitui o teste que fazia download do MNIST por uma versão determinística.
import numpy as np
import pytest
from PIL import Image

from patchkit import ProcessedDataset, OptimizedPatchExtractor

def to_uint8_img(t):
    # compatível com tensor torch ou numpy
    import torch
    if hasattr(t, 'dim'):
        # torch tensor
        if t.dim() == 3 and t.shape[0] in (1, 3):
            arr = (t.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            if t.shape[0] == 1:
                return Image.fromarray(arr.squeeze(0), mode="L")
            return Image.fromarray(np.moveaxis(arr, 0, 2), mode="RGB")
        elif t.dim() == 2:
            arr = (t.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L")
    elif isinstance(t, np.ndarray):
        arr = (np.clip(t,0,1) * 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
    raise ValueError(f"Formato de tensor/numpy não suportado: {getattr(t,'shape',None)}")

@pytest.mark.timeout(60)
def test_processed_dataset_and_patch_extraction(tmp_path, tiny_synthetic, cache_dir):
    # usa dataset sintético (não faz download)
    ds = tiny_synthetic(n=3, size=(28,28))
    cfg = {
        'target_size': (28, 28),
        'resize_alg': None,
        'image_format': None,
        'quality': None,
        'quantization_levels': 2,
        'quantization_method': 'uniform'
    }
    pd = ProcessedDataset(ds, cache_dir=cache_dir, cache_rebuild=True, **cfg)
    assert pd.data.shape[0] == 3
    assert pd.data.shape[-2:] == (28, 28)

    # Extrai patches da primeira imagem
    first_img = to_uint8_img(pd.data[0].squeeze(0))
    extractor = OptimizedPatchExtractor(
        patch_size=(4, 4), stride=2, cache_dir=cache_dir,
        image_size=(28, 28), max_memory_cache=10
    )
    patches = extractor.process(first_img, index=0)
    assert patches.shape[1:] == (4, 4)
    assert patches.shape[0] > 0
    assert np.all(np.isin(patches[0].numpy(), [0, 255]))