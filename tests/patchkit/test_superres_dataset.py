import pytest
import torch
from PIL import Image

from patchkit import SuperResPatchDataset

def test_superres_structure(tmp_path, tiny_synthetic):
    """
    Usa a fábrica tiny_synthetic (fixture em conftest.py) em vez de uma
    implementação local equivocada de TinySynthetic.
    """
    base_ds = tiny_synthetic(n=2, size=(28, 28))
    low_cfg = {
        'target_size': (14, 14),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None,
        'quantization_levels': 2,
        'quantization_method': 'uniform'
    }
    high_cfg = {
        'target_size': (28, 28),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None,
        'quantization_levels': None,
        'quantization_method': 'uniform'
    }

    ds = SuperResPatchDataset(
        original_ds=base_ds,
        low_res_config=low_cfg,
        high_res_config=high_cfg,
        small_patch_size=(2, 2),
        large_patch_size=(4, 4),
        stride=1,
        scale_factor=2,
        cache_dir=str(tmp_path / "cache"),
        cache_rebuild=True,
        max_memory_cache=5
    )

    # tamanho total = num_imagens * patches_por_imagem
    assert ds.num_images == len(base_ds)
    assert ds.num_patches_per_image > 0

    # Recupera o primeiro item de patches (X, y) e valida estrutura
    X, y = ds[0]
    label, img_idx, patch_idx, small_patch = X

    assert isinstance(label, int)
    assert isinstance(img_idx, int)
    assert isinstance(patch_idx, int)
    # small_patch pode ser [H,W] ou [C,H,W]; aceita ambos
    assert small_patch.ndim in (2, 3)