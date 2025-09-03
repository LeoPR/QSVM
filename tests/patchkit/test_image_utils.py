import pytest
import numpy as np
from PIL import Image

import torch

from patchkit.image_utils import resize, to_pil, to_tensor

def _make_gradient_tensor(size=(28,28)):
    H,W = size
    arr = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H,1))
    t = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
    return t

def test_resize_backend_equivalence(tmp_path):
    """
    Teste simples que compara backend 'pil' vs 'torch' para um caso
    determinístico (gradiente). Verifica shapes e que a média absoluta
    das diferenças seja pequena (tolerância relaxada).
    """
    t = _make_gradient_tensor((28,28))  # (1,H,W)
    # resize to 14x14 with PIL backend (returns tensor)
    out_pil = resize(t, target_size=(14,14), alg=Image.BICUBIC, backend='pil', return_type='tensor')
    assert isinstance(out_pil, torch.Tensor)
    assert out_pil.ndim == 3  # (C,H,W)
    assert out_pil.shape[-2:] == (14,14)

    # resize to 14x14 with torch backend
    out_torch = resize(t, target_size=(14,14), alg=Image.BICUBIC, backend='torch', return_type='tensor')
    assert isinstance(out_torch, torch.Tensor)
    assert out_torch.shape == out_pil.shape

    # Compare mean absolute difference (relaxed tolerance)
    mad = float(torch.abs(out_pil - out_torch).mean())
    assert mad < 0.05, f"Mean abs diff between backends too large: {mad}"

    # Round-trip: upsample back to 28 and compare means roughly
    up_pil = resize(out_pil, target_size=(28,28), alg=Image.BICUBIC, backend='pil', return_type='tensor')
    up_torch = resize(out_torch, target_size=(28,28), alg=Image.BICUBIC, backend='torch', return_type='tensor')
    mad2 = float(torch.abs(up_pil - up_torch).mean())
    assert mad2 < 0.05, f"Round-trip mean abs diff too large: {mad2}"