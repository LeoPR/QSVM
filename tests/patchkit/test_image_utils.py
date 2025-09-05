import numpy as np
import pytest

from patchkit.image_utils import to_pil

# Verificações de disponibilidade do PIL e (opcional) torch
try:
    from PIL import Image
except Exception:
    Image = None  # noqa: F401

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False


def _require_pil():
    if Image is None:
        pytest.skip("Pillow (PIL) não disponível")


def _require_torch():
    if not _has_torch:
        pytest.skip("torch não disponível")


def test_to_pil_grayscale_from_float_array():
    """Float array em [0,1] deve ser convertido para imagem L uint8 corretamente."""
    _require_pil()
    H, W = 28, 28
    arr = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape((H, W))
    img = to_pil(arr, mode='L')
    assert isinstance(img, Image.Image)
    assert img.mode == 'L'
    assert img.size == (W, H)
    # verificar que os valores foram escalados para uint8
    arr_out = np.array(img)
    assert arr_out.dtype == np.uint8
    assert arr_out.min() >= 0 and arr_out.max() <= 255


def test_to_pil_grayscale_from_uint8_array():
    """Array uint8 2D deve ser mantido sem alterações estranhas."""
    _require_pil()
    H, W = 16, 12
    arr = (np.random.RandomState(0).randint(0, 256, size=(H, W))).astype(np.uint8)
    img = to_pil(arr, mode='L')
    assert img.mode == 'L'
    assert img.size == (W, H)
    arr_out = np.array(img)
    assert arr_out.dtype == np.uint8
    np.testing.assert_array_equal(arr_out, arr)


def test_to_pil_rgb_from_array():
    """Array HxWx3 uint8 deve virar imagem RGB."""
    _require_pil()
    H, W = 24, 20
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(255, 0, H, dtype=np.uint8)[:, None]
    rgb[..., 2] = 128
    img = to_pil(rgb, mode='RGB')
    assert img.mode == 'RGB'
    assert img.size == (W, H)
    arr_out = np.array(img)
    assert arr_out.shape == (H, W, 3)
    assert arr_out.dtype == np.uint8
    np.testing.assert_array_equal(arr_out[..., 2], 128)


def test_to_pil_replicate_grayscale_to_rgb():
    """Imagem grayscale replicada corretamente em 3 canais quando pedido RGB."""
    _require_pil()
    H, W = 10, 8
    gray = (np.random.RandomState(1).randint(0, 256, size=(H, W))).astype(np.uint8)
    img = to_pil(gray, mode='RGB')
    assert img.mode == 'RGB'
    arr_out = np.array(img)
    assert arr_out.shape == (H, W, 3)
    # todos os canais devem ser iguais (replicação)
    np.testing.assert_array_equal(arr_out[..., 0], arr_out[..., 1])
    np.testing.assert_array_equal(arr_out[..., 1], arr_out[..., 2])


def test_to_pil_from_torch_tensor_chw_and_1chw():
    """Tensors torch no formato CxHxW e 1xHxW são aceitos."""
    _require_pil()
    _require_torch()
    H, W = 14, 14
    # tensor (1,H,W) float em [0,1]
    t1 = torch.linspace(0.0, 1.0, H * W, dtype=torch.float32).reshape(1, H, W)
    img1 = to_pil(t1, mode='L')
    assert img1.mode == 'L'
    assert img1.size == (W, H)
    # tensor (3,H,W) uint8
    t3 = torch.zeros((3, H, W), dtype=torch.uint8)
    t3[0] = torch.arange(W, dtype=torch.uint8).unsqueeze(0).repeat(H, 1)
    t3[2] = 128
    img3 = to_pil(t3, mode='RGB')
    assert img3.mode == 'RGB'
    assert img3.size == (W, H)


def test_to_pil_infer_mode_when_none():
    """Quando mode=None, to_pil deve inferir e não usar parâmetro depreciado."""
    _require_pil()
    H, W = 12, 9
    arr = np.random.RandomState(2).rand( H, W ).astype(np.float32)  # valores em [0,1]
    img = to_pil(arr, mode=None)
    assert isinstance(img, Image.Image)
    # inferido como L (já que arr é 2D), e dtype convertido para uint8
    assert img.mode in ('L',)
    arr_out = np.array(img)
    assert arr_out.dtype == np.uint8


def test_to_pil_handles_unusual_channel_counts():
    """Se o array tiver canais diferentes de 1 ou 3, to_pil tenta ajustar/recortar/replicar."""
    _require_pil()
    H, W = 6, 6
    # 4 canais -> deve truncar para 3
    arr4 = (np.random.RandomState(3).randint(0, 256, size=(H, W, 4))).astype(np.uint8)
    img4 = to_pil(arr4, mode='RGB')
    assert img4.mode == 'RGB'
    assert np.array(img4).shape[2] == 3
    # 2 canais -> replicar/concat para 3 canais (comportamento definido no utilitário)
    arr2 = (np.random.RandomState(4).randint(0, 256, size=(H, W, 2))).astype(np.uint8)
    img2 = to_pil(arr2, mode='RGB')
    assert img2.mode == 'RGB'
    assert np.array(img2).shape[2] == 3