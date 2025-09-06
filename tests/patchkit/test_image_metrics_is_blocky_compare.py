import builtins
import numpy as np
import pytest

from patchkit import optional_deps
from patchkit.image_metrics import ImageMetrics


def test_is_blocky_boolean_threshold():
    # smooth
    smooth = np.linspace(0.0, 1.0, 256, dtype=np.float64).reshape(16, 16)
    # blocky (8x8 blocks)
    blocky = np.zeros((16, 16), dtype=np.float64)
    vals = [0.0, 0.25, 0.5, 0.75]
    idx = 0
    for by in range(0, 16, 8):
        for bx in range(0, 16, 8):
            blocky[by:by+8, bx:bx+8] = vals[idx % len(vals)]
            idx += 1

    # Em vez de depender de um threshold absoluto (que pode variar com escala),
    # comparamos os scores e então definimos um threshold intermediário.
    score_smooth = ImageMetrics.detect_jpeg_blocking(smooth, block_size=8)
    score_blocky = ImageMetrics.detect_jpeg_blocking(blocky, block_size=8)

    assert score_blocky > score_smooth

    mid_threshold = 0.5 * (score_blocky + score_smooth)
    assert ImageMetrics.is_blocky(blocky, block_size=8, threshold=mid_threshold) is True
    assert ImageMetrics.is_blocky(smooth, block_size=8, threshold=mid_threshold) is False


def test_is_blocky_adaptive_default():
    # Testa o comportamento adaptativo (threshold=None, adaptive=True)
    smooth = np.linspace(0.0, 1.0, 256, dtype=np.float64).reshape(16, 16)
    blocky = np.zeros((16, 16), dtype=np.float64)
    vals = [0.0, 0.25, 0.5, 0.75]
    idx = 0
    for by in range(0, 16, 8):
        for bx in range(0, 16, 8):
            blocky[by:by+8, bx:bx+8] = vals[idx % len(vals)]
            idx += 1

    assert ImageMetrics.is_blocky(blocky, block_size=8, threshold=None, adaptive=True) is True
    assert ImageMetrics.is_blocky(smooth, block_size=8, threshold=None, adaptive=True) is False


def test_is_blocky_small_image_returns_false():
    tiny = np.zeros((4, 4), dtype=np.float64)
    assert ImageMetrics.detect_jpeg_blocking(tiny, block_size=8) == 0.0
    assert ImageMetrics.is_blocky(tiny, block_size=8, threshold=0.01) is False
    # adaptive também deve retornar False para imagem muito pequena
    assert ImageMetrics.is_blocky(tiny, block_size=8, threshold=None, adaptive=True) is False


def test_compare_returns_dict_and_handles_missing_ssim(monkeypatch):
    # Garantir que o cache interno de optional_deps esteja limpo para simular ausência
    optional_deps.clear_optional_cache()

    # Remover qualquer módulo já carregado de 'skimage' (e submódulos) de sys.modules
    import sys
    for mod_name in list(sys.modules.keys()):
        if mod_name == "skimage" or mod_name.startswith("skimage."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    # Bloquear futuras importações de 'skimage' via importlib.import_module (o que o loader usa)
    import importlib

    orig_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "skimage" or (isinstance(name, str) and name.startswith("skimage.")):
            raise ImportError("mocked missing skimage")
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    # Opcional: também fazer find_spec "não encontrar" skimage (coerência com has_package)
    import importlib.util as importlib_util

    orig_find_spec = importlib_util.find_spec

    def fake_find_spec(name, package=None):
        if name == "skimage" or (isinstance(name, str) and name.startswith("skimage.")):
            return None
        return orig_find_spec(name, package)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)

    # Agora, rodar o compare
    a = np.zeros((8, 8), dtype=np.uint8)
    b = a.copy()
    res = ImageMetrics.compare(a, b)
    assert isinstance(res, dict)
    assert "psnr" in res
    assert res["psnr"] == float("inf")
    assert "ssim" in res and res["ssim"] is None
    assert res.get("ssim_available") is False


def test_compare_includes_metrics_with_ssim_present():
    pytest.importorskip("skimage")
    rng = np.random.default_rng(1)
    a = rng.random((16, 16)).astype(np.float64)
    b = a.copy()
    r = ImageMetrics.compare(a, b)
    assert r["psnr"] == float("inf")
    # ssim deve estar presente e ser aproximadamente 1.0
    assert r.get("ssim_available") is True
    assert isinstance(r.get("ssim"), float)