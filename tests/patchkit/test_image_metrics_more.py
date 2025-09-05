import builtins
import numpy as np
import pytest
import importlib

from patchkit.image_metrics import ImageMetrics


def test_psnr_uint8_vs_float_equivalence():
    """PSNR deve ser (praticamente) igual para representações equivalentes uint8 e float [0,1]."""
    rng = np.random.default_rng(1234)
    img_uint8 = rng.integers(0, 256, size=(32, 24, 3), dtype=np.uint8)
    img_float = (img_uint8.astype(np.float64) / 255.0).copy()

    psnr_uint8 = ImageMetrics.compute_psnr(img_uint8, img_uint8)
    psnr_float = ImageMetrics.compute_psnr(img_float, img_float)

    # idênticas -> mse 0 -> psnr = inf
    assert psnr_uint8 == float("inf")
    assert psnr_float == float("inf")

    # pequenas diferenças: perturbar uma cópia e comparar valores (deve ser finito)
    img_uint8_2 = img_uint8.copy()
    # garantir operação segura sem overflow convertendo para int antes da soma
    img_uint8_2[0, 0, 0] = (int(img_uint8_2[0, 0, 0]) + 10) % 256
    img_float_2 = (img_uint8_2.astype(np.float64) / 255.0)

    p1 = ImageMetrics.compute_psnr(img_uint8, img_uint8_2)
    p2 = ImageMetrics.compute_psnr(img_float, img_float_2)
    # Valores próximos (tolerância pequena)
    assert np.isfinite(p1) and np.isfinite(p2)
    assert pytest.approx(p1, rel=1e-6) == p2


def test_psnr_grayscale_and_rgb_consistent():
    """PSNR deve funcionar para grayscale e RGB; conversões não deveriam gerar erro."""
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(20, 20), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    psnr_gray = ImageMetrics.compute_psnr(gray, gray)
    psnr_rgb = ImageMetrics.compute_psnr(rgb, rgb)
    assert psnr_gray == float("inf")
    assert psnr_rgb == float("inf")


def test_ssim_multichannel_and_dtype_handling():
    """SSIM multichannel: teste básico; pular se skimage não estiver instalado."""
    pytest.importorskip("skimage")  # pula automática se skimage ausente

    rng = np.random.default_rng(42)
    a = rng.random((16, 16, 3)).astype(np.float64)
    b = a.copy()
    # idênticas -> SSIM == 1.0 (aprox)
    s = ImageMetrics.compute_ssim(a, b)
    assert pytest.approx(1.0, rel=1e-9) == s

    # alterar um pixel e checar que SSIM cai (menor que 1)
    b[0, 0, 0] = b[0, 0, 0] * 0.5
    s2 = ImageMetrics.compute_ssim(a, b)
    assert s2 < s


def test_detect_jpeg_blocking_rgb_and_threshold():
    """Detecta blockiness também em imagens RGB (conversão para luminância)."""
    # smooth RGB gradiente
    grad = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
    smooth_rgb = np.stack([grad, grad, grad], axis=-1)

    # blocky RGB: blocos 4x4 com valores constantes
    blocky = np.zeros((8, 8, 3), dtype=np.float64)
    values = [0.0, 0.5, 1.0, 0.25]
    idx = 0
    for by in range(0, 8, 4):
        for bx in range(0, 8, 4):
            blocky[by:by+4, bx:bx+4, :] = values[idx % len(values)]
            idx += 1

    score_smooth = ImageMetrics.detect_jpeg_blocking(smooth_rgb, block_size=4)
    score_blocky = ImageMetrics.detect_jpeg_blocking(blocky, block_size=4)

    assert score_blocky >= score_smooth
    assert (score_blocky - score_smooth) >= 1e-8


def test_compute_ssim_raises_informative_when_skimage_missing(monkeypatch):
    """
    Simula ausência do pacote skimage e verifica que compute_ssim lança ImportError
    com mensagem informativa.
    """
    # Guardar referência ao import original
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Se tentar importar algo da árvore 'skimage', simular ImportError
        if name == "skimage" or (isinstance(name, str) and name.startswith("skimage.")):
            raise ImportError("mocked missing skimage")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Now call compute_ssim and expect ImportError
    with pytest.raises(ImportError) as excinfo:
        ImageMetrics.compute_ssim(np.zeros((4, 4)), np.zeros((4, 4)))
    msg = str(excinfo.value)
    assert "scikit-image" in msg or "skimage" in msg

    # monkeypatch ensures original import is restored at teardown