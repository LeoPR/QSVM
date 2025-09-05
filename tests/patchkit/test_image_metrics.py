import numpy as np
import pytest

from patchkit.image_metrics import ImageMetrics


def test_compute_psnr_identical_images():
    """PSNR deve ser infinito para imagens idênticas e finito para imagens diferentes."""
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros_like(a)
    # idênticas -> mse 0 -> psnr = inf
    psnr_identical = ImageMetrics.compute_psnr(a, b)
    assert psnr_identical == float("inf")

    # imagens diferentes -> psnr finito
    b[0, 0] = 255
    psnr_diff = ImageMetrics.compute_psnr(a, b)
    assert np.isfinite(psnr_diff)
    assert psnr_diff > 0.0


def test_compute_ssim_identical_images():
    """
    SSIM requer scikit-image; pular o teste se não instalado.
    Para imagens idênticas, SSIM deve ser 1.0.
    """
    pytest.importorskip("skimage")  # pula o teste se skimage não estiver disponível

    # usar floats em [0,1] também é suportado
    a = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
    b = a.copy()
    s = ImageMetrics.compute_ssim(a, b)
    # estruturas idênticas -> SSIM == 1.0
    assert pytest.approx(1.0, rel=1e-9) == s


def test_detect_jpeg_blocking_relative():
    """
    Verifica que a heurística detect_jpeg_blocking retorna valor maior para imagem com blocos 8x8
    do que para uma imagem suave (mesma resolução).
    """
    # imagem suave (gradiente)
    smooth = np.linspace(0.0, 1.0, 256, dtype=np.float64).reshape(16, 16)

    # imagem com blocos 8x8 distintos
    blocky = np.zeros((16, 16), dtype=np.float64)
    values = [0.0, 0.25, 0.5, 0.75]
    idx = 0
    for by in range(0, 16, 8):
        for bx in range(0, 16, 8):
            blocky[by:by+8, bx:bx+8] = values[idx % len(values)]
            idx += 1

    score_smooth = ImageMetrics.detect_jpeg_blocking(smooth, block_size=8)
    score_blocky = ImageMetrics.detect_jpeg_blocking(blocky, block_size=8)

    # esperamos que a imagem com blocos apresente maior blockiness que a suave
    assert score_blocky >= score_smooth
    # e idealmente bem maior em cenários sintéticos
    assert (score_blocky - score_smooth) >= 1e-6




