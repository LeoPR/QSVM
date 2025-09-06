"""
Métricas e utilitários para avaliação de qualidade de imagens.

Agora delega o loader de dependências opcionais ao patchkit.optional_deps:
- compute_ssim usa ssim_callable injetado ou optional_deps.get_ssim_func() (lazy cached).
"""
from __future__ import annotations
from typing import Optional, Sequence, Dict, Any, Callable

import numpy as np

from patchkit.optional_deps import get_ssim_func  # loader centralizado

__all__ = ["ImageMetrics"]


def _to_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] in (3, 4):
            rgb = arr[..., :3].astype(np.float64)
            if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1.0:
                r = rgb[..., 0]
                g = rgb[..., 1]
                b = rgb[..., 2]
            else:
                r = rgb[..., 0] / 255.0
                g = rgb[..., 1] / 255.0
                b = rgb[..., 2] / 255.0
            y = 0.299 * r + 0.587 * g + 0.114 * b
            return y
        return np.mean(arr, axis=2)
    raise ValueError("Imagem deve ser 2D (grayscale) ou 3D (H,W,C).")


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        a_clipped = np.clip(a, 0.0, 1.0)
        return (a_clipped * 255.0).round().astype(np.uint8)
    return a.astype(np.uint8)


def _box_blur_channel(channel: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(channel, pad, mode="reflect")
    H, W = channel.shape
    out = np.empty_like(channel)
    for i in range(H):
        for j in range(W):
            window = padded[i : i + k, j : j + k]
            out[i, j] = window.mean()
    return out


def _box_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if kernel_size <= 1:
        return arr
    if arr.ndim == 2:
        return _box_blur_channel(arr, kernel_size)
    elif arr.ndim == 3:
        channels = []
        for c in range(arr.shape[2]):
            channels.append(_box_blur_channel(arr[..., c], kernel_size))
        return np.stack(channels, axis=-1)
    else:
        raise ValueError("Imagem deve ser 2D ou 3D.")


class ImageMetrics:
    """Classe utilitária com métodos estáticos para métricas de qualidade de imagem."""

    @staticmethod
    def compute_psnr(a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None) -> float:
        A = np.asarray(a)
        B = np.asarray(b)
        if A.shape != B.shape:
            raise ValueError("Imagens devem ter a mesma forma para PSNR")

        if np.issubdtype(A.dtype, np.floating):
            Af = np.clip(A, 0.0, 1.0).astype(np.float64)
        else:
            Af = A.astype(np.float64) / 255.0
        if np.issubdtype(B.dtype, np.floating):
            Bf = np.clip(B, 0.0, 1.0).astype(np.float64)
        else:
            Bf = B.astype(np.float64) / 255.0

        mse = np.mean((Af - Bf) ** 2)
        if mse == 0:
            return float("inf")
        if data_range is None:
            data_range = 1.0
        psnr = 10.0 * np.log10((data_range ** 2) / mse)
        return float(psnr)

    @staticmethod
    def compute_ssim(a: np.ndarray, b: np.ndarray, *,
                     data_range: Optional[float] = None,
                     multichannel: Optional[bool] = None,
                     ssim_callable: Optional[Callable] = None,
                     **skimage_kwargs) -> float:
        """
        Calcula SSIM entre duas imagens.

        - ssim_callable: se fornecido, será usado diretamente (evita import interno).
                         Caso contrário, tentamos obter a função via optional_deps.get_ssim_func() (lazy import + cache).
        - Mantemos compatibilidade com versões antigas/novas do skimage (channel_axis vs multichannel).
        """
        if ssim_callable is None:
            ssim_func = get_ssim_func()
        else:
            ssim_func = ssim_callable

        A = np.asarray(a)
        B = np.asarray(b)
        if A.shape != B.shape:
            raise ValueError("Imagens devem ter a mesma forma para SSIM")

        if multichannel is None:
            multichannel = (A.ndim == 3 and A.shape[2] in (3, 4))

        if data_range is None:
            if np.issubdtype(A.dtype, np.floating) or np.issubdtype(B.dtype, np.floating):
                data_range = 1.0
            else:
                data_range = 255.0

        call_kwargs = dict(data_range=data_range, **skimage_kwargs)
        if multichannel:
            try:
                return float(ssim_func(A, B, channel_axis=-1, **call_kwargs))
            except TypeError:
                try:
                    return float(ssim_func(A, B, multichannel=True, **call_kwargs))
                except TypeError as e:
                    raise TypeError(
                        "structural_similarity: assinatura inesperada; channel_axis e multichannel falharam."
                    ) from e
        else:
            try:
                return float(ssim_func(A, B, **call_kwargs))
            except TypeError:
                return float(ssim_func(A, B, multichannel=False, **call_kwargs))

    @staticmethod
    def detect_jpeg_blocking(img: np.ndarray, block_size: int = 8, factor: float = 1.5) -> float:
        arr = np.asarray(img)
        gray = _to_gray(arr)
        if gray.dtype != np.float64:
            if np.issubdtype(gray.dtype, np.floating) and gray.max() <= 1.0:
                grayf = gray.astype(np.float64)
            else:
                grayf = gray.astype(np.float64) / 255.0
        else:
            grayf = gray

        H, W = grayf.shape
        if H < block_size or W < block_size:
            return 0.0

        vert_diffs = []
        for x in range(block_size, W, block_size):
            col_diff = np.abs(grayf[:, x] - grayf[:, x - 1])
            vert_diffs.append(np.mean(col_diff))
        hor_diffs = []
        for y in range(block_size, H, block_size):
            row_diff = np.abs(grayf[y, :] - grayf[y - 1, :])
            hor_diffs.append(np.mean(row_diff))

        if len(vert_diffs) == 0 and len(hor_diffs) == 0:
            return 0.0

        mean_block_border = 0.0
        count = 0
        if vert_diffs:
            mean_block_border += np.mean(vert_diffs)
            count += 1
        if hor_diffs:
            mean_block_border += np.mean(hor_diffs)
            count += 1
        mean_block_border = mean_block_border / count

        grad_h = np.mean(np.abs(grayf[:, 1:] - grayf[:, :-1]))
        grad_v = np.mean(np.abs(grayf[1:, :] - grayf[:-1, :]))
        global_grad = 0.5 * (grad_h + grad_v)

        if global_grad <= 1e-12:
            return float(mean_block_border)

        blockiness_score = float(mean_block_border / (global_grad * factor))
        return blockiness_score

    @staticmethod
    def is_blocky(img: np.ndarray,
                  block_size: int = 8,
                  factor: float = 1.5,
                  threshold: Optional[float] = None,
                  adaptive: bool = True,
                  adaptive_factor: float = 1.5,
                  smooth_kernel: int = 3) -> bool:
        score = ImageMetrics.detect_jpeg_blocking(img, block_size=block_size, factor=factor)
        if threshold is not None:
            return float(score) >= float(threshold)

        if adaptive:
            smoothed = _box_blur(img, kernel_size=max(3, int(smooth_kernel)))
            score_smooth = ImageMetrics.detect_jpeg_blocking(smoothed, block_size=block_size, factor=factor)
            eps = 1e-12
            ratio = score / (score_smooth + eps)
            return float(ratio) >= float(adaptive_factor)
        else:
            return float(score) >= float(adaptive_factor)

    @staticmethod
    def compare(a: np.ndarray, b: np.ndarray,
                metrics: Optional[Sequence[str]] = None,
                ssim_kwargs: Optional[dict] = None,
                ssim_callable: Optional[Callable] = None) -> Dict[str, Any]:
        if metrics is None:
            metrics = ("psnr", "ssim", "blockiness")
        metrics = tuple(metrics)
        result: Dict[str, Any] = {}
        if "psnr" in metrics:
            result["psnr"] = ImageMetrics.compute_psnr(a, b)
        if "blockiness" in metrics or "is_blocky" in metrics:
            blockiness = ImageMetrics.detect_jpeg_blocking(a)
            result["blockiness"] = blockiness
            result["is_blocky"] = ImageMetrics.is_blocky(a)
        if "ssim" in metrics:
            ssim_kwargs = ssim_kwargs or {}
            try:
                result["ssim"] = ImageMetrics.compute_ssim(a, b, ssim_callable=ssim_callable, **ssim_kwargs)
                result["ssim_available"] = True
            except ImportError:
                result["ssim"] = None
                result["ssim_available"] = False
            except ValueError:
                result["ssim"] = None
                result["ssim_available"] = True
        return result