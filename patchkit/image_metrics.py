"""
Métricas e utilitários para avaliação de qualidade de imagens.

Objetivo:
- Fornecer implementações pequenas e reutilizáveis de PSNR (numpy),
  detecção simples de JPEG blocking (numpy) e wrapper para SSIM (skimage quando disponível).
- Não poluir o pacote com dependências opcionais: funções que exigem scikit-image
  só importam/uso quando chamadas; se ausente levantam erro informativo.

Uso:
- from patchkit.image_metrics import ImageMetrics
- psnr = ImageMetrics.compute_psnr(img_a, img_b)
- ssim = ImageMetrics.compute_ssim(img_a, img_b)  # requer scikit-image
- blocked = ImageMetrics.detect_jpeg_blocking(img)

As entradas podem ser numpy arrays HxW (grayscale) ou HxWxC (RGB). Floats em [0,1]
serão automaticamente escalados para uint8 para as métricas que requerem esse formato.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

__all__ = ["ImageMetrics"]


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Converte HxW ou HxWxC para grayscale HxW (float64 em 0..1 se input float, senão uint8)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Assumimos último eixo canais
        if arr.shape[2] == 3 or arr.shape[2] == 4:
            # converter para luminância Y usando coeficientes BT.601 aproximados
            rgb = arr[..., :3].astype(np.float64)
            # Normalizar se float em [0,1]
            if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1.0:
                r = rgb[..., 0]
                g = rgb[..., 1]
                b = rgb[..., 2]
            else:
                # se uint8 ou floats >1 assumimos 0..255
                r = rgb[..., 0] / 255.0
                g = rgb[..., 1] / 255.0
                b = rgb[..., 2] / 255.0
            y = 0.299 * r + 0.587 * g + 0.114 * b
            # retornar no mesmo tipo do input? Usaremos float em [0,1]
            return y
        else:
            # canais não padrão: média simples
            return np.mean(arr, axis=2)
    raise ValueError("Imagem deve ser 2D (grayscale) ou 3D (H,W,C).")


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Garante numpy array uint8 HxW ou HxWxC. Floats em [0,1] são convertidos."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        # assume valores em 0..1 (se estiverem em 0..255, essa função ainda clipa)
        a_clipped = np.clip(a, 0.0, 1.0)
        return (a_clipped * 255.0).round().astype(np.uint8)
    return a.astype(np.uint8)


class ImageMetrics:
    """Classe utilitária com métodos estáticos para métricas de qualidade de imagem."""

    @staticmethod
    def compute_psnr(a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None) -> float:
        """
        Calcula PSNR entre duas imagens.
        - a, b: arrays HxW ou HxWxC, floats em [0,1] ou uint8 [0,255].
        - data_range: faixa dos dados (por exemplo 255.0). Se None, inferimos a partir do dtype.
        Retorna PSNR em decibéis (float).
        """
        A = np.asarray(a)
        B = np.asarray(b)
        if A.shape != B.shape:
            raise ValueError("Imagens devem ter a mesma forma para PSNR")

        # Tratamento simples: converter para float64 em 0..1
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
            data_range = 1.0  # já normalizamos para 0..1
        psnr = 10.0 * np.log10((data_range ** 2) / mse)
        return float(psnr)

    @staticmethod
    def compute_ssim(a: np.ndarray, b: np.ndarray, *,
                     data_range: Optional[float] = None,
                     multichannel: Optional[bool] = None,
                     **skimage_kwargs) -> float:
        """
        Calcula SSIM entre duas imagens usando skimage.metrics.structural_similarity.
        - Requer scikit-image; se não instalado, lança ImportError com instrução.
        - a, b: arrays HxW ou HxWxC.
        - data_range: faixa dos dados; se None, inferimos.
        - multichannel: se None, inferimos a partir da dimensão do array.
        - skimage_kwargs: argumentos adicionais passados para structural_similarity.

        Compatibilidade: a API do skimage mudou (multichannel -> channel_axis).
        Aqui tentamos chamar de forma compatível com versões novas e antigas:
        - se a função suportar channel_axis, chamamos com channel_axis=-1 para imagens com canais.
        - se isso falhar (TypeError), tentamos chamar com o argumento multichannel (API antiga).
        """
        try:
            from skimage.metrics import structural_similarity as ssim_func  # type: ignore
        except Exception as e:
            raise ImportError(
                "compute_ssim requer scikit-image (skimage). Instale com: pip install scikit-image"
            ) from e

        A = np.asarray(a)
        B = np.asarray(b)
        if A.shape != B.shape:
            raise ValueError("Imagens devem ter a mesma forma para SSIM")

        # Inferir multichannel se não fornecido
        if multichannel is None:
            multichannel = (A.ndim == 3 and A.shape[2] in (3, 4))

        # Inferir data_range se não fornecido
        if data_range is None:
            # se floats assumimos 0..1, se inteiros assumimos 0..255
            if np.issubdtype(A.dtype, np.floating) or np.issubdtype(B.dtype, np.floating):
                data_range = 1.0
            else:
                data_range = 255.0

        # Tentar compatibilidade com versões novas (channel_axis) e antigas (multichannel)
        call_kwargs = dict(data_range=data_range, **skimage_kwargs)
        if multichannel:
            # Preferir channel_axis para versões recentes do skimage
            try:
                # channel_axis=-1 indica que o último eixo são canais
                return float(ssim_func(A, B, channel_axis=-1, **call_kwargs))
            except TypeError:
                # fallback para API antiga que aceita 'multichannel'
                try:
                    return float(ssim_func(A, B, multichannel=True, **call_kwargs))
                except TypeError as e:
                    raise TypeError(
                        "structural_similarity signature inesperada; não foi possível "
                        "chamar com channel_axis nem com multichannel."
                    ) from e
        else:
            # imagem grayscale; apenas chamar normalmente
            try:
                return float(ssim_func(A, B, **call_kwargs))
            except TypeError:
                # algumas versões podem exigir explicitamente multichannel=False
                return float(ssim_func(A, B, multichannel=False, **call_kwargs))

    @staticmethod
    def detect_jpeg_blocking(img: np.ndarray, block_size: int = 8, factor: float = 1.5) -> float:
        """
        Heurística simples para detectar blocking artifacts de JPEG.
        - img: array HxW (grayscale) ou HxWxC (converte para luminância).
        - block_size: tamanho do bloco (8).
        - factor: threshold multiplicador para média das diferenças.

        Retorna um valor de 'blockiness' (float). Valores maiores indicam mais blockiness.
        - Para uma decisão booleana, compare com um threshold empiricamente decidido.
        """
        arr = np.asarray(img)
        # converter para grayscale float em 0..1
        gray = _to_gray(arr)
        if gray.dtype != np.float64:
            # normalizar: se já float com max<=1 ok; senão converte
            if np.issubdtype(gray.dtype, np.floating) and gray.max() <= 1.0:
                grayf = gray.astype(np.float64)
            else:
                grayf = gray.astype(np.float64) / 255.0
        else:
            grayf = gray

        H, W = grayf.shape
        if H < block_size or W < block_size:
            # imagem muito pequena — heurística pouco confiável
            return 0.0

        # calcular diferenças verticais e horizontais ao longo das fronteiras de blocos
        vert_diffs = []
        for x in range(block_size, W, block_size):
            # diffs entre coluna x-1 e x
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

        # comparar com média global de diferenças (aleatória) para normalizar
        # calculamos a média absoluta de gradiente (diferenças adjacentes em toda a imagem)
        grad_h = np.mean(np.abs(grayf[:, 1:] - grayf[:, :-1]))
        grad_v = np.mean(np.abs(grayf[1:, :] - grayf[:-1, :]))
        global_grad = 0.5 * (grad_h + grad_v)

        # evitar divisão por zero
        if global_grad <= 1e-12:
            return float(mean_block_border)

        blockiness_score = float(mean_block_border / (global_grad * factor))
        return blockiness_score