"""
Utilitários compartilhados para exemplos em examples/patchs.

Funções incluídas (módulo pensado para importação direta nos scripts de exemplo):
- ensure_active(result, patches): normaliza retorno de filter_active_patches.
- to_uint8_img(t): converte tensor/ndarray para PIL.Image ('L' ou 'RGB').
- to_uint8_img_tensor(t): variação que aceita tensor [C,H,W] ou [H,W] e retorna PIL.Image.
- normalize_patches_from_extractor(raw): normaliza saída do OptimizedPatchExtractor para [N,C,ph,pw] float [0,1].

Não altera lógica de nenhum script; apenas reduz duplicação.
"""

from typing import Any
import numpy as _np
import torch as _t
from PIL import Image
import torchvision.transforms.functional as TF


def ensure_active(result: Any, patches: Any):
    """
    Normaliza o retorno de filter_active_patches.
    - Se `result` for tupla (e.g. (indices, scores) ou (patches, scores)), usa result[0].
    - Se esse elemento for um vetor 1D de índices, indexa `patches` retornando os patches ativos.
    - Caso contrário, retorna o candidato (que pode ser já um array/tensor de patches).

    Retorna: array/tensor com os patches ativos (ou o valor original se não puder indexar).
    """
    cand = result[0] if isinstance(result, tuple) else result

    # Se cand é tensor 1D de índices
    if isinstance(cand, _t.Tensor) and cand.ndim == 1:
        try:
            idxs = cand.long().cpu().numpy().astype(int)
            return patches[idxs]
        except Exception:
            return cand

    # Se cand é lista/ndarray 1D de índices
    if isinstance(cand, (list, tuple, _np.ndarray)):
        arr = _np.asarray(cand)
        if arr.ndim == 1:
            try:
                idxs = arr.astype(int)
                return patches[idxs]
            except Exception:
                return cand

    # Caso já seja uma coleção de patches (torch tensor ou ndarray)
    return cand


def to_uint8_img(t: Any) -> Image.Image:
    """
    Converte tensor [C,H,W] (C=1 ou 3) ou ndarray/HxW para PIL.Image em modo 'L' ou 'RGB'.
    - Aceita torch.Tensor float em [0,1] ou uint8 0..255.
    - Aceita numpy arrays com shapes (H,W), (C,H,W) ou (H,W,C).
    """
    # Torch tensor
    if isinstance(t, _t.Tensor):
        arr = t.detach().cpu()
        # se float em [0,1], escalar
        if arr.dtype.is_floating_point:
            arr = (arr.clamp(0, 1) * 255.0).round().to(_t.uint8)
        else:
            arr = arr.to(_t.uint8)
        arr = arr.numpy()
        # C,H,W -> H,W,C ou H,W
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            if arr.shape[0] == 1:
                return Image.fromarray(arr.squeeze(0), mode="L")
            return Image.fromarray(_np.moveaxis(arr, 0, 2), mode="RGB")
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        raise ValueError(f"to_uint8_img: formato tensor inesperado: {arr.shape}")

    # numpy array
    if isinstance(t, _np.ndarray):
        arr = t
        if arr.dtype != _np.uint8:
            # assume float em [0,1] ou valores maiores -> normaliza/clipa e converte
            arr = (_np.clip(arr, 0.0, 1.0) * 255.0).round().astype(_np.uint8) if _np.issubdtype(arr.dtype, _np.floating) else arr.astype(_np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3:
            # pode ser C,H,W ou H,W,C -> detectar
            if arr.shape[0] in (1, 3):
                if arr.shape[0] == 1:
                    return Image.fromarray(arr.squeeze(0), mode="L")
                return Image.fromarray(_np.moveaxis(arr, 0, 2), mode="RGB")
            if arr.shape[2] in (1, 3):
                if arr.shape[2] == 1:
                    return Image.fromarray(arr.squeeze(2), mode="L")
                return Image.fromarray(arr, mode="RGB")
        raise ValueError(f"to_uint8_img: formato ndarray inesperado: {arr.shape}")

    # PIL.Image passthrough
    if isinstance(t, Image.Image):
        return t

    raise TypeError("to_uint8_img: tipo não suportado (esperado torch.Tensor, numpy.ndarray ou PIL.Image)")


def to_uint8_img_tensor(t: Any) -> Image.Image:
    """
    Variante com nome alternativo — mantém compatibilidade com exemplos que chamam
    'to_uint8_img_tensor'. Chama to_uint8_img internamente.
    """
    return to_uint8_img(t)


def normalize_patches_from_extractor(raw: Any) -> _t.Tensor:
    """
    Converte a saída do OptimizedPatchExtractor (que pode ser torch.Tensor, ndarray ou lista de PIL/ndarray)
    para torch.Tensor com shape [N, C, ph, pw] e dtype float32 em [0,1].
    - Mantém comportamento compatível com implementações existentes.
    """
    if isinstance(raw, _t.Tensor):
        t = raw
    else:
        if isinstance(raw, _np.ndarray):
            t = _t.from_numpy(raw)
        else:
            # raw pode ser lista de arrays/PIL; tentar converter cada item via image_utils equivalentes (usa torchvision)
            try:
                lst = []
                for x in raw:
                    if isinstance(x, Image.Image):
                        arr = _np.asarray(x)
                        # garantir formato H,W ou H,W,C
                        if arr.ndim == 2:
                            arr = arr[None, :, :]
                        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
                            arr = _np.moveaxis(arr, 2, 0)
                        lst.append(_t.from_numpy(arr))
                if lst:
                    t = _t.stack(lst, dim=0)
                else:
                    # fallback: tentar converter com numpy.array
                    t = _t.from_numpy(_np.array(raw))
            except Exception as e:
                raise TypeError("normalize_patches_from_extractor: formato de saída do extractor não reconhecido") from e

    # garantir dimensões [N,C,ph,pw]
    if t.ndim == 3:
        t = t.unsqueeze(1)
    if t.ndim == 4:
        if not _t.is_floating_point(t):
            t = t.to(_t.float32) / 255.0
        else:
            t = t.to(_t.float32)
    else:
        raise ValueError("normalize_patches_from_extractor: formato de patches não suportado")
    return t