"""
Utilitários leves para conversões PIL <-> torch.Tensor e resize com dois backends:
- backend='pil'  : usa Pillow (comportamento atual via PIL)
- backend='torch': usa torch.nn.functional.interpolate (tensor-based, batched)

Funções principais:
- to_pil(x, mode=None)   -> PIL.Image
- to_tensor(x, dtype=...) -> torch.Tensor (C,H,W) float32 em [0,1]
- resize(img, target_size, alg=Image.BICUBIC, backend='pil', return_type='tensor')

Este módulo é seguro quanto a tipos comuns (PIL.Image, torch.Tensor, numpy.ndarray).
"""
from typing import Tuple, Union, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

__all__ = ["to_pil", "to_tensor", "resize"]

_to_pil = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


def to_pil(x: Union[Image.Image, torch.Tensor, np.ndarray], mode: Optional[str] = None) -> Image.Image:
    """
    Garante que o resultado seja PIL.Image.
    - aceita PIL.Image (retorna igual)
    - aceita torch.Tensor (C,H,W) ou (H,W) float in [0,1] or uint8 [0,255]
    - aceita numpy.ndarray (H,W) or (H,W,C)
    - mode: opcional, 'L' ou 'RGB' para forçar conversão de canal
    """
    if isinstance(x, Image.Image):
        return x

    if isinstance(x, torch.Tensor):
        # ToPILImage lida corretamente com float tensors em [0,1] e uint8
        pil = _to_pil(x)
        if mode is None:
            return pil
        return pil.convert(mode)

    if isinstance(x, np.ndarray):
        arr = x
        # floats assumed in [0,1]
        if np.issubdtype(arr.dtype, np.floating):
            arr_u8 = np.clip(arr, 0.0, 1.0)
            arr_u8 = (arr_u8 * 255.0).round().astype(np.uint8)
        else:
            arr_u8 = arr.astype(np.uint8)

        # If requested mode, coerce channels appropriately
        if mode is not None:
            mode = mode.upper()
            if mode == "L":
                if arr_u8.ndim == 3 and arr_u8.shape[2] == 3:
                    # convert RGB -> L
                    r = arr_u8[..., 0].astype(np.float32) / 255.0
                    g = arr_u8[..., 1].astype(np.float32) / 255.0
                    b = arr_u8[..., 2].astype(np.float32) / 255.0
                    gray = (0.299 * r + 0.587 * g + 0.114 * b)
                    arr_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                elif arr_u8.ndim == 3 and arr_u8.shape[2] == 1:
                    arr_u8 = arr_u8[..., 0]
                # else if 2D, already fine
            elif mode == "RGB":
                # Ensure HxWx3 uint8
                if arr_u8.ndim == 2:
                    arr_u8 = np.stack([arr_u8, arr_u8, arr_u8], axis=-1)
                elif arr_u8.ndim == 3:
                    if arr_u8.shape[2] == 3:
                        pass  # already RGB
                    elif arr_u8.shape[2] > 3:
                        arr_u8 = arr_u8[..., :3]
                    else:
                        # fewer than 3 channels (e.g. 1 or 2): replicate first channel to form RGB
                        arr_u8 = np.concatenate([arr_u8[..., :1]] * 3, axis=-1)

        return Image.fromarray(arr_u8)

    raise TypeError(f"Unsupported type for to_pil: {type(x)}")


def to_tensor(x: Union[Image.Image, torch.Tensor, np.ndarray], dtype=torch.float32) -> torch.Tensor:
    """
    Garante que o resultado seja torch.Tensor float32 no formato (C,H,W), com valores em [0,1].
    Aceita PIL.Image, numpy.ndarray (H,W) ou (H,W,C) ou torch.Tensor (C,H,W) ou (H,W).
    """
    if isinstance(x, torch.Tensor):
        t = x
        # Normalize uint8
        if t.dtype == torch.uint8:
            t = t.to(dtype).div(255.0)
        else:
            t = t.to(dtype)
            # Se tensor numérico tiver máximos >1, assume escala 0..255
            try:
                if t.numel() > 0 and t.max().item() > 1.0:
                    t = t / 255.0
            except Exception:
                # fallback: não alterar
                pass
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t

    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype == np.uint8:
            t = torch.from_numpy(arr).to(dtype).div(255.0)
        elif np.issubdtype(arr.dtype, np.floating):
            # floats: if values >1 assume 0..255, caso contrário 0..1
            maxv = np.nanmax(arr) if arr.size > 0 else 0.0
            if maxv > 1.0:
                t = torch.from_numpy((np.clip(arr, 0.0, 255.0) / 255.0).astype(np.float32)).to(dtype)
            else:
                t = torch.from_numpy(arr.astype(np.float32)).to(dtype)
        else:
            # inteiros diferentes de uint8
            t = torch.from_numpy(arr).to(dtype)
            if t.numel() > 0 and t.max().item() > 1.0:
                t = t / 255.0

        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            # HWC -> CHW
            t = t.permute(2, 0, 1)
        return t

    if isinstance(x, Image.Image):
        # torchvision.transforms.ToTensor garante float32 em [0,1] e shape C,H,W
        return _to_tensor(x).to(dtype)

    raise TypeError(f"Unsupported type for to_tensor: {type(x)}")


def _resize_pil(img: Image.Image, size: Tuple[int, int], alg=Image.BICUBIC) -> Image.Image:
    """Redimensiona via Pillow. size é (W, H) na API do PIL."""
    return img.resize((int(size[0]), int(size[1])), resample=alg)


def _resize_tensor(t: torch.Tensor, size: Tuple[int, int], mode: str = 'bilinear') -> torch.Tensor:
    """
    Redimensiona tensor usando torch.nn.functional.interpolate.
    - t: (C,H,W) ou (N,C,H,W)
    - size: (out_h, out_w)
    - mode: 'nearest' ou 'bilinear'
    - retorna tensor float na faixa [0,1]
    """
    if t.ndim == 3:
        t = t.unsqueeze(0)
        squeeze = True
    elif t.ndim == 4:
        squeeze = False
    else:
        raise ValueError("Tensor must be 3D (C,H,W) or 4D (N,C,H,W)")

    # interpolate expects size=(out_h, out_w)
    # align_corners: use False for bilinear to avoid unexpected coordinate warping
    align_corners = False if mode in ('bilinear', 'bicubic') else None
    out = F.interpolate(t, size=(int(size[0]), int(size[1])), mode=mode, align_corners=align_corners)
    if squeeze:
        out = out.squeeze(0)
    return out


def resize(
    img: Union[Image.Image, torch.Tensor, np.ndarray],
    target_size: Tuple[int, int],
    alg=Image.BICUBIC,
    backend: str = "pil",
    return_type: str = "tensor"
) -> Union[Image.Image, torch.Tensor]:
    """
    Resize com escolha de backend.
    - target_size: (H, W)
    - alg: PIL resample alg when backend='pil' (Image.NEAREST, Image.BILINEAR, ...)
    - backend: 'pil' or 'torch'
    - return_type: 'tensor' or 'pil' (what to return)
    """
    if backend not in ('pil', 'torch'):
        raise ValueError("backend must be 'pil' or 'torch'")
    if return_type not in ('tensor', 'pil'):
        raise ValueError("return_type must be 'tensor' or 'pil'")

    if backend == 'pil':
        pil = to_pil(img)
        # PIL expects (width, height)
        pil_out = _resize_pil(pil, (target_size[1], target_size[0]), alg=alg)
        if return_type == 'pil':
            return pil_out
        return to_tensor(pil_out)
    else:
        t = to_tensor(img)
        mode = 'nearest' if alg == Image.NEAREST else 'bilinear'
        out = _resize_tensor(t, (int(target_size[0]), int(target_size[1])), mode=mode)
        if return_type == 'tensor':
            return out
        return to_pil(out)