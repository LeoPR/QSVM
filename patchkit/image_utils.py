"""
Utilitários leves para conversões PIL <-> torch.Tensor e resize com dois backends:
- backend='pil'  : usa Pillow (comportamento atual)
- backend='torch': usa torch.nn.functional.interpolate (tensor-based, batched)

Objetivo: oferecer uma camada de compatibilidade de baixo risco. Não altera comportamento
se for usado com backend='pil' por padrão.
"""
from typing import Tuple, Union
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

_to_pil = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


def to_pil(x: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    """Garante que o resultado seja PIL.Image.
    - aceita PIL.Image (retorna igual)
    - aceita torch.Tensor (C,H,W) ou (H,W) float in [0,1] or uint8 [0,255]
    - aceita numpy.ndarray (H,W) or (H,W,C)
    """
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, torch.Tensor):
        # ToPILImage handles float tensors in [0,1] and uint8 tensors
        return _to_pil(x)
    if isinstance(x, np.ndarray):
        # If float in [0,1], scale to 0-255 for PIL expectation of uint8 optionally
        if x.dtype == np.float32 or x.dtype == np.float64:
            arr = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = x
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported type for to_pil: {type(x)}")


def to_tensor(x: Union[Image.Image, torch.Tensor, np.ndarray], dtype=torch.float32) -> torch.Tensor:
    """Garante que o resultado seja torch.Tensor float32 no formato (C,H,W), com valores em [0,1]."""
    if isinstance(x, torch.Tensor):
        t = x
        # Normalize if uint8
        if t.dtype == torch.uint8:
            t = t.to(dtype).div(255.0)
        else:
            t = t.to(dtype)
        # Ensure shape C,H,W
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            t = torch.from_numpy(x).to(dtype).div(255.0)
        else:
            t = torch.from_numpy(x).to(dtype)
            if t.max() > 1.0:
                t = t / 255.0
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            # HWC -> CHW
            t = t.permute(2, 0, 1)
        return t
    if isinstance(x, Image.Image):
        return _to_tensor(x).to(dtype)
    raise TypeError(f"Unsupported type for to_tensor: {type(x)}")


def _resize_pil(img: Image.Image, size: Tuple[int, int], alg=Image.BICUBIC) -> Image.Image:
    """Resize via Pillow. size is (W, H) or (width, height) accepted by PIL."""
    return img.resize((int(size[0]), int(size[1])), resample=alg)


def _resize_tensor(t: torch.Tensor, size: Tuple[int, int], mode='bilinear') -> torch.Tensor:
    """Resize tensor using torch.nn.functional.interpolate.
    - t: (C,H,W) or (N,C,H,W)
    - size: (H, W) target spatial size
    - returns tensor same dtype (float) scaled in [0,1]
    """
    if t.ndim == 3:
        t = t.unsqueeze(0)  # make batched
        squeeze = True
    elif t.ndim == 4:
        squeeze = False
    else:
        raise ValueError("Tensor must be 3D (C,H,W) or 4D (N,C,H,W)")

    # interpolate expects size=(out_h, out_w)
    out = F.interpolate(t, size=(int(size[0]), int(size[1])), mode=mode, align_corners=(mode != 'nearest' and mode != 'area'))
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
    """Resize com escolha de backend.
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
        # PIL uses (width, height), but we keep (H,W) externally; convert:
        pil_out = _resize_pil(pil, (target_size[1], target_size[0]), alg=alg)
        if return_type == 'pil':
            return pil_out
        return to_tensor(pil_out)
    else:
        # tensor backend: work in tensors
        t = to_tensor(img)
        out = _resize_tensor(t, (int(target_size[0]), int(target_size[1])),
                             mode='nearest' if alg == Image.NEAREST else 'bilinear')
        if return_type == 'tensor':
            return out
        return to_pil(out)