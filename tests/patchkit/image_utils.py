"""
Helpers para converter numpy arrays / torch tensors em PIL.Image de forma segura,
sem usar o parâmetro 'mode' (depreciado no Pillow).

Uso:
    from tests.patchkit.image_utils import to_pil
    img = to_pil(arr, mode='L')  # ou mode='RGB' ou mode=None para inferir
"""

from typing import Optional, Any
import numpy as np

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

try:
    from PIL import Image
except Exception:  # pragma: no cover - se PIL não estiver disponível, testes que dependem dele devem ser pulados
    Image = None  # type: ignore

def _to_numpy(x: Any) -> np.ndarray:
    """Converte tensor/array para numpy.ndarray sem copiar quando possível."""
    if _has_torch and isinstance(x, torch.Tensor):
        # move para cpu se necessário e converte
        if x.device.type != "cpu":
            x = x.cpu()
        arr = x.detach().numpy()
        return arr
    if isinstance(x, np.ndarray):
        return x
    # tentar converter outros iteráveis
    return np.asarray(x)


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Garante dtype uint8. Se arr for float presumimos intervalo [0,1] e escalamos.
    Se for inteiro, fazemos cast seguro.
    """
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        # pressupõe valores em [0,1], aplica clip por segurança
        arr2 = np.clip(arr, 0.0, 1.0)
        arr2 = (arr2 * 255.0).round().astype(np.uint8)
        return arr2
    # se for inteiro diferente de uint8, fazemos cast direto (pode truncar)
    return arr.astype(np.uint8)


def to_pil(x: Any, mode: Optional[str] = None):
    """
    Converte numpy array ou torch tensor para PIL.Image de forma segura.
    - x: np.ndarray ou torch.Tensor (2D grayscale, 3D HxWxC, ou 3D CxHxW para tensores)
    - mode: 'L', 'RGB' ou None. Se mode for fornecido, tentamos obedecê-lo ajustando shape/dtype.
    Retorna PIL.Image.
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) não está disponível no ambiente")

    arr = _to_numpy(x)

    # Se tensor com shape (C,H,W), converter para (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and (arr.shape[1] != 3):
        # p.ex. (C,H,W) -> (H,W,C)
        arr = np.transpose(arr, (1, 2, 0))

    # Caso comum: tensors com 1xHxW (camada extra)
    if arr.ndim == 3 and arr.shape[2] == 1:
        # tornar (H,W)
        arr2 = arr[:, :, 0]
        arr = arr2

    if mode is None:
        # deixar Pillow inferir, apenas garantir dtype uint8 quando apropriado
        if np.issubdtype(arr.dtype, np.floating):
            arr = _ensure_uint8(arr)
        return Image.fromarray(arr)

    mode = mode.upper()
    if mode == 'L':
        # garantir 2D (H,W) uint8
        if arr.ndim == 3:
            # se (H,W,3) -> converter para grayscale (média ponderada simples)
            if arr.shape[2] == 3:
                # converter para float [0,1] se necessário, depois média e voltar a uint8
                if np.issubdtype(arr.dtype, np.floating):
                    a = np.clip(arr, 0.0, 1.0)
                    gray = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2])
                else:
                    a = arr.astype(np.float32) / 255.0
                    gray = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2])
                arr = _ensure_uint8(gray)
            else:
                # último canal possivelmente 1 -> squeeze
                arr = arr[..., 0]
                arr = _ensure_uint8(arr)
        else:
            arr = _ensure_uint8(arr)
        return Image.fromarray(arr)

    if mode == 'RGB':
        # garantir shape (H,W,3) e dtype uint8
        if arr.ndim == 2:
            # grayscale -> replicate canais
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] != 3:
            # tentar ajustar canais
            if arr.shape[2] > 3:
                arr = arr[..., :3]
            else:
                # menos canais (p.ex. 1), replicar último canal
                arr = np.concatenate([arr[..., :1]] * 3, axis=-1)
        arr = _ensure_uint8(arr)
        return Image.fromarray(arr)

    # Para modos não contemplados, coercionar dtype e chamar fromarray sem mode
    if np.issubdtype(arr.dtype, np.floating):
        arr = _ensure_uint8(arr)
    else:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)