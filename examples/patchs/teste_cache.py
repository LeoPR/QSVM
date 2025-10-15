"""
Pequeno módulo de wrappers de dataset com cache em camadas:
 - DatasetAdapter: adapta torchvision datasets ou pastas simples para um format comum.
 - QuantizeWrapper: aplica quantização e cacheia.
 - ArtefactWrapper: aplica artefatos (ex.: compressão JPEG) e cacheia.
 - PatchCacheWrapper: extrai patches (p/ usar OptimizedPatchExtractor ou fallback) e cacheia.

Interface: todos suportam __len__() e __getitem__(idx) -> retorna um dicionário com chaves:
  {
    "pil": PIL.Image (modo 'L'),
    "tensor": torch.Tensor shape [C,H,W] float [0,1]
  }
PatchCacheWrapper retorna "patches": torch.Tensor [N,C,ph,pw] além de "pil"/"tensor".

Uso recomendado:
  base = DatasetAdapter(torchvision.datasets.MNIST(...))
  q = QuantizeWrapper(base, levels=16, cache_dir=".cache/quan")
  a = ArtefactWrapper(q, jpeg_quality=50, cache_dir=".cache/art")
  p = PatchCacheWrapper(a, patch_size=(2,2), stride=1, cache_dir=".cache/patches")
"""
import os
import io
import json
import hashlib
from typing import Optional, Tuple, Dict, Any
import torch
import torchvision
from PIL import Image, ImageFilter
import numpy as np

# Tenta usar zstd se disponível para compressão de cache; senão fallback para não-comprimido
try:
    import zstandard as zstd
    _HAS_ZSTD = True
except Exception:
    _HAS_ZSTD = False

# Config centralizada de saídas/caches
from examples.patchs.config import CACHE_QUAN_TEST_DIR, CACHE_ART_TEST_DIR, CACHE_PATCH_TEST_DIR  # alteração pontual

# utilidades
def _ensure_dir(d: Optional[str]):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def _params_hash(params: Dict[str, Any]) -> str:
    # json determinístico simples
    j = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(j.encode('utf-8')).hexdigest()[:12]

def _save_obj(obj, path: str):
    # salva com torch.save (obj pode ser tensor ou dict com tensors). opcionalmente comprime com zstd
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(data)
        with open(path, 'wb') as f:
            f.write(compressed)
    else:
        with open(path, 'wb') as f:
            f.write(data)

def _load_obj(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    if _HAS_ZSTD:
        try:
            dctx = zstd.ZstdDecompressor()
            data = dctx.decompress(data)
        except Exception:
            pass
    buf = io.BytesIO(data)
    return torch.load(buf, map_location='cpu')


def _pil_to_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format='PNG')  # PNG é lossless e rápido para hashing
    return buf.getvalue()

# Adapter para datasets torchvision ou diretórios (retorna PIL e tensor float [0,1])
class DatasetAdapter:
    def __init__(self, dataset, as_gray: bool = True):
        """
        dataset: pode ser uma instancia de torchvision.datasets (que retorna tensor) ou um iterável de paths.
        as_gray: converte para 'L' (single channel) para este exemplo.
        """
        self._src = dataset
        self.as_gray = as_gray

    def __len__(self):
        return len(self._src)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._src[idx]
        # torchvision MNIST returns (tensor, label) if dataset
        if isinstance(item, tuple) and len(item) >= 1:
            img = item[0]
            # se é tensor [C,H,W] float -> converter para PIL
            if torch.is_tensor(img):
                pil = torchvision.transforms.functional.to_pil_image(img)
            elif isinstance(img, Image.Image):
                pil = img
            else:
                # assume path-like string
                pil = Image.open(img).convert('L' if self.as_gray else 'RGB')
        elif isinstance(item, Image.Image):
            pil = item
        elif isinstance(item, str):
            pil = Image.open(item).convert('L' if self.as_gray else 'RGB')
        else:
            raise TypeError("Formato de dataset não suportado pelo DatasetAdapter")
        if self.as_gray:
            pil = pil.convert('L')
        tensor = torchvision.transforms.functional.to_tensor(pil).float()
        return {"pil": pil, "tensor": tensor}

# Quantize wrapper
class QuantizeWrapper:
    def __init__(self, source, levels: int = 256, cache_dir: Optional[str] = None):
        """
        source: dataset-like (implementando __len__ e __getitem__(idx) -> dict with 'pil'/'tensor')
        levels: número de níveis (ex: 256, 16, 2)
        cache_dir: se fornecido, salva versões quantizadas aqui
        """
        self.src = source
        self.levels = int(levels)
        self.cache_dir = cache_dir
        if cache_dir:
            _ensure_dir(cache_dir)

    def __len__(self):
        return len(self.src)

    def _cache_path(self, idx: int, pil: Image.Image) -> str:
        params = {"levels": self.levels}
        h = _params_hash(params)
        img_hash = _hash_bytes(_pil_to_bytes(pil))
        return os.path.join(self.cache_dir, f"q_idx{idx}_p{img_hash}_{h}.pt") if self.cache_dir else ""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.src[idx]
        pil = base["pil"]
        tensor = base["tensor"]
        if self.cache_dir:
            path = self._cache_path(idx, pil)
            if os.path.exists(path):
                try:
                    out = _load_obj(path)
                    return out
                except Exception:
                    pass
        # quantize tensor (assume single channel)
        # tensor em [C,H,W] float
        t = tensor.clone()
        # map [0,1] -> discrete levels 0..levels-1 -> back to normalized float
        levels = self.levels
        if levels <= 1:
            q = (t > 0.5).float()
        else:
            q = torch.floor(t * (levels - 1) + 0.5) / float(levels - 1)
        out = {"pil": Image.fromarray((q.squeeze(0).mul(255).round().byte().cpu().numpy())), "tensor": q}
        if self.cache_dir:
            try:
                _save_obj(out, path)
            except Exception:
                pass
        return out

# Artefacts wrapper (ex.: JPEG compression)
class ArtefactWrapper:
    def __init__(self, source, jpeg_quality: Optional[int] = None, gaussian_radius: Optional[float] = None, cache_dir: Optional[str] = None):
        """
        Aplica artefatos à imagem de origem (source).
        - jpeg_quality: se definido, aplica compressão JPEG in-memory e reabre.
        - gaussian_radius: se definido, aplica GaussianBlur com raio indicado.
        cache_dir: onde salvar versões com artefatos.
        """
        self.src = source
        self.jpeg_quality = jpeg_quality
        self.gaussian_radius = gaussian_radius
        self.cache_dir = cache_dir
        if cache_dir:
            _ensure_dir(cache_dir)

    def __len__(self):
        return len(self.src)

    def _cache_path(self, idx: int, pil: Image.Image) -> str:
        params = {"jpeg": self.jpeg_quality, "gauss": self.gaussian_radius}
        h = _params_hash(params)
        img_hash = _hash_bytes(_pil_to_bytes(pil))
        return os.path.join(self.cache_dir, f"art_idx{idx}_p{img_hash}_{h}.pt") if self.cache_dir else ""

    def _apply_jpeg(self, pil: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert('L')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.src[idx]
        pil = base["pil"]
        tensor = base["tensor"]
        if self.cache_dir:
            path = self._cache_path(idx, pil)
            if os.path.exists(path):
                try:
                    out = _load_obj(path)
                    return out
                except Exception:
                    pass
        pil2 = pil.copy()
        if self.jpeg_quality is not None:
            pil2 = self._apply_jpeg(pil2, self.jpeg_quality)
        if self.gaussian_radius is not None:
            pil2 = pil2.filter(ImageFilter.GaussianBlur(self.gaussian_radius))
        t2 = torchvision.transforms.functional.to_tensor(pil2).float()
        out = {"pil": pil2, "tensor": t2}
        if self.cache_dir:
            try:
                _save_obj(out, path)
            except Exception:
                pass
        return out

# PatchCacheWrapper: extrai patches a partir da imagem transformada e cacheia os patches
class PatchCacheWrapper:
    def __init__(self, source, patch_size: Tuple[int, int] = (2,2), stride: int = 1, cache_dir: Optional[str] = None, use_extractor=True):
        """
        source: dataset-like (pil/tensor)
        patch_size: tuple (h,w)
        stride: stride
        cache_dir: onde salvar patches extraídos
        use_extractor: se True, tenta usar patchkit.patches.OptimizedPatchExtractor se disponível; caso contrário usa unfold local.
        """
        self.src = source
        self.patch_size = patch_size
        self.stride = stride
        self.cache_dir = cache_dir
        self.use_extractor = use_extractor
        if cache_dir:
            _ensure_dir(cache_dir)
        # tentar importar extractor
        try:
            from patchkit.patches import OptimizedPatchExtractor
            self._Extractor = OptimizedPatchExtractor
        except Exception:
            self._Extractor = None

    def __len__(self):
        return len(self.src)

    def _cache_path(self, idx: int, pil: Image.Image) -> str:
        params = {"ps": self.patch_size, "stride": self.stride}
        h = _params_hash(params)
        img_hash = _hash_bytes(_pil_to_bytes(pil))
        return os.path.join(self.cache_dir, f"patch_idx{idx}_p{img_hash}_{h}.pt") if self.cache_dir else ""

    def _extract_local(self, pil: Image.Image):
        # extrai patches usando torchvision unfold: retorna [N, C, ph, pw] float [0,1]
        t = torchvision.transforms.functional.to_tensor(pil).float()
        if t.dim() == 2:
            t = t.unsqueeze(0)
        C, H, W = t.shape
        ph, pw = self.patch_size
        # pad? assumimos compatibilidade
        patches = t.unfold(1, ph, self.stride).unfold(2, pw, self.stride)
        n_rows, n_cols = patches.shape[1], patches.shape[2]
        patches = patches.permute(1,2,0,3,4).contiguous().view(n_rows * n_cols, C, ph, pw)
        return patches

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.src[idx]
        pil = base["pil"]
        tensor = base["tensor"]
        if self.cache_dir:
            path = self._cache_path(idx, pil)
            if os.path.exists(path):
                try:
                    out = _load_obj(path)
                    return out
                except Exception:
                    pass
        # extrair patches (prefer extractor se ativado)
        patches = None
        if self.use_extractor and self._Extractor is not None:
            try:
                # usar extractor.process com index=None para evitar cache colidindo com index fixo
                ex = self._Extractor(patch_size=self.patch_size, stride=self.stride, cache_dir=self.cache_dir or None, image_size=(pil.height, pil.width))
                raw = ex.process(pil, index=None)
                # normalizar caso necessário: aceitar [L,H,W] ou [L,C,H,W]
                import torch as _t
                if isinstance(raw, _t.Tensor):
                    r = raw
                else:
                    # se extractor retornou lista/ndarray, tentar converter
                    try:
                        import numpy as _np
                        if isinstance(raw, _np.ndarray):
                            r = _t.from_numpy(raw)
                        else:
                            # fallback: build stack
                            lst = [torchvision.transforms.functional.to_tensor(x) for x in raw]
                            r = _t.stack(lst, dim=0)
                    except Exception:
                        r = None
                if r is not None:
                    # garantir float [0,1]
                    if not torch.is_floating_point(r):
                        r = r.float() / 255.0
                    patches = r
            except Exception:
                patches = None
        if patches is None:
            patches = self._extract_local(pil)  # float [0,1]
        out = {"pil": pil, "tensor": tensor, "patches": patches}
        if self.cache_dir:
            try:
                _save_obj(out, path)
            except Exception:
                pass
        return out

# exemplo de uso simples (para você testar)
if __name__ == "__main__":  # teste rápido
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    base = DatasetAdapter(ds)
    q = QuantizeWrapper(base, levels=16, cache_dir=CACHE_QUAN_TEST_DIR)   # alteração pontual
    a = ArtefactWrapper(q, jpeg_quality=50, cache_dir=CACHE_ART_TEST_DIR) # alteração pontual
    p = PatchCacheWrapper(a, patch_size=(2,2), stride=1, cache_dir=CACHE_PATCH_TEST_DIR, use_extractor=False)  # alteração pontual
    item = p[0]
    print("len patches:", item["patches"].shape)