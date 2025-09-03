import pytest
import numpy as np
import torch
from torchvision import transforms
from types import SimpleNamespace

# --- Compatibilidade com Pillow: shim para Image.fromarray(mode=...) deprecated ---
# Evita DeprecationWarning quando testes usam Image.fromarray(..., mode=...).
try:
    from PIL import Image as PILImage  # noqa: N812
    _pil_fromarray_orig = PILImage.fromarray
    def _pil_fromarray_compat(obj, mode=None):
        # If mode is None, call original
        if mode is None:
            return _pil_fromarray_orig(obj)
        # Ensure numpy array for processing
        arr = obj
        if not isinstance(arr, np.ndarray):
            return _pil_fromarray_orig(arr)
        # Handle common modes used in tests: 'L' and 'RGB'
        if mode == 'L':
            # make sure uint8 2D array
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr2 = arr.squeeze(2)
            else:
                arr2 = arr
            if arr2.dtype != np.uint8:
                # If float in [0,1], scale; else cast safely
                if np.issubdtype(arr2.dtype, np.floating):
                    arr2 = (np.clip(arr2, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr2 = arr2.astype(np.uint8)
            return _pil_fromarray_orig(arr2)
        elif mode == 'RGB':
            # ensure shape (H,W,3) and dtype uint8
            arr2 = arr
            if arr2.ndim == 2:
                arr2 = np.stack([arr2, arr2, arr2], axis=-1)
            if arr2.shape[-1] != 3:
                # try to broadcast or truncate/expand channels
                if arr2.ndim == 3 and arr2.shape[2] > 3:
                    arr2 = arr2[..., :3]
                else:
                    # fallback: stack or tile last channel
                    arr2 = np.stack([arr2[..., 0]]*3, axis=-1) if arr2.ndim == 3 else np.stack([arr2]*3, axis=-1)
            if arr2.dtype != np.uint8:
                if np.issubdtype(arr2.dtype, np.floating):
                    arr2 = (np.clip(arr2, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr2 = arr2.astype(np.uint8)
            return _pil_fromarray_orig(arr2)
        else:
            # Unknown mode: try to coerce dtype safely then call original without mode
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    arr2 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr2 = arr.astype(np.uint8)
                return _pil_fromarray_orig(arr2)
            return _pil_fromarray_orig(arr)
    # Patch PIL.Image.fromarray for the test run
    PILImage.fromarray = _pil_fromarray_compat  # type: ignore
except Exception:
    # If PIL not available here, tests that require it will skip later.
    pass

# --- Fixtures compartilhadas disponíveis para toda a suíte de testes ---


@pytest.fixture
def tiny_synthetic():
    """
    Fábrica para criar pequenos datasets sintéticos.
    Uso:
        ds = tiny_synthetic(n=3, size=(28,28), pattern="gradient")
    Retorna um torch.utils.data.Dataset com itens (tensor, label).
    """
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, tensors, labels):
            self._tensors = tensors
            self._labels = labels
        def __len__(self):
            return len(self._tensors)
        def __getitem__(self, idx):
            return self._tensors[idx], self._labels[idx]

    def _factory(n=3, size=(28,28), pattern="gradient"):
        H, W = size
        imgs = []
        labels = []
        for i in range(n):
            if pattern == "gradient":
                arr = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H,1))
            elif pattern == "checkerboard" or pattern == "checker":
                xx, yy = np.meshgrid(np.arange(W), np.arange(H))
                arr = ((xx // 4 + yy // 4) % 2).astype(np.float32)
            elif pattern == "binary":
                rng = np.random.RandomState(123 + i)
                arr = (rng.rand(H, W) > 0.7).astype(np.float32)
            else:
                rng = np.random.RandomState(123 + i)
                arr = rng.rand(H, W).astype(np.float32)
            t = torch.from_numpy(arr).unsqueeze(0).to(torch.float32)  # (1,H,W)
            imgs.append(t)
            labels.append(i % 10)
        return SimpleDataset(imgs, labels)
    return _factory


@pytest.fixture(scope="session")
def mnist_cache_dir(tmp_path_factory):
    p = tmp_path_factory.getbasetemp() / "mnist_cache"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


@pytest.fixture
def mnist_sample(mnist_cache_dir):
    """
    Factory returning a small Subset of MNIST. Pula o teste se torchvision/MNIST não puder ser obtido.
    Uso: ds = mnist_sample(n=6, train=False)
    """
    def _factory(n=6, train=False, transform=None):
        try:
            from torchvision.datasets import MNIST
        except Exception:
            pytest.skip("torchvision MNIST não disponível")
        if transform is None:
            transform = transforms.ToTensor()
        try:
            ds = MNIST(root=mnist_cache_dir, train=train, download=True, transform=transform)
        except Exception as e:
            pytest.skip(f"Não foi possível obter MNIST: {e}")
        if n <= 0:
            pytest.skip("mnist_sample requires n>0")
        from torch.utils.data import Subset
        return Subset(ds, list(range(min(n, len(ds)))))
    return _factory


@pytest.fixture
def project_dirs(tmp_path):
    """
    Fixture que fornece diretórios de projeto. Retorna um objeto que suporta
    acesso por atributo (project_dirs.data) e por índice (project_dirs['data']).
    """
    class ProjectDirs(dict):
        def __init__(self, root, cache, data):
            super().__init__(root=str(root), cache=str(cache), data=str(data))
            self.root = str(root)
            self.cache = str(cache)
            self.data = str(data)
        # __getitem__ já presente em dict
        # manter compatibilidade de atributo via __getattr__ (opcional)
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    root = tmp_path / "project_root"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    return ProjectDirs(root=root, cache=cache, data=data)