import pytest
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image

# --- Helpers / fixtures usados nos testes ---

@pytest.fixture
def tiny_synthetic():
    """
    Retorna uma fábrica que cria um pequeno dataset (torch.utils.data.Dataset)
    com padrão determinístico para testes rápidos.
    Uso:
        ds = tiny_synthetic(n=3, size=(28,28), pattern="gradient")
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
            elif pattern == "checker":
                # checkerboard of 0/1
                xx, yy = np.meshgrid(np.arange(W), np.arange(H))
                arr = ((xx // 4 + yy // 4) % 2).astype(np.float32)
            elif pattern == "binary":
                # binary image 0/255
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
def sample_pil_images():
    """
    Retorna um dicionário de PIL images de exemplo usados em alguns testes.
    Corrige o uso deprecated de Image.fromarray(..., mode=...) garantindo uint8 dtype.
    """
    images = {}
    H, W = 28, 28

    # grayscale gradient 0..255 uint8
    grad = (np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1)))
    images['gradient'] = Image.fromarray(grad)  # mode inferred as 'L'

    # binary 0/255
    binary_array = (np.indices((H, W)).sum(axis=0) % 2).astype(np.uint8) * 255
    # Antes: Image.fromarray(binary_array, mode='L')  # deprecated
    images['binary'] = Image.fromarray(binary_array)  # dtype uint8 -> mode 'L' inferido

    # RGB sample
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)  # red gradient
    rgb[..., 1] = np.linspace(255, 0, H, dtype=np.uint8)[:, None]  # green gradient
    rgb[..., 2] = 128
    images['rgb'] = Image.fromarray(rgb)  # mode 'RGB'

    return images