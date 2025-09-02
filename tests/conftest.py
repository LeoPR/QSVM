import shutil
import pytest
import torch
from pathlib import Path

@pytest.fixture(scope="session")
def project_dirs():
    """
    Cria e garante que os diretórios padrão do projeto existem.
    Útil para testes de integração que precisam dos caminhos reais.
    """
    dirs = {
        'data': Path(".data"),
        'cache': Path(".cache"),
        'output': Path("outputs")
    }

    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)

    return dirs

@pytest.fixture
def cache_dir(tmp_path):
    """
    Pasta temporária para caches/outputs dos testes.
    Use str(cache_dir) ao passar para APIs que esperam path string.
    """
    d = tmp_path / "cache"
    d.mkdir()
    yield str(d)
    try:
        shutil.rmtree(str(d))
    except Exception:
        pass


@pytest.fixture
def tiny_synthetic():
    """
    Tiny synthetic dataset reutilizável (grayscale tensor [1,H,W]).
    CORRIGIDO: Não mais loop infinito
    """

    class TinySynthetic(torch.utils.data.Dataset):
        def __init__(self, n=8, size=(28, 28), pattern='gradient'):
            self.n = n
            self.size = size
            self.pattern = pattern
            # Pre-generate all items to avoid infinite loops
            self._items = self._generate_all_items()

        def _generate_all_items(self):
            """Generate all items upfront"""
            items = []
            H, W = self.size

            for idx in range(self.n):  # Fixed: iterate over n, not infinite
                if self.pattern == 'gradient':
                    y = torch.linspace(0, 1, steps=H).view(H, 1).repeat(1, W)
                    x = torch.linspace(0, 1, steps=W).view(1, W).repeat(H, 1)
                    img = 0.5 * y + 0.5 * x
                elif self.pattern == 'checkerboard':
                    img = torch.zeros(H, W)
                    img[::2, ::2] = 1.0
                    img[1::2, 1::2] = 1.0
                elif self.pattern == 'circle':
                    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                    center_y, center_x = H // 2, W // 2
                    radius = min(H, W) // 4
                    dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                    img = (dist <= radius).float()
                else:
                    img = torch.rand(H, W)

                img = img.clamp(0, 1).unsqueeze(0)  # [1,H,W]
                label = idx % 10
                items.append((img, label))

            return items

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            if idx >= self.n:  # Explicit bounds check
                raise IndexError(f"Index {idx} out of bounds for dataset of size {self.n}")
            return self._items[idx]

    return TinySynthetic