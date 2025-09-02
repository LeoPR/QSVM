import shutil
import pytest
import torch

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
    Retorna uma classe/fábrica que pode ser instanciada nos testes.
    """
    class TinySynthetic(torch.utils.data.Dataset):
        def __init__(self, n=8, size=(28, 28)):
            self.n = n
            self.size = size
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            H, W = self.size
            y = torch.linspace(0,1,steps=H).view(H,1).repeat(1,W)
            x = torch.linspace(0,1,steps=W).view(1,W).repeat(H,1)
            img = 0.5*y + 0.5*x
            img = img.clamp(0,1).unsqueeze(0)  # [1,H,W]
            label = idx % 10
            return img, label
    return TinySynthetic