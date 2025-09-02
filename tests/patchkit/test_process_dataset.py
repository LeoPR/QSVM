import torch
from torchvision import transforms
import pytest

from patchkit import ProcessedDataset
from PIL import Image

class TinySynthetic(torch.utils.data.Dataset):
    """Dataset sintético pequeno, sem dependência de download, grayscale."""
    def __init__(self, n=8, size=(28,28)):
        self.n = n
        self.size = size
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        # gradiente + ruído leve
        H, W = self.size
        y = torch.linspace(0,1,steps=H).view(H,1).repeat(1,W)
        x = torch.linspace(0,1,steps=W).view(1,W).repeat(H,1)
        img = 0.5*y + 0.5*x
        img = img.clamp(0,1).unsqueeze(0)  # [1,H,W]
        label = idx % 10
        return img, label

@pytest.mark.parametrize("qlevels", [None, 2, 4])
@pytest.mark.parametrize("fmt,qual", [(None,None), ("JPEG",30), ("PNG",None)])
def test_processed_dataset(tmp_path, qlevels, fmt, qual):
    ds = TinySynthetic(n=5, size=(28,28))
    cfg = {
        'target_size': (14,14),
        'resize_alg': Image.BICUBIC,
        'image_format': fmt,
        'quality': qual,
        'quantization_levels': qlevels,
        'quantization_method': 'uniform'
    }
    cache_dir = str(tmp_path / "cache")
    pd = ProcessedDataset(ds, cache_dir=cache_dir, cache_rebuild=True, **cfg)

    # shape
    assert pd.data.shape[0] == len(ds)
    assert pd.data.shape[-2:] == (14,14)
    # valores
    assert float(pd.data.min()) >= 0.0 and float(pd.data.max()) <= 1.0
    if qlevels == 2:
        uniq = set(float(x) for x in torch.unique(pd.data))
        assert uniq.issubset({0.0,1.0})
    # reload cache
    pd2 = ProcessedDataset(ds, cache_dir=cache_dir, cache_rebuild=False, **cfg)
    assert pd2.data.shape == pd.data.shape