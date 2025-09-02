import torch
from PIL import Image
from torchvision import transforms

from patchkit import SuperResPatchDataset

class TinySynthetic(torch.utils.data.Dataset):
    def __init__(self, n=3, size=(28,28)):
        self.n=n; self.size=size
    def __len__(self): return self.n
    def __getitem__(self, idx):
        H,W = self.size
        t = torch.zeros(1,H,W,dtype=torch.float32)
        t[0, H//4:3*H//4, W//4:3*W//4] = 1.0
        return t, idx%10

def test_superres_structure(tmp_path):
    base_ds = TinySynthetic(n=2, size=(28,28))
    low_cfg = {'target_size': (14,14), 'resize_alg': Image.BICUBIC,
               'image_format': None, 'quality': None,
               'quantization_levels': 2, 'quantization_method': 'uniform'}
    high_cfg = {'target_size': (28,28), 'resize_alg': Image.BICUBIC,
                'image_format': None, 'quality': None,
                'quantization_levels': None, 'quantization_method': 'uniform'}
    ds = SuperResPatchDataset(original_ds=base_ds,
                              low_res_config=low_cfg, high_res_config=high_cfg,
                              small_patch_size=(2,2), large_patch_size=(4,4),
                              stride=1, scale_factor=2,
                              cache_dir=str(tmp_path/"cache"), cache_rebuild=True, max_memory_cache=5)
    # tamanho total = num_imagens * patches_por_imagem
    assert ds.num_images == len(base_ds)
    assert ds.num_patches_per_image > 0
    X, y = ds[0]
    label, img_idx, patch_idx, small_patch = X
    assert isinstance(label, int)
    assert small_patch.ndim in (2,3)  # [H,W] uint8