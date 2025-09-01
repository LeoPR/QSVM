import os
import numpy as np
import pytest
from PIL import Image

import torch
from torchvision import datasets, transforms

from patchkit import ProcessedDataset, OptimizedPatchExtractor

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def to_uint8_img(t: torch.Tensor) -> Image.Image:
    if t.dim() == 3 and t.shape[0] in (1, 3):
        arr = (t.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        if t.shape[0] == 1:
            return Image.fromarray(arr.squeeze(0), mode="L")
        return Image.fromarray(np.moveaxis(arr, 0, 2), mode="RGB")
    elif t.dim() == 2:
        arr = (t.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Formato de tensor não suportado: {tuple(t.shape)}")

@pytest.mark.timeout(60)
def test_processed_dataset_and_patch_extraction():
    # Baixa MNIST (primeiras 3 imagens)
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(mnist, list(range(3)))

    cfg = {
        'target_size': (28, 28),
        'resize_alg': None,
        'image_format': None,
        'quality': None,
        'quantization_levels': 2,
        'quantization_method': 'uniform'
    }
    pd = ProcessedDataset(subset, cache_dir=CACHE_DIR, cache_rebuild=True, **cfg)
    # Verifica que as imagens processadas têm o formato esperado
    assert pd.data.shape[0] == 3
    assert pd.data.shape[-2:] == (28, 28)

    # Extrai patches da primeira imagem
    first_img = to_uint8_img(pd.data[0].squeeze(0))
    extractor = OptimizedPatchExtractor(
        patch_size=(4, 4), stride=2, cache_dir=CACHE_DIR,
        image_size=(28, 28), max_memory_cache=10
    )
    patches = extractor.process(first_img, index=0)
    # Verifica que extraiu patches > 0 e com shape correto
    assert patches.shape[1:] == (4, 4)
    assert patches.shape[0] > 0
    # Opcional: verifica que patch é imagem binária (0 ou 255)
    assert np.all(np.isin(patches[0].numpy(), [0, 255]))

# Pronto para rodar com pytest!