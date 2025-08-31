"""
Exemplo de processamento e extração de patches, com salvamento de algumas imagens
em examples/outputs. Execute via:

  python -m examples.teste_process_data
"""

import os
import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms

from patchkit import ProcessedDataset, OptimizedPatchExtractor

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


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


def main():
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(mnist, list(range(3)))

    cfg = {
        'target_size': (28, 28),
        'resize_alg': None,
        'image_format': None,
        'quality': None,
        'quantization_levels': 2,
        'quantization_method': 'uniform'
    }

    pd = ProcessedDataset(subset, cache_dir="./cache_example_process", cache_rebuild=True, **cfg)

    # Salvar primeiras 3 imagens processadas como PNG binário
    for i in range(3):
        img = to_uint8_img(pd.data[i].squeeze(0))  # [1,H,W] -> PIL L
        out_path = os.path.join(OUT_DIR, f"processed_bin_{i}.png")
        img.save(out_path)
        print(f"[OK] Salvou {out_path}")

    # Extrair patches da primeira imagem e salvar alguns patches
    first_img = to_uint8_img(pd.data[0].squeeze(0))
    # Convertemos de volta pra PIL (já está PIL), configurar extractor
    extractor = OptimizedPatchExtractor(patch_size=(4, 4), stride=2, cache_dir="./cache_example_process",
                                       image_size=(28, 28), max_memory_cache=10)
    patches = extractor.process(first_img, index=0)
    # Salvar 12 patches (se existir)
    for k in range(min(12, patches.shape[0])):
        p = patches[k]
        if p.dim() == 2:
            pil = Image.fromarray(p.numpy(), mode="L")
        else:
            pil = Image.fromarray(np.moveaxis(p.numpy(), 0, 2), mode="RGB")
        pil = pil.resize((32, 32), Image.NEAREST)
        pil.save(os.path.join(OUT_DIR, f"patch_{k:02d}.png"))

    print("[OK] Patches salvos em", OUT_DIR)


if __name__ == "__main__":
    main()