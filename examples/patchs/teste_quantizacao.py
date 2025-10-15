#!/usr/bin/env python3
import os
import torch
from torchvision import datasets, transforms
from patchkit import ImageQuantizer
from patchkit.patches import filter_active_patches
from PIL import Image
import numpy as np

# Parâmetros
DATASET = "mnist"
OUT_DIR = f"./examples/outputs/{DATASET}"
PATCH_SIZE = (7, 7)
STRIDE = 3

def to_uint8_img(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if t.dim() == 2:
        return Image.fromarray(arr, mode="L")
    elif t.dim() == 3:
        if t.shape[0] == 1:
            return Image.fromarray(arr.squeeze(0), mode="L")
        elif t.shape[0] == 3:
            return Image.fromarray(np.moveaxis(arr, 0, 2), mode="RGB")
        else:
            raise ValueError("Formato não suportado")
    else:
        raise ValueError("Formato não suportado")

def print_quant_info(img, q, desc):
    print(f"\n=== {desc} ===")
    print(f"Shape: {q.shape}")
    uniq = torch.unique(q)
    print(f"Valores únicos: {[float(x) for x in uniq]}")
    print(f"Min: {float(q.min()):.3f}, Max: {float(q.max()):.3f}")
    print(f"Primeiros valores: {q.flatten()[0:10].tolist()}")

def _ensure_active(result, patches):
    """
    Normaliza o retorno de filter_active_patches.
    - se result é tupla, assume-se result[0] é a estrutura principal;
    - se result[0] for índices 1D, indexa patches;
    - caso contrário, retorna result[0] ou result.
    """
    import numpy as _np, torch as _t
    cand = result[0] if isinstance(result, tuple) else result
    # se cand é lista/array/torch tensor 1D -> interpret as indices
    if isinstance(cand, (list, tuple, _np.ndarray)) and _np.asarray(cand).ndim == 1:
        idxs = _np.asarray(cand).astype(int)
        try:
            return patches[idxs]
        except Exception:
            return cand
    if isinstance(cand, _t.Tensor) and cand.ndim == 1:
        idxs = cand.long().cpu().numpy().astype(int)
        try:
            return patches[idxs]
        except Exception:
            return cand
    return cand

def main():
    # Carregar MNIST (train), pegar 3 amostras de cada classe 1, 8
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    wanted_classes = [1, 8]
    samples_per_class = 3
    selected = []
    for label in wanted_classes:
        idx = (mnist.targets == label).nonzero(as_tuple=True)[0][:samples_per_class]
        for i in idx:
            img, lbl = mnist[i]
            selected.append((img, lbl))

    for i, (img, lbl) in enumerate(selected):
        # Subpasta para cada classe
        class_dir = os.path.join(OUT_DIR, f"class_{lbl}")
        os.makedirs(class_dir, exist_ok=True)

        orig_path = os.path.join(class_dir, f"sample_{i}_original.png")
        to_uint8_img(img.squeeze(0)).save(orig_path)

        quant_methods = [
            ("Uniform2", 2, "uniform"),
            ("Uniform4", 4, "uniform"),
            ("Otsu", None, "otsu"),
            ("Adaptive4", 4, "adaptive"),
        ]
        try:
            import sklearn
            quant_methods.append(("Kmeans4", 4, "kmeans"))
        except ImportError:
            print("[SKIP] kmeans: sklearn não encontrado.")

        for mname, levels, method in quant_methods:
            # Subpasta para cada método/configuração
            conf_dir = os.path.join(class_dir,
                f"{mname}_ps{PATCH_SIZE[0]}x{PATCH_SIZE[1]}_str{STRIDE}")
            os.makedirs(conf_dir, exist_ok=True)

            if levels:
                qimg = ImageQuantizer.quantize(img, levels=levels, method=method)
            else:
                qimg = ImageQuantizer.quantize(img, method=method)
            print_quant_info(img, qimg, f"{mname} quantization")

            # Salvar imagem quantizada
            q_path = os.path.join(conf_dir, f"sample_{i}_quant.png")
            to_uint8_img(qimg.squeeze(0)).save(q_path)

            # Extrair patches da imagem quantizada
            img_patch = qimg.squeeze(0)
            patches = img_patch.unfold(0, PATCH_SIZE[0], STRIDE).unfold(1, PATCH_SIZE[1], STRIDE)
            patches = patches.contiguous().view(-1, PATCH_SIZE[0], PATCH_SIZE[1])
            res = filter_active_patches(patches, min_mean=0.05, max_mean=0.95)
            active_patches = _ensure_active(res, patches)

            print(f"Patches ativos ({mname}): {active_patches.shape[0]}/{patches.shape[0]}")

            # Salvar até 5 patches ativos
            for k in range(min(5, active_patches.shape[0])):
                patch_img = to_uint8_img(active_patches[k])
                patch_path = os.path.join(conf_dir, f"patch_{k}.png")
                patch_img.save(patch_path)

if __name__ == "__main__":
    main()