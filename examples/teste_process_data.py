import os
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

from patchkit import ProcessedDataset, OptimizedPatchExtractor
from patchkit.patches import filter_active_patches

OUT_ROOT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_ROOT, exist_ok=True)

def to_uint8_img(t: torch.Tensor) -> Image.Image:
    """
    Converte tensor [1,H,W] ou [C,H,W] para PIL.Image (L ou RGB).
    Assume valores em [0,1] ou uint8 0-255.
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.max() <= 1.0:
            arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
        else:
            arr = t.numpy().astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            if arr.shape[0] == 1:
                return Image.fromarray(arr.squeeze(0), mode="L")
            return Image.fromarray(np.moveaxis(arr, 0, 2), mode="RGB")
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        raise ValueError(f"Tensor shape não suportado: {tuple(arr.shape)}")
    elif isinstance(t, np.ndarray):
        if t.ndim == 2:
            return Image.fromarray(t, mode="L")
        if t.ndim == 3 and t.shape[2] == 3:
            return Image.fromarray(t, mode="RGB")
        raise ValueError("ndarray formato não suportado")
    else:
        raise ValueError("Tipo não suportado para conversão")

def main(n_classes=(1, 8), samples_per_class=3,
         pd_cfg=None, patch_size=(4,4), stride=2, max_patches_save=12):
    """
    Executa processamento e salva imagens + patches ativos organizados em subpastas.

    pd_cfg: dicionário para ProcessedDataset (target_size, quantization_levels, ...)
    """
    if pd_cfg is None:
        pd_cfg = {
            'target_size': (28, 28),
            'resize_alg': None,
            'image_format': None,
            'quality': None,
            'quantization_levels': 2,
            'quantization_method': 'uniform'
        }

    # Carrega MNIST (pequeno subset por classe)
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Recolhe índices por classe
    selected = []
    for label in n_classes:
        indices = (mnist.targets == label).nonzero(as_tuple=True)[0][:samples_per_class]
        for idx in indices:
            img, lbl = mnist[int(idx)]
            selected.append((img, int(lbl)))

    # Cria ProcessedDataset para as amostras selecionadas (usa subset para não rebaixar todo MNIST)
    subset = torch.utils.data.Subset(mnist, [int(i) for i in range(len(mnist))])  # we'll map later
    # Para demonstration, vamos construir um subset com as imagens que selecionamos
    sel_indices = []
    for _, lbl in selected:
        # já temos os pares, mas precisamos dos índices originais
        pass
    # Simples: criar subset direto usando os índices que coletamos antes
    # (refazer a coleta para guardar índices)
    selected = []
    sel_indices = []
    for label in n_classes:
        idxs = (mnist.targets == label).nonzero(as_tuple=True)[0][:samples_per_class]
        for idx in idxs:
            sel_indices.append(int(idx))
            img, lbl = mnist[int(idx)]
            selected.append((img, int(lbl)))

    subset = torch.utils.data.Subset(mnist, sel_indices)

    pd = ProcessedDataset(subset, cache_dir=os.path.join(OUT_ROOT, "cache_processed"),
                          cache_rebuild=True, **pd_cfg)

    # Para cada item processado, salvar estrutura organizada
    for i in range(len(pd.data)):
        img_tensor = pd.data[i]                # [C,H,W] já processado (float [0,1])
        # Note: temos acesso ao label via subset indices order -> sel_indices[i]
        orig_idx = sel_indices[i]
        label = int(mnist.targets[orig_idx].item())

        sample_dir = os.path.join(OUT_ROOT, f"mnist", f"class_{label}", f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        # Salvar imagem processada (após quantização/preprocess)
        proc_path = os.path.join(sample_dir, "processed.png")
        to_uint8_img(img_tensor.squeeze(0)).save(proc_path)
        print(f"[OK] Saved processed image: {proc_path}")

        # Extrair patches (usando OptimizedPatchExtractor para reproduzir cache/extractor behavior)
        extractor = OptimizedPatchExtractor(patch_size=patch_size, stride=stride,
                                           cache_dir=os.path.join(OUT_ROOT, "cache_patches"),
                                           image_size=img_tensor.shape[-2:])
        # extractor.process espera PIL image (o método interno usa pil_to_tensor), então convertemos:
        pil_img = to_uint8_img(img_tensor.squeeze(0))
        patches = extractor.process(pil_img, index=orig_idx)  # uint8 patches: [L, H, W] ou [L, C, H, W]

        # Filtra patches "ativos"
        active = filter_active_patches(patches, min_mean=0.05, max_mean=0.95)
        print(f"Sample {i} class {label}: patches active {active.shape[0]}/{patches.shape[0]}")

        # Salvar alguns patches ativos
        patches_dir = os.path.join(sample_dir, "patches_active")
        os.makedirs(patches_dir, exist_ok=True)
        nsave = min(max_patches_save, int(active.shape[0]))
        for k in range(nsave):
            p = active[k]
            # p pode ser [H,W] uint8 ou [C,H,W]
            if isinstance(p, torch.Tensor):
                arr = p.cpu().numpy()
            else:
                arr = np.array(p)
            # converter para PIL incluindo canais
            if arr.ndim == 2:
                pil = Image.fromarray(arr, mode="L")
            elif arr.ndim == 3 and arr.shape[0] in (1,3):
                pil = Image.fromarray(np.moveaxis(arr, 0, 2))
            elif arr.ndim == 3 and arr.shape[2] in (1,3):
                pil = Image.fromarray(arr)
            else:
                # fallback: convertendo flatten
                arr2 = (arr.mean(axis=0)).astype(np.uint8)
                pil = Image.fromarray(arr2, mode="L")
            pil = pil.resize((32, 32), Image.NEAREST)
            outp = os.path.join(patches_dir, f"patch_{k:02d}.png")
            pil.save(outp)
        print(f"[OK] Saved {nsave} active patches in {patches_dir}")

    print("[DONE] Example processing completed. Outputs in:", OUT_ROOT)

if __name__ == "__main__":
    main()