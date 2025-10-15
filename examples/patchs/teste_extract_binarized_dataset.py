import os
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

from patchkit import ProcessedDataset
from patchkit.patches import OptimizedPatchExtractor, filter_active_patches

# uso do config centralizado (apenas alteração pontual)
from examples.patchs.config import OUTPUTS_ROOT
# utilitários compartilhados (substituem funções locais duplicadas)
from examples.patchs.utils import ensure_active, to_uint8_img_tensor

OUT_ROOT = os.path.join(OUTPUTS_ROOT, "binarized_datasets")
os.makedirs(OUT_ROOT, exist_ok=True)

def save_active_patches_from_pil(pil_img, out_dir, patch_size=(4,4), stride=2, max_save=12,
                                 min_mean=0.05, max_mean=0.95):
    os.makedirs(out_dir, exist_ok=True)
    extractor = OptimizedPatchExtractor(patch_size=patch_size, stride=stride,
                                       cache_dir=os.path.join(out_dir, "cache_patches"),
                                       image_size=pil_img.size[::-1])
    patches = extractor.process(pil_img, index=0)
    # patches returned uint8: [L, H, W] ou [L, C, H, W]
    res = filter_active_patches(patches, min_mean=min_mean, max_mean=max_mean)
    active = ensure_active(res, patches)
    nsave = min(int(active.shape[0]), max_save)
    for k in range(nsave):
        p = active[k]
        arr = p.cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p)
        if arr.ndim == 2:
            pil = Image.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[0] in (1,3):
            pil = Image.fromarray(np.moveaxis(arr, 0, 2))
        else:
            arr2 = (arr.mean(axis=0)).astype(np.uint8)
            pil = Image.fromarray(arr2, mode="L")
        pil = pil.resize((32,32), Image.NEAREST)
        pil.save(os.path.join(out_dir, f"patch_{k:02d}.png"))
    return active.shape[0], patches.shape[0]

def process_and_save(target_size, quant_levels, quant_method, dataset_name="mnist",
                     classes=(1,8), samples_per_class=3, patch_size=(4,4), stride=2):
    pd_cfg = {
        'target_size': target_size,
        'resize_alg': None,
        'image_format': None,
        'quality': None,
        'quantization_levels': quant_levels,
        'quantization_method': quant_method
    }

    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    sel_indices = []
    for c in classes:
        idxs = (mnist.targets == c).nonzero(as_tuple=True)[0][:samples_per_class]
        sel_indices.extend([int(x) for x in idxs])
    sel_indices = list(dict.fromkeys(sel_indices))

    subset = torch.utils.data.Subset(mnist, sel_indices)
    pd = ProcessedDataset(subset, cache_dir=os.path.join(OUT_ROOT, "cache_processed"),
                          cache_rebuild=True, **pd_cfg)

    size_dir = os.path.join(OUT_ROOT, f"{dataset_name}_{target_size[0]}x{target_size[1]}_q{quant_levels}_{quant_method}")
    os.makedirs(size_dir, exist_ok=True)

    for i in range(len(pd.data)):
        img_tensor = pd.data[i]
        orig_idx = sel_indices[i]
        label = int(mnist.targets[orig_idx].item())

        sample_dir = os.path.join(size_dir, f"class_{label}", f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        proc_path = os.path.join(sample_dir, "processed.png")
        to_uint8_img_tensor(img_tensor.squeeze(0)).save(proc_path)

        patches_dir = os.path.join(sample_dir, "patches_active")
        active_count, total_count = save_active_patches_from_pil(Image.open(proc_path).convert("L"), patches_dir,
                                                                 patch_size=patch_size, stride=stride,
                                                                 max_save=12)
        print(f"Saved sample {i} label {label}: processed -> {proc_path}, active patches {active_count}/{total_count}")

if __name__ == "__main__":
    process_and_save(target_size=(14,14), quant_levels=2, quant_method='uniform',
                     classes=(1,8), samples_per_class=3, patch_size=(4,4), stride=2)