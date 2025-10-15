import os
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

from patchkit import ProcessedDataset, OptimizedPatchExtractor
from patchkit.patches import filter_active_patches

# uso do config centralizado (apenas alteração pontual)
from examples.patchs.config import OUTPUTS_ROOT
# utilitários compartilhados (substituem funções locais duplicadas)
from examples.patchs.utils import ensure_active, to_uint8_img

os.makedirs(OUTPUTS_ROOT, exist_ok=True)

def main(n_classes=(1, 8), samples_per_class=3,
         pd_cfg=None, patch_size=(4,4), stride=2, max_patches_save=12):
    if pd_cfg is None:
        pd_cfg = {
            'target_size': (28, 28),
            'resize_alg': None,
            'image_format': None,
            'quality': None,
            'quantization_levels': 2,
            'quantization_method': 'uniform'
        }

    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    selected = []
    for label in n_classes:
        indices = (mnist.targets == label).nonzero(as_tuple=True)[0][:samples_per_class]
        for idx in indices:
            for i in indices:
                pass

    selected = []
    sel_indices = []
    for label in n_classes:
        idxs = (mnist.targets == label).nonzero(as_tuple=True)[0][:samples_per_class]
        for idx in idxs:
            sel_indices.append(int(idx))
            img, lbl = mnist[int(idx)]
            selected.append((img, int(lbl)))

    subset = torch.utils.data.Subset(mnist, sel_indices)

    pd = ProcessedDataset(subset, cache_dir=os.path.join(OUTPUTS_ROOT, "cache_processed"),
                          cache_rebuild=True, **pd_cfg)

    for i in range(len(pd.data)):
        img_tensor = pd.data[i]
        orig_idx = sel_indices[i]
        label = int(mnist.targets[orig_idx].item())

        sample_dir = os.path.join(OUTPUTS_ROOT, f"mnist", f"class_{label}", f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        proc_path = os.path.join(sample_dir, "processed.png")
        to_uint8_img(img_tensor.squeeze(0)).save(proc_path)
        print(f"[OK] Saved processed image: {proc_path}")

        extractor = OptimizedPatchExtractor(patch_size=patch_size, stride=stride,
                                           cache_dir=os.path.join(OUTPUTS_ROOT, "cache_patches"),
                                           image_size=img_tensor.shape[-2:])
        pil_img = to_uint8_img(img_tensor.squeeze(0))
        patches = extractor.process(pil_img, index=orig_idx)

        # normalize possible return types of patches -> ensure numpy/torch array-like
        # patches may be torch tensor or ndarray/list; convert to torch if needed
        if not isinstance(patches, torch.Tensor):
            try:
                import numpy as _np
                patches = torch.from_numpy(_np.array(patches))
            except Exception:
                pass

        res = filter_active_patches(patches, min_mean=0.05, max_mean=0.95)
        active_patches = ensure_active(res, patches)
        print(f"Sample {i} class {label}: patches active {active_patches.shape[0]}/{patches.shape[0]}")

        patches_dir = os.path.join(sample_dir, "patches_active")
        os.makedirs(patches_dir, exist_ok=True)
        nsave = min(max_patches_save, int(active_patches.shape[0]))
        for k in range(nsave):
            p = active_patches[k]
            if isinstance(p, torch.Tensor):
                arr = p.cpu().numpy()
            else:
                arr = np.array(p)
            if arr.ndim == 2:
                pil = Image.fromarray(arr, mode="L")
            elif arr.ndim == 3 and arr.shape[0] in (1,3):
                pil = Image.fromarray(np.moveaxis(arr, 0, 2))
            elif arr.ndim == 3 and arr.shape[2] in (1,3):
                pil = Image.fromarray(arr)
            else:
                arr2 = (arr.mean(axis=0)).astype(np.uint8)
                pil = Image.fromarray(arr2, mode="L")
            pil = pil.resize((32, 32), Image.NEAREST)
            outp = os.path.join(patches_dir, f"patch_{k:02d}.png")
            pil.save(outp)
        print(f"[OK] Saved {nsave} active patches in {patches_dir}")

    print("[DONE] Example processing completed. Outputs in:", OUTPUTS_ROOT)

if __name__ == "__main__":
    main()