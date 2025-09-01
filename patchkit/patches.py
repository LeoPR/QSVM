import os
import io
import tempfile
import hashlib
import torch
import zstandard as zstd
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from collections import OrderedDict
from Logger import Logger

def filter_active_patches(patches, min_mean=0.1, max_mean=0.9):
    """
    Retorna apenas os patches cuja média está entre min_mean e max_mean.
    Útil para filtrar patches "ativos" (nem só pretos, nem só brancos).
    """
    p = patches.float()
    if p.max() > 1.0:
        p = p / 255.0
    means = p.mean(dim=(-2, -1))
    if means.dim() > 1:
        means = means.mean(dim=-1)
    sel = (means > min_mean) & (means < max_mean)
    return patches[sel]


class OptimizedPatchExtractor:
    def __init__(self, patch_size, stride, cache_dir, image_size, max_memory_cache=100):
        if cache_dir is None:
            # respeita variável de ambiente pra facilitar CI/local dev
            cache_dir = os.environ.get("QSVM_CACHE_DIR",
                                       os.path.join(tempfile.gettempdir(), "qsvm_cache"))
        self.cache_dir = os.path.join(cache_dir, "patches_optimized")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.patch_size = patch_size  # (h,w)
        self.stride = stride
        self.image_size = image_size  # (h,w)
        self.max_memory_cache = max_memory_cache

        self.num_patches_h = (image_size[0] - patch_size[0]) // stride + 1
        self.num_patches_w = (image_size[1] - patch_size[1]) // stride + 1
        self.num_patches_per_image = self.num_patches_h * self.num_patches_w

        self.cache_dir = os.path.join(cache_dir, "patches_optimized")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.memory_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_path(self, image_index):
        config_str = (f"img_{image_index}_size_{self.image_size[0]}x{self.image_size[1]}_"
                      f"patch_{self.patch_size[0]}x{self.patch_size[1]}_stride_{self.stride}")
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{config_str}_{config_hash}.pt.zst")

    def _extract_all_patches(self, image):
        tensor = TF.pil_to_tensor(image)  # uint8 [C,H,W]
        C, H, W = tensor.shape
        tensor = tensor.unsqueeze(0).float()  # [1,C,H,W] in 0-255
        ph, pw = self.patch_size
        patches_unf = F.unfold(tensor, kernel_size=(ph, pw), stride=self.stride)  # [1, C*ph*pw, L]
        patches_flat = patches_unf.squeeze(0).transpose(0, 1)  # [L, C*ph*pw]
        patches = patches_flat.view(-1, C, ph, pw)
        patches = patches.round().to(torch.uint8)
        if C == 1:
            return patches.squeeze(1)
        return patches

    def _save_compressed_patches(self, patches, cache_path):
        buffer = io.BytesIO()
        torch.save(patches, buffer)
        buffer.seek(0)
        with open(cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=3, threads=-1)
            compressed = cctx.compress(buffer.read())
            f.write(compressed)

    def _load_compressed_patches(self, cache_path):
        try:
            with open(cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(f.read())
            buffer = io.BytesIO(decompressed)
            patches = torch.load(buffer, map_location="cpu")
            if patches.shape[0] != self.num_patches_per_image:
                Logger.warning("Invalid patch count in cache")
                return None
            return patches
        except Exception as e:
            Logger.warning(f"Failed to load patches: {e}")
            return None

    def _update_memory_cache(self, image_index, patches):
        if image_index in self.memory_cache:
            del self.memory_cache[image_index]
        self.memory_cache[image_index] = patches
        if len(self.memory_cache) > self.max_memory_cache:
            evicted = next(iter(self.memory_cache))
            del self.memory_cache[evicted]

    def process(self, image, index=None):
        if index is None:
            return self._extract_all_patches(image)

        if index in self.memory_cache:
            self.cache_hits += 1
            patches = self.memory_cache[index]
            # move to end
            del self.memory_cache[index]
            self.memory_cache[index] = patches
            return patches

        self.cache_misses += 1
        cache_path = self._get_cache_path(index)
        if os.path.exists(cache_path):
            patches = self._load_compressed_patches(cache_path)
            if patches is not None:
                self._update_memory_cache(index, patches)
                return patches

        patches = self._extract_all_patches(image)
        self._save_compressed_patches(patches, cache_path)
        self._update_memory_cache(index, patches)
        return patches

    def get_patch(self, image, index, patch_idx):
        all_patches = self.process(image, index)
        return all_patches[patch_idx]

    def clear_memory_cache(self):
        self.memory_cache.clear()

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'max_cache_size': self.max_memory_cache
        }

    def validate_and_clean_cache(self):
        invalid = 0
        files = os.listdir(self.cache_dir)
        for f in files:
            if f.endswith('.pt.zst'):
                p = os.path.join(self.cache_dir, f)
                try:
                    patches = self._load_compressed_patches(p)
                    if patches is None or patches.shape[0] != self.num_patches_per_image:
                        os.remove(p)
                        invalid += 1
                except Exception:
                    os.remove(p)
                    invalid += 1
        return invalid

    def reconstruct_image(self, patches, device=torch.device("cpu")):
        """
        Reconstruct an image from a set of patches using F.fold (vectorized).

        Accepts:
            patches: Tensor [L, H, W] (uint8) or [L, C, H, W] (uint8)
        Returns:
            - single-channel: Tensor [recon_h, recon_w] uint8
            - multi-channel: Tensor [C, recon_h, recon_w] uint8
        """
        num_h = self.num_patches_h
        num_w = self.num_patches_w
        patch_h, patch_w = self.patch_size

        recon_h = patch_h + (num_h - 1) * self.stride
        recon_w = patch_w + (num_w - 1) * self.stride

        # Detect format of patches and prepare tensor [L, C, ph, pw]
        if patches.dim() == 3:
            # [L, H, W] -> [L, 1, H, W]
            patches_f = patches.unsqueeze(1).float() / 255.0
        elif patches.dim() == 4:
            # [L, C, H, W]
            patches_f = patches.float() / 255.0
        else:
            raise ValueError("Unsupported patches shape for reconstruction")

        L, C, ph, pw = patches_f.shape
        # Move to requested device (CPU by default). fold will sum contributions.
        patches_f = patches_f.to(device)

        # Prepare for fold: need [1, C*ph*pw, L]
        patches_flat = patches_f.view(L, -1).transpose(0, 1).unsqueeze(0)  # [1, C*ph*pw, L]

        # Sum contributions (reconstruction) using fold
        recon_sum = F.fold(patches_flat, output_size=(recon_h, recon_w),
                           kernel_size=(ph, pw), stride=self.stride)  # [1, C, H, W]

        # Create weight matrix (how many times each pixel was summed) using ones
        ones = torch.ones_like(patches_f, dtype=patches_f.dtype, device=device)
        ones_flat = ones.view(L, -1).transpose(0, 1).unsqueeze(0)  # [1, C*ph*pw, L]
        weight = F.fold(ones_flat, output_size=(recon_h, recon_w),
                        kernel_size=(ph, pw), stride=self.stride)  # [1, C, H, W]

        # Avoid division by zero
        recon = recon_sum / weight.clamp(min=1e-6)

        # Convert back to uint8
        recon = (recon * 255.0).round().to(torch.uint8).squeeze(0)  # [C, H, W] or [H, W] if C==1

        if recon.shape[0] == 1:
            return recon.squeeze(0)  # [H, W]
        return recon  # [C, H, W]


PatchExtractor = OptimizedPatchExtractor
