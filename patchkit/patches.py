import os
import io
import hashlib
import torch
import zstandard as zstd
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from collections import OrderedDict
from Logger import Logger

class OptimizedPatchExtractor:
    def __init__(self, patch_size, stride, cache_dir, image_size, max_memory_cache=100):
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
        patches_flat = patches_unf.squeeze(0).transpose(0,1)  # [L, C*ph*pw]
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

PatchExtractor = OptimizedPatchExtractor