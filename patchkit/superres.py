from .processed import ProcessedDataset
from .patches import OptimizedPatchExtractor
from Logger import Logger
from torchvision import transforms

class SuperResPatchDataset:
    def __init__(self, original_ds, low_res_config, high_res_config,
                 small_patch_size, large_patch_size, stride, scale_factor,
                 cache_dir="./cache", cache_rebuild=False, max_memory_cache=100):
        assert large_patch_size[0] == small_patch_size[0] * scale_factor
        assert large_patch_size[1] == small_patch_size[1] * scale_factor
        self.original_ds = original_ds
        self.scale_factor = scale_factor
        self.stride = stride

        low_cache = cache_dir + "/low_res"
        high_cache = cache_dir + "/high_res"

        self.low_res_ds = ProcessedDataset(original_ds, cache_dir=low_cache, cache_rebuild=cache_rebuild, **low_res_config)
        self.high_res_ds = ProcessedDataset(original_ds, cache_dir=high_cache, cache_rebuild=cache_rebuild, **high_res_config)

        self.small_patch_extractor = OptimizedPatchExtractor(small_patch_size, stride, cache_dir + "/low_patches", low_res_config['target_size'], max_memory_cache)
        self.large_patch_extractor = OptimizedPatchExtractor(large_patch_size, stride * scale_factor, cache_dir + "/high_patches", high_res_config['target_size'], max_memory_cache)

        assert self.small_patch_extractor.num_patches_per_image == self.large_patch_extractor.num_patches_per_image

        self.num_images = len(self.low_res_ds)
        self.num_patches_per_image = self.small_patch_extractor.num_patches_per_image

    def __len__(self):
        return self.num_images * self.num_patches_per_image

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        img_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image
        label = self.low_res_ds.labels[img_idx].item()
        low_tensor = self.low_res_ds.data[img_idx]
        high_tensor = self.high_res_ds.data[img_idx]
        low_pil = transforms.ToPILImage()(low_tensor.squeeze())
        high_pil = transforms.ToPILImage()(high_tensor.squeeze())
        small_patch = self.small_patch_extractor.get_patch(low_pil, img_idx, patch_idx)
        large_patch = self.large_patch_extractor.get_patch(high_pil, img_idx, patch_idx)
        X = (label, img_idx, patch_idx, small_patch)
        y = large_patch
        return X, y

    def prefetch_image(self, img_idx):
        low_tensor = self.low_res_ds.data[img_idx]
        high_tensor = self.high_res_ds.data[img_idx]
        low_pil = transforms.ToPILImage()(low_tensor.squeeze())
        high_pil = transforms.ToPILImage()(high_tensor.squeeze())
        self.small_patch_extractor.process(low_pil, img_idx)
        self.large_patch_extractor.process(high_pil, img_idx)