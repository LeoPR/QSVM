import os
import zstandard as zstd
import torch
import io
from PIL import Image
from torchvision import transforms, datasets
from Logger import Logger
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
import hashlib

Logger.set_log_level("INFO")

RESIZE_ALG_MAP = {
    Image.NEAREST: "nearest",
    Image.BILINEAR: "bilinear",
    Image.BICUBIC: "bicubic",
    Image.LANCZOS: "lanczos",
}


class ProcessedDataset(torch.utils.data.Dataset):
    """
    Creates a processed version of a dataset with optional resizing and compression artifacts.
    Automatically infers processing needs based on parameters.

    Features:
    - Automatic resizing detection (if target_size != original size)
    - Automatic compression detection (based on image_format)
    - Compatible with torch.utils.data.ConcatDataset
    - Maintains original dataset structure for easy pairing
    - Caches processed data with zstd compression

    Args:
        original_ds: Dataset containing images (PIL Images or torch Tensors) and labels
        target_size: (width, height) for output images
        resize_alg: PIL resize algorithm (e.g., Image.BICUBIC)
        image_format: None/'RAW' for no compression, 'JPEG'/'PNG' for compression
        quality: Compression quality (1-100) for lossy formats
        cache_dir: Directory for processed data caching
        cache_rebuild: If True, rebuild cache even if it exists
    """

    def __init__(self, original_ds, target_size, resize_alg,
                 image_format=None, quality=None, cache_dir="./cache", cache_rebuild=False):
        self.original_ds = original_ds
        self.target_size = target_size
        self.resize_alg = resize_alg
        self.image_format = image_format.upper() if image_format else None
        self.quality = quality
        self.cache_rebuild = cache_rebuild

        # Auto-detect processing needs
        self.needs_resize = (self.target_size != self._get_original_size())
        self.needs_compression = self.image_format in ['JPEG', 'PNG']

        # Create unique cache name based on parameters
        self.cache_path = self._create_cache_path(cache_dir)

        # Load or process data
        if os.path.exists(self.cache_path) and not self.cache_rebuild:
            self.data, self.labels = self._load_cache()
        else:
            self.data, self.labels = self._process_and_cache()

    def _get_original_size(self):
        """Get size from first image (assumes consistent sizes)"""
        img, _ = self.original_ds[0]
        if isinstance(img, Image.Image):
            return img.size  # (width, height)
        elif isinstance(img, torch.Tensor):
            return (img.shape[-1], img.shape[-2])  # (width, height)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def _create_cache_path(self, cache_dir):
        """Generate unique cache filename based on parameters"""
        original_size = self._get_original_size()
        resize_info = (f"{original_size[0]}x{original_size[1]}"
                       f"_{self.target_size[0]}x{self.target_size[1]}")

        alg_name = RESIZE_ALG_MAP.get(self.resize_alg, "unknown")
        format_info = f"{self.image_format}_q{self.quality}" if self.needs_compression else "raw"

        return os.path.join(
            cache_dir,
            f"processed_{resize_info}_{alg_name}_{format_info}.pt.zst"
        )

    def _process_image(self, img):
        """Process single image with inferred transformations"""
        # Convert tensor to PIL Image if needed
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # Apply resizing if needed
        if self.needs_resize:
            img = img.resize(self.target_size, self.resize_alg)

        # Apply compression artifacts if needed
        if self.needs_compression:
            buffer = io.BytesIO()
            save_kwargs = {'format': self.image_format}
            if self.quality is not None:
                save_kwargs['quality'] = self.quality
            img.save(buffer, **save_kwargs)
            buffer.seek(0)
            img = Image.open(buffer)

        return transforms.ToTensor()(img)

    def _process_and_cache(self):
        """Process all images and save to cache with zstd compression"""
        processed_images = []
        labels = []

        for img, label in tqdm(self.original_ds, desc="Processing Dataset"):
            processed_img = self._process_image(img)
            processed_images.append(processed_img)
            labels.append(label)

        data_tensor = torch.stack(processed_images)
        labels_tensor = torch.tensor(labels)

        data_dict = {
            'data': data_tensor,
            'labels': labels_tensor,
            'config': self._get_config()
        }

        buffer = io.BytesIO()
        torch.save(data_dict, buffer)
        buffer.seek(0)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=1, threads=-1)
            Logger.micro("Compressing processed data stream for caching...")
            with cctx.stream_writer(f) as compressor:
                compressor.write(buffer.read())

        Logger.info(f"Processed and cached dataset saved to {self.cache_path}")

        return data_tensor, labels_tensor

    def _load_cache(self):
        """Load cached data with zstd decompression and validation"""
        Logger.info(f"Loading dataset from cache at {self.cache_path}")
        try:
            with open(self.cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                Logger.micro("Decompressing cached data stream...")
                with dctx.stream_reader(f) as reader:
                    decompressed_data = reader.read()
            buffer = io.BytesIO(decompressed_data)
            loaded = torch.load(buffer)

            # Validate cache contents
            if 'data' not in loaded or 'labels' not in loaded:
                Logger.warning("Invalid cache format, rebuilding...")
                return self._process_and_cache()

            # Validate dataset size matches
            if len(loaded['data']) != len(self.original_ds):
                Logger.warning(
                    f"Cache size mismatch (cached: {len(loaded['data'])}, expected: {len(self.original_ds)}), rebuilding...")
                return self._process_and_cache()

            Logger.info("Loaded dataset from cache successfully.")
            return loaded['data'], loaded['labels']
        except Exception as e:
            Logger.warning(f"Failed to load cache: {e}. Rebuilding...")
            return self._process_and_cache()

    def _get_config(self):
        """Get configuration dictionary for verification"""
        return {
            'target_size': self.target_size,
            'resize_alg': RESIZE_ALG_MAP.get(self.resize_alg, "unknown"),
            'image_format': self.image_format,
            'quality': self.quality
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return (processed_image, original_image, label) for explicit pairing"""
        return self.data[idx], self.original_ds[idx][0], self.labels[idx]


class OptimizedPatchExtractor:
    """
    Optimized patch extractor that caches all patches from an image together.
    Uses in-memory LRU cache for recently accessed images.
    """

    def __init__(self, patch_size, stride, cache_dir, image_size, max_memory_cache=100):
        """
        Initialize the OptimizedPatchExtractor.

        Parameters:
            patch_size (tuple): Dimensions of each patch (height, width).
            stride (int): Stride between patches (assumes same for height and width).
            cache_dir (str): Base directory for caching extracted patches.
            image_size (tuple): Image dimensions (height, width).
            max_memory_cache (int): Maximum number of images to keep in memory cache.
        """
        self.patch_size = patch_size  # (height, width)
        self.stride = stride
        self.image_size = image_size  # (height, width)
        self.max_memory_cache = max_memory_cache

        # Calculate number of patches
        self.num_patches_h = (self.image_size[0] - self.patch_size[0]) // self.stride + 1
        self.num_patches_w = (self.image_size[1] - self.patch_size[1]) // self.stride + 1
        self.num_patches_per_image = self.num_patches_h * self.num_patches_w

        # Create cache directory
        self.cache_dir = os.path.join(cache_dir, "patches_optimized")
        os.makedirs(self.cache_dir, exist_ok=True)

        # LRU cache for in-memory storage
        self.memory_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

        Logger.info(
            f"OptimizedPatchExtractor initialized: patch_size={self.patch_size}, "
            f"stride={self.stride}, patches_per_image={self.num_patches_per_image}"
        )

    def _get_cache_path(self, image_index):
        """Generate cache filename for an image's patches."""
        # Create a unique identifier for this configuration
        config_str = (f"img_{image_index}_"
                      f"size_{self.image_size[0]}x{self.image_size[1]}_"
                      f"patch_{self.patch_size[0]}x{self.patch_size[1]}_"
                      f"stride_{self.stride}")

        # Use hash for shorter filenames
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

        return os.path.join(self.cache_dir, f"{config_str}_{config_hash}.pt.zst")

    def _extract_all_patches(self, image):
        """Extract all patches from an image at once."""
        Logger.micro(f"Extracting all {self.num_patches_per_image} patches from image")

        # Convert image to tensor (shape: [C, H, W])
        tensor = transforms.ToTensor()(image)
        # Add batch dimension: [1, C, H, W]
        tensor = tensor.unsqueeze(0)

        # Use unfold to extract all patches at once
        patch_h, patch_w = self.patch_size
        patches = F.unfold(tensor, kernel_size=(patch_h, patch_w), stride=self.stride)

        # Reshape: [1, C*patch_h*patch_w, num_patches] -> [num_patches, patch_h, patch_w]
        patches = patches.squeeze(0).transpose(0, 1)
        patches = patches.reshape(-1, patch_h, patch_w)

        # Convert to uint8 for storage efficiency
        patches = (patches * 255).round().to(torch.uint8)

        return patches

    def _update_memory_cache(self, image_index, patches):
        """Update LRU memory cache."""
        # Remove from current position if exists
        if image_index in self.memory_cache:
            del self.memory_cache[image_index]

        # Add to end (most recently used)
        self.memory_cache[image_index] = patches

        # Evict oldest if cache is full
        if len(self.memory_cache) > self.max_memory_cache:
            evicted_idx = next(iter(self.memory_cache))
            del self.memory_cache[evicted_idx]
            Logger.micro(f"Evicted image {evicted_idx} from memory cache")

    def _save_compressed_patches(self, patches, cache_path):
        """Save patches with zstd compression."""
        buffer = io.BytesIO()
        torch.save(patches, buffer)
        buffer.seek(0)

        with open(cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=3)  # Higher compression for patches
            compressed = cctx.compress(buffer.read())
            f.write(compressed)

        Logger.micro(f"Saved compressed patches to {cache_path}")

    def _load_compressed_patches(self, cache_path):
        """Load patches with zstd decompression and error handling."""
        try:
            with open(cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(f.read())

            buffer = io.BytesIO(decompressed)
            patches = torch.load(buffer)

            # Validate patches shape
            if patches.shape[0] != self.num_patches_per_image:
                Logger.warning(
                    f"Invalid patch count in cache (expected {self.num_patches_per_image}, got {patches.shape[0]})")
                return None

            return patches
        except Exception as e:
            Logger.warning(f"Failed to load patches from {cache_path}: {e}")
            return None

    def process(self, image, index=None):
        """
        Extract or retrieve all patches from an image.

        Parameters:
            image (PIL.Image): Input image.
            index (int, optional): Image index for caching.

        Returns:
            patches (torch.Tensor): All patches from the image.
        """
        if index is None:
            # No caching if index not provided
            return self._extract_all_patches(image)

        # Check memory cache first
        if index in self.memory_cache:
            self.cache_hits += 1
            Logger.micro(f"Memory cache hit for image {index} (hit rate: {self.get_cache_stats()['hit_rate']:.2%})")
            # Move to end (most recently used)
            patches = self.memory_cache[index]
            del self.memory_cache[index]
            self.memory_cache[index] = patches
            return patches

        self.cache_misses += 1

        # Check disk cache
        cache_path = self._get_cache_path(index)

        if os.path.exists(cache_path):
            Logger.micro(f"Loading patches for image {index} from disk cache")
            patches = self._load_compressed_patches(cache_path)

            if patches is not None:
                self._update_memory_cache(index, patches)
                return patches
            else:
                Logger.warning(f"Invalid cache for image {index}, re-extracting patches")
                # Continue to extraction below

        # Extract patches
        Logger.micro(f"Extracting patches for image {index} (cache miss)")
        patches = self._extract_all_patches(image)

        # Save to disk cache
        self._save_compressed_patches(patches, cache_path)

        # Update memory cache
        self._update_memory_cache(index, patches)

        return patches

    def get_patch(self, image, index, patch_idx):
        """
        Get a specific patch from an image.

        Parameters:
            image (PIL.Image): Input image.
            index (int): Image index.
            patch_idx (int): Patch index within the image.

        Returns:
            patch (torch.Tensor): The requested patch.
        """
        all_patches = self.process(image, index)
        return all_patches[patch_idx]

    def clear_memory_cache(self):
        """Clear the in-memory cache."""
        self.memory_cache.clear()
        Logger.info("Cleared memory cache")

    def get_cache_stats(self):
        """Get cache statistics."""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'max_cache_size': self.max_memory_cache
        }

    def validate_and_clean_cache(self):
        """
        Validate cache files and remove invalid ones.
        Returns number of invalid caches cleaned.
        """
        invalid_count = 0
        cache_files = os.listdir(self.cache_dir)

        for cache_file in cache_files:
            if cache_file.endswith('.pt.zst'):
                cache_path = os.path.join(self.cache_dir, cache_file)
                try:
                    patches = self._load_compressed_patches(cache_path)
                    if patches is None or patches.shape[0] != self.num_patches_per_image:
                        os.remove(cache_path)
                        invalid_count += 1
                        Logger.info(f"Removed invalid cache: {cache_file}")
                except Exception as e:
                    os.remove(cache_path)
                    invalid_count += 1
                    Logger.info(f"Removed corrupted cache: {cache_file}")

        if invalid_count > 0:
            Logger.info(f"Cleaned {invalid_count} invalid cache files")

        return invalid_count

    def reconstruct_image(self, patches, device=torch.device("cpu")):
        """
        Reconstruct an image from a set of patches.
        """
        num_h = self.num_patches_h
        num_w = self.num_patches_w
        patch_h, patch_w = self.patch_size

        recon_h = patch_h + (num_h - 1) * self.stride
        recon_w = patch_w + (num_w - 1) * self.stride

        reconstructed = torch.zeros((recon_h, recon_w), dtype=torch.float32, device=device)
        weight = torch.zeros((recon_h, recon_w), dtype=torch.float32, device=device)

        for i in range(num_h):
            for j in range(num_w):
                idx = i * num_w + j
                top = i * self.stride
                left = j * self.stride
                reconstructed[top:top + patch_h, left:left + patch_w] += patches[idx].float() / 255.0
                weight[top:top + patch_h, left:left + patch_w] += 1

        reconstructed /= weight.clamp(min=1e-6)
        reconstructed = (reconstructed * 255).round().to(torch.uint8)
        return reconstructed


# Mantém compatibilidade com código existente
PatchExtractor = OptimizedPatchExtractor


class SuperResPatchDataset(torch.utils.data.Dataset):
    """
    Dataset for super-resolution tasks using patches from low and high resolution versions.
    Now uses optimized patch extraction with image-level caching.
    """

    def __init__(self, original_ds, low_res_config, high_res_config, small_patch_size, large_patch_size,
                 stride, scale_factor, cache_dir="./cache", cache_rebuild=False, max_memory_cache=100):

        assert large_patch_size[0] == small_patch_size[0] * scale_factor
        assert large_patch_size[1] == small_patch_size[1] * scale_factor
        assert high_res_config['target_size'][0] == low_res_config['target_size'][0] * scale_factor
        assert high_res_config['target_size'][1] == low_res_config['target_size'][1] * scale_factor

        self.original_ds = original_ds
        self.scale_factor = scale_factor
        self.stride = stride

        # Processed datasets for low and high res
        low_cache_dir = os.path.join(cache_dir, "low_res")
        high_cache_dir = os.path.join(cache_dir, "high_res")
        os.makedirs(low_cache_dir, exist_ok=True)
        os.makedirs(high_cache_dir, exist_ok=True)

        self.low_res_ds = ProcessedDataset(
            original_ds, cache_dir=low_cache_dir,
            cache_rebuild=cache_rebuild, **low_res_config
        )
        self.high_res_ds = ProcessedDataset(
            original_ds, cache_dir=high_cache_dir,
            cache_rebuild=cache_rebuild, **high_res_config
        )

        # Optimized patch extractors
        low_patch_cache = os.path.join(cache_dir, "low_patches")
        high_patch_cache = os.path.join(cache_dir, "high_patches")

        self.small_patch_extractor = OptimizedPatchExtractor(
            small_patch_size, stride, low_patch_cache,
            low_res_config['target_size'], max_memory_cache
        )
        self.large_patch_extractor = OptimizedPatchExtractor(
            large_patch_size, stride * scale_factor, high_patch_cache,
            high_res_config['target_size'], max_memory_cache
        )

        assert self.small_patch_extractor.num_patches_per_image == \
               self.large_patch_extractor.num_patches_per_image

        self.num_images = len(self.low_res_ds)
        self.num_patches_per_image = self.small_patch_extractor.num_patches_per_image

        Logger.info(f"SuperResPatchDataset initialized: {self.num_images} images, "
                    f"{self.num_patches_per_image} patches per image")

    def __len__(self):
        return self.num_images * self.num_patches_per_image

    def __getitem__(self, idx):
        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        img_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image

        # Additional validation
        if img_idx >= self.num_images:
            raise IndexError(f"Image index {img_idx} out of range (max: {self.num_images - 1})")

        label = self.low_res_ds.labels[img_idx].item()

        # Get processed image tensors and convert to PIL for patch extraction
        low_tensor = self.low_res_ds.data[img_idx]
        high_tensor = self.high_res_ds.data[img_idx]

        low_pil = transforms.ToPILImage()(low_tensor.squeeze())
        high_pil = transforms.ToPILImage()(high_tensor.squeeze())

        # Get specific patch (will load/cache all patches for the image)
        small_patch = self.small_patch_extractor.get_patch(low_pil, img_idx, patch_idx)
        large_patch = self.large_patch_extractor.get_patch(high_pil, img_idx, patch_idx)

        X = (label, img_idx, patch_idx, small_patch)
        y = large_patch

        return X, y

    def prefetch_image(self, img_idx):
        """
        Prefetch all patches for a specific image into memory cache.
        Useful for sequential processing of patches from the same image.
        """
        low_tensor = self.low_res_ds.data[img_idx]
        high_tensor = self.high_res_ds.data[img_idx]

        low_pil = transforms.ToPILImage()(low_tensor.squeeze())
        high_pil = transforms.ToPILImage()(high_tensor.squeeze())

        # This will load all patches into memory cache
        self.small_patch_extractor.process(low_pil, img_idx)
        self.large_patch_extractor.process(high_pil, img_idx)

        Logger.micro(f"Prefetched all patches for image {img_idx}")

    def get_cache_stats(self):
        """Get cache statistics from both extractors."""
        return {
            'small_patches': self.small_patch_extractor.get_cache_stats(),
            'large_patches': self.large_patch_extractor.get_cache_stats()
        }

    def clear_memory_caches(self):
        """Clear memory caches of both extractors."""
        self.small_patch_extractor.clear_memory_cache()
        self.large_patch_extractor.clear_memory_cache()

    def reconstruct_low(self, img_idx, device=torch.device("cpu")):
        low_tensor = self.low_res_ds.data[img_idx]
        low_pil = transforms.ToPILImage()(low_tensor.squeeze())
        patches = self.small_patch_extractor.process(low_pil, img_idx)
        return self.small_patch_extractor.reconstruct_image(patches, device)

    def reconstruct_high(self, img_idx, device=torch.device("cpu")):
        high_tensor = self.high_res_ds.data[img_idx]
        high_pil = transforms.ToPILImage()(high_tensor.squeeze())
        patches = self.large_patch_extractor.process(high_pil, img_idx)
        return self.large_patch_extractor.reconstruct_image(patches, device)


def show_sample_images(samples, sample_index, zoom_ratio=1.0, patch_coords=None):
    """
    Display sample images side by side, with optional patch location highlighting.
    """
    Logger.info(f"Displaying sample images for sample index: {sample_index}")
    num_images = len(samples)
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    if num_images == 1:
        axes = [axes]

    for ax, (title, tensor) in zip(axes, samples.items()):
        pil_image = transforms.ToPILImage()(tensor)
        width, height = pil_image.size
        ax.imshow(pil_image, cmap='gray')
        ax.set_title(f"{title}\nSize: {width}x{height}")
        ax.axis("off")

        # Draw patch rectangle if provided
        if patch_coords and title in patch_coords:
            top, left, patch_h, patch_w = patch_coords[title]
            rect = plt.Rectangle((left, top), patch_w, patch_h,
                                 linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    fig.suptitle(f"Sample Index: {sample_index}")
    if zoom_ratio != 1.0:
        current_size = fig.get_size_inches()
        new_size = (current_size[0] * zoom_ratio, current_size[1] * zoom_ratio)
        fig.set_size_inches(new_size)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=True)
    Logger.info("Displayed sample images successfully")


def test_optimized_dataset():
    """
    Test function to demonstrate the optimized SuperResPatchDataset.
    Shows cache efficiency and performance improvements.
    """
    Logger.step("Testing Optimized SuperResPatchDataset")

    # Configuration
    low_res_config = {
        'target_size': (14, 14),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None
    }
    high_res_config = {
        'target_size': (28, 28),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None
    }
    small_patch_size = (2, 2)
    large_patch_size = (4, 4)
    stride = 1
    scale_factor = 2
    cache_dir = "./cache"
    cache_rebuild = False
    max_memory_cache = 50  # Keep 50 images in memory

    # Load MNIST dataset - use a small subset for testing
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Use subset of 100 images for testing
    subset_indices = list(range(100))
    original_ds = torch.utils.data.Subset(full_dataset, subset_indices)

    # Create the optimized dataset
    dataset = SuperResPatchDataset(
        original_ds=original_ds,
        low_res_config=low_res_config,
        high_res_config=high_res_config,
        small_patch_size=small_patch_size,
        large_patch_size=large_patch_size,
        stride=stride,
        scale_factor=scale_factor,
        cache_dir=cache_dir,
        cache_rebuild=cache_rebuild,
        max_memory_cache=max_memory_cache
    )

    Logger.info(f"Dataset length: {len(dataset)}")
    Logger.info(f"Number of images: {dataset.num_images}")
    Logger.info(f"Patches per image: {dataset.num_patches_per_image}")

    # Test sequential access (simulating training on patches from same image)
    Logger.step("Testing sequential patch access from same image")

    img_idx = 0
    start_patch = img_idx * dataset.num_patches_per_image
    end_patch = start_patch + 10  # Access first 10 patches

    # First access will load all patches
    for patch_idx in range(start_patch, end_patch):
        X, y = dataset[patch_idx]
        if patch_idx == start_patch:
            Logger.info(f"First patch access - will cache all {dataset.num_patches_per_image} patches")

    # Check cache statistics
    stats = dataset.get_cache_stats()
    Logger.info(f"Cache stats after sequential access:")
    Logger.info(f"  Small patches - Hit rate: {stats['small_patches']['hit_rate']:.2%}")
    Logger.info(f"  Large patches - Hit rate: {stats['large_patches']['hit_rate']:.2%}")

    # Test random access from different images
    Logger.step("Testing random access from different images")

    import random
    # Ensure we don't go out of bounds
    max_index = len(dataset) - 1
    num_samples = min(20, max_index)
    random_indices = random.sample(range(max_index + 1), num_samples)

    for idx in random_indices:
        X, y = dataset[idx]

    # Check cache statistics again
    stats = dataset.get_cache_stats()
    Logger.info(f"Cache stats after random access:")
    Logger.info(f"  Small patches - Hit rate: {stats['small_patches']['hit_rate']:.2%}")
    Logger.info(f"  Large patches - Hit rate: {stats['large_patches']['hit_rate']:.2%}")
    Logger.info(f"  Memory cache sizes: {stats['small_patches']['memory_cache_size']}/{max_memory_cache}")

    # Test reconstruction
    Logger.step("Testing image reconstruction")

    img_idx = 1
    label = dataset.low_res_ds.labels[img_idx].item()

    # Reconstruct images
    recon_low = dataset.reconstruct_low(img_idx)
    recon_high = dataset.reconstruct_high(img_idx)

    # Get original processed images for comparison
    low_tensor = dataset.low_res_ds.data[img_idx]
    high_tensor = dataset.high_res_ds.data[img_idx]

    # Get a sample patch for visualization
    patch_idx = dataset.num_patches_per_image // 2
    global_idx = img_idx * dataset.num_patches_per_image + patch_idx
    X, y = dataset[global_idx]
    small_patch = X[3]
    large_patch = y

    # Calculate patch coordinates
    num_patches_w = dataset.small_patch_extractor.num_patches_w
    row = patch_idx // num_patches_w
    col = patch_idx % num_patches_w
    low_top = row * stride
    low_left = col * stride
    high_top = row * stride * scale_factor
    high_left = col * stride * scale_factor

    patch_coords = {
        'Low Resolution': (low_top, low_left, small_patch_size[0], small_patch_size[1]),
        'High Resolution': (high_top, high_left, large_patch_size[0], large_patch_size[1])
    }

    # Prepare patches for display
    small_patch_img = small_patch.float() / 255.0
    large_patch_img = large_patch.float() / 255.0
    small_upsampled = F.interpolate(
        small_patch_img.unsqueeze(0).unsqueeze(0),
        scale_factor=10, mode='nearest'
    ).squeeze()
    large_upsampled = F.interpolate(
        large_patch_img.unsqueeze(0).unsqueeze(0),
        scale_factor=10, mode='nearest'
    ).squeeze()

    recon_low_tensor = (recon_low.float() / 255.0).unsqueeze(0)
    recon_high_tensor = (recon_high.float() / 255.0).unsqueeze(0)

    # Display all images
    samples = {
        'Low Resolution': low_tensor,
        'High Resolution': high_tensor,
        f'Small Patch (idx {patch_idx})': small_upsampled.unsqueeze(0),
        f'Large Patch (idx {patch_idx})': large_upsampled.unsqueeze(0),
        'Reconstructed Low': recon_low_tensor,
        'Reconstructed High': recon_high_tensor
    }

    show_sample_images(
        samples,
        sample_index=f"Image {img_idx} (Label: {label})",
        patch_coords=patch_coords,
        zoom_ratio=1.5
    )

    # Performance comparison
    Logger.step("Performance Comparison")

    import time

    # Clear caches for fair comparison
    dataset.clear_memory_caches()

    # Measure time for accessing all patches from one image
    # Use a valid image index (must be less than dataset.num_images)
    img_idx = min(10, dataset.num_images - 1)  # Use image 10 or last available
    start_idx = img_idx * dataset.num_patches_per_image
    end_idx = start_idx + dataset.num_patches_per_image

    # First access (cache miss)
    start_time = time.time()
    for idx in range(start_idx, end_idx):
        X, y = dataset[idx]
    first_access_time = time.time() - start_time

    # Second access (cache hit)
    start_time = time.time()
    for idx in range(start_idx, end_idx):
        X, y = dataset[idx]
    second_access_time = time.time() - start_time

    Logger.info(f"Time to access {dataset.num_patches_per_image} patches from one image:")
    Logger.info(f"  First access (cache miss): {first_access_time:.3f}s")
    Logger.info(f"  Second access (cache hit): {second_access_time:.3f}s")
    Logger.info(f"  Speedup: {first_access_time / second_access_time:.1f}x")

    # Final cache statistics
    final_stats = dataset.get_cache_stats()
    Logger.info("\nFinal Cache Statistics:")
    Logger.info(f"  Small patches:")
    Logger.info(
        f"    Total accesses: {final_stats['small_patches']['cache_hits'] + final_stats['small_patches']['cache_misses']}")
    Logger.info(f"    Hit rate: {final_stats['small_patches']['hit_rate']:.2%}")
    Logger.info(f"    Memory usage: {final_stats['small_patches']['memory_cache_size']}/{max_memory_cache} images")
    Logger.info(f"  Large patches:")
    Logger.info(
        f"    Total accesses: {final_stats['large_patches']['cache_hits'] + final_stats['large_patches']['cache_misses']}")
    Logger.info(f"    Hit rate: {final_stats['large_patches']['hit_rate']:.2%}")
    Logger.info(f"    Memory usage: {final_stats['large_patches']['memory_cache_size']}/{max_memory_cache} images")

    Logger.info("\nOptimized dataset test completed successfully!")

    return dataset


def benchmark_cache_performance():
    """
    Benchmark the performance improvements of the optimized caching system.
    """
    Logger.step("Benchmarking Cache Performance")

    import time
    import random

    # Configuration
    config = {
        'low_res_config': {
            'target_size': (14, 14),
            'resize_alg': Image.BICUBIC,
            'image_format': None,
            'quality': None
        },
        'high_res_config': {
            'target_size': (28, 28),
            'resize_alg': Image.BICUBIC,
            'image_format': None,
            'quality': None
        },
        'small_patch_size': (2, 2),
        'large_patch_size': (4, 4),
        'stride': 1,
        'scale_factor': 2,
        'cache_dir': "./cache",
        'cache_rebuild': False
    }

    # Load subset of MNIST for benchmarking
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Use subset for faster benchmarking
    subset_indices = list(range(100))  # First 100 images
    subset = torch.utils.data.Subset(full_dataset, subset_indices)

    # Create optimized dataset with different cache sizes
    cache_sizes = [0, 10, 50, 100]
    results = {}

    for cache_size in cache_sizes:
        Logger.info(f"\nTesting with memory cache size: {cache_size}")

        dataset = SuperResPatchDataset(
            original_ds=subset,
            **config,
            max_memory_cache=cache_size
        )

        # Clear any existing memory cache
        dataset.clear_memory_caches()

        # Test 1: Sequential access (typical training pattern)
        sequential_times = []
        num_test_images = min(10, len(subset))  # Test on up to 10 images
        for img_idx in range(num_test_images):
            start_idx = img_idx * dataset.num_patches_per_image
            end_idx = start_idx + dataset.num_patches_per_image

            start_time = time.time()
            for idx in range(start_idx, end_idx):
                X, y = dataset[idx]
            sequential_times.append(time.time() - start_time)

        # Test 2: Random access
        random_times = []
        num_random_samples = min(100, len(dataset))
        random_indices = random.sample(range(len(dataset)), num_random_samples)

        for idx in random_indices:
            start_time = time.time()
            X, y = dataset[idx]
            random_times.append(time.time() - start_time)

        # Get cache statistics
        stats = dataset.get_cache_stats()

        results[cache_size] = {
            'sequential_avg': sum(sequential_times) / len(sequential_times),
            'random_avg': sum(random_times) / len(random_times),
            'hit_rate': stats['small_patches']['hit_rate'],
            'sequential_times': sequential_times,
            'random_times': random_times
        }

        Logger.info(f"  Sequential access avg: {results[cache_size]['sequential_avg']:.4f}s")
        Logger.info(f"  Random access avg: {results[cache_size]['random_avg']:.4f}s")
        Logger.info(f"  Cache hit rate: {results[cache_size]['hit_rate']:.2%}")

    # Visualize results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Average access times
    ax = axes[0]
    cache_sizes_str = [str(s) for s in cache_sizes]
    sequential_avgs = [results[s]['sequential_avg'] for s in cache_sizes]
    random_avgs = [results[s]['random_avg'] for s in cache_sizes]

    x = range(len(cache_sizes))
    width = 0.35
    ax.bar([i - width / 2 for i in x], sequential_avgs, width, label='Sequential', color='blue')
    ax.bar([i + width / 2 for i in x], random_avgs, width, label='Random', color='orange')
    ax.set_xlabel('Memory Cache Size')
    ax.set_ylabel('Average Access Time (s)')
    ax.set_title('Access Time vs Cache Size')
    ax.set_xticks(x)
    ax.set_xticklabels(cache_sizes_str)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Hit rates
    ax = axes[1]
    hit_rates = [results[s]['hit_rate'] * 100 for s in cache_sizes]
    ax.plot(cache_sizes, hit_rates, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Memory Cache Size')
    ax.set_ylabel('Cache Hit Rate (%)')
    ax.set_title('Cache Hit Rate vs Cache Size')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # Plot 3: Speedup
    ax = axes[2]
    baseline_seq = results[0]['sequential_avg']
    baseline_rand = results[0]['random_avg']
    speedup_seq = [baseline_seq / results[s]['sequential_avg'] for s in cache_sizes]
    speedup_rand = [baseline_rand / results[s]['random_avg'] for s in cache_sizes]

    ax.plot(cache_sizes, speedup_seq, 'b-o', label='Sequential', linewidth=2, markersize=8)
    ax.plot(cache_sizes, speedup_rand, 'r-o', label='Random', linewidth=2, markersize=8)
    ax.set_xlabel('Memory Cache Size')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Speedup vs Cache Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Cache Performance Benchmark Results', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

    Logger.info("\n" + "=" * 50)
    Logger.info("BENCHMARK SUMMARY")
    Logger.info("=" * 50)
    Logger.info(f"Best cache size for sequential access: {cache_sizes[sequential_avgs.index(min(sequential_avgs))]}")
    Logger.info(f"Best cache size for random access: {cache_sizes[random_avgs.index(min(random_avgs))]}")
    Logger.info(f"Maximum speedup achieved (sequential): {max(speedup_seq):.2f}x")
    Logger.info(f"Maximum speedup achieved (random): {max(speedup_rand):.2f}x")

    return results


if __name__ == "__main__":
    # Run the optimized test
    Logger.macro("Running Optimized SuperResPatchDataset Test")
    dataset = test_optimized_dataset()

    # Run performance benchmark
    Logger.macro("Running Cache Performance Benchmark")
    benchmark_results = benchmark_cache_performance()

    Logger.macro("All tests completed successfully!")