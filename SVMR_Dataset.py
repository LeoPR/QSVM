import os
import zstandard as zstd
import torch
import io
from PIL import Image
from torchvision import transforms, datasets
from Logger import Logger  # Import the Logger class
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

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

        with open(self.cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=1, threads=-1)
            Logger.micro("Compressing processed data stream for caching...")
            with cctx.stream_writer(f) as compressor:
                compressor.write(buffer.read())

        Logger.info(f"Processed and cached dataset saved to {self.cache_path}")

        return data_tensor, labels_tensor

    def _load_cache(self):
        """Load cached data with zstd decompression"""
        Logger.info(f"Loading dataset from cache at {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            Logger.micro("Decompressing cached data stream...")
            with dctx.stream_reader(f) as reader:
                decompressed_data = reader.read()
        buffer = io.BytesIO(decompressed_data)
        loaded = torch.load(buffer)
        Logger.info("Loaded dataset from cache successfully.")
        return loaded['data'], loaded['labels']

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


class PatchExtractor:
    def __init__(self, patch_size, stride, cache_dir, image_size):
        """
        Initialize the PatchExtractor.

        Parameters:
            patch_size (tuple): Dimensions of each patch (height, width).
            stride (int): Stride between patches (assumes same for height and width).
            cache_dir (str): Base directory for caching extracted patches.
            image_size (tuple): Image dimensions (height, width).
        """
        self.processor_type = 'patch'
        self.image_format = 'none'
        self.image_size = image_size  # (height, width)
        self.patch_size = patch_size  # (height, width)
        self.stride = stride
        self.resize_alg = None
        self.target_size = patch_size
        self.do_compression = False
        self.do_resize = False

        # Calculate number of patches
        self.num_patches_h = (self.image_size[0] - self.patch_size[0]) // self.stride + 1
        self.num_patches_w = (self.image_size[1] - self.patch_size[1]) // self.stride + 1
        self.num_patches_per_image = self.num_patches_h * self.num_patches_w

        # Create a subfolder for patch caching.
        self.cache_dir = os.path.join(cache_dir, "patches")
        os.makedirs(self.cache_dir, exist_ok=True)
        Logger.info(
            f"PatchExtractor initialized with patch size {self.patch_size} and stride {self.stride}. Cache directory: {self.cache_dir}")

    def process(self, image, index=None):
        """
        Extract patches from a PIL image using the specified patch size and stride.

        Parameters:
            image (PIL.Image): Input image.
            index (int, optional): Optional index for generating a unique cache filename.

        Returns:
            patches (torch.Tensor): Tensor of extracted patches with shape (num_patches, patch_height, patch_width).
        """
        # Build a unique cache filename incorporating image dimensions, patch size, stride, and index.
        fname = f"h{image.size[1]}_w{image.size[0]}_p{self.patch_size[0]}x{self.patch_size[1]}_s{self.stride}"
        if index is not None:
            fname += f"_i{index}"
        fname += ".pt"
        cache_path = os.path.join(self.cache_dir, fname)

        if os.path.exists(cache_path):
            Logger.micro(f"Patch cache exists for image {index} at {cache_path}. Loading cached patches.")
            return torch.load(cache_path)

        Logger.micro(f"Extracting patches from image {index}: original size {image.size}")

        # Convert image to tensor (shape: [C, H, W]).
        tensor = transforms.ToTensor()(image)
        # Add a batch dimension: [1, C, H, W].
        tensor = tensor.unsqueeze(0)

        # Use unfold to extract patches.
        patch_h, patch_w = self.patch_size
        patches = F.unfold(tensor, kernel_size=(patch_h, patch_w), stride=self.stride)
        # The result has shape [1, C*patch_h*patch_w, L] where L is the number of patches.
        # Reshape it to [L, patch_h, patch_w] assuming C=1 (grayscale).
        patches = patches.squeeze(0).transpose(0, 1)
        patches = patches.reshape(-1, patch_h, patch_w)
        patches = (patches * 255).round().to(torch.uint8)

        Logger.micro(f"Extracted {patches.shape[0]} patches from image {index}.")
        torch.save(patches, cache_path)
        Logger.micro(f"Cached extracted patches for image {index} at {cache_path}")
        return patches

    def reconstruct_image(self, patches, device=torch.device("cpu")):
        """
        Reconstruct an image from a set of patches using class parameters.
        Handles non-square grids properly.
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

        reconstructed /= weight.clamp(min=1e-6)  # Avoid division by zero
        reconstructed = (reconstructed * 255).round().to(torch.uint8)
        return reconstructed


class SuperResPatchDataset(torch.utils.data.Dataset):
    """
    Dataset for super-resolution tasks using patches from low and high resolution versions of MNIST.
    Provides small patches from low-res as input (X) and corresponding larger patches from high-res as target (Y).
    Includes metadata: label (class), image index, patch index, small_patch.

    Args:
        original_ds: Original MNIST dataset (e.g., torchvision.datasets.MNIST)
        low_res_config: Dict with 'target_size', 'resize_alg', 'image_format', 'quality'
        high_res_config: Similar dict for high-res (e.g., original size)
        small_patch_size: Tuple (height, width) for small patches (e.g., (2,2))
        large_patch_size: Tuple (height, width) for large patches (e.g., (4,4))
        stride: Stride for small patches (large stride will be stride * scale_factor)
        scale_factor: Integer scale between low and high res (e.g., 2)
        cache_dir: Base cache directory
        cache_rebuild: If True, rebuild all caches
    """
    def __init__(self, original_ds, low_res_config, high_res_config, small_patch_size, large_patch_size,
                 stride, scale_factor, cache_dir="./cache", cache_rebuild=False):
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

        self.low_res_ds = ProcessedDataset(original_ds, cache_dir=low_cache_dir, cache_rebuild=cache_rebuild, **low_res_config)
        self.high_res_ds = ProcessedDataset(original_ds, cache_dir=high_cache_dir, cache_rebuild=cache_rebuild, **high_res_config)

        # Patch extractors
        low_patch_cache = os.path.join(cache_dir, "low_patches")
        high_patch_cache = os.path.join(cache_dir, "high_patches")
        self.small_patch_extractor = PatchExtractor(small_patch_size, stride, low_patch_cache, low_res_config['target_size'])
        self.large_patch_extractor = PatchExtractor(large_patch_size, stride * scale_factor, high_patch_cache, high_res_config['target_size'])

        assert self.small_patch_extractor.num_patches_per_image == self.large_patch_extractor.num_patches_per_image

        self.num_images = len(self.low_res_ds)
        self.num_patches_per_image = self.small_patch_extractor.num_patches_per_image

    def __len__(self):
        return self.num_images * self.num_patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image

        label = self.low_res_ds.labels[img_idx].item()

        # Get processed image tensors and convert to PIL for patch extraction
        low_tensor = self.low_res_ds.data[img_idx]
        high_tensor = self.high_res_ds.data[img_idx]

        low_pil = transforms.ToPILImage()(low_tensor.squeeze())  # Assuming grayscale
        high_pil = transforms.ToPILImage()(high_tensor.squeeze())

        # Extract patches (cached per image)
        small_patches = self.small_patch_extractor.process(low_pil, img_idx)
        large_patches = self.large_patch_extractor.process(high_pil, img_idx)

        small_patch = small_patches[patch_idx]
        large_patch = large_patches[patch_idx]

        X = (label, img_idx, patch_idx, small_patch)
        y = large_patch

        return X, y

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

    Parameters:
      samples: Dictionary where each key is a title and each value is an image tensor.
      sample_index: Identifier for the sample (shown in the figure title).
      zoom_ratio: Factor to scale the figure size.
      patch_coords: Dict with keys matching sample titles containing (top, left, height, width) for patch rectangles.
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
            rect = plt.Rectangle((left, top), patch_w, patch_h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    fig.suptitle(f"Sample Index: {sample_index}")
    if zoom_ratio != 1.0:
        current_size = fig.get_size_inches()
        new_size = (current_size[0] * zoom_ratio, current_size[1] * zoom_ratio)
        fig.set_size_inches(new_size)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=True)
    Logger.info("Displayed sample images successfully")


def test_superres_dataset():
    """
    Test function to demonstrate the SuperResPatchDataset class.
    Loads MNIST, creates low and high res versions, selects an informative patch,
    shows original and resized images with patch locations, the patches themselves,
    and the reconstructed images in a single figure.
    """
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
    cache_rebuild = False  # Set to True to force rebuild

    # Load MNIST dataset (train set)
    transform = transforms.Compose([transforms.ToTensor()])
    original_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Create the SuperResPatchDataset
    dataset = SuperResPatchDataset(
        original_ds=original_ds,
        low_res_config=low_res_config,
        high_res_config=high_res_config,
        small_patch_size=small_patch_size,
        large_patch_size=large_patch_size,
        stride=stride,
        scale_factor=scale_factor,
        cache_dir=cache_dir,
        cache_rebuild=cache_rebuild
    )

    print(f"Dataset length: {len(dataset)}")

    # Select a sample image index
    img_idx = 0

    # Get low and high res images
    low_tensor = dataset.low_res_ds.data[img_idx]  # (1, 14, 14)
    high_tensor = dataset.high_res_ds.data[img_idx]  # (1, 28, 28)
    label = dataset.low_res_ds.labels[img_idx].item()

    # Convert to PIL for patch extraction
    low_pil = transforms.ToPILImage()(low_tensor.squeeze())
    high_pil = transforms.ToPILImage()(high_tensor.squeeze())

    # Extract all patches
    small_patches = dataset.small_patch_extractor.process(low_pil, img_idx)
    large_patches = dataset.large_patch_extractor.process(high_pil, img_idx)

    # Select an informative patch (highest variance, not all dark or bright)
    variances = torch.var(small_patches.float(), dim=(1, 2))
    valid_patches = (variances > variances.mean()) & (small_patches.max(dim=1)[0].max(dim=1)[0] < 255) & (small_patches.min(dim=1)[0].min(dim=1)[0] > 0)
    if valid_patches.any():
        patch_idx = torch.argmax(variances[valid_patches]).item()
        patch_idx = (valid_patches.nonzero(as_tuple=False)[patch_idx] % dataset.num_patches_per_image).item()
    else:
        patch_idx = 0  # Fallback to first patch if none are valid

    # Get the selected patch
    small_patch = small_patches[patch_idx]
    large_patch = large_patches[patch_idx]

    # Calculate patch coordinates for visualization
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

    # Prepare patches for display (upsampled for visibility)
    small_patch_img = small_patch.float() / 255.0
    large_patch_img = large_patch.float() / 255.0
    small_upsampled = torch.nn.functional.interpolate(small_patch_img.unsqueeze(0).unsqueeze(0), scale_factor=10, mode='nearest').squeeze(0).squeeze(0)
    large_upsampled = torch.nn.functional.interpolate(large_patch_img.unsqueeze(0).unsqueeze(0), scale_factor=10, mode='nearest').squeeze(0).squeeze(0)

    # Reconstruct images from patches
    recon_low = dataset.reconstruct_low(img_idx)
    recon_high = dataset.reconstruct_high(img_idx)
    recon_low_tensor = (recon_low.float() / 255.0).unsqueeze(0)
    recon_high_tensor = (recon_high.float() / 255.0).unsqueeze(0)

    # Display all images in a single figure
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

    # Print patch details
    print(f"\nSelected Patch {patch_idx}:")
    print(f"  Label: {label}, Image Index: {img_idx}, Patch Index: {patch_idx}")
    print(f"  Small Patch (2x2):\n{small_patch}")
    print(f"  Large Patch (4x4):\n{large_patch}")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_superres_dataset()