import os
import zstandard as zstd
import torch
import io
from PIL import Image
from torchvision import transforms
from Logger import Logger  # Import the Logger class
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

RESIZE_ALG_MAP = {
    Image.NEAREST: "nearest",
    Image.BILINEAR: "bilinear",
    Image.BICUBIC: "bicubic",
    Image.LANCZOS: "lanczos",
}

class CachedDataset:
    def __init__(self, config, dataset, processor, cache_rebuild: bool = False):
        """
        Initialize the CachedDataset loader.

        Parameters:
            config (dict): Configuration dictionary with keys such as "image_format" and "cache_dir".
            dataset: The raw dataset that contains data to process. Expected to have a .data attribute.
            processor: An object with a 'process' method for processing each image.
                       It should have the following attributes:
                         - target_size (tuple): The target dimensions (e.g., (width, height)).
                         - do_compression (bool): Flag to include compression info in the filename.
                         - quality (int): Compression quality level, if applicable.
                         - resize_alg: The algorithm to use for resizing.
        """
        self.config = config
        self.dataset = dataset
        self.processor = processor
        self.cached_data = None
        self.cache_rebuild = cache_rebuild
        self.dataset_size = len(self.dataset)
        # Include the resize algorithm name in the cache filename.
        alg_name = RESIZE_ALG_MAP.get(processor.resize_alg, "unknown")
        self.cache_filename = f"MNIST_{self.dataset_size}_{processor.image_format}_{processor.processor_type}_r{processor.target_size[0]}x{processor.target_size[1]}_{alg_name}"
        if processor.do_compression:
            self.cache_filename += f"_c{processor.quality}"
        self.cache_filename += ".pt.zst"
        self.cache_path = os.path.join(config["cache_dir"], self.cache_filename)
        Logger.info(f"Cache filename set to {self.cache_filename}")

    def load_data(self):
        """
        Loads the processed dataset from cache if available; otherwise, processes the dataset,
        saves it to cache, and returns the processed data.

        Returns:
            The processed dataset.
        """
        if os.path.exists(self.cache_path) and not self.cache_rebuild:
            Logger.info(f"Cache found at {self.cache_path}. Loading low-resolution train dataset from outer cache.")
            with open(self.cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                Logger.micro("Decompressing cached data stream...")
                with dctx.stream_reader(f) as reader:
                    decompressed_data = reader.read()
                buffer = io.BytesIO(decompressed_data)
                self.cached_data = torch.load(buffer)
            Logger.info("Loaded dataset from cache successfully.")
        else:
            Logger.info(
                "Cache not found or rebuild requested. Processing low-resolution train dataset and saving to outer cache.")
            # Process each image in the dataset.
            self.cached_data = []
            for i, img in enumerate(self.dataset):
                Logger.step(f"Processing image {i}")
                ii = img.numpy()
                iii = Image.fromarray(ii, mode='L')
                processed_tensor = self.processor.process(iii, index=i)
                #self.cached_data.append(processed_tensor)
                if processed_tensor.ndim == 2:
                    self.cached_data.append(processed_tensor)
                elif processed_tensor.ndim == 3:
                    self.cached_data.extend(processed_tensor) #list(processed_tensor.unbind(0)))

            with open(self.cache_path, 'wb') as f:
                cctx = zstd.ZstdCompressor(level=1, threads=34)
                Logger.micro("Compressing processed data stream for caching...")
                with cctx.stream_writer(f) as compressor:
                    torch.save(self.cached_data, compressor)
            Logger.info(f"Processed and cached dataset saved to {self.cache_path}")
        return self.cached_data


# ------------------------
# Improved Low-Resolution Dataset
# ------------------------

class LowResDataset(torch.utils.data.Dataset):
    """
    Creates a low-resolution version of a dataset with optional resizing and compression artifacts.
    Automatically infers processing needs based on parameters.

    Features:
    - Automatic resizing detection (if target_size != original size)
    - Automatic compression detection (based on image_format)
    - Compatible with torch.utils.data.ConcatDataset
    - Maintains original dataset structure for easy pairing

    Args:
        original_ds: Dataset containing PIL Images and labels
        target_size: (width, height) for output images
        resize_alg: PIL resize algorithm (e.g., Image.BICUBIC)
        image_format: None/'RAW' for no compression, 'JPEG'/'PNG' for compression
        quality: Compression quality (1-100) for lossy formats
        cache_dir: Directory for processed data caching
    """

    def __init__(self, original_ds, target_size, resize_alg,
                 image_format=None, quality=None, cache_dir="./cache"):
        self.original_ds = original_ds
        self.target_size = target_size
        self.resize_alg = resize_alg
        self.image_format = image_format.upper() if image_format else None
        self.quality = quality

        # Auto-detect processing needs
        self.needs_resize = (self.target_size != self._get_original_size())
        self.needs_compression = self.image_format in ['JPEG', 'PNG']

        # Create unique cache name based on parameters
        self.cache_path = self._create_cache_path(cache_dir)

        # Load or process data
        if os.path.exists(self.cache_path):
            self.data, self.labels = self._load_cache()
        else:
            self.data, self.labels = self._process_and_cache()

    def _get_original_size(self):
        """Get size from first image (assumes consistent sizes)"""
        img, _ = self.original_ds[0]
        return img.size  # (width, height)

    def _create_cache_path(self, cache_dir):
        """Generate unique cache filename based on parameters"""
        original_size = self._get_original_size()
        resize_info = (f"{original_size[0]}x{original_size[1]}"
                       f"_{self.target_size[0]}x{self.target_size[1]}")

        alg_name = RESIZE_ALG_MAP.get(self.resize_alg, "unknown")
        format_info = f"{self.image_format}_q{self.quality}" if self.needs_compression else "raw"

        return os.path.join(
            cache_dir,
            f"lowres_{resize_info}_{alg_name}_{format_info}.pt"
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
        """Process all images and save to cache"""
        processed_images = []
        labels = []

        for img, label in tqdm(self.original_ds, desc="Processing Low-Res"):
            processed_img = self._process_image(img)
            processed_images.append(processed_img)
            labels.append(label)

        data_tensor = torch.stack(processed_images)
        labels_tensor = torch.tensor(labels)

        torch.save({
            'data': data_tensor,
            'labels': labels_tensor,
            'config': self._get_config()
        }, self.cache_path)

        return data_tensor, labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return (lowres_image, original_image, label) for explicit pairing"""
        return self.data[idx], self.original_ds[idx][0], self.labels[idx]


class PatchExtractor:
    def __init__(self, patch_size, stride, cache_dir, image_size):
        """
        Initialize the PatchExtractor.

        Parameters:
            patch_size (tuple): Dimensions of each patch (height, width).
            stride (int or tuple): Stride between patches.
            cache_dir (str): Base directory for caching extracted patches.
        """
        self.processor_type = 'patch'
        self.image_format = 'none'
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.resize_alg = None
        self.target_size = patch_size
        self.do_compression = False
        self.do_resize = False
        self.num_patches_per_image = self.calc_n_patchs_p_img()

        # Create a subfolder for patch caching.
        self.cache_dir = os.path.join(cache_dir, "patches")
        os.makedirs(self.cache_dir, exist_ok=True)
        Logger.info(
            f"PatchExtractor initialized with patch size {self.patch_size} and stride {self.stride}. Cache directory: {self.cache_dir}")

    def calc_n_patchs_p_img(self):
        num_patches_width = (self.image_size[0] - self.patch_size[0]) // self.stride + 1
        num_patches_height = (self.image_size[1] - self.patch_size[1]) // self.stride + 1
        return num_patches_width * num_patches_height


    def process(self, image, index=None):
        """
        Extract patches from a PIL image using the specified patch size and stride.

        Parameters:
            image (PIL.Image): Input image.
            index (int, optional): Optional index for generating a unique cache filename.

        Returns:
            patches (torch.Tensor): Tensor of extracted patches with shape (num_patches, channels, patch_height, patch_width).
        """
        # Build a unique cache filename incorporating image dimensions, patch size, stride, and index.
        fname = f"w{image.size[0]}_h{image.size[1]}_p{self.patch_size[0]}x{self.patch_size[1]}_s{self.stride}"
        if index is not None:
            fname += f"_i{index}"
        fname += ".pt"
        cache_path = os.path.join(self.cache_dir, fname)

        if os.path.exists(cache_path):
            Logger.info(f"Patch cache exists for image {index} at {cache_path}. Loading cached patches.")
            return torch.load(cache_path)

        Logger.step(f"Extracting patches from image {index}: original size {image.size}")

        # Convert image to tensor (shape: [C, H, W]).
        tensor = transforms.ToTensor()(image)
        # Add a batch dimension: [1, C, H, W].
        tensor = tensor.unsqueeze(0)

        # Use unfold to extract patches.
        patch_h, patch_w = self.patch_size
        patches = F.unfold(tensor, kernel_size=(patch_h, patch_w), stride=self.stride)
        # The result has shape [1, C*patch_h*patch_w, L] where L is the number of patches.
        # Reshape it to [L, C, patch_h, patch_w].
        patches = patches.squeeze(0).transpose(0, 1)
        patches = patches.reshape(-1, patch_h, patch_w)
        patches = (patches * 255).round().to(torch.uint8)

        Logger.micro(f"Extracted {patches.shape[0]} patches from image {index}.")
        torch.save(patches, cache_path)
        Logger.info(f"Cached extracted patches for image {index} at {cache_path}")
        return patches

    def get_patch_from_index(self, dataset, index):
        start_idx = index * self.num_patches_per_image
        end_idx = start_idx + self.num_patches_per_image
        dataset_sample = dataset[start_idx:end_idx, :] #.to(self.device)
        return dataset_sample

    def reconstruct_image(self, patches, device=torch.device("cpu")):
        """
        Reconstruct an image from a set of patches.

        This method now uses the already available parameters in the class to simplify the logic.
        """
        grid_size = int(patches.shape[0] ** 0.5)  # Number of patches along one axis (square grid)
        recon_h = self.patch_size[0] + (grid_size - 1) * self.stride
        recon_w = recon_h

        reconstructed = torch.zeros((recon_h, recon_w), dtype=torch.float32, device=device)
        weight = torch.zeros((recon_h, recon_w), dtype=torch.float32, device=device)

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                top = i * self.stride
                left = j * self.stride
                reconstructed[top:top + self.patch_size[0], left:left + self.patch_size[1]] += patches[
                                                                                                   idx].float() / 255.0
                weight[top:top + self.patch_size[0], left:left + self.patch_size[1]] += 1

        reconstructed /= weight
        reconstructed = (reconstructed * 255).round().to(torch.uint8)
        return reconstructed


def show_sample_images(samples, sample_index, zoom_ratio=1.0):
    """
    Display sample images side by side.

    Parameters:
      samples: Dictionary where each key is a title and each value is an image tensor.
      sample_index: Identifier for the sample (shown in the figure title).
      zoom_ratio: Factor to scale the figure size.
    """
    Logger.info(f"Displaying sample images for sample index: {sample_index}")
    num_images = len(samples)
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    if num_images == 1:
        axes = [axes]
    for ax, (title, tensor) in zip(axes, samples.items()):
        pil_image = transforms.ToPILImage()(tensor)
        width, height = pil_image.size
        ax.imshow(pil_image)
        ax.set_title(f"{title}\nSize: {width}x{height}")
        ax.axis("off")
    fig.suptitle(f"Sample Index: {sample_index}")
    if zoom_ratio != 1.0:
        current_size = fig.get_size_inches()
        new_size = (current_size[0] * zoom_ratio, current_size[1] * zoom_ratio)
        fig.set_size_inches(new_size)
    plt.show(block=True)
    Logger.info("Displayed sample images successfully")
