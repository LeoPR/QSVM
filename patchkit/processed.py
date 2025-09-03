import os
import io
import torch
import zstandard as zstd
from .image_utils import resize as image_resize, to_pil, to_tensor
from PIL import Image
from torchvision import transforms
from Logger import Logger
from .quantize import ImageQuantizer


class ProcessedDataset(torch.utils.data.Dataset):
    """
    Dataset that processes images (resize, compress artifacts, quantize) and caches result.
    FIXED: Loop infinito, target_size order, zstd threading
    """

    def __init__(self, original_ds, target_size, resize_alg=Image.BICUBIC,
                 image_format=None, quality=None, resize_backend='torch',
                 quantization_levels=None, quantization_method='uniform',
                 cache_dir="./cache", cache_rebuild=False):
        if target_size is not None:
            if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
                raise ValueError(f"target_size must be (height, width), got {target_size}")
            h, w = target_size
            if h <= 0 or w <= 0:
                raise ValueError(f"target_size must have positive dimensions, got {target_size}")

        if quantization_levels is not None and quantization_levels < 2:
            raise ValueError(f"quantization_levels must be >= 2, got {quantization_levels}")

        self.original_ds = original_ds
        self.target_size = target_size
        self.resize_backend = resize_backend
        self.resize_alg = resize_alg
        self.image_format = image_format.upper() if image_format else None
        self.quality = quality
        self.quantization_levels = quantization_levels
        self.quantization_method = quantization_method
        self.cache_rebuild = cache_rebuild

        self.needs_resize = (self.target_size is not None)
        self.needs_compression = self.image_format in ['JPEG', 'PNG']
        self.needs_quantization = quantization_levels is not None

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = self._create_cache_path(cache_dir)

        if os.path.exists(self.cache_path) and not self.cache_rebuild:
            self.data, self.labels = self._load_cache()
        else:
            self.data, self.labels = self._process_and_cache()

    def _create_cache_path(self, cache_dir):
        base = f"{len(self.original_ds)}"
        quant = f"_q{self.quantization_levels}_{self.quantization_method}" if self.needs_quantization else ""
        fmt = f"_{self.image_format}_q{self.quality}" if self.needs_compression else ""
        resize = f"_{self.target_size[0]}x{self.target_size[1]}" if self.target_size is not None else ""
        fname = f"processed_{base}{resize}{fmt}{quant}.pt.zst"
        return os.path.join(cache_dir, fname)

    def _process_image(self, img):
        """Process single image using image_utils.resize to respect backend selection.
        Ensures correct types for later compression/quantization steps.
        """
        # Decide return type for resize: if we will compress (save to a format) we need PIL.Image,
        # otherwise we can stay with tensor for quantization and downstream processing.
        desired_return = 'pil' if self.needs_compression else 'tensor'

        # Perform resize only if required
        if self.needs_resize and self.target_size is not None:
            # image_resize expects target_size (H, W)
            img = image_resize(
                img,
                target_size=self.target_size,
                alg=self.resize_alg or Image.BICUBIC,
                backend=self.resize_backend,
                return_type=desired_return
            )

        # If compression is required, ensure we have a PIL image to save into the buffer
        if self.needs_compression:
            if not isinstance(img, Image.Image):
                img = to_pil(img)
            buffer = io.BytesIO()
            save_kwargs = {'format': self.image_format}
            if self.quality is not None:
                save_kwargs['quality'] = self.quality
            img.save(buffer, **save_kwargs)
            buffer.seek(0)
            # Re-open to normalize the PIL Image object (Pillow handles the decoding)
            img = Image.open(buffer).convert(img.mode)

        # Ensure we have a tensor (C,H,W) in float [0,1] for quantization / stacking
        if not isinstance(img, torch.Tensor):
            img_tensor = to_tensor(img)
        else:
            img_tensor = to_tensor(img)  # ensures dtype and shape

        if self.needs_quantization:
            img_tensor = ImageQuantizer.quantize(
                img_tensor,
                levels=self.quantization_levels,
                method=self.quantization_method,
                dithering=(self.quantization_method == 'uniform' and self.quantization_levels == 2)
            )

        return img_tensor

    def _process_and_cache(self):
        """Process all images with bounds checking to prevent infinite loops"""
        processed_images = []
        labels = []

        # Get dataset size with proper bounds
        try:
            total = len(self.original_ds)
        except Exception:
            total = 0
            Logger.warning("Could not determine dataset size")

        if total == 0:
            Logger.info("Empty dataset")
            data_tensor = torch.empty((0, 1, 1, 1))
            labels_tensor = torch.empty((0,), dtype=torch.long)
            self._save_cache({'data': data_tensor, 'labels': labels_tensor, 'config': self._get_config()})
            return data_tensor, labels_tensor

        Logger.info(f"Processing {total} images...")

        # Explicit bounds checking to prevent infinite loops
        for i, item in enumerate(self.original_ds):
            try:
                # Support datasets that return either (img, label) or (data, label, ...)
                if isinstance(item, tuple) or isinstance(item, list):
                    img, label = item[0], item[1]
                else:
                    # If dataset yields a single value, consider label unknown
                    img, label = item, None

                processed_img = self._process_image(img)
                processed_images.append(processed_img)
                labels.append(label)

                # Progress logging
                if total > 10 and (i + 1) % max(1, total // 10) == 0:
                    Logger.info(f"Processed {i + 1}/{total} ({100 * (i + 1) / total:.0f}%)")

            except Exception as e:
                Logger.error(f"Failed to process image {i}: {e}")
                continue

        if not processed_images:
            raise RuntimeError("No images were successfully processed")

        data_tensor = torch.stack(processed_images)
        # Convert labels to tensor (use -1 for unknown labels)
        labels_safe = [(-1 if l is None else int(l)) for l in labels]
        labels_tensor = torch.tensor(labels_safe, dtype=torch.long)

        data_dict = {'data': data_tensor, 'labels': labels_tensor, 'config': self._get_config()}
        self._save_cache(data_dict)

        Logger.info(f"Processed and cached dataset saved to {self.cache_path}")
        return data_tensor, labels_tensor

    def _save_cache(self, data_dict):
        """Save with safer zstd compression options (single thread for portability)"""
        buffer = io.BytesIO()
        torch.save(data_dict, buffer)
        buffer.seek(0)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Use single thread zstd for portability (Windows / CI)
        with open(self.cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=1, threads=-1)
            with cctx.stream_writer(f) as compressor:
                compressor.write(buffer.read())

    def _load_cache(self):
        """Load cache with proper error handling"""
        try:
            with open(self.cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    decompressed = reader.read()
            buffer = io.BytesIO(decompressed)
            loaded = torch.load(buffer, map_location="cpu")

            if 'data' not in loaded or 'labels' not in loaded:
                Logger.warning("Invalid cache format; rebuilding.")
                return self._process_and_cache()

            # If original dataset length differs, rebuild (safe)
            try:
                orig_len = len(self.original_ds)
            except Exception:
                orig_len = None

            if orig_len is not None and len(loaded['data']) != orig_len:
                Logger.warning("Cache size mismatch; rebuilding.")
                return self._process_and_cache()

            return loaded['data'], loaded['labels']

        except Exception as e:
            Logger.warning(f"Failed to load cache: {e}; rebuilding.")
            return self._process_and_cache()

    def _get_config(self):
        cfg = {
            'target_size': self.target_size,
            'resize_alg': str(self.resize_alg),
            'image_format': self.image_format,
            'quality': self.quality,
            'resize_backend': self.resize_backend
        }
        if self.needs_quantization:
            cfg['quantization_levels'] = self.quantization_levels
            cfg['quantization_method'] = self.quantization_method
        return cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return (processed_image, label) which is what's expected by most tests.
        # label is an int; if unknown, will be -1.
        return self.data[idx], int(self.labels[idx].item())