import os
import io
import torch
import zstandard as zstd
from PIL import Image
from torchvision import transforms
from Logger import Logger
from .quantize import ImageQuantizer

class ProcessedDataset(torch.utils.data.Dataset):
    """
    Dataset that processes images (resize, compress artifacts, quantize) and caches result.
    """

    def __init__(self, original_ds, target_size, resize_alg=None,
                 image_format=None, quality=None,
                 quantization_levels=None, quantization_method='uniform',
                 cache_dir="./cache", cache_rebuild=False):
        self.original_ds = original_ds
        self.target_size = target_size
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
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        if self.needs_resize and self.target_size is not None:
            img = img.resize(self.target_size, self.resize_alg or Image.BICUBIC)

        if self.needs_compression:
            buffer = io.BytesIO()
            save_kwargs = {'format': self.image_format}
            if self.quality is not None:
                save_kwargs['quality'] = self.quality
            img.save(buffer, **save_kwargs)
            buffer.seek(0)
            img = Image.open(buffer)

        img_tensor = transforms.ToTensor()(img)

        if self.needs_quantization:
            img_tensor = ImageQuantizer.quantize(
                img_tensor,
                levels=self.quantization_levels,
                method=self.quantization_method,
                dithering=(self.quantization_method == 'uniform' and self.quantization_levels == 2)
            )

        return img_tensor

    def _process_and_cache(self):
        processed_images = []
        labels = []
        for img, label in self.original_ds:
            processed_images.append(self._process_image(img))
            labels.append(label)

        data_tensor = torch.stack(processed_images)
        labels_tensor = torch.tensor(labels)

        data_dict = {'data': data_tensor, 'labels': labels_tensor, 'config': self._get_config()}

        buffer = io.BytesIO()
        torch.save(data_dict, buffer)
        buffer.seek(0)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=1, threads=-1)
            with cctx.stream_writer(f) as compressor:
                compressor.write(buffer.read())

        Logger.info(f"Processed and cached dataset saved to {self.cache_path}")
        return data_tensor, labels_tensor

    def _load_cache(self):
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
            # basic size validation
            if len(loaded['data']) != len(self.original_ds):
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
            'quality': self.quality
        }
        if self.needs_quantization:
            cfg['quantization_levels'] = self.quantization_levels
            cfg['quantization_method'] = self.quantization_method
        return cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.original_ds[idx][0], self.labels[idx]