import os
import zstandard as zstd
import torch
import io
from PIL import Image
from torchvision import transforms, datasets
from Logger import Logger
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm
from collections import OrderedDict
import hashlib
import numpy as np

Logger.set_log_level("INFO")

RESIZE_ALG_MAP = {
    Image.NEAREST: "nearest",
    Image.BILINEAR: "bilinear",
    Image.BICUBIC: "bicubic",
    Image.LANCZOS: "lanczos",
}


class ImageQuantizer:
    """
    Quantiza imagens para diferentes níveis de cores.
    Útil para criar datasets com diferentes níveis de complexidade visual.
    """

    @staticmethod
    def quantize(image, levels=2, method='uniform', dithering=False):
        """
        Quantiza uma imagem para um número específico de níveis.

        Args:
            image: Tensor ou PIL Image
            levels: Número de níveis de quantização (2 = preto e branco)
            method: 'uniform', 'kmeans', 'otsu', 'adaptive'
            dithering: Aplicar Floyd-Steinberg dithering (apenas para uniform)

        Returns:
            Imagem quantizada como tensor
        """
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)

        if method == 'uniform':
            return ImageQuantizer._uniform_quantize(image, levels, dithering)
        elif method == 'kmeans':
            return ImageQuantizer._kmeans_quantize(image, levels)
        elif method == 'otsu':
            return ImageQuantizer._otsu_binarize(image)
        elif method == 'adaptive':
            return ImageQuantizer._adaptive_quantize(image, levels)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

    @staticmethod
    def _uniform_quantize(image, levels, dithering=False):
        """Quantização uniforme com opção de dithering."""
        if dithering and levels == 2:
            # Floyd-Steinberg dithering para binarização
            return ImageQuantizer._floyd_steinberg_dither(image)

        # Quantização simples uniforme
        if levels == 2:
            return (image > 0.5).float()

        step = 1.0 / (levels - 1)
        quantized = torch.round(image / step) * step
        return quantized.clamp(0, 1)

    @staticmethod
    def _floyd_steinberg_dither(image):
        """
        Aplica Floyd-Steinberg dithering para binarização.
        Produz preto e branco puro preservando detalhes visuais.
        """
        img = image.clone()
        h, w = img.shape[-2:]

        for y in range(h):
            for x in range(w):
                old_pixel = img[..., y, x]
                new_pixel = torch.round(old_pixel)
                img[..., y, x] = new_pixel
                error = old_pixel - new_pixel

                # Distribuir erro para pixels vizinhos
                if x + 1 < w:
                    img[..., y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        img[..., y + 1, x - 1] += error * 3 / 16
                    img[..., y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        img[..., y + 1, x + 1] += error * 1 / 16

        return img.clamp(0, 1)

    @staticmethod
    def _kmeans_quantize(image, levels):
        """Quantização usando K-means clustering.

        Observações:
        - Se `image` estiver na GPU, trazemos para CPU com detach().cpu() antes de converter
          para NumPy, evitando erros e dinâmica do autograd.
        - Reconstruímos o tensor final e movemos de volta para o mesmo device do `image`.
        - Usamos float32 para reduzir uso de memória.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            Logger.warning("sklearn not available, falling back to uniform quantization")
            return ImageQuantizer._uniform_quantize(image, levels, False)

        # Reshape para lista de pixels (garantir CPU, detach e array contíguo)
        original_shape = tuple(image.shape)
        pixels_np = image.detach().cpu().flatten().numpy().reshape(-1, 1)

        # K-means clustering (sklearn trabalha em CPU)
        kmeans = KMeans(n_clusters=levels, random_state=42, n_init=10)
        kmeans.fit(pixels_np)

        # Mapear cada pixel para o centro do cluster mais próximo
        quantized_np = kmeans.cluster_centers_[kmeans.labels_].reshape(original_shape)

        # Converter de volta para tensor no mesmo device do input, com dtype float32
        quantized_t = torch.from_numpy(np.ascontiguousarray(quantized_np)).to(image.device).float()

        return quantized_t

    @staticmethod
    def _otsu_binarize(image):
        """Binarização usando método de Otsu.

        Observações:
        - Trazer para CPU e desconectar do grafo antes da conversão para NumPy.
        - Retornar tensor no mesmo dispositivo do `image` (útil quando você roda em GPU).
        """
        try:
            from skimage.filters import threshold_otsu
        except ImportError:
            Logger.warning("skimage not available, using simple threshold")
            return (image > 0.5).float()

        # Garantir CPU, detach e numpy contíguo
        img_np = image.detach().cpu().numpy()
        threshold = threshold_otsu(img_np)
        binary_np = (img_np > threshold).astype(np.float32)

        # Converter de volta para tensor no mesmo device do input
        binary_t = torch.from_numpy(np.ascontiguousarray(binary_np)).to(image.device).float()

        return binary_t

    @staticmethod
    def _adaptive_quantize(image, levels):
        """Quantização adaptativa baseada no histograma."""
        # Calcular histograma
        hist, bins = torch.histogram(image.flatten(), bins=256)
        hist = hist.float()

        # Encontrar limiares que dividem o histograma em áreas iguais
        cumsum = torch.cumsum(hist, dim=0)
        total = cumsum[-1]

        thresholds = []
        for i in range(1, levels):
            target = i * total / levels
            # searchsorted retorna tensor; converter para int Python
            idx = int(torch.searchsorted(cumsum, target).item())
            # Garantir faixa válida para indexar em 'bins'
            idx = max(0, min(idx, len(bins) - 1))
            thresholds.append(bins[idx].item())

        if len(thresholds) == 0:
            return torch.zeros_like(image)

        # Quantização vetorizada:
        # bucketize conta quantos thresholds são estritamente menores que cada pixel (right=False),
        # replicando a lógica "image > thresh".
        thresholds_t = torch.tensor(thresholds, dtype=image.dtype, device=image.device)
        levels_idx = torch.bucketize(image, thresholds_t, right=False)  # valores em [0, levels-1]
        quantized = levels_idx.to(image.dtype) / (levels - 1)

        return quantized


def select_informative_patch(patches, num_candidates=5):
    """
    Seleciona um patch informativo baseado em múltiplas métricas.

    Critérios:
    - Variância (diversidade de valores)
    - Entropia (quantidade de informação)
    - Gradientes (bordas e texturas)
    - Range dinâmico (evita patches muito escuros ou claros)
    - Contraste local

    Returns:
        best_idx: Índice do melhor patch
        scores: Scores de todos os patches
    """
    patches_float = patches.float() / 255.0
    n_patches = patches.shape[0]

    scores = torch.zeros(n_patches)

    for i in range(n_patches):
        patch = patches_float[i]

        # 1. Variância (diversidade de valores)
        variance = torch.var(patch).item()

        # 2. Entropia aproximada (baseada em histograma)
        hist = torch.histc(patch, bins=16, min=0, max=1)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remover zeros para evitar log(0)
        entropy = -(hist * torch.log(hist)).sum().item() if len(hist) > 0 else 0

        # 3. Magnitude dos gradientes (detecta bordas)
        if patch.shape[0] >= 3 and patch.shape[1] >= 3:
            # Calcular gradientes simples
            dx = torch.abs(patch[1:, :] - patch[:-1, :]).mean().item()
            dy = torch.abs(patch[:, 1:] - patch[:, :-1]).mean().item()
            gradient_mag = (dx + dy) / 2
        else:
            gradient_mag = variance  # Fallback para patches muito pequenos

        # 4. Range dinâmico (penaliza muito escuro ou muito claro)
        mean_val = patch.mean().item()
        range_score = 4 * mean_val * (1 - mean_val)  # Máximo em 0.5

        # 5. Contraste local
        min_val = patch.min().item()
        max_val = patch.max().item()
        contrast = max_val - min_val

        # Score combinado (pesos ajustáveis)
        scores[i] = (
                0.25 * variance +
                0.20 * entropy / 3.0 +  # Normalizar entropia
                0.20 * gradient_mag +
                0.20 * range_score +
                0.15 * contrast
        )

    # Selecionar top candidatos e escolher o melhor
    top_k = min(num_candidates, n_patches)
    top_indices = torch.topk(scores, top_k).indices

    # Entre os top candidatos, escolher o com melhor range dinâmico
    best_idx = top_indices[0]
    for idx in top_indices:
        patch_mean = patches_float[idx].mean()
        if 0.3 <= patch_mean <= 0.7:  # Preferir patches com brilho médio
            best_idx = idx
            break

    return best_idx.item(), scores


class ProcessedDataset(torch.utils.data.Dataset):
    """
    Creates a processed version of a dataset with optional resizing, compression and quantization.

    Features:
    - Automatic resizing detection
    - Compression artifacts simulation
    - Quantization (reduce color levels, including binarization)
    - Caches processed data with zstd compression
    """

    def __init__(self, original_ds, target_size, resize_alg,
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

        # Auto-detect processing needs
        self.needs_resize = (self.target_size != self._get_original_size())
        self.needs_compression = self.image_format in ['JPEG', 'PNG']
        self.needs_quantization = quantization_levels is not None

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

        # Add quantization info
        if self.needs_quantization:
            quant_info = f"_quant{self.quantization_levels}_{self.quantization_method}"
        else:
            quant_info = ""

        return os.path.join(
            cache_dir,
            f"processed_{resize_info}_{alg_name}_{format_info}{quant_info}.pt.zst"
        )

    def _process_image(self, img):
        """Process single image with all transformations"""
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

        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)

        # Apply quantization if configured
        if self.needs_quantization:
            img_tensor = ImageQuantizer.quantize(
                img_tensor,
                levels=self.quantization_levels,
                method=self.quantization_method,
                dithering=(self.quantization_method == 'uniform' and self.quantization_levels == 2)
            )

        return img_tensor

    def _process_and_cache(self):
        """Process all images and save to cache with zstd compression"""
        processed_images = []
        labels = []

        desc = "Processing Dataset"
        if self.needs_quantization:
            desc += f" (Quantizing to {self.quantization_levels} levels)"

        for img, label in tqdm(self.original_ds, desc=desc):
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
        """Load cached data with zstd decompression and validation (GPU/CPU safe)"""
        Logger.info(f"Loading dataset from cache at {self.cache_path}")
        try:
            with open(self.cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                Logger.micro("Decompressing cached data stream...")
                with dctx.stream_reader(f) as reader:
                    decompressed_data = reader.read()
            buffer = io.BytesIO(decompressed_data)
            # Garantir load seguro independente do device onde foi salvo
            loaded = torch.load(buffer, map_location="cpu")

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
        config = {
            'target_size': self.target_size,
            'resize_alg': RESIZE_ALG_MAP.get(self.resize_alg, "unknown"),
            'image_format': self.image_format,
            'quality': self.quality
        }

        if self.needs_quantization:
            config['quantization_levels'] = self.quantization_levels
            config['quantization_method'] = self.quantization_method

        return config

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
        """Extract all patches from an image at once (uses pil_to_tensor for efficiency).

        Returns:
            - If single-channel image: Tensor [num_patches, H, W] (uint8)  <-- backwards compatible
            - If multi-channel image: Tensor [num_patches, C, H, W] (uint8)
        """
        Logger.micro(f"Extracting all {self.num_patches_per_image} patches from image")

        # pil_to_tensor retorna uint8 com shape [C, H, W] diretamente (evita *255 e round)
        tensor = TF.pil_to_tensor(image)  # uint8 [C, H, W]
        C, H, W = tensor.shape
        # Colocar batch e converter para float para usar unfold (preserva range 0-255)
        tensor = tensor.unsqueeze(0).float()  # [1, C, H, W] float in [0,255]

        patch_h, patch_w = self.patch_size
        # F.unfold opera com [N, C, H, W] e retorna [N, C*kh*kw, L]
        patches_unf = F.unfold(tensor, kernel_size=(patch_h, patch_w), stride=self.stride)
        # [1, C*kh*kw, L] -> [L, C*kh*kw]
        patches_flat = patches_unf.squeeze(0).transpose(0, 1)
        # reshape -> [L, C, kh, kw]
        patches = patches_flat.view(-1, C, patch_h, patch_w)
        # Converter para uint8 para armazenar/serializar eficientemente
        patches = patches.round().to(torch.uint8)

        # Compatibilidade com código legado: para imagens single-channel, retornar [L,H,W]
        if C == 1:
            return patches.squeeze(1)  # [L, H, W]
        return patches  # [L, C, H, W]

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
        """Save patches with zstd compression using threads for speed."""
        buffer = io.BytesIO()
        torch.save(patches, buffer)
        buffer.seek(0)

        # Usar threads para acelerar compressão em CPUs multi-core
        with open(cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=3, threads=-1)
            # Para arquivos não muito grandes é OK fazer compress() direto; threads acelera
            compressed = cctx.compress(buffer.read())
            f.write(compressed)

        Logger.micro(f"Saved compressed patches to {cache_path}")

    def _load_compressed_patches(self, cache_path):
        """Load patches with zstd decompression and error handling (CPU-safe)"""
        try:
            with open(cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(f.read())

            buffer = io.BytesIO(decompressed)
            # Carregar no CPU por segurança (compatibilidade com diferentes dispositivos)
            patches = torch.load(buffer, map_location="cpu")

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

        # Detectar formato de patches e preparar tensor [L, C, ph, pw]
        if patches.dim() == 3:
            # [L, H, W] -> [L, 1, H, W]
            patches_f = patches.unsqueeze(1).float() / 255.0
        elif patches.dim() == 4:
            # [L, C, H, W]
            patches_f = patches.float() / 255.0
        else:
            raise ValueError("Unsupported patches shape for reconstruction")

        L, C, ph, pw = patches_f.shape
        # Mover para device solicitado (CPU por padrão). fold fará soma sobre posições.
        patches_f = patches_f.to(device)

        # Preparar para fold: precisamos [1, C*ph*pw, L]
        patches_flat = patches_f.view(L, -1).transpose(0, 1).unsqueeze(0)  # [1, C*ph*pw, L]

        # Somatório das contribuições (reconstrução) usando fold
        recon_sum = F.fold(patches_flat, output_size=(recon_h, recon_w),
                           kernel_size=(ph, pw), stride=self.stride)  # [1, C, H, W]

        # Criar matriz de pesos (quantas vezes cada pixel foi somado) usando ones
        ones = torch.ones_like(patches_f, dtype=patches_f.dtype, device=device)
        ones_flat = ones.view(L, -1).transpose(0, 1).unsqueeze(0)  # [1, C*ph*pw, L]
        weight = F.fold(ones_flat, output_size=(recon_h, recon_w),
                        kernel_size=(ph, pw), stride=self.stride)  # [1, C, H, W]

        # Evitar divisão por zero
        recon = recon_sum / weight.clamp(min=1e-6)

        # Converter de volta para uint8
        recon = (recon * 255.0).round().to(torch.uint8).squeeze(0)  # [C, H, W] ou [H, W] se C==1

        if recon.shape[0] == 1:
            return recon.squeeze(0)  # [H, W]
        return recon  # [C, H, W]


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


def show_sample_images(samples, sample_index, zoom_ratio=0.5, patch_coords=None, cmap='viridis'):
    """
    Display sample images side by side with colorful visualization.

    Parameters:
        samples: Dictionary where each key is a title and each value is an image tensor
        sample_index: Identifier for the sample (shown in the figure title)
        zoom_ratio: Factor to scale the figure size (0.5 = half size)
        patch_coords: Dict with patch rectangle coordinates
        cmap: Colormap to use (default 'viridis' for colorful display)
    """
    Logger.info(f"Displaying sample images for sample index: {sample_index}")
    num_images = len(samples)

    # Reduced figure size (half of original)
    fig, axes = plt.subplots(1, num_images, figsize=(3 * num_images * zoom_ratio, 3 * zoom_ratio))

    if num_images == 1:
        axes = [axes]

    for ax, (title, tensor) in zip(axes, samples.items()):
        # Convert to numpy for display
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            # Single channel image
            img_array = tensor.squeeze(0).numpy()
        else:
            img_array = tensor.numpy()

        # Display with colorful colormap
        im = ax.imshow(img_array, cmap=cmap)

        # Add title with size info
        if hasattr(tensor, 'shape'):
            height, width = img_array.shape[-2:]
            ax.set_title(f"{title}\nSize: {width}x{height}", fontsize=8)
        else:
            ax.set_title(title, fontsize=8)

        ax.axis("off")

        # Draw patch rectangle if provided
        if patch_coords and title in patch_coords:
            top, left, patch_h, patch_w = patch_coords[title]
            rect = plt.Rectangle((left, top), patch_w, patch_h,
                                 linewidth=10, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    # Add main title
    fig.suptitle(f"Sample Index: {sample_index}", fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show with block=False for non-blocking display
    plt.show(block=True)
    Logger.info("Displayed sample images successfully")


def test_optimized_dataset():
    """
    Test function to demonstrate the optimized SuperResPatchDataset.
    Shows cache efficiency and performance improvements.
    """
    Logger.step("Testing Optimized SuperResPatchDataset")

    # Configuration with quantization options
    low_res_config = {
        'target_size': (14, 14),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None,
        'quantization_levels': None,  # Can be 2 for binary, 4 for 4-level, etc.
        'quantization_method': 'uniform'  # or 'otsu', 'kmeans', 'adaptive'
    }
    high_res_config = {
        'target_size': (28, 28),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None,
        'quantization_levels': None,  # Keep high-res without quantization usually
        'quantization_method': 'uniform'
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

    # Test reconstruction with informative patch selection
    Logger.step("Testing image reconstruction with informative patch selection")

    img_idx = 1
    label = dataset.low_res_ds.labels[img_idx].item()

    # Reconstruct images
    recon_low = dataset.reconstruct_low(img_idx)
    recon_high = dataset.reconstruct_high(img_idx)

    # Get original processed images for comparison
    low_tensor = dataset.low_res_ds.data[img_idx]
    high_tensor = dataset.high_res_ds.data[img_idx]

    # Get all patches and select most informative one
    low_pil = transforms.ToPILImage()(low_tensor.squeeze())
    high_pil = transforms.ToPILImage()(high_tensor.squeeze())

    small_patches = dataset.small_patch_extractor.process(low_pil, img_idx)
    large_patches = dataset.large_patch_extractor.process(high_pil, img_idx)

    # Select informative patch using new selection function
    patch_idx, scores = select_informative_patch(small_patches, num_candidates=10)
    Logger.info(f"Selected patch {patch_idx} as most informative (score: {scores[patch_idx]:.4f})")

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

    # Display all images with reduced size and colorful visualization
    samples = {
        'Low Res': low_tensor,
        'High Res': high_tensor,
        f'Small Patch #{patch_idx}': small_upsampled.unsqueeze(0),
        f'Large Patch #{patch_idx}': large_upsampled.unsqueeze(0),
        'Recon Low': recon_low_tensor,
        'Recon High': recon_high_tensor
    }

    show_sample_images(
        samples,
        sample_index=f"Image {img_idx} (Label: {label})",
        patch_coords=patch_coords,
        zoom_ratio=1,  # Half size window
        cmap='viridis'  # Colorful display
    )

    # Performance comparison
    Logger.step("Performance Comparison")

    import time

    # Clear caches for fair comparison
    dataset.clear_memory_caches()

    # Measure time for accessing all patches from one image
    img_idx = min(10, dataset.num_images - 1)
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


def test_quantization():
    """Test different quantization methods on sample images."""
    Logger.step("Testing Image Quantization Methods")

    # Load a sample image
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Get a sample image
    img_tensor, label = mnist[0]

    # Test different quantization methods
    methods = [
        ('Original', None, None),
        ('Binary (Uniform)', 2, 'uniform'),
        ('Binary (Otsu)', 2, 'otsu'),
        ('4-Level (Uniform)', 4, 'uniform'),
        ('4-Level (Adaptive)', 4, 'adaptive'),
        ('8-Level (Uniform)', 8, 'uniform'),
    ]

    # Create figure with smaller size
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    axes = axes.flatten()

    for idx, (title, levels, method) in enumerate(methods):
        if levels is None:
            # Original image
            quantized = img_tensor
        else:
            # Apply quantization
            quantized = ImageQuantizer.quantize(
                img_tensor,
                levels=levels,
                method=method,
                dithering=(method == 'uniform' and levels == 2)
            )

        # Display
        axes[idx].imshow(quantized.squeeze(), cmap='plasma')
        axes[idx].set_title(title, fontsize=8)
        axes[idx].axis('off')

    plt.suptitle(f"Quantization Methods Comparison (Label: {label})", fontsize=10)
    plt.tight_layout()
    plt.show()

    Logger.info("Quantization test completed!")


if __name__ == "__main__":
    # Run the optimized test
    Logger.macro("Running Optimized SuperResPatchDataset Test")
    dataset = test_optimized_dataset()

    # Test quantization methods
    Logger.macro("Testing Quantization Methods")
    test_quantization()

    Logger.macro("All tests completed successfully!")