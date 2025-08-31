import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from Logger import Logger

class ImageQuantizer:
    """
    Image quantization utilities.
    Methods return torch tensors with dtype torch.float32 in range [0,1],
    on the same device as the input (if input is a tensor).
    """

    @staticmethod
    def quantize(image, levels=2, method='uniform', dithering=False):
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)

        if not isinstance(image, torch.Tensor):
            raise ValueError("image must be a PIL.Image or torch.Tensor")

        if method == 'uniform':
            out = ImageQuantizer._uniform_quantize(image, levels, dithering)
        elif method == 'kmeans':
            out = ImageQuantizer._kmeans_quantize(image, levels)
        elif method == 'otsu':
            out = ImageQuantizer._otsu_binarize(image)
        elif method == 'adaptive':
            out = ImageQuantizer._adaptive_quantize(image, levels)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        # Ensure consistent dtype/range and device
        out = out.clamp(0, 1).to(torch.float32)
        try:
            img_device = image.device
            out = out.to(img_device)
        except Exception:
            # In case input was CPU tensor but had no device attr or PIL handling
            pass
        return out

    @staticmethod
    def _uniform_quantize(image, levels, dithering=False):
        if dithering and levels == 2:
            return ImageQuantizer._floyd_steinberg_dither(image)

        if levels == 2:
            return (image > 0.5).float()

        step = 1.0 / (levels - 1)
        quantized = torch.round(image / step) * step
        return quantized

    @staticmethod
    def _floyd_steinberg_dither(image):
        img = image.clone().detach().cpu()
        h, w = img.shape[-2:]
        for y in range(h):
            for x in range(w):
                old = img[..., y, x]
                new = torch.round(old)
                img[..., y, x] = new
                err = old - new
                if x + 1 < w:
                    img[..., y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        img[..., y + 1, x - 1] += err * 3 / 16
                    img[..., y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        img[..., y + 1, x + 1] += err * 1 / 16
        return img.clamp(0, 1)

    @staticmethod
    def _kmeans_quantize(image, levels):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            Logger.warning("sklearn not available; falling back to uniform quantization")
            return ImageQuantizer._uniform_quantize(image, levels, False)

        img = image.detach().cpu().numpy()
        orig_shape = img.shape
        # If multichannel, cluster by channel vector per pixel
        if img.ndim == 3:
            C, H, W = orig_shape
            pixels = img.transpose(1, 2, 0).reshape(-1, C)
        else:
            pixels = img.flatten().reshape(-1, 1)

        kmeans = KMeans(n_clusters=levels, random_state=42, n_init=10)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        quant_np = centers[labels].reshape(-1, *centers.shape[1:]) if centers.ndim > 1 else centers[labels]

        if img.ndim == 3:
            quant_np = quant_np.reshape(H, W, C).transpose(2, 0, 1)
        else:
            quant_np = quant_np.reshape(orig_shape)

        quant_t = torch.from_numpy(np.ascontiguousarray(quant_np)).float()
        return quant_t

    @staticmethod
    def _otsu_binarize(image):
        try:
            from skimage.filters import threshold_otsu
        except ImportError:
            Logger.warning("skimage not available; using simple 0.5 threshold")
            return (image > 0.5).float()

        img = image.detach().cpu().numpy()
        thresh = threshold_otsu(img)
        binary = (img > thresh).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(binary)).float()

    @staticmethod
    def _adaptive_quantize(image, levels):
        """
        Adaptive quantization using histogram-based thresholding into 'levels' bins.
        Returns tensor in [0,1] with discrete levels 0..levels-1 scaled by (levels-1).
        Robust to PyTorch versions: uses torch.histogram when available, otherwise torch.histc.
        """
        img = image.detach().cpu()
        flat = img.flatten()

        # Compute histogram and bin edges robustly
        try:
            hist_tuple = torch.histogram(flat, bins=256, range=(0.0, 1.0))
            hist = hist_tuple[0].to(torch.float32)
            edges = hist_tuple[1].to(torch.float32)  # length bins+1
        except Exception:
            # Fallback to histc (older PyTorch): returns only counts, need to infer edges
            hist = torch.histc(flat, bins=256, min=0.0, max=1.0).to(torch.float32)
            edges = torch.linspace(0.0, 1.0, steps=257)

        cumsum = torch.cumsum(hist, dim=0)
        total = float(cumsum[-1].item())
        if total == 0:
            return torch.zeros_like(img)

        thresholds = []
        for i in range(1, levels):
            target = i * total / levels
            idx = int(torch.searchsorted(cumsum, torch.tensor(target)).item())
            # clamp idx into valid bin range
            idx = max(0, min(idx, edges.numel() - 2))
            # use left edge of bin as threshold
            thresh_value = float(edges[idx].item())
            thresholds.append(thresh_value)

        if len(thresholds) == 0:
            return torch.zeros_like(img)

        thresholds_t = torch.tensor(thresholds, dtype=img.dtype)
        levels_idx = torch.bucketize(img, thresholds_t, right=False)
        quantized = levels_idx.to(img.dtype) / max(1, (levels - 1))
        return quantized