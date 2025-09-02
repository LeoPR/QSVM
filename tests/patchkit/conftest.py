"""
Fixtures específicas para testes do patchkit
"""
import pytest
import torch
from PIL import Image
import numpy as np

from patchkit import OptimizedPatchExtractor, ProcessedDataset, SuperResPatchDataset


@pytest.fixture
def standard_extractor(cache_dir):
    """
    Extractor padrão para a maioria dos testes.
    Configuração comum: patches 4x4, stride 2, imagem 28x28
    """
    return OptimizedPatchExtractor(
        patch_size=(4, 4),
        stride=2,
        cache_dir=cache_dir,
        image_size=(28, 28),
        max_memory_cache=10
    )


@pytest.fixture
def small_extractor(cache_dir):
    """
    Extractor para patches pequenos: 2x2, stride 1, imagem 14x14
    """
    return OptimizedPatchExtractor(
        patch_size=(2, 2),
        stride=1,
        cache_dir=cache_dir,
        image_size=(14, 14),
        max_memory_cache=5
    )


@pytest.fixture
def sample_pil_images():
    """
    Conjunto de imagens PIL para testar diferentes casos.
    """
    images = {}

    # Gradiente horizontal (grayscale)
    gradient_array = np.tile(np.linspace(0, 255, 28, dtype=np.uint8), (28, 1))
    images['gradient'] = Image.fromarray(gradient_array, mode='L')

    # Imagem binária (preto e branco)
    binary_array = np.zeros((28, 28), dtype=np.uint8)
    binary_array[:14, :] = 255
    images['binary'] = Image.fromarray(binary_array, mode='L')

    # Padrão xadrez
    checker = np.zeros((28, 28), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    images['checkerboard'] = Image.fromarray(checker, mode='L')

    # Imagem pequena (8x8)
    small_array = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    images['small'] = Image.fromarray(small_array, mode='L')

    return images


@pytest.fixture
def processed_dataset_configs():
    """
    Configurações comuns para ProcessedDataset.
    """
    return {
        'basic': {
            'target_size': (28, 28),
            'resize_alg': Image.BICUBIC,
            'image_format': None,
            'quality': None,
            'quantization_levels': None,
            'quantization_method': 'uniform'
        },
        'binary': {
            'target_size': (14, 14),
            'resize_alg': Image.BICUBIC,
            'image_format': None,
            'quality': None,
            'quantization_levels': 2,
            'quantization_method': 'uniform'
        },
        'compressed': {
            'target_size': (28, 28),
            'resize_alg': Image.BICUBIC,
            'image_format': 'JPEG',
            'quality': 50,
            'quantization_levels': None,
            'quantization_method': 'uniform'
        }
    }


@pytest.fixture
def superres_configs():
    """
    Configurações para SuperResPatchDataset.
    """
    low_res = {
        'target_size': (14, 14),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None,
        'quantization_levels': 2,
        'quantization_method': 'uniform'
    }

    high_res = {
        'target_size': (28, 28),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None,
        'quantization_levels': None,
        'quantization_method': 'uniform'
    }

    return {
        'low_res_config': low_res,
        'high_res_config': high_res,
        'small_patch_size': (2, 2),
        'large_patch_size': (4, 4),
        'stride': 1,
        'scale_factor': 2
    }


@pytest.fixture
def make_synthetic_dataset():
    """
    Factory para criar datasets sintéticos com diferentes características.
    """

    def _factory(n_samples=10, size=(28, 28), pattern='mixed', classes=None):
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.n_samples = n_samples
                self.size = size
                self.pattern = pattern
                self.classes = classes or list(range(min(10, n_samples)))

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                H, W = self.size

                if self.pattern == 'gradient':
                    img = self._make_gradient(H, W, idx)
                elif self.pattern == 'circles':
                    img = self._make_circle(H, W, idx)
                elif self.pattern == 'mixed':
                    # Alternar padrões baseado no índice
                    if idx % 3 == 0:
                        img = self._make_gradient(H, W, idx)
                    elif idx % 3 == 1:
                        img = self._make_circle(H, W, idx)
                    else:
                        img = self._make_noise(H, W, idx)
                else:
                    img = self._make_noise(H, W, idx)

                img = img.clamp(0, 1).unsqueeze(0)  # [1, H, W]
                label = self.classes[idx % len(self.classes)]
                return img, label

            def _make_gradient(self, H, W, seed):
                torch.manual_seed(seed)  # Determinístico
                angle = torch.rand(1) * 2 * np.pi
                x = torch.arange(W).float()
                y = torch.arange(H).float()
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                direction = torch.cos(angle) * xx + torch.sin(angle) * yy
                return (direction - direction.min()) / (direction.max() - direction.min())

            def _make_circle(self, H, W, seed):
                torch.manual_seed(seed)
                center_y = torch.randint(H // 4, 3 * H // 4, (1,)).item()
                center_x = torch.randint(W // 4, 3 * W // 4, (1,)).item()
                radius = torch.randint(H // 8, H // 4, (1,)).item()

                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                return (dist <= radius).float()

            def _make_noise(self, H, W, seed):
                torch.manual_seed(seed)
                return torch.rand(H, W)

        return SyntheticDataset()

    return _factory


@pytest.fixture(scope="session")
def benchmark_timer():
    """
    Timer simples para benchmarks em testes.
    """
    import time

    class Timer:
        def __init__(self):
            self.times = {}

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.elapsed = time.time() - self.start

        def record(self, name):
            self.times[name] = getattr(self, 'elapsed', 0)

        def get_results(self):
            return self.times.copy()

    return Timer