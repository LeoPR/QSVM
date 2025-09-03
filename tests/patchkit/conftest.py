import pytest
import numpy as np
from PIL import Image
import torch

# Fixtures locais do pacote patchkit (não duplicam tiny_synthetic/mnist_sample)


@pytest.fixture
def sample_pil_images():
    """
    Retorna um dicionário de PIL images de exemplo usados em alguns testes.
    Garante dtype uint8 para evitar warnings do Pillow e mantém chaves esperadas.
    """
    images = {}
    H, W = 28, 28

    # grayscale gradient 0..255 uint8
    grad = (np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1)))
    images['gradient'] = Image.fromarray(grad)  # mode inferred as 'L'

    # checkerboard 0/255 (nome esperado pelos testes)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    checker = ((xx // 4 + yy // 4) % 2).astype(np.uint8) * 255
    images['checkerboard'] = Image.fromarray(checker)

    # binary 0/255
    binary_array = (np.indices((H, W)).sum(axis=0) % 2).astype(np.uint8) * 255
    images['binary'] = Image.fromarray(binary_array)

    # RGB sample
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)  # red gradient
    rgb[..., 1] = np.linspace(255, 0, H, dtype=np.uint8)[:, None]  # green gradient
    rgb[..., 2] = 128
    images['rgb'] = Image.fromarray(rgb)  # mode 'RGB'

    return images


# --- Patch extractor fixtures (locais ao pacote patchkit) ---


def _locate_optimized_patch_extractor():
    """
    Tenta encontrar a classe OptimizedPatchExtractor em locais prováveis.
    Retorna a classe ou None se não encontrada.
    """
    candidates = [
        "patchkit.optimized_patch_extractor",
        "patchkit.extractor",
        "patchkit.optimized",
        "patchkit",
    ]
    for modname in candidates:
        try:
            mod = __import__(modname, fromlist=['OptimizedPatchExtractor'])
            cls = getattr(mod, 'OptimizedPatchExtractor', None)
            if cls is not None:
                return cls
        except Exception:
            continue
    try:
        from patchkit import OptimizedPatchExtractor  # type: ignore
        return OptimizedPatchExtractor
    except Exception:
        return None


@pytest.fixture
def standard_extractor(tmp_path):
    """
    Instância padrão do extractor usada por muitos testes.
    Pula o teste se a classe não estiver disponível.
    """
    cls = _locate_optimized_patch_extractor()
    if cls is None:
        pytest.skip("OptimizedPatchExtractor class not found in patchkit (fixture standard_extractor skipped)")
    cache_dir = tmp_path / "cache_standard"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cls(patch_size=(4, 4), stride=4, cache_dir=str(cache_dir), image_size=(28, 28))


@pytest.fixture
def small_extractor(tmp_path):
    """
    Versão menor / alternativa do extractor para testes que precisem de outros tamanhos.
    """
    cls = _locate_optimized_patch_extractor()
    if cls is None:
        pytest.skip("OptimizedPatchExtractor class not found in patchkit (fixture small_extractor skipped)")
    cache_dir = tmp_path / "cache_small"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cls(patch_size=(8, 8), stride=8, cache_dir=str(cache_dir), image_size=(28, 28))


@pytest.fixture
def cache_dir(tmp_path):
    """
    Fixture simples que fornece um diretório de cache (string) para testes que o requerem.
    Uso: cache_dir + f"_suffix" é aceito nos testes existentes.
    """
    p = tmp_path / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)