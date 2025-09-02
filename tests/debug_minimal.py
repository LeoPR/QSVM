"""
Teste minimal para debug do travamento
"""
import pytest
import torch
import time
import tempfile
from pathlib import Path


# Teste 1: Fixture básica
def test_basic_torch():
    """Testa se torch funciona básico"""
    print("🔍 Testing basic torch...")
    x = torch.rand(3, 3)
    assert x.shape == (3, 3)
    print("✅ Torch OK")


# Teste 2: Fixture synthetic
def test_tiny_synthetic_simple():
    """Testa fixture tiny_synthetic isoladamente"""
    print("🔍 Testing tiny_synthetic...")

    class TinySynthetic(torch.utils.data.Dataset):
        def __init__(self, n=2, size=(8, 8)):  # MUITO pequeno para debug
            print(f"  🔨 Creating TinySynthetic n={n}, size={size}")
            self.n = n
            self.size = size

        def __len__(self):
            print(f"  📏 __len__ called, returning {self.n}")
            return self.n

        def __getitem__(self, idx):
            print(f"  🎯 __getitem__ called with idx={idx}")
            if idx >= self.n:
                raise IndexError(f"Index {idx} >= {self.n}")

            H, W = self.size
            # Gradiente SUPER simples
            img = torch.linspace(0, 1, H * W).view(1, H, W)
            label = idx
            print(f"  ✅ Generated img {img.shape}, label {label}")
            return img, label

    ds = TinySynthetic(n=2, size=(4, 4))
    print(f"✅ Dataset created, len={len(ds)}")

    # Testar getitem
    img, label = ds[0]
    print(f"✅ Got item 0: {img.shape}, {label}")
    print("✅ TinySynthetic OK")


# Teste 3: ProcessedDataset básico
def test_processed_dataset_minimal():
    """Testa ProcessedDataset com config mínima"""
    print("🔍 Testing ProcessedDataset minimal...")

    # Dataset tiny
    class TinyDs(torch.utils.data.Dataset):
        def __len__(self): return 1

        def __getitem__(self, idx):
            return torch.full((1, 4, 4), 0.5), 0

    ds = TinyDs()
    print("✅ Tiny dataset created")

    # Cache temporário
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"  📁 Using temp cache: {tmp_dir}")

        try:
            # Import aqui para ver se trava
            print("  📦 Importing ProcessedDataset...")
            from patchkit import ProcessedDataset
            print("  ✅ Import OK")

            print("  🔨 Creating ProcessedDataset...")
            start_time = time.time()

            processed = ProcessedDataset(
                ds,
                target_size=(4, 4),  # Mesmo tamanho - sem resize
                resize_alg=None,
                image_format=None,  # Sem compressão
                quality=None,
                quantization_levels=None,  # Sem quantização
                quantization_method='uniform',
                cache_dir=tmp_dir,
                cache_rebuild=True
            )

            elapsed = time.time() - start_time
            print(f"  ⏱️ ProcessedDataset created in {elapsed:.2f}s")

            print(f"  📊 Dataset length: {len(processed)}")
            print(f"  📊 Data shape: {processed.data.shape}")
            print("✅ ProcessedDataset minimal OK")

        except Exception as e:
            print(f"❌ ProcessedDataset failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# Teste 4: Com quantização
def test_processed_dataset_with_quantization():
    """Testa com quantização (possível culpado)"""
    print("🔍 Testing ProcessedDataset with quantization...")

    class TinyDs(torch.utils.data.Dataset):
        def __len__(self): return 1

        def __getitem__(self, idx):
            return torch.rand(1, 8, 8), 0  # Random para quantizar

    ds = TinyDs()

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            from patchkit import ProcessedDataset

            print("  🔨 Creating with quantization...")
            start_time = time.time()

            processed = ProcessedDataset(
                ds,
                target_size=(8, 8),
                quantization_levels=2,  # Esta é a possível culpada
                quantization_method='uniform',
                cache_dir=tmp_dir,
                cache_rebuild=True
            )

            elapsed = time.time() - start_time
            print(f"  ⏱️ With quantization: {elapsed:.2f}s")
            print("✅ Quantization OK")

        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            raise


if __name__ == "__main__":
    print("🚀 Running debug tests manually...")
    test_basic_torch()
    test_tiny_synthetic_simple()
    test_processed_dataset_minimal()
    test_processed_dataset_with_quantization()
    print("🎉 All debug tests passed!")