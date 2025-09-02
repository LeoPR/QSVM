"""
Teste ULTRA-SIMPLES para ProcessedDataset
Sem fixtures externas, sem parametrização complexa, datasets mínimos
"""
import pytest
import torch
import tempfile
from PIL import Image

from patchkit import ProcessedDataset


class MicroSyntheticDataset(torch.utils.data.Dataset):
    """Dataset microscópio para debug"""

    def __init__(self, n=2):
        self.n = n
        # Pre-generate para evitar problemas
        self.items = []
        for i in range(n):
            # Tensor 2x2 super simples
            if i % 2 == 0:
                img = torch.full((1, 2, 2), 0.2)  # Escuro
            else:
                img = torch.full((1, 2, 2), 0.8)  # Claro
            self.items.append((img, i))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.items[idx]


class TestProcessedDatasetUltraSimple:
    """Testes mínimos sem complexidade"""

    def test_no_processing(self):
        """Teste mais básico: sem resize, sem quantização, sem compressão"""
        ds = MicroSyntheticDataset(n=2)

        with tempfile.TemporaryDirectory() as cache_dir:
            processed = ProcessedDataset(
                ds,
                target_size=None,  # SEM resize
                quantization_levels=None,  # SEM quantização
                image_format=None,  # SEM compressão
                cache_dir=cache_dir,
                cache_rebuild=True
            )

            # Verificações básicas
            assert len(processed) == 2
            assert processed.data.shape == (2, 1, 2, 2)

            # Test getitem
            item = processed[0]
            assert len(item) == 2  # (data, label)

    def test_simple_resize(self):
        """Resize básico 2x2 -> 3x3"""
        ds = MicroSyntheticDataset(n=2)

        with tempfile.TemporaryDirectory() as cache_dir:
            processed = ProcessedDataset(
                ds,
                target_size=(3, 3),
                resize_alg=Image.NEAREST,  # Mais rápido
                cache_dir=cache_dir,
                cache_rebuild=True
            )

            assert len(processed) == 2
            assert processed.data.shape == (2, 1, 3, 3)

    def test_simple_quantization(self):
        """Quantização binária básica"""
        ds = MicroSyntheticDataset(n=2)

        with tempfile.TemporaryDirectory() as cache_dir:
            processed = ProcessedDataset(
                ds,
                target_size=(2, 2),  # Mesmo tamanho
                quantization_levels=2,  # Binário
                quantization_method='uniform',
                cache_dir=cache_dir,
                cache_rebuild=True
            )

            assert len(processed) == 2

            # Deve ser binário (0.0 ou 1.0)
            unique_vals = torch.unique(processed.data)
            assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_cache_reuse(self):
        """Cache simples"""
        ds = MicroSyntheticDataset(n=2)

        with tempfile.TemporaryDirectory() as cache_dir:
            # Primeira execução
            processed1 = ProcessedDataset(
                ds, target_size=(2, 2),
                cache_dir=cache_dir, cache_rebuild=True
            )

            # Segunda execução (deve usar cache)
            processed2 = ProcessedDataset(
                ds, target_size=(2, 2),
                cache_dir=cache_dir, cache_rebuild=False
            )

            torch.testing.assert_close(processed1.data, processed2.data)

    def test_empty_dataset(self):
        """Dataset vazio"""

        class EmptyDataset(torch.utils.data.Dataset):
            def __len__(self): return 0

            def __getitem__(self, idx): raise IndexError()

        ds = EmptyDataset()

        with tempfile.TemporaryDirectory() as cache_dir:
            processed = ProcessedDataset(
                ds, cache_dir=cache_dir, cache_rebuild=True
            )

            assert len(processed) == 0


# Para teste manual
if __name__ == "__main__":
    print("🧪 Running manual tests...")

    test_instance = TestProcessedDatasetUltraSimple()

    try:
        print("1. Testing no processing...")
        test_instance.test_no_processing()
        print("✅ PASS")

        print("2. Testing simple resize...")
        test_instance.test_simple_resize()
        print("✅ PASS")

        print("3. Testing quantization...")
        test_instance.test_simple_quantization()
        print("✅ PASS")

        print("4. Testing cache...")
        test_instance.test_cache_reuse()
        print("✅ PASS")

        print("5. Testing empty dataset...")
        test_instance.test_empty_dataset()
        print("✅ PASS")

        print("🎉 ALL TESTS PASSED!")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()