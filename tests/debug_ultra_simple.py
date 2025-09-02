"""
Debug ULTRA-SIMPLES para identificar travamento
"""
import sys
import time
import tempfile
from pathlib import Path


def print_step(msg):
    print(f"🔍 {msg}")
    sys.stdout.flush()


def test_step_1_basic_imports():
    """Teste 1: Imports básicos"""
    print_step("Step 1: Testing basic imports...")

    try:
        import torch
        print_step("✅ torch OK")

        import PIL
        print_step("✅ PIL OK")

        from torchvision import transforms
        print_step("✅ torchvision OK")

        return True
    except Exception as e:
        print_step(f"❌ Basic imports failed: {e}")
        return False


def test_step_2_tiny_dataset():
    """Teste 2: Dataset tiny"""
    print_step("Step 2: Testing tiny dataset...")

    try:
        import torch

        class MicroDataset(torch.utils.data.Dataset):
            def __init__(self):
                print_step("  Creating MicroDataset...")
                self.data = [
                    torch.full((1, 2, 2), 0.3),  # Super tiny 2x2
                    torch.full((1, 2, 2), 0.7)
                ]
                self.labels = [0, 1]
                print_step(f"  Dataset created with {len(self.data)} items")

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                print_step(f"  Getting item {idx}")
                return self.data[idx], self.labels[idx]

        ds = MicroDataset()
        print_step(f"✅ Dataset length: {len(ds)}")

        # Test getitem
        item0 = ds[0]
        print_step(f"✅ Got item 0: shape {item0[0].shape}, label {item0[1]}")

        return ds
    except Exception as e:
        print_step(f"❌ Tiny dataset failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_step_3_patchkit_import():
    """Teste 3: Import do patchkit"""
    print_step("Step 3: Testing patchkit import...")

    try:
        print_step("  Importing patchkit...")

        # Import por partes
        print_step("  Importing quantize...")
        from patchkit.quantize import ImageQuantizer
        print_step("  ✅ ImageQuantizer imported")

        print_step("  Importing ProcessedDataset...")
        from patchkit import ProcessedDataset
        print_step("  ✅ ProcessedDataset imported")

        return True
    except Exception as e:
        print_step(f"❌ patchkit import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_4_processed_dataset_minimal():
    """Teste 4: ProcessedDataset minimal"""
    print_step("Step 4: Testing ProcessedDataset minimal...")

    ds = test_step_2_tiny_dataset()
    if ds is None:
        return False

    try:
        from patchkit import ProcessedDataset

        with tempfile.TemporaryDirectory(prefix="debug_") as cache_dir:
            print_step(f"  Using cache dir: {cache_dir}")

            print_step("  Creating ProcessedDataset (NO processing)...")
            start_time = time.time()

            # Configuração MÍNIMA - sem resize, sem quantização, sem compressão
            processed = ProcessedDataset(
                ds,
                target_size=None,  # SEM resize
                resize_alg=None,
                image_format=None,  # SEM compressão
                quality=None,
                quantization_levels=None,  # SEM quantização
                quantization_method='uniform',
                cache_dir=cache_dir,
                cache_rebuild=True
            )

            elapsed = time.time() - start_time
            print_step(f"  ✅ ProcessedDataset created in {elapsed:.2f}s")
            print_step(f"  ✅ Length: {len(processed)}")
            print_step(f"  ✅ Data shape: {processed.data.shape}")

            # Test getitem
            item = processed[0]
            print_step(f"  ✅ Got item: {len(item)} elements")

            return True

    except Exception as e:
        print_step(f"❌ ProcessedDataset minimal failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_5_with_resize():
    """Teste 5: Com resize"""
    print_step("Step 5: Testing with resize...")

    ds = test_step_2_tiny_dataset()
    if ds is None:
        return False

    try:
        from patchkit import ProcessedDataset
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="debug_resize_") as cache_dir:
            print_step("  Creating with resize 2x2 -> 3x3...")

            processed = ProcessedDataset(
                ds,
                target_size=(3, 3),  # Resize pequeno
                resize_alg=Image.NEAREST,  # Mais rápido
                cache_dir=cache_dir,
                cache_rebuild=True
            )

            print_step(f"  ✅ With resize - shape: {processed.data.shape}")
            return True

    except Exception as e:
        print_step(f"❌ With resize failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_6_with_quantization():
    """Teste 6: Com quantização"""
    print_step("Step 6: Testing with quantization...")

    ds = test_step_2_tiny_dataset()
    if ds is None:
        return False

    try:
        from patchkit import ProcessedDataset

        with tempfile.TemporaryDirectory(prefix="debug_quant_") as cache_dir:
            print_step("  Creating with quantization...")

            processed = ProcessedDataset(
                ds,
                target_size=(2, 2),  # Mesmo tamanho
                quantization_levels=2,  # Binário
                quantization_method='uniform',
                cache_dir=cache_dir,
                cache_rebuild=True
            )

            print_step(f"  ✅ With quantization - shape: {processed.data.shape}")
            unique_vals = processed.data.unique()
            print_step(f"  ✅ Unique values: {unique_vals.tolist()}")
            return True

    except Exception as e:
        print_step(f"❌ With quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes em sequência"""
    print_step("=== DEBUG ULTRA-SIMPLES ===")

    tests = [
        test_step_1_basic_imports,
        test_step_2_tiny_dataset,
        test_step_3_patchkit_import,
        test_step_4_processed_dataset_minimal,
        test_step_5_with_resize,
        test_step_6_with_quantization
    ]

    for i, test_func in enumerate(tests, 1):
        print_step(f"\n--- Test {i}/6: {test_func.__name__} ---")

        start_time = time.time()
        try:
            result = test_func()
            elapsed = time.time() - start_time

            if result:
                print_step(f"✅ PASS ({elapsed:.2f}s)")
            else:
                print_step(f"❌ FAIL ({elapsed:.2f}s)")
                break

        except Exception as e:
            elapsed = time.time() - start_time
            print_step(f"💥 EXCEPTION ({elapsed:.2f}s): {e}")
            import traceback
            traceback.print_exc()
            break

        # Pausa entre testes para identificar travamentos
        if elapsed > 10.0:
            print_step(f"⚠️  WARNING: Test took {elapsed:.1f}s - might be hanging")

    print_step("\n=== DEBUG COMPLETED ===")


if __name__ == "__main__":
    main()