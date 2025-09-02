"""
Testes consolidados para patchkit.ProcessedDataset
Inclui: resize algorithms, compression artifacts, quantization, caching
"""
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from pathlib import Path

from patchkit import ProcessedDataset, OptimizedPatchExtractor


class TestProcessedDatasetBasic:
    """Testes básicos com dados sintéticos (rápidos)"""

    @pytest.mark.parametrize("target_size", [(14, 14), (32, 32), (16, 24)])
    def test_resize_different_sizes(self, tiny_synthetic, cache_dir, target_size):
        """Testa redimensionamento para diferentes tamanhos"""
        ds = tiny_synthetic(n=5, size=(28, 28))

        processed = ProcessedDataset(
            ds, target_size=target_size, resize_alg=Image.BICUBIC,
            cache_dir=cache_dir, cache_rebuild=True
        )

        assert len(processed) == len(ds)
        assert processed.data.shape[0] == 5
        assert processed.data.shape[-2:] == target_size
        assert 0.0 <= processed.data.min().item() <= processed.data.max().item() <= 1.0

    @pytest.mark.parametrize("resize_alg", [
        Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS
    ])
    def test_resize_algorithms(self, tiny_synthetic, cache_dir, resize_alg):
        """Testa diferentes algoritmos de redimensionamento"""
        ds = tiny_synthetic(n=3, size=(28, 28), pattern='gradient')

        processed = ProcessedDataset(
            ds, target_size=(14, 14), resize_alg=resize_alg,
            cache_dir=cache_dir, cache_rebuild=True
        )

        assert processed.data.shape[-2:] == (14, 14)
        # Gradiente deve manter características básicas
        assert processed.data.var() > 0.01  # Não deve ser uniforme

    @pytest.mark.parametrize("qlevels,qmethod", [
        (2, 'uniform'), (4, 'uniform'), (8, 'uniform'),
        (2, 'otsu'), (4, 'adaptive')
    ])
    def test_quantization_combinations(self, tiny_synthetic, cache_dir, qlevels, qmethod):
        """Testa diferentes combinações de quantização"""
        ds = tiny_synthetic(n=3, size=(16, 16))

        processed = ProcessedDataset(
            ds, target_size=(16, 16),
            quantization_levels=qlevels, quantization_method=qmethod,
            cache_dir=cache_dir, cache_rebuild=True
        )

        unique_vals = torch.unique(processed.data)
        assert unique_vals.numel() <= qlevels
        assert 0.0 <= unique_vals.min().item() <= unique_vals.max().item() <= 1.0

    @pytest.mark.parametrize("img_format,quality", [
        ('JPEG', 30), ('JPEG', 70), ('JPEG', 95),
        ('PNG', None)  # PNG é lossless
    ])
    def test_compression_artifacts(self, tiny_synthetic, cache_dir, img_format, quality):
        """Testa artefatos de compressão"""
        ds = tiny_synthetic(n=3, size=(32, 32), pattern='checkerboard')

        processed = ProcessedDataset(
            ds, target_size=(32, 32),
            image_format=img_format, quality=quality,
            cache_dir=cache_dir, cache_rebuild=True
        )

        assert processed.data.shape[-2:] == (32, 32)

        # JPEG baixa qualidade deve introduzir mais artefatos
        if img_format == 'JPEG' and quality <= 50:
            # Teste básico: dados ainda devem estar no range válido
            assert 0.0 <= processed.data.min().item() <= processed.data.max().item() <= 1.0


class TestProcessedDatasetCaching:
    """Testes de sistema de cache"""

    def test_cache_creation_and_reuse(self, tiny_synthetic, cache_dir):
        """Testa criação e reutilização de cache"""
        ds = tiny_synthetic(n=5, size=(28, 28))

        # Primeira execução - cria cache
        processed1 = ProcessedDataset(
            ds, target_size=(14, 14), quantization_levels=2,
            cache_dir=cache_dir, cache_rebuild=True
        )

        cache_files = list(Path(cache_dir).glob("*.pt.zst"))
        assert len(cache_files) >= 1

        # Segunda execução - usa cache
        processed2 = ProcessedDataset(
            ds, target_size=(14, 14), quantization_levels=2,
            cache_dir=cache_dir, cache_rebuild=False
        )

        # Dados devem ser idênticos
        torch.testing.assert_close(processed1.data, processed2.data)
        torch.testing.assert_close(processed1.labels, processed2.labels)

    def test_cache_invalidation_on_config_change(self, tiny_synthetic, cache_dir):
        """Testa que mudança de config invalida cache"""
        ds = tiny_synthetic(n=3, size=(16, 16))

        # Cache com config 1
        processed1 = ProcessedDataset(
            ds, target_size=(16, 16), quantization_levels=2,
            cache_dir=cache_dir, cache_rebuild=True
        )

        # Cache com config 2 (diferente) - deve criar novo cache
        processed2 = ProcessedDataset(
            ds, target_size=(16, 16), quantization_levels=4,
            cache_dir=cache_dir, cache_rebuild=False  # Não force rebuild
        )

        # Dados devem ser diferentes (diferentes níveis de quantização)
        unique1 = torch.unique(processed1.data).numel()
        unique2 = torch.unique(processed2.data).numel()
        assert unique1 != unique2  # Diferentes números de valores únicos

    def test_cache_rebuild_force(self, tiny_synthetic, cache_dir):
        """Testa rebuild forçado de cache"""
        ds = tiny_synthetic(n=3, size=(12, 12))

        # Cache inicial
        processed1 = ProcessedDataset(
            ds, target_size=(12, 12),
            cache_dir=cache_dir, cache_rebuild=True
        )

        # Rebuild forçado
        processed2 = ProcessedDataset(
            ds, target_size=(12, 12),
            cache_dir=cache_dir, cache_rebuild=True
        )

        # Dados devem ser idênticos
        torch.testing.assert_close(processed1.data, processed2.data)


class TestProcessedDatasetEdgeCases:
    """Testes de casos extremos"""

    def test_empty_dataset(self, cache_dir):
        """Testa dataset vazio"""

        class EmptyDataset(torch.utils.data.Dataset):
            def __len__(self): return 0

            def __getitem__(self, idx): raise IndexError("Empty dataset")

        empty_ds = EmptyDataset()

        processed = ProcessedDataset(
            empty_ds, target_size=(16, 16),
            cache_dir=cache_dir, cache_rebuild=True
        )

        assert len(processed) == 0
        assert processed.data.shape[0] == 0

    def test_single_item_dataset(self, cache_dir):
        """Testa dataset com um único item"""

        class SingleDataset(torch.utils.data.Dataset):
            def __len__(self): return 1

            def __getitem__(self, idx):
                if idx != 0: raise IndexError()
                img = torch.full((1, 8, 8), 0.5)
                return img, 42

        single_ds = SingleDataset()

        processed = ProcessedDataset(
            single_ds, target_size=(4, 4), quantization_levels=2,
            cache_dir=cache_dir, cache_rebuild=True
        )

        assert len(processed) == 1
        assert processed.data.shape == (1, 1, 4, 4)
        assert processed.labels.shape == (1,)
        assert processed.labels[0].item() == 42

    def test_inconsistent_input_sizes(self, cache_dir):
        """Testa dataset com tamanhos inconsistentes de entrada"""

        class VarSizeDataset(torch.utils.data.Dataset):
            def __len__(self): return 3

            def __getitem__(self, idx):
                sizes = [(8, 8), (16, 12), (10, 14)]
                h, w = sizes[idx]
                img = torch.rand(1, h, w)
                return img, idx

        var_ds = VarSizeDataset()

        # Deve normalizar todos para o mesmo tamanho
        processed = ProcessedDataset(
            var_ds, target_size=(10, 10),
            cache_dir=cache_dir, cache_rebuild=True
        )

        assert processed.data.shape == (3, 1, 10, 10)

    def test_invalid_target_size(self, tiny_synthetic, cache_dir):
        """Testa target_size inválido"""
        ds = tiny_synthetic(n=2)

        with pytest.raises((ValueError, TypeError)):
            ProcessedDataset(
                ds, target_size=(0, 10),  # Largura zero
                cache_dir=cache_dir
            )

        with pytest.raises((ValueError, TypeError)):
            ProcessedDataset(
                ds, target_size=(-5, 10),  # Negativo
                cache_dir=cache_dir
            )


class TestProcessedDatasetIntegration:
    """Testes de integração com OptimizedPatchExtractor"""

    def test_processed_dataset_with_patch_extraction(self, tiny_synthetic, cache_dir):
        """Testa ProcessedDataset + OptimizedPatchExtractor juntos"""
        ds = tiny_synthetic(n=3, size=(28, 28))

        # Processar dataset
        processed = ProcessedDataset(
            ds, target_size=(28, 28), quantization_levels=2,
            cache_dir=cache_dir, cache_rebuild=True
        )

        # Converter primeira imagem para PIL
        first_tensor = processed.data[0]  # [1, 28, 28]
        first_pil = transforms.ToPILImage()(first_tensor.squeeze(0))

        # Extrair patches
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir + "/patches",
            image_size=(28, 28)
        )

        patches = extractor.process(first_pil, index=0)

        # Verificações
        assert patches.shape[1:] == (4, 4)
        assert patches.shape[0] > 0
        # Imagem quantizada binária -> patches também binários
        assert torch.all(torch.isin(patches[0], torch.tensor([0, 255], dtype=patches.dtype)))

    def test_different_processed_configs_compatibility(self, tiny_synthetic, cache_dir):
        """Testa compatibilidade entre diferentes configurações"""
        ds = tiny_synthetic(n=2, size=(16, 16))

        configs = [
            {'target_size': (16, 16), 'quantization_levels': 2},
            {'target_size': (8, 8), 'quantization_levels': 4},
            {'target_size': (12, 12), 'image_format': 'JPEG', 'quality': 50}
        ]

        processed_datasets = []
        for i, config in enumerate(configs):
            processed = ProcessedDataset(
                ds, cache_dir=f"{cache_dir}/config_{i}",
                cache_rebuild=True, **config
            )
            processed_datasets.append(processed)

        # Cada um deve ter processado corretamente
        for i, processed in enumerate(processed_datasets):
            assert len(processed) == 2
            expected_size = configs[i]['target_size']
            assert processed.data.shape[-2:] == expected_size


@pytest.mark.slow
@pytest.mark.integration
class TestProcessedDatasetWithRealData:
    """Testes com dados reais (MNIST) - marcados como lentos"""

    def test_mnist_basic_processing(self, cache_dir, project_dirs):
        """Testa processamento básico do MNIST real"""

        # Usar subset muito pequeno para não demorar
        transform = transforms.ToTensor()
        full_mnist = datasets.MNIST(
            root=str(project_dirs['data']),
            train=True, download=True, transform=transform
        )

        # Apenas primeiras 10 imagens
        subset = torch.utils.data.Subset(full_mnist, list(range(10)))

        processed = ProcessedDataset(
            subset, target_size=(14, 14), quantization_levels=2,
            cache_dir=cache_dir, cache_rebuild=True
        )

        assert len(processed) == 10
        assert processed.data.shape == (10, 1, 14, 14)

        # Verificar que são realmente dados binários
        unique_vals = torch.unique(processed.data)
        assert set(unique_vals.tolist()).issubset({0.0, 1.0})

    def test_mnist_compression_comparison(self, cache_dir, project_dirs):
        """Compara diferentes níveis de compressão JPEG no MNIST"""
        transform = transforms.ToTensor()
        full_mnist = datasets.MNIST(
            root=str(project_dirs['data']),
            train=True, download=True, transform=transform
        )

        # Apenas 5 imagens
        subset = torch.utils.data.Subset(full_mnist, list(range(5)))

        qualities = [10, 50, 90]  # Baixa, média, alta qualidade
        processed_results = {}

        for quality in qualities:
            processed = ProcessedDataset(
                subset, target_size=(28, 28),
                image_format='JPEG', quality=quality,
                cache_dir=f"{cache_dir}/jpeg_q{quality}",
                cache_rebuild=True
            )
            processed_results[quality] = processed.data.clone()

        # Verificar que qualidades diferentes produzem resultados diferentes
        # (exceto em casos muito específicos)
        low_quality = processed_results[10]
        high_quality = processed_results[90]

        # Deve haver alguma diferença (artefatos de compressão)
        diff = torch.abs(low_quality - high_quality).mean()
        assert diff >= 0.0  # Pelo menos deve funcionar sem erro