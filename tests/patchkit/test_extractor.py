"""
Testes consolidados para patchkit.OptimizedPatchExtractor
Inclui: extraction, caching, reconstruction, edge cases
"""
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from patchkit import OptimizedPatchExtractor


class TestPatchExtractionBasic:
    """Testes básicos de extração de patches"""

    def test_extract_patches_standard_config(self, sample_pil_images, cache_dir):
        """Testa extração com configuração padrão"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28)
        )

        patches = extractor.process(sample_pil_images['gradient'], index=0)

        # Verificações básicas
        assert isinstance(patches, torch.Tensor)
        assert patches.dtype == torch.uint8
        assert patches.shape[1:] == (4, 4)  # [L, H, W] para grayscale
        assert patches.shape[0] == extractor.num_patches_per_image

        # Valores no range correto
        assert 0 <= patches.min().item() <= patches.max().item() <= 255

    def test_patch_count_calculation(self, sample_pil_images, cache_dir):
        """Testa cálculo correto do número de patches"""
        test_cases = [
            # (image_size, patch_size, stride, expected_patches)
            ((28, 28), (4, 4), 2, 13 * 13),  # (28-4)//2 + 1 = 13
            ((14, 14), (2, 2), 1, 13 * 13),  # (14-2)//1 + 1 = 13
            ((16, 16), (8, 8), 4, 3 * 3),  # (16-8)//4 + 1 = 3
            ((10, 8), (4, 4), 2, 4 * 3),  # h=(10-4)//2+1=4, w=(8-4)//2+1=3
        ]

        for img_size, patch_size, stride, expected in test_cases:
            extractor = OptimizedPatchExtractor(
                patch_size=patch_size, stride=stride,
                cache_dir=cache_dir + f"_{img_size}_{patch_size}_{stride}",
                image_size=img_size
            )

            assert extractor.num_patches_per_image == expected

    def test_different_patch_sizes(self, sample_pil_images, cache_dir):
        """Testa diferentes tamanhos de patch"""
        patch_sizes = [(2, 2), (4, 4), (8, 8), (3, 5)]  # Incluindo não-quadrado

        for patch_size in patch_sizes:
            extractor = OptimizedPatchExtractor(
                patch_size=patch_size, stride=2,
                cache_dir=cache_dir + f"_ps_{patch_size[0]}x{patch_size[1]}",
                image_size=(28, 28)
            )

            patches = extractor.process(sample_pil_images['binary'], index=0)

            assert patches.shape[1:] == patch_size
            assert patches.shape[0] > 0

    def test_different_strides(self, sample_pil_images, cache_dir):
        """Testa diferentes valores de stride"""
        strides = [1, 2, 4, 8]

        for stride in strides:
            extractor = OptimizedPatchExtractor(
                patch_size=(4, 4), stride=stride,
                cache_dir=cache_dir + f"_stride_{stride}",
                image_size=(28, 28)
            )

            patches = extractor.process(sample_pil_images['checkerboard'], index=0)

            expected_count = ((28 - 4) // stride + 1) ** 2
            assert patches.shape[0] == expected_count

    def test_single_patch_access(self, sample_pil_images, standard_extractor):
        """Testa acesso a patch individual"""
        img = sample_pil_images['gradient']

        # Extrair patch específico
        patch_idx = 5
        single_patch = standard_extractor.get_patch(img, index=0, patch_idx=patch_idx)

        # Extrair todos e comparar
        all_patches = standard_extractor.process(img, index=0)
        expected_patch = all_patches[patch_idx]

        torch.testing.assert_close(single_patch, expected_patch)


class TestPatchExtractionCaching:
    """Testes do sistema de cache"""

    def test_memory_cache_functionality(self, sample_pil_images, cache_dir):
        """Testa cache em memória"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28),
            max_memory_cache=5
        )

        img = sample_pil_images['gradient']

        # Primeira chamada - cache miss
        patches1 = extractor.process(img, index=0)
        assert extractor.cache_misses == 1
        assert extractor.cache_hits == 0

        # Segunda chamada - cache hit
        patches2 = extractor.process(img, index=0)
        assert extractor.cache_hits == 1

        # Dados devem ser idênticos
        torch.testing.assert_close(patches1, patches2)

    def test_disk_cache_persistence(self, sample_pil_images, cache_dir):
        """Testa persistência do cache em disco"""
        config = {
            'patch_size': (4, 4), 'stride': 2,
            'cache_dir': cache_dir, 'image_size': (28, 28)
        }

        # Primeira instância - cria cache
        extractor1 = OptimizedPatchExtractor(**config)
        patches1 = extractor1.process(sample_pil_images['binary'], index=0)

        # Verificar que arquivo de cache foi criado
        cache_files = list(Path(cache_dir + "/patches_optimized").glob("*.pt.zst"))
        assert len(cache_files) >= 1

        # Segunda instância - usa cache existente
        extractor2 = OptimizedPatchExtractor(**config)
        patches2 = extractor2.process(sample_pil_images['binary'], index=0)

        # Dados devem ser idênticos
        torch.testing.assert_close(patches1, patches2)

    def test_cache_size_limit(self, sample_pil_images, cache_dir):
        """Testa limite de tamanho do cache em memória"""
        max_cache = 3
        extractor = OptimizedPatchExtractor(
            patch_size=(2, 2), stride=1,
            cache_dir=cache_dir, image_size=(8, 8),
            max_memory_cache=max_cache
        )

        # Processar mais imagens que o limite do cache
        for i in range(max_cache + 2):
            extractor.process(sample_pil_images['gradient'], index=i)

        # Cache não deve exceder o limite
        assert len(extractor.memory_cache) <= max_cache

    def test_cache_statistics(self, sample_pil_images, cache_dir):
        """Testa estatísticas do cache"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28)
        )

        img = sample_pil_images['checkerboard']

        # Operações para gerar estatísticas
        extractor.process(img, index=0)  # miss
        extractor.process(img, index=0)  # hit
        extractor.process(img, index=1)  # miss

        stats = extractor.get_cache_stats()

        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 2
        assert stats['hit_rate'] == 1 / 3
        assert stats['memory_cache_size'] == 2

    def test_cache_clearing(self, sample_pil_images, cache_dir):
        """Testa limpeza do cache"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28)
        )

        # Preencher cache
        extractor.process(sample_pil_images['gradient'], index=0)
        extractor.process(sample_pil_images['binary'], index=1)

        assert len(extractor.memory_cache) == 2

        # Limpar cache
        extractor.clear_memory_cache()

        assert len(extractor.memory_cache) == 0


class TestPatchReconstruction:
    """Testes de reconstrução de imagem a partir de patches"""

    def test_perfect_reconstruction_no_overlap(self, cache_dir):
        """Testa reconstrução perfeita sem overlapping (stride = patch_size)"""
        # Imagem 8x8, patches 4x4, stride 4 -> sem overlap
        img_array = np.tile(np.arange(8, dtype=np.uint8), (8, 1))  # gradiente horizontal
        img = Image.fromarray(img_array, mode='L')

        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=4,
            cache_dir=cache_dir, image_size=(8, 8)
        )

        patches = extractor.process(img, index=0)
        reconstructed = extractor.reconstruct_image(patches)

        # Deve ser reconstituição perfeita
        original_tensor = torch.from_numpy(img_array).to(torch.uint8)
        torch.testing.assert_close(reconstructed, original_tensor, rtol=0, atol=0)

    def test_reconstruction_with_overlap(self, sample_pil_images, standard_extractor):
        """Testa reconstrução com overlapping (stride < patch_size)"""
        img = sample_pil_images['gradient']

        patches = standard_extractor.process(img, index=0)
        reconstructed = standard_extractor.reconstruct_image(patches)

        # Deve ter formato correto
        assert reconstructed.shape == (28, 28)
        assert reconstructed.dtype == torch.uint8
        assert 0 <= reconstructed.min().item() <= reconstructed.max().item() <= 255

        # Para gradiente, deve preservar características básicas
        original_array = np.array(img)
        recon_array = reconstructed.numpy()

        # Correlação deve ser alta (não perfeita devido ao overlapping averaging)
        correlation = np.corrcoef(original_array.flatten(), recon_array.flatten())[0, 1]
        assert correlation > 0.8  # Alta correlação

    def test_reconstruction_different_devices(self, sample_pil_images, standard_extractor):
        """Testa reconstrução em diferentes devices"""
        patches = standard_extractor.process(sample_pil_images['binary'], index=0)

        # CPU reconstruction
        recon_cpu = standard_extractor.reconstruct_image(patches, device=torch.device('cpu'))
        assert recon_cpu.device.type == 'cpu'

        # GPU reconstruction (se disponível)
        if torch.cuda.is_available():
            recon_gpu = standard_extractor.reconstruct_image(patches, device=torch.device('cuda'))
            assert recon_gpu.device.type == 'cuda'

            # Resultados devem ser iguais
            torch.testing.assert_close(recon_cpu, recon_gpu.cpu())

    def test_reconstruction_multichannel(self, cache_dir):
        """Testa reconstrução com imagens RGB (multicanal)"""
        # Criar imagem RGB sintética
        rgb_array = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        rgb_img = Image.fromarray(rgb_array, mode='RGB')

        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=4,  # Sem overlap para reconstrução mais precisa
            cache_dir=cache_dir, image_size=(16, 16)
        )

        patches = extractor.process(rgb_img, index=0)  # [L, C, H, W] com C=3
        reconstructed = extractor.reconstruct_image(patches)

        assert reconstructed.shape == (3, 16, 16)  # [C, H, W]
        assert reconstructed.dtype == torch.uint8


class TestPatchExtractionEdgeCases:
    """Testes de casos extremos"""

    def test_stride_larger_than_patch(self, cache_dir):
        """Testa stride maior que patch size (gaps na reconstrução)"""
        img_array = np.full((16, 16), 128, dtype=np.uint8)  # Imagem uniforme
        img = Image.fromarray(img_array, mode='L')

        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=8,  # stride > patch_size
            cache_dir=cache_dir, image_size=(16, 16)
        )

        patches = extractor.process(img, index=0)

        expected_patches = ((16 - 4) // 8 + 1) ** 2  # = 2 * 2 = 4
        assert patches.shape[0] == expected_patches
        assert patches.shape[1:] == (4, 4)

    def test_patch_larger_than_image_fails(self, cache_dir):
        """Testa que patch maior que imagem falha adequadamente"""
        img_array = np.zeros((4, 4), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')

        extractor = OptimizedPatchExtractor(
            patch_size=(8, 8), stride=1,  # patch > image
            cache_dir=cache_dir, image_size=(4, 4)
        )

        # Deve falhar ou retornar resultado válido (dependendo da implementação)
        try:
            patches = extractor.process(img, index=0)
            # Se não falhar, deve ter formato válido
            assert patches.ndim >= 3
        except (ValueError, RuntimeError):
            # É aceitável falhar
            pass

    def test_very_small_image(self, cache_dir):
        """Testa imagem muito pequena"""
        img_array = np.array([[255]], dtype=np.uint8)  # 1x1 pixel
        img = Image.fromarray(img_array, mode='L')

        extractor = OptimizedPatchExtractor(
            patch_size=(1, 1), stride=1,
            cache_dir=cache_dir, image_size=(1, 1)
        )

        patches = extractor.process(img, index=0)

        assert patches.shape == (1, 1, 1)  # 1 patch de 1x1
        assert patches[0, 0, 0].item() == 255

    def test_non_square_patches_and_images(self, cache_dir):
        """Testa patches e imagens não-quadradas"""
        # Imagem retangular
        img_array = np.random.randint(0, 256, (12, 20), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')

        # Patches retangulares
        extractor = OptimizedPatchExtractor(
            patch_size=(3, 5), stride=2,
            cache_dir=cache_dir, image_size=(12, 20)
        )

        patches = extractor.process(img, index=0)

        expected_h = (12 - 3) // 2 + 1  # 5
        expected_w = (20 - 5) // 2 + 1  # 8
        expected_patches = expected_h * expected_w  # 40

        assert patches.shape == (expected_patches, 3, 5)

    def test_invalid_image_input(self, cache_dir):
        """Testa entrada inválida"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28)
        )

        with pytest.raises((TypeError, AttributeError, ValueError)):
            extractor.process(None, index=0)

        with pytest.raises((TypeError, AttributeError, ValueError)):
            extractor.process("not_an_image", index=0)

    def test_index_out_of_bounds(self, sample_pil_images, standard_extractor):
        """Testa índice de patch fora dos limites"""
        img = sample_pil_images['gradient']

        total_patches = standard_extractor.num_patches_per_image

        with pytest.raises(IndexError):
            standard_extractor.get_patch(img, index=0, patch_idx=total_patches)  # Índice igual ao total

        with pytest.raises(IndexError):
            standard_extractor.get_patch(img, index=0, patch_idx=-1)  # Índice negativo


class TestPatchExtractionUtilities:
    """Testes de funcionalidades utilitárias"""

    def test_cache_validation_and_cleaning(self, sample_pil_images, cache_dir):
        """Testa validação e limpeza de cache"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28)
        )

        # Criar cache válido
        extractor.process(sample_pil_images['binary'], index=0)

        # Cache deve estar válido
        invalid_count = extractor.validate_and_clean_cache()
        assert invalid_count == 0

        # TODO: Simular cache corrompido e testar limpeza
        # (requer manipulação manual de arquivos de cache)

    def test_no_index_processing(self, sample_pil_images, cache_dir):
        """Testa processamento sem índice (sem cache)"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28)
        )

        img = sample_pil_images['gradient']

        # Sem índice - não deve usar cache
        patches1 = extractor.process(img, index=None)
        patches2 = extractor.process(img, index=None)

        # Dados devem ser iguais mas cache não deve ser usado
        torch.testing.assert_close(patches1, patches2)
        assert extractor.cache_hits == 0  # Nenhum cache hit

    def test_processing_multiple_images_efficiently(self, sample_pil_images, cache_dir):
        """Testa processamento eficiente de múltiplas imagens"""
        extractor = OptimizedPatchExtractor(
            patch_size=(4, 4), stride=2,
            cache_dir=cache_dir, image_size=(28, 28),
            max_memory_cache=10
        )

        images = list(sample_pil_images.values())

        # Processar todas as imagens
        all_patches = []
        for i, img in enumerate(images):
            patches = extractor.process(img, index=i)
            all_patches.append(patches)

        # Verificar que todas têm o mesmo número de patches
        patch_counts = [p.shape[0] for p in all_patches]
        assert len(set(patch_counts)) == 1  # Todos iguais

        # Cache deve ter sido utilizado adequadamente
        stats = extractor.get_cache_stats()
        assert stats['memory_cache_size'] <= extractor.max_memory_cache