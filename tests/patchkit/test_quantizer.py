"""
Testes consolidados para patchkit.quantize.ImageQuantizer
Combina funcionalidade básica + edge cases + validações
"""
import pytest
import torch
import numpy as np
from PIL import Image

from patchkit.quantize import ImageQuantizer


class TestImageQuantizerBasic:
    """Testes básicos de funcionalidade"""

    @pytest.mark.parametrize("levels", [2, 4, 8])
    def test_uniform_quantization_levels(self, levels):
        """Testa quantização uniforme com diferentes níveis"""
        img = torch.linspace(0, 1, steps=64).view(1, 8, 8)
        quantized = ImageQuantizer.quantize(img, levels=levels, method='uniform', dithering=False)

        # Verifica range [0,1]
        assert 0.0 <= quantized.min().item() <= quantized.max().item() <= 1.0

        # Verifica tipos e shapes
        assert isinstance(quantized, torch.Tensor)
        assert quantized.shape == img.shape
        assert quantized.dtype == torch.float32

        unique_vals = torch.unique(quantized)

        if levels == 2:
            # Binário: apenas 0.0 e 1.0
            expected = {0.0, 1.0}
            actual = set(float(x) for x in unique_vals)
            assert actual.issubset(expected)
        else:
            # Níveis discretos: múltiplos de 1/(levels-1)
            step = 1.0 / (levels - 1)
            # Converte para índices inteiros [0, 1, 2, ..., levels-1]
            indices = set(round(float(x) / step) for x in unique_vals)
            expected_indices = set(range(levels))
            assert indices.issubset(expected_indices)

    @pytest.mark.parametrize("levels", [4, 8])
    def test_kmeans_quantization(self, levels):
        """Testa quantização por K-means"""
        try:
            import sklearn  # noqa
        except ImportError:
            pytest.skip("scikit-learn não disponível")

        # Imagem com gradiente para ter múltiplos clusters
        img = torch.rand(1, 12, 12)
        quantized = ImageQuantizer.quantize(img, levels=levels, method='kmeans')

        # Verificações básicas
        assert quantized.shape == img.shape
        assert 0.0 <= quantized.min().item() <= quantized.max().item() <= 1.0

        # Deve ter no máximo 'levels' valores únicos
        unique_vals = torch.unique(quantized)
        assert unique_vals.numel() <= levels

    @pytest.mark.parametrize("levels", [2])
    def test_otsu_quantization(self, levels):
        """Testa quantização Otsu (binária)"""
        # Imagem com contraste para Otsu funcionar bem
        img = torch.cat([
            torch.zeros(1, 6, 12),  # região escura
            torch.ones(1, 6, 12)  # região clara
        ], dim=1)

        quantized = ImageQuantizer.quantize(img, levels=levels, method='otsu')

        # Otsu deve retornar apenas valores binários
        unique_vals = torch.unique(quantized)
        actual = set(float(x) for x in unique_vals)
        assert actual.issubset({0.0, 1.0})
        assert quantized.shape == img.shape

    @pytest.mark.parametrize("levels", [3, 6])
    def test_adaptive_quantization(self, levels):
        """Testa quantização adaptativa"""
        # Imagem com distribuição não uniforme
        img = torch.cat([
            torch.full((1, 4, 8), 0.1),  # região escura
            torch.full((1, 4, 8), 0.5),  # região média
            torch.full((1, 4, 8), 0.9),  # região clara
        ], dim=1)

        quantized = ImageQuantizer.quantize(img, levels=levels, method='adaptive')

        assert quantized.shape == img.shape
        assert 0.0 <= quantized.min().item() <= quantized.max().item() <= 1.0

        # Deve ter valores discretos
        unique_vals = torch.unique(quantized)
        assert unique_vals.numel() <= levels

    def test_dithering_option(self):
        """Testa opção de dithering para quantização binária"""
        img = torch.rand(1, 8, 8)

        # Sem dithering
        q1 = ImageQuantizer.quantize(img, levels=2, method='uniform', dithering=False)

        # Com dithering
        q2 = ImageQuantizer.quantize(img, levels=2, method='uniform', dithering=True)

        # Ambos devem ser válidos e binários
        for q in [q1, q2]:
            assert 0.0 <= q.min().item() <= q.max().item() <= 1.0
            unique_vals = set(float(x) for x in torch.unique(q))
            assert unique_vals.issubset({0.0, 1.0})

        # Com dithering deve ser diferente (exceto em casos muito específicos)
        # Não testamos diferença exata pois pode variar


class TestImageQuantizerEdgeCases:
    """Testes de casos extremos e validação de entrada"""

    def test_invalid_quantization_levels(self):
        """Testa níveis de quantização inválidos"""
        img = torch.rand(1, 8, 8)

        # Níveis muito baixos devem falhar
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            ImageQuantizer.quantize(img, levels=1, method='uniform')

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            ImageQuantizer.quantize(img, levels=0, method='uniform')

    def test_invalid_method(self):
        """Testa método de quantização inválido"""
        img = torch.rand(1, 8, 8)

        with pytest.raises(ValueError, match="Unknown quantization method"):
            ImageQuantizer.quantize(img, levels=2, method='nonexistent_method')

    def test_nan_input_handling(self):
        """Testa tratamento de entrada com NaN"""
        img = torch.rand(1, 8, 8)
        img[0, 0, 0] = float('nan')

        # Deve falhar ou retornar resultado válido (dependendo da implementação)
        try:
            result = ImageQuantizer.quantize(img, levels=2, method='uniform')
            # Se não falhar, resultado deve estar no range válido (exceto NaN)
            finite_mask = torch.isfinite(result)
            if finite_mask.any():
                finite_vals = result[finite_mask]
                assert 0.0 <= finite_vals.min().item() <= finite_vals.max().item() <= 1.0
        except (ValueError, RuntimeError, TypeError):
            # É aceitável falhar com entrada inválida
            pass

    def test_inf_input_handling(self):
        """Testa tratamento de entrada com infinito"""
        img = torch.rand(1, 8, 8)
        img[0, 0, 0] = float('inf')
        img[0, 0, 1] = float('-inf')

        # Deve falhar ou normalizar adequadamente
        try:
            result = ImageQuantizer.quantize(img, levels=2, method='uniform')
            # Se não falhar, valores finitos devem estar no range
            finite_mask = torch.isfinite(result)
            if finite_mask.any():
                finite_vals = result[finite_mask]
                assert 0.0 <= finite_vals.min().item() <= finite_vals.max().item() <= 1.0
        except (ValueError, RuntimeError, TypeError):
            # É aceitável falhar com entrada inválida
            pass

    def test_empty_tensor(self):
        """Testa tensor vazio"""
        empty_img = torch.empty(1, 0, 0)

        try:
            result = ImageQuantizer.quantize(empty_img, levels=2, method='uniform')
            assert result.shape == empty_img.shape
        except (ValueError, RuntimeError, IndexError):
            # É aceitável falhar com tensor vazio
            pass

    def test_single_pixel(self):
        """Testa imagem de um pixel"""
        single_pixel = torch.tensor([[[0.5]]])

        result = ImageQuantizer.quantize(single_pixel, levels=2, method='uniform')

        assert result.shape == single_pixel.shape
        assert result.item() in [0.0, 1.0]  # Deve ser quantizado para 0 ou 1

    def test_uniform_input(self):
        """Testa imagem completamente uniforme"""
        uniform_img = torch.full((1, 8, 8), 0.7)

        result = ImageQuantizer.quantize(uniform_img, levels=4, method='uniform')

        # Imagem uniforme deve continuar uniforme após quantização
        unique_vals = torch.unique(result)
        assert unique_vals.numel() == 1  # Apenas um valor único
        assert 0.0 <= unique_vals.item() <= 1.0

    def test_already_quantized_input(self):
        """Testa entrada já quantizada"""
        # Imagem já binária
        binary_img = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

        result = ImageQuantizer.quantize(binary_img, levels=2, method='uniform')

        # Deve permanecer praticamente igual
        torch.testing.assert_close(result, binary_img, rtol=1e-5, atol=1e-5)


class TestImageQuantizerInputTypes:
    """Testes de diferentes tipos de entrada"""

    def test_pil_image_input(self):
        """Testa entrada PIL Image"""
        # Criar PIL Image grayscale
        pil_img = Image.new('L', (8, 8), 128)  # Cinza médio

        result = ImageQuantizer.quantize(pil_img, levels=2, method='uniform')

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape[0] == 1  # Single channel
        assert 0.0 <= result.min().item() <= result.max().item() <= 1.0

    def test_invalid_input_type(self):
        """Testa tipo de entrada inválido"""
        # Numpy array deve falhar (não suportado diretamente)
        numpy_img = np.random.rand(8, 8)

        with pytest.raises(ValueError, match="image must be a PIL.Image or torch.Tensor"):
            ImageQuantizer.quantize(numpy_img, levels=2, method='uniform')

    def test_multichannel_tensor(self):
        """Testa tensor multicanal (RGB)"""
        rgb_img = torch.rand(3, 8, 8)  # RGB

        result = ImageQuantizer.quantize(rgb_img, levels=4, method='uniform')

        assert result.shape == rgb_img.shape
        assert 0.0 <= result.min().item() <= result.max().item() <= 1.0

    def test_device_preservation(self):
        """Testa preservação do device do tensor"""
        img = torch.rand(1, 8, 8)

        result = ImageQuantizer.quantize(img, levels=2, method='uniform')

        # Deve estar no mesmo device que a entrada
        assert result.device == img.device

        # Se GPU disponível, testar também
        if torch.cuda.is_available():
            img_cuda = img.cuda()
            result_cuda = ImageQuantizer.quantize(img_cuda, levels=2, method='uniform')
            assert result_cuda.device == img_cuda.device


class TestImageQuantizerFallbacks:
    """Testes de fallbacks quando dependências opcionais não estão disponíveis"""

    def test_kmeans_fallback_without_sklearn(self, monkeypatch):
        """Testa fallback quando sklearn não está disponível"""

        # Mock sklearn import failure
        def mock_import_fail(*args, **kwargs):
            raise ImportError("sklearn not available")

        monkeypatch.setattr("builtins.__import__", mock_import_fail)

        img = torch.rand(1, 8, 8)

        # Deve fazer fallback para uniform quantization
        result = ImageQuantizer.quantize(img, levels=4, method='kmeans')

        assert isinstance(result, torch.Tensor)
        assert 0.0 <= result.min().item() <= result.max().item() <= 1.0

    def test_otsu_fallback_without_skimage(self, monkeypatch):
        """Testa fallback quando skimage não está disponível"""

        # Mock skimage import failure
        def mock_import_fail(*args, **kwargs):
            raise ImportError("skimage not available")

        monkeypatch.setattr("builtins.__import__", mock_import_fail)

        img = torch.rand(1, 8, 8)

        # Deve fazer fallback para threshold simples em 0.5
        result = ImageQuantizer.quantize(img, levels=2, method='otsu')

        assert isinstance(result, torch.Tensor)
        unique_vals = set(float(x) for x in torch.unique(result))
        assert unique_vals.issubset({0.0, 1.0})