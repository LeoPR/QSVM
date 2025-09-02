import pytest
import torch

from patchkit.quantize import ImageQuantizer

@pytest.mark.parametrize("levels", [2, 4, 8])
def test_uniform_levels(levels):
    img = torch.linspace(0, 1, steps=64).view(1, 8, 8)
    q = ImageQuantizer.quantize(img, levels=levels, method='uniform', dithering=False)
    uniq = torch.unique(q)
    if levels == 2:
        assert set(float(x) for x in uniq) <= {0.0, 1.0}
    else:
        # valores discretos em múltiplos de 1/(levels-1)
        step = 1.0 / (levels - 1)
        vals = set(round(float(x) / step) for x in uniq)
        # todos inteiros entre 0 e levels-1
        assert vals.issubset(set(range(levels)))

@pytest.mark.parametrize("levels", [4, 8])
def test_kmeans_levels(levels):
    try:
        import sklearn  # noqa
    except Exception:
        pytest.skip("scikit-learn não disponível")
    img = torch.rand(1, 12, 12)
    q = ImageQuantizer.quantize(img, levels=levels, method='kmeans')
    uniq = torch.unique(q)
    # deve ter até 'levels' centróides
    assert uniq.numel() <= levels
    # valores no range [0,1]
    assert float(q.min()) >= 0.0 and float(q.max()) <= 1.0

@pytest.mark.parametrize("levels", [2])
def test_otsu_binary(levels):
    img = torch.rand(1, 12, 12)
    q = ImageQuantizer.quantize(img, levels=levels, method='otsu')
    uniq = torch.unique(q)
    # Otsu deve retornar binário puro (0.0 ou 1.0)
    assert set(float(x) for x in uniq).issubset({0.0, 1.0})