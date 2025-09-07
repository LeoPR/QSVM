import types
import numpy as np
import pytest
import importlib

from patchkit import optional_deps
from patchkit.image_metrics import ImageMetrics


def test_get_ssim_func_imports_once_with_cache_and_clear(monkeypatch):
    # Garantir cache limpo
    optional_deps.clear_optional_cache()

    calls = {"count": 0}
    orig_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "skimage.metrics":
            calls["count"] += 1
            # retornar um "módulo" falso com structural_similarity
            mod = types.SimpleNamespace()
            def dummy_ssim(a, b, **kwargs):
                return 1.0
            mod.structural_similarity = dummy_ssim
            return mod
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    # Primeira chamada deve importar (count=1)
    ssim1 = optional_deps.get_ssim_func()
    assert callable(ssim1)
    assert calls["count"] == 1

    # Segunda chamada deve vir do cache (count permanece 1)
    ssim2 = optional_deps.get_ssim_func()
    assert ssim2 is ssim1
    assert calls["count"] == 1

    # Limpar cache e chamar de novo deve importar de novo (count=2)
    optional_deps.clear_optional_cache()
    ssim3 = optional_deps.get_ssim_func()
    assert callable(ssim3)
    assert calls["count"] == 2


def test_prewarm_ssim_false_when_missing(monkeypatch):
    # Garantir cache limpo
    optional_deps.clear_optional_cache()

    # Bloquear import do skimage.metrics
    orig_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "skimage" or (isinstance(name, str) and name.startswith("skimage.")):
            raise ImportError("mocked missing skimage")
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    ok = optional_deps.prewarm_ssim()
    assert ok is False


def test_compute_ssim_with_injected_callable_skips_import(monkeypatch):
    # Se alguém tentar importar skimage, falhar — para garantir que a injeção evita import
    orig_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "skimage" or (isinstance(name, str) and name.startswith("skimage.")):
            raise AssertionError("structural_similarity should not be imported when ssim_callable is provided")
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    # Callable SSIM "fake"
    def fake_ssim(a, b, **kwargs):
        # retorna 1.0 se a e b forem idênticas; qualquer outra coisa, 0.5
        if np.array_equal(a, b):
            return 1.0
        return 0.5

    a = np.zeros((8, 8), dtype=np.float64)
    b = a.copy()
    s = ImageMetrics.compute_ssim(a, b, ssim_callable=fake_ssim)
    assert s == pytest.approx(1.0)