import pytest

def has_skimage() -> bool:
    """Retorna True se scikit-image estiver disponível."""
    try:
        import skimage.metrics  # noqa: F401
        return True
    except Exception:
        return False

def has_lpips() -> bool:
    """Retorna True se lpips estiver disponível."""
    try:
        import lpips  # noqa: F401
        return True
    except Exception:
        return False

def has_sklearn() -> bool:
    """Retorna True se scikit-learn estiver disponível."""
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False

# Pytest skip markers (avaliados em import time — conveniente para decorar testes)
skip_if_no_skimage = pytest.mark.skipif(not has_skimage(), reason="scikit-image is required for this test")
skip_if_no_lpips = pytest.mark.skipif(not has_lpips(), reason="lpips is required for this test")
skip_if_no_sklearn = pytest.mark.skipif(not has_sklearn(), reason="scikit-learn is required for this test")

def require_skimage():
    """Chamar no início do teste para pular dinamicamente se ausente."""
    if not has_skimage():
        pytest.skip("scikit-image not available")

def require_lpips():
    if not has_lpips():
        pytest.skip("lpips not available")

def require_sklearn():
    if not has_sklearn():
        pytest.skip("scikit-learn not available")