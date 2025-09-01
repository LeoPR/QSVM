import shutil
import pytest

@pytest.fixture
def cache_dir(tmp_path):
    """
    Pasta temporária para caches/outputs dos testes.
    Os arquivos são removidos automaticamente ao final do teste.
    Use str(cache_dir) ao passar para APIs que esperam path string.
    """
    d = tmp_path / "cache"
    d.mkdir()
    yield str(d)
    # cleanup explícito (geralmente tmp_path é limpo pelo pytest, mas garantimos)
    try:
        shutil.rmtree(str(d))
    except Exception:
        pass