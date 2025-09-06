"""
Loader centralizado para dependências opcionais (lazy import + cache).

Uso:
- from patchkit.optional_deps import get_ssim_func, prewarm_ssim, has_package
- get_ssim_func() -> retorna callable structural_similarity (ou levanta ImportError se ausente)
- prewarm_ssim() -> tenta carregar e cachear, retorna True se carregou com sucesso
- check_optional_deps / print_check -> utilitários simples para CI/local

Decisão de design:
- usa importlib.util.find_spec para checar disponibilidade sem importar.
- importa e cacheia a função structural_similarity na primeira chamada (thread-safe).
"""
from __future__ import annotations
from typing import Callable, Optional, Dict, Iterable
import importlib
import importlib.util
import threading

__all__ = [
    "has_package",
    "check_optional_deps",
    "print_check",
    "get_ssim_func",
    "prewarm_ssim",
    "clear_optional_cache",
]

_lock = threading.Lock()
_cache: Dict[str, object] = {}


def has_package(name: str) -> bool:
    """
    Verifica se um pacote está disponível (não importa o módulo).
    name: nome de import (ex.: 'skimage', 'numpy').
    """
    return importlib.util.find_spec(name) is not None


def check_optional_deps(pkgs: Iterable[str] = ("skimage",)) -> Dict[str, bool]:
    """
    Retorna um dicionário com disponibilidade (True/False) para as chaves em pkgs.
    Observação: usar nomes de import (ex.: 'skimage' para scikit-image).
    """
    results = {}
    for pkg in pkgs:
        results[pkg] = has_package(pkg)
    return results


def print_check(pkgs: Iterable[str] = ("skimage",)):
    """Imprime um resumo simples sobre dependências opcionais."""
    res = check_optional_deps(pkgs)
    for pkg, ok in res.items():
        status = "OK" if ok else "MISSING"
        print(f"{pkg}: {status}")
    missing = [p for p, ok in res.items() if not ok]
    if missing:
        print("\nPara instalar, por exemplo:")
        print("  pip install " + " ".join(missing))


# --------------------
# Loader específico: structural_similarity (ssim)
# --------------------
def _load_ssim_callable() -> Callable:
    """
    Importa e retorna skimage.metrics.structural_similarity. Lança ImportError com
    mensagem instrutiva se skimage não estiver instalado.
    """
    # import local (pode lançar ImportError)
    try:
        # import_module garante carregamento numa string dinâmica
        mod = importlib.import_module("skimage.metrics")
    except Exception as e:
        raise ImportError(
            "compute_ssim requer scikit-image (skimage). Instale com: pip install scikit-image"
        ) from e

    # structural_similarity deve estar em skimage.metrics
    ssim_func = getattr(mod, "structural_similarity", None)
    if ssim_func is None:
        # fallback: tentar importar diretamente (raro)
        try:
            ssim_func = importlib.import_module("skimage.metrics._structural_similarity").structural_similarity  # type: ignore
        except Exception:
            raise ImportError(
                "scikit-image foi carregado mas structural_similarity não foi encontrada."
            )
    return ssim_func


def get_ssim_func() -> Callable:
    """
    Retorna a função structural_similarity (callable). Faz lazy import + cache.
    Lança ImportError se skimage não estiver instalado.
    """
    key = "ssim_func"
    if key in _cache:
        return _cache[key]  # type: ignore
    with _lock:
        if key in _cache:
            return _cache[key]  # type: ignore
        ssim = _load_ssim_callable()
        _cache[key] = ssim
        return ssim


def prewarm_ssim() -> bool:
    """
    Tenta carregar e cachear a função ssim. Retorna True em sucesso, False se ausente.
    Útil para setup de workers (pré-carregar dependências pesadas).
    """
    try:
        get_ssim_func()
        return True
    except ImportError:
        return False


def clear_optional_cache():
    """
    Limpa o cache interno de loaders opcionais. Útil em testes que simulam ausência
    de pacotes (para forçar novo import).
    """
    with _lock:
        _cache.clear()