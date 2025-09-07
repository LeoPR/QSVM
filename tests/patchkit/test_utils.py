"""
Pequenos utilitários de teste reutilizáveis.

Uso:
  from tests.patchkit.test_utils import simulate_missing_module

simulate_missing_module(monkeypatch, prefix, clear_cache_call=None)
  - monkeypatch: fixture do pytest
  - prefix: 'skimage' ou 'sklearn' etc.
  - clear_cache_call: callable opcional para limpar caches (ex: patchkit.optional_deps.clear_optional_cache)
Retorna: None (monkeypatch aplica as alterações e o teardown do pytest restaura)
"""
import sys
import importlib
import importlib.util as importlib_util
from typing import Callable, Optional


def simulate_missing_module(monkeypatch, prefix: str, clear_cache_call: Optional[Callable] = None):
    """
    Simula ausência de um pacote (prefix) para testes:
      - limpa sys.modules das chaves que começam com prefix
      - monkeypatch de importlib.import_module para levantar ImportError para nomes que começam com prefix
      - monkeypatch de importlib.util.find_spec para retornar None para prefix (coerência)
      - opcional: chama clear_cache_call() se fornecido

    Exemplo:
      from patchkit import optional_deps
      simulate_missing_module(monkeypatch, "skimage", clear_cache_call=optional_deps.clear_optional_cache)
    """
    # limpar cache/fábricas externas (se fornecido)
    if clear_cache_call is not None:
        try:
            clear_cache_call()
        except Exception:
            # não falhar aqui — utilitário de teste deve ser resiliente
            pass

    # remover qualquer módulo já carregado do prefix
    for mod_name in list(sys.modules.keys()):
        if mod_name == prefix or mod_name.startswith(prefix + "."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    # monkeypatch import_module
    orig_import_module = importlib.import_module

    def _fake_import_module(name, package=None):
        if name == prefix or (isinstance(name, str) and name.startswith(prefix + ".")):
            raise ImportError(f"mocked missing {prefix}")
        return orig_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    # monkeypatch find_spec para coerência com has_package checks
    orig_find_spec = importlib_util.find_spec

    def _fake_find_spec(name, package=None):
        if name == prefix or (isinstance(name, str) and name.startswith(prefix + ".")):
            return None
        return orig_find_spec(name, package)

    monkeypatch.setattr(importlib_util, "find_spec", _fake_find_spec)