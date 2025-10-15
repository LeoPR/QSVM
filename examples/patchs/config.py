# Configuração comum para os exemplos de "patchs".
# Centraliza diretórios de saída/caches com caminhos relativos a esta pasta.

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_ROOT = str(BASE_DIR / "outputs")

# Caches usados pelos exemplos de teste
CACHE_QUAN_TEST_DIR = str(Path(OUTPUTS_ROOT) / "cache_quan_test")
CACHE_ART_TEST_DIR = str(Path(OUTPUTS_ROOT) / "cache_art_test")
CACHE_PATCH_TEST_DIR = str(Path(OUTPUTS_ROOT) / "cache_patch_test")