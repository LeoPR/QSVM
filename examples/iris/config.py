from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_ROOT = str(BASE_DIR / "outputs")

# Parâmetros úteis (mudar aqui se quiser)
RANDOM_SEED = 42
TEST_SIZE = 0.3

# garante existência da pasta de saída
os.makedirs(OUTPUTS_ROOT, exist_ok=True)