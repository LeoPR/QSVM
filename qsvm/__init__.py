# Pacote qsvm: exporta modelos principais
# Mantenha simples: exponha os modelos do subpackage `models`.
from .models import ClassicalSVM

__all__ = ["ClassicalSVM"]