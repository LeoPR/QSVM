# Inicializador do subpackage qsvm.models
# Exponha os modelos aqui (cada modelo fica em um arquivo separado).
from .svm import ClassicalSVM

__all__ = [
    "ClassicalSVM",
]