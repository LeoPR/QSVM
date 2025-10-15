"""
qkernel_utils.py — utilitários mínimos para kernel quântico com PennyLane.

Fornece:
- get_best_device(wires): escolhe o melhor backend disponível.
- make_overlap_qkernel(n_qubits, device): cria uma função de kernel k(x1,x2) via overlap (U(x1) adj(U(x2))).
- compute_kernel_matrix(X1, X2, kfunc): monta matriz K para SVC precomputed.

Observações:
- Espera que as features de entrada já estejam escaladas para ângulos (ex.: [0, 2π]).
- n_qubits pode ser igual ao número de features (ou menor, usando apenas as primeiras).
"""

from typing import Callable
import numpy as np
import pennylane as qml


def get_best_device(wires: int) -> str:
    """
    Escolhe o melhor device disponível, priorizando lightning.qubit.
    """
    for dev in ("lightning.qubit", "default.qubit"):
        try:
            _ = qml.device(dev, wires=max(1, wires))
            return dev
        except Exception:
            continue
    return "default.qubit"


def make_overlap_qkernel(n_qubits: int, device: str | None = None) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Cria uma função k(x1, x2) que retorna a probabilidade de |0...0> após
    aplicar U(x1) e depois adj(U(x2)) (overlap-based kernel).
    - U(x): RY(x_i) em cada qubit, seguido de anel de CNOT (leve acoplamento).
    """
    dev_name = device or get_best_device(n_qubits)
    dev = qml.device(dev_name, wires=n_qubits)

    def _encode(x: np.ndarray):
        L = min(n_qubits, len(x))
        for i in range(L):
            qml.RY(float(x[i]), wires=i)
        # entanglement ring simples
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

    @qml.qnode(dev)
    def _qnode(x1: np.ndarray, x2: np.ndarray):
        _encode(x1)
        qml.adjoint(_encode)(x2)
        return qml.probs(wires=range(n_qubits))

    def k(x1: np.ndarray, x2: np.ndarray) -> float:
        probs = _qnode(x1, x2)
        return float(probs[0])  # prob de |0...0>, aproxima |<ψ(x1)|ψ(x2)>|^2

    return k


def compute_kernel_matrix(X1: np.ndarray, X2: np.ndarray, kfunc: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """
    Constrói a matriz de kernel K[i,j] = k(X1[i], X2[j]).
    """
    K = np.zeros((len(X1), len(X2)), dtype=float)
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i, j] = kfunc(X1[i], X2[j])
    return K