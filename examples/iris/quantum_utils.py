"""
Kernel quântico mínimo para Iris usando PennyLane.

Ideia:
- Codifica x com rotações RY por feature em n_qubits.
- Aplica U(x1) seguido de adjoint(U(x2)); |<φ(x1)|φ(x2)>|^2 = prob(|0...0>).
- Retorna K(i,j) = prob0(x_i, x_j).
"""
from typing import Optional
import numpy as np

try:
    import pennylane as qml
except Exception as e:
    qml = None  # será verificado em tempo de uso


def get_best_device(n_qubits: int) -> str:
    """
    Tenta 'lightning.qubit' (se instalado) e cai em 'default.qubit'.
    """
    if qml is None:
        raise ImportError("PennyLane não está instalado. Instale com: pip install pennylane")
    for dev_name in ("lightning.qubit", "default.qubit"):
        try:
            _ = qml.device(dev_name, wires=max(1, n_qubits))
            return dev_name
        except Exception:
            continue
    return "default.qubit"


class QuantumKernelPL:
    """
    Kernel quântico com encoding RY por feature e sobreposição via adjoint.
    Uso:
      qk = QuantumKernelPL(n_qubits=4)  # ou deixe None para pegar de X
      Ktr = qk.compute_matrix(X_train, X_train)
      Kte = qk.compute_matrix(X_test, X_train)
    """
    def __init__(self, n_qubits: Optional[int] = None, device_type: Optional[str] = None):
        self.n_qubits_cfg = n_qubits
        self.device_type = device_type
        self._qnode = None
        self._nq = None

    def _build_qnode(self, n_qubits: int):
        if qml is None:
            raise ImportError("PennyLane não está instalado. Instale com: pip install pennylane")
        dev_name = self.device_type or get_best_device(n_qubits)
        dev = qml.device(dev_name, wires=n_qubits)

        def encode(x):
            L = min(len(x), n_qubits)
            for i in range(L):
                qml.RY(float(x[i]), wires=i)

        @qml.qnode(dev)
        def kernel_qnode(x1, x2):
            encode(x1)
            qml.adjoint(encode)(x2)
            return qml.probs(wires=list(range(n_qubits)))

        self._qnode = kernel_qnode
        self._nq = n_qubits

    def compute_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Retorna K de shape (len(X1), len(X2)) com prob0.
        X1/X2 devem estar na mesma escala angular (ex.: [0, 2π]).
        """
        n_qubits = self.n_qubits_cfg or X1.shape[1]
        if self._qnode is None or self._nq != n_qubits:
            self._build_qnode(n_qubits)

        K = np.zeros((len(X1), len(X2)), dtype=float)
        for i in range(len(X1)):
            xi = np.asarray(X1[i], dtype=float)
            for j in range(len(X2)):
                xj = np.asarray(X2[j], dtype=float)
                probs = self._qnode(xi, xj)
                K[i, j] = float(probs[0])
        return K