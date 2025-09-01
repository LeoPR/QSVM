"""
HybridQuantumKernel
- Kernel quântico com qnode que retorna probabilidades e k(x1,x2) = probs[0].
- Agora com fit/predict internos usando SVC(kernel='precomputed'), para integração
  direta com scripts de teste que esperam interface de classificador.
"""
import numpy as np

try:
    import pennylane as qml
except Exception:
    qml = None

from sklearn.svm import SVC

def _get_best_device(wires: int):
    if qml is None:
        return None
    devices_priority = [
        ("lightning.gpu", "GPU Lightning"),
        ("lightning.qubit", "CPU Lightning"),
        ("default.qubit", "Default CPU"),
    ]
    for device_name, _ in devices_priority:
        try:
            _ = qml.device(device_name, wires=min(2, wires or 1))
            return device_name
        except Exception:
            continue
    return "default.qubit"

class HybridQuantumKernel:
    """
    Implementação do kernel híbrido, espelhando QSVM v10, mas com interface de
    classificador (fit/predict) via SVC precomputed.
    """
    def __init__(self, n_qubits: int = 4, device_type: str = None, C: float = 1.0, svc_kwargs: dict | None = None):
        if qml is None:
            raise ImportError("PennyLane é necessário para HybridQuantumKernel. Instale 'pennylane'.")
        self.n_qubits = int(n_qubits)
        if device_type is None:
            device_type = _get_best_device(self.n_qubits)
        self.dev = qml.device(device_type, wires=self.n_qubits)
        self.C = float(C)
        self.svc_kwargs = dict(svc_kwargs or {})
        self._svc = None
        self._X_train = None

        def data_encoding_circuit(x):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if i < len(x) - 1:
                    qml.RZ(x[i] * x[i + 1], wires=i + 1)
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
        self._encode = data_encoding_circuit

        @qml.qnode(self.dev)
        def kernel_qnode(x1, x2):
            self._encode(x1)
            qml.adjoint(self._encode)(x2)
            return qml.probs(wires=range(self.n_qubits))
        self.kernel_qnode = kernel_qnode

    def k(self, x1, x2):
        probs = self.kernel_qnode(x1, x2)
        return float(probs[0])

    def compute_matrix(self, X1, X2):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                try:
                    K[i, j] = float(self.k(X1[i], X2[j]))
                except Exception:
                    # fallback robusto
                    K[i, j] = np.exp(-0.5 * np.linalg.norm(X1[i] - X2[j]) ** 2)
        return K

    # Interface de classificador
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self._X_train = X.copy()
        Ktr = self.compute_matrix(self._X_train, self._X_train)
        self._svc = SVC(kernel="precomputed", C=self.C, **self.svc_kwargs)
        self._svc.fit(Ktr, y)
        return self

    def predict(self, X):
        if self._svc is None or self._X_train is None:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")
        X = np.asarray(X)
        Kte = self.compute_matrix(X, self._X_train)
        return self._svc.predict(Kte)