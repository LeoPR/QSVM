"""
Hybrid model:
- Extracts quantum features for each input (vector of expectations) using a small circuit,
  then concatenates these features with classical features and trains a classical estimator.
- If pennylane not available, the class will raise ImportError.
"""
import numpy as np

try:
    import pennylane as qml
except Exception:
    qml = None

from sklearn.base import clone
from sklearn.svm import SVC

class HybridModel:
    def __init__(self, quantum_feature_map=None, n_qubits=None, backend="default.qubit", device_kwargs=None,
                 classical_estimator=None):
        if qml is None:
            raise ImportError("PennyLane não encontrado. Instale 'pennylane' para usar HybridModel.")
        self.qml = qml
        self.quantum_feature_map = quantum_feature_map  # callable(x) -> expectation vector or None to use default
        self.n_qubits = n_qubits
        self.backend = backend
        self.device_kwargs = device_kwargs or {}
        self.classical_estimator = classical_estimator or SVC(kernel="rbf", probability=True)
        self._estimator = clone(self.classical_estimator)
        self._device = None
        self._qnode = None

    def _default_feature_map(self, x):
        # returns vector of expectations (length n_qubits)
        for i in range(self.n_qubits):
            qml.RY(float(x[i % len(x)]), wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _build_qnode(self):
        dev = qml.device(self.backend, wires=self.n_qubits, **self.device_kwargs)
        @qml.qnode(dev, interface="autograd")
        def feature_circuit(x):
            # encode
            for i in range(self.n_qubits):
                qml.RY(float(x[i % len(x)]), wires=i)
            # small entangling
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            # measure PauliZ expectations
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self._qnode = feature_circuit
        self._device = dev

    def _quantum_features(self, X):
        X = np.asarray(X)
        if self.n_qubits is None:
            self.n_qubits = max(1, X.shape[1])
        if self._qnode is None:
            self._build_qnode()
        feats = []
        for x in X:
            vec = np.array(self._qnode(x), dtype=float)
            feats.append(vec)
        return np.vstack(feats)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        qf = self._quantum_features(X)
        Xcat = np.hstack([X, qf])
        self._estimator = clone(self.classical_estimator)
        self._estimator.fit(Xcat, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        qf = self._quantum_features(X)
        Xcat = np.hstack([X, qf])
        return self._estimator.predict(Xcat)

    def predict_proba(self, X):
        X = np.asarray(X)
        qf = self._quantum_features(X)
        Xcat = np.hstack([X, qf])
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(Xcat)
        raise RuntimeError("classical estimator has no predict_proba")