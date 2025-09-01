"""
Quantum kernel SVM wrapper.
- Requer pennylane (pip install pennylane).
- Usa um feature map (angle embedding + entanglers) para calcular kernel matrix.
- Treina sklearn.svm.SVC(kernel='precomputed') sobre a Gram matrix quântica.
"""
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except Exception:
    qml = None
    pnp = None

from sklearn.svm import SVC

class QuantumKernelSVM:
    def __init__(self, n_qubits=None, feature_map_layers=1, backend="default.qubit", device_kwargs=None,
                 svc_kwargs=None):
        if qml is None:
            raise ImportError("PennyLane não encontrado. Instale 'pennylane' para usar QuantumKernelSVM.")
        self.qml = qml
        self.n_qubits = n_qubits
        self.feature_map_layers = feature_map_layers
        self.backend = backend
        self.device_kwargs = device_kwargs or {}
        self.svc_kwargs = svc_kwargs or {"kernel": "precomputed", "C": 1.0}
        self._device = None
        self._kernel_qnode = None
        self._X_train = None
        self._svc = SVC(**self.svc_kwargs)

    def _build_device(self, n_qubits):
        self._device = qml.device(self.backend, wires=n_qubits, **self.device_kwargs)

    def _feature_map(self, x):
        # x is length n_qubits (or will be mapped/resized)
        # simple angle embedding + entangling layers
        for i in range(len(x)):
            qml.RY(float(x[i]), wires=i)
        # entangling layers
        for _ in range(self.feature_map_layers):
            for i in range(len(x) - 1):
                qml.CNOT(wires=[i, i + 1])
            # optional rotational layer
            for i in range(len(x)):
                qml.RZ(0.1, wires=i)

    def _make_qnode(self, n_qubits):
        dev = qml.device(self.backend, wires=n_qubits, **self.device_kwargs)
        @qml.qnode(dev, interface="autograd")
        def kernel_circuit(x1, x2):
            # Encode x1, then inverse encode x2 and measure fidelity-like overlap
            self._feature_map(x1)
            qml.adjoint(self._feature_map)(x2)
            # measure fidelity via overlap with all-zero projector approximated by PauliZ product
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return kernel_circuit

    def _kernel_between(self, x1, x2):
        # compute similarity as product over expectation values transformed to scalar
        # simpler: use statevector overlap via qml.state() if backend supports it.
        # fallback: use inner product of expectation vector.
        try:
            # Try state overlap if device supports state
            dev = qml.device(self.backend, wires=len(x1), **self.device_kwargs)
            @qml.qnode(dev, interface="autograd")
            def overlap(a, b):
                self._feature_map(a)
                return qml.state()
            s1 = overlap(x1)
            @qml.qnode(dev, interface="autograd")
            def overlap2(a, b):
                self._feature_map(b)
                return qml.state()
            s2 = overlap2(x2)
            # compute fidelity (absolute inner product squared)
            fid = np.abs(np.vdot(s1, s2)) ** 2
            return float(fid)
        except Exception:
            # fallback: compute expectation vector similarity
            if self._kernel_qnode is None:
                self._kernel_qnode = self._make_qnode(len(x1))
            v1 = np.array(self._kernel_qnode(x1, x1), dtype=float)
            v2 = np.array(self._kernel_qnode(x2, x2), dtype=float)
            # cosine similarity
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom == 0:
                return 0.0
            return float(np.dot(v1, v2) / denom)

    def _gram_matrix(self, X1, X2):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2), dtype=float)
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_between(X1[i], X2[j])
        return K

    def fit(self, X, y):
        X = np.asarray(X)
        self._X_train = X.copy()
        # adapt n_qubits
        n_qubits = self.n_qubits or X.shape[1]
        self._build_device(n_qubits)
        # compute Gram matrix
        K = self._gram_matrix(X, X)
        self._svc = SVC(**self.svc_kwargs)
        self._svc.fit(K, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        K_test = self._gram_matrix(X, self._X_train)
        return self._svc.predict(K_test)

    def decision_function(self, X):
        X = np.asarray(X)
        K_test = self._gram_matrix(X, self._X_train)
        return self._svc.decision_function(K_test)