"""
Variational Fully Quantum (VFQ) classifier (simple example).
- Requer PennyLane.
- Implementação simples: parametrized circuit -> expectation -> sigmoid -> binary label.
- Para multiclass, usa estratégia one-vs-rest externa (MultiOutputWrapper).
"""

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except Exception:
    qml = None
    pnp = None

class VariationalFullyQuantum:
    def __init__(self, n_qubits=None, n_layers=2, backend="default.qubit", device_kwargs=None, shots=None,
                 lr=0.1, epochs=50, batch_size=None, seed=42):
        if qml is None:
            raise ImportError("PennyLane não encontrado. Instale 'pennylane' para usar VariationalFullyQuantum.")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.device_kwargs = device_kwargs or {}
        self.shots = shots
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._device = None
        self.params = None
        self._constructed = False

    def _construct_circuit(self, n_qubits, n_layers):
        dev = qml.device(self.backend, wires=n_qubits, shots=self.shots, **self.device_kwargs)
        @qml.qnode(dev, interface="autograd")
        def circuit(x, params):
            # encoding
            for i in range(n_qubits):
                qml.RY(x[i % len(x)], wires=i)
            # variational layers
            p = params.reshape(n_layers, n_qubits)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(p[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))  # single expectation as score
        self._qnode = circuit
        self._device = dev
        self._constructed = True

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_qubits is None:
            self.n_qubits = max(1, X.shape[1])
        self._construct_circuit(self.n_qubits, self.n_layers)
        # init params
        rng = np.random.RandomState(self.seed)
        params = pnp.array(rng.normal(scale=0.1, size=(self.n_layers * self.n_qubits)), requires_grad=True)
        opt = qml.AdamOptimizer(self.lr)
        # simple binary logistic loss (y in {0,1})
        def loss(p, batch_x, batch_y):
            preds = []
            for xx in batch_x:
                val = self._qnode(xx, p)
                preds.append(val)
            preds = pnp.array(preds)
            # map expectation [-1,1] -> [0,1]
            probs = (1 + preds) / 2
            # binary cross entropy
            eps = 1e-8
            return -pnp.mean(batch_y * pnp.log(probs + eps) + (1 - batch_y) * pnp.log(1 - probs + eps))

        n = X.shape[0]
        bs = self.batch_size or n
        for ep in range(self.epochs):
            # simple full-batch or mini-batch
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                batch_x = X[idx]
                batch_y = y[idx]
                params = opt.step(lambda p: loss(p, batch_x, batch_y), params)
            if (ep+1) % 10 == 0:
                l = loss(params, X, y)
                # printing lightly; user can modify
                print(f"[VFQ] epoch {ep+1}/{self.epochs} loss={float(l):.4f}")
        self.params = params
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        preds = []
        for xx in X:
            val = self._qnode(xx, self.params)
            prob = float((1 + val) / 2.0)
            preds.append([1-prob, prob])
        return np.array(preds)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)