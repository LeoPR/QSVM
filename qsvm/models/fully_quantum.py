"""
FullyQuantumSVM (migração do QSVM_penny_v10_experiments.py)
- Kernel via SWAP-test (estimate inner product probability on ancilla=0)
- Resolve sistema linear clássico (A = K + C I) para alpha
- Suporte a multiclass via estratégia one-vs-rest (computação de alpha por classe)
- API: fit(X, y), predict(X)
"""
import numpy as np

try:
    import pennylane as qml
except Exception:
    qml = None

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

class FullyQuantumSVM:
    """
    Fully quantum kernel SVM (swap-test inner product estimate) + classical solver.
    Mirrors the implementation used in QSVM_penny_v10_experiments.py, with multiclass support.
    """
    def __init__(self, n_qubits=4, C=1.0, device_type=None):
        if qml is None:
            raise ImportError("PennyLane is required for FullyQuantumSVM. Install 'pennylane'.")
        self.n_qubits = int(n_qubits)
        self.C = float(C)
        if device_type is None:
            device_type = _get_best_device(2 * self.n_qubits + 2)
        self.device_type = device_type
        self.alpha = None              # for binary legacy (kept for API)
        self.alpha_per_class = None    # dict label -> alpha vector (for multiclass)
        self.classes_ = None
        self.X_train = None
        self.y_train = None

    def _inner_product_prob0(self, x1, x2):
        # small wires didactic swap-test (like v10). Robust fallback if device fails.
        dev = qml.device(self.device_type, wires=2 * 2 + 1)
        @qml.qnode(dev)
        def inner():
            for i in range(min(2, len(x1))):
                if abs(x1[i]) > 1e-9:
                    qml.RY(2 * np.arctan2(abs(x1[i]), 1), wires=i)
            for i in range(min(2, len(x2))):
                if abs(x2[i]) > 1e-9:
                    qml.RY(2 * np.arctan2(abs(x2[i]), 1), wires=i + 2)
            qml.Hadamard(wires=4)
            qml.CSWAP(wires=[4, 0, 2])
            qml.CSWAP(wires=[4, 1, 3])
            qml.Hadamard(wires=4)
            return qml.probs(wires=4)
        try:
            probs = inner()
            return float(probs[0])
        except Exception:
            return None

    def _kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    val = self._inner_product_prob0(X[i], X[j])
                    if val is None:
                        raise RuntimeError("quantum inner product failed")
                    K[i, j] = val
                except Exception:
                    K[i, j] = np.exp(-0.5 * np.linalg.norm(X[i] - X[j]) ** 2)
        return K

    def _solve(self, K, y):
        A = K + self.C * np.eye(len(K))
        try:
            return np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, y, rcond=None)[0]

    def fit(self, X, y):
        """
        Fit supports both binary and multiclass:
        - binary: computes self.alpha
        - multiclass: computes self.alpha_per_class {label: alpha_vector} using one-vs-rest
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = X.copy()
        self.y_train = y.copy()
        classes = np.unique(y)
        K = self._kernel_matrix(X)

        if len(classes) <= 2:
            # binary case: map labels to ±1 if necessary
            if set(classes) <= {0,1}:
                y_bin = np.where(y == classes[0], -1.0, 1.0) if len(classes)==2 else np.where(y==classes[0], 1.0, -1.0)
            else:
                # assume original labels are already ±1
                y_bin = y.astype(float)
            self.alpha = self._solve(K, y_bin)
            self.classes_ = classes.tolist()
            self.alpha_per_class = None
        else:
            # multiclass: one-vs-rest using same kernel matrix K (efficient)
            self.alpha_per_class = {}
            for cl in classes:
                y_bin = np.where(y == cl, 1.0, -1.0)
                alpha_c = self._solve(K, y_bin)
                self.alpha_per_class[cl] = alpha_c
            self.classes_ = classes.tolist()
            self.alpha = None
        return self

    def _score_for_sample(self, xt, alpha_vec):
        # compute s = sum_i alpha_i * y_i * (2*k - 1)
        s = 0.0
        for i, xi in enumerate(self.X_train):
            k = self._inner_product_prob0(xt, xi)
            if k is None:
                k = np.exp(-0.5 * np.linalg.norm(xt - xi) ** 2)
            s += alpha_vec[i] * self.y_train[i] * (2 * k - 1)
        return s

    def predict(self, Xtest):
        Xtest = np.asarray(Xtest)
        if self.alpha_per_class is not None:
            # multiclass: compute score per class and choose argmax
            preds = []
            for xt in Xtest:
                scores = []
                for cl in self.classes_:
                    alpha_c = self.alpha_per_class[cl]
                    s = self._score_for_sample(xt, alpha_c)
                    scores.append(s)
                idx = int(np.argmax(scores))
                preds.append(self.classes_[idx])
            return np.array(preds)
        else:
            # binary legacy
            if self.alpha is None:
                raise ValueError("Model not trained. Call fit() first.")
            preds = []
            for xt in Xtest:
                s = 0.0
                for i, xi in enumerate(self.X_train):
                    k = self._inner_product_prob0(xt, xi)
                    if k is None:
                        k = np.exp(-0.5 * np.linalg.norm(xt - xi) ** 2)
                    s += self.alpha[i] * self.y_train[i] * (2 * k - 1)
                preds.append(np.sign(s))
            return np.array(preds)