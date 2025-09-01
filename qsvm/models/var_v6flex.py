"""
VariationalQuantumSVM_V6Flex
Migration of the v6-flex variational model (analytical device).
- Multiclass via one-vs-rest (treina parâmetros independentes por classe).
- Usa pennylane.numpy para parâmetros treináveis; converte para floats puros
  ao armazenar (evita 'setting an array element with a sequence').
"""
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except Exception:
    qml = None
    pnp = None

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

class VariationalQuantumSVM_V6Flex:
    def __init__(self, n_qubits=4, n_layers=2, device_type=None, lr: float = 0.05,
                 use_rx: bool = False, loss: str = "mse", entangler: str = "line"):
        if qml is None:
            raise ImportError("PennyLane é necessário para VariationalQuantumSVM_V6Flex. Instale 'pennylane'.")
        if pnp is None:
            raise ImportError("pennylane.numpy (pnp) é necessário.")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.lr = float(lr)
        self.use_rx = bool(use_rx)
        self.loss = loss.lower()
        self.entangler = entangler
        if device_type is None:
            device_type = _get_best_device(self.n_qubits)
        self.dev = qml.device(device_type, wires=self.n_qubits)
        self.params = None
        self.params_per_class = None  # dict label -> params (np.ndarray float)
        self.history = []

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def qnode(x, params):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            idx = 0
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[idx], wires=i); idx += 1
                    qml.RZ(params[idx], wires=i); idx += 1
                    if self.use_rx:
                        qml.RX(params[idx], wires=i); idx += 1
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.entangler == "circular" and self.n_qubits > 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))
        self._qnode = qnode

    def _loss(self, y, yhat):
        if self.loss == "mse":
            return (y - yhat) ** 2
        elif self.loss in ("hinge", "svm"):
            return np.maximum(0.0, 1.0 - y * yhat)
        elif self.loss in ("mae", "l1"):
            return np.abs(y - yhat)
        else:
            return (y - yhat) ** 2

    def cost(self, params, X, y):
        preds = np.array([self._qnode(x, params) for x in X])
        return float(np.mean(self._loss(y, preds)))

    def _convert_params(self, params):
        params = np.asarray(params)
        # Se for array 2D ou dtype=object, achate e force float
        if params.dtype == "object" or params.ndim > 1:
            params = np.array(params.flat, dtype=float)
        else:
            params = params.astype(float).flatten()
        return params

    def _train_single(self, X, y_bin, n_epochs, lr, early_stopping, patience, min_delta):
        per_qubit = 2 + (1 if self.use_rx else 0)
        n_params = max(1, self.n_layers * self.n_qubits * per_qubit)
        init = np.random.normal(0, 0.1, n_params)
        params = pnp.array(init, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=lr)
        best = float("inf"); best_params = params.copy(); no_improve = 0
        history = []
        for _ in range(n_epochs):
            params, c = opt.step_and_cost(lambda p: self.cost(p, X, y_bin), params)
            c = float(c)
            history.append(c)
            if early_stopping:
                if best - c > float(min_delta):
                    best = c; best_params = params.copy(); no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
        params_np = self._convert_params(params)
        return params_np, history

    def fit(self, X, y, n_epochs=50, lr=None, verbose=False,
            early_stopping=False, patience=10, min_delta=1e-4):
        lr = self.lr if lr is None else lr
        X = np.asarray(X); y = np.asarray(y)
        classes = np.unique(y)
        self.history = []

        if len(classes) <= 2:
            # Binário -> mapeia para ±1 se necessário
            if set(classes) <= {0, 1}:
                if len(classes) == 2:
                    y_bin = np.where(y == classes[0], -1.0, 1.0)
                else:
                    y_bin = np.where(y == classes[0], 1.0, -1.0)
            else:
                y_bin = y.astype(float)
            params, hist = self._train_single(X, y_bin, n_epochs, lr, early_stopping, patience, min_delta)
            self.params = params
            self.params_per_class = None
            self.history = hist
        else:
            self.params_per_class = {}
            all_hist = {}
            for cl in classes:
                y_bin = np.where(y == cl, 1.0, -1.0)
                params_c, hist_c = self._train_single(X, y_bin, n_epochs, lr, early_stopping, patience, min_delta)
                params_c = self._convert_params(params_c)
                self.params_per_class[int(cl)] = params_c
                all_hist[int(cl)] = hist_c
            self.params = None
            self.history = all_hist
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.params_per_class is not None:
            preds = []
            class_labels = list(self.params_per_class.keys())
            for x in X:
                scores = [float(self._qnode(x, self.params_per_class[cl])) for cl in class_labels]
                cl_idx = int(np.argmax(scores))
                preds.append(class_labels[cl_idx])
            return np.array(preds, dtype=int)
        else:
            if self.params is None:
                raise ValueError("Modelo não treinado. Chame fit() primeiro.")
            preds = [np.sign(self._qnode(x, self.params)) for x in X]
            return np.array(preds, dtype=float)