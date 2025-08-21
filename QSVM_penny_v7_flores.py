#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSVM_penny_v7_flores.py
------------------------------------------------------------------
- Base: estrutura do v5 (clássicos, híbrido, FULLY QUANTUM, variacional original)
- Melhorias do v6: normalização [0, 2π] p/ ângulos, n_qubits=4 por padrão,
  optimizer Adam + parameter-shift opcional, classe variacional aprimorada.
- Objetivo: permitir comparar lado a lado os "sabores" e recuperar
  o desempenho do variacional original do v5 mantendo tudo no mesmo arquivo.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pennylane as qml
from pennylane import numpy as pnp

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    try:
        qml.numpy.random.seed(seed)
    except Exception:
        pass


def get_best_device(wires: int):
    """Retorna o melhor backend disponível em ordem de preferência."""
    devices_priority = [
        ("lightning.gpu", "GPU Lightning"),
        ("lightning.qubit", "CPU Lightning"),
        ("default.qubit", "Default CPU"),
    ]
    for device_name, description in devices_priority:
        try:
            # smoke test
            _ = qml.device(device_name, wires=min(2, wires or 1))
            print(f"✅ Usando: {description}")
            return device_name
        except Exception:
            continue
    print("Usando: Default CPU (fallback)")
    return "default.qubit"


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

def load_iris_data(return_features=False):
    """Iris binário (Setosa vs Versicolor), X normalizado (StandardScaler), y em {-1,+1}."""
    iris = load_iris()
    X, y = iris.data, iris.target

    mask = y != 2  # binário
    X = X[mask]
    y = y[mask]
    y = np.where(y == 0, -1, 1)  # SVM-friendly

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if return_features:
        return X_scaled, y, iris.feature_names
    return X_scaled, y


def normalize_data_for_quantum(X):
    """Mapeia features para [0, 2π] (ideal p/ codificação de ângulos)."""
    return MinMaxScaler(feature_range=(0, 2 * np.pi)).fit_transform(X)


# ------------------------------------------------------------------
# Kernels Clássicos
# ------------------------------------------------------------------

def rbf_kernel_classical(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def compute_classical_kernel_matrix(X1, X2, kernel_func, **kwargs):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j], **kwargs)
    return K


# ------------------------------------------------------------------
# Config Q (n_qubits = min(n_features, 4) por padrão)
# ------------------------------------------------------------------

def _infer_qubits(n_features: int, max_qubits: int = 4):
    return max(1, min(max_qubits, n_features))


# ------------------------------------------------------------------
# QSVM Híbrido (kernel quântico por fidelidade) – base v5
# ------------------------------------------------------------------

class HybridQuantumKernel:
    def __init__(self, n_qubits: int, device_type: str):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_type, wires=n_qubits)

        def data_encoding_circuit(x):
            # Camada 1
            for i in range(min(len(x), n_qubits)):
                qml.RY(x[i], wires=i)
            # Interações não lineares (v5)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if i < len(x) - 1:
                    qml.RZ(x[i] * x[i + 1], wires=i + 1)
            # Camada 3
            for i in range(min(len(x), n_qubits)):
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
        return probs[0]  # fidelidade ao |0...0>

    def compute_matrix(self, X1, X2):
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                K[i, j] = self.k(X1[i], X2[j])
        return K


# ------------------------------------------------------------------
# QSVM Completamente Quântico – restaurado do v5 (com mín. ajustes)
# ------------------------------------------------------------------

class FullyQuantumSVM:
    """
    Versão didática inspirada em Rebentrost et al. (2014).
    Usa SWAP test p/ kernel e resolve o sistema linear de forma clássica
    (HHL real não implementado aqui).
    """

    def __init__(self, n_qubits=4, C=1.0, device_type=None):
        self.n_qubits = n_qubits
        self.C = C
        if device_type is None:
            device_type = get_best_device(2 * n_qubits + 2)
        self.device_type = device_type
        self.alpha = None
        self.X_train = None
        self.y_train = None

    def _inner_product_prob0(self, x1, x2):
        """Teste SWAP com ancilla -> prob de |0>"""
        dev = qml.device(self.device_type, wires=2 * 2 + 1)  # 2+2 + ancilla
        @qml.qnode(dev)
        def inner():
            # |x1> em (0,1)
            for i in range(min(2, len(x1))):
                if abs(x1[i]) > 1e-9:
                    qml.RY(2 * np.arctan2(abs(x1[i]), 1), wires=i)
            # |x2> em (2,3)
            for i in range(min(2, len(x2))):
                if abs(x2[i]) > 1e-9:
                    qml.RY(2 * np.arctan2(abs(x2[i]), 1), wires=i + 2)
            # SWAP test
            qml.Hadamard(wires=4)
            qml.CSWAP(wires=[4, 0, 2])
            qml.CSWAP(wires=[4, 1, 3])
            qml.Hadamard(wires=4)
            return qml.probs(wires=4)
        return inner()[0]

    def _kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    K[i, j] = self._inner_product_prob0(X[i], X[j])
                except Exception:
                    # fallback RBF suave
                    K[i, j] = np.exp(-0.5 * np.linalg.norm(X[i] - X[j]) ** 2)
        return K

    def _solve(self, K, y):
        A = K + self.C * np.eye(len(K))
        try:
            return np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, y, rcond=None)[0]

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        K = self._kernel_matrix(X)
        self.alpha = self._solve(K, y)
        return self

    def predict(self, Xtest):
        if self.alpha is None:
            raise ValueError("Treine o modelo primeiro.")
        preds = []
        for xt in Xtest:
            s = 0.0
            for i, xi in enumerate(self.X_train):
                k = 2 * self._inner_product_prob0(xt, xi) - 1
                s += self.alpha[i] * self.y_train[i] * k
            preds.append(np.sign(s))
        return np.array(preds)


# ------------------------------------------------------------------
# QSVM Variacional – ORIGINAL do v5 (hinge loss + gradiente numérico)
# com opção de parameter-shift + Adam (sem alterar o default)
# ------------------------------------------------------------------

class VariationalQuantumSVM_V5:
    def __init__(self, n_qubits=4, n_layers=2, device_type=None,
                 use_param_shift: bool = False, lr: float = 0.05):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_param_shift = use_param_shift
        self.lr = lr
        if device_type is None:
            device_type = get_best_device(n_qubits)
        self.dev = qml.device(device_type, wires=n_qubits)
        self.params = None

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def qnode(x, params):
            # Codificação
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            # Ansatz do v5: RY, RZ, RX + CNOTs (mais expressivo)
            for L in range(self.n_layers):
                for i in range(self.n_qubits):
                    base = L * self.n_qubits * 3 + i * 3
                    qml.RY(params[base + 0], wires=i)
                    qml.RZ(params[base + 1], wires=i)
                    qml.RX(params[base + 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        self._qnode = qnode

    def _hinge_loss(self, y, yhat):
        m = y * yhat
        return np.maximum(0.0, 1.0 - m)

    def cost(self, params, X, y):
        preds = np.array([self._qnode(x, params) for x in X])
        return float(np.mean(self._hinge_loss(y, preds)))

    def fit(self, X, y, n_epochs=100, lr=None):
        n_params = self.n_layers * self.n_qubits * 3
        self.params = np.random.normal(0, 0.1, n_params)
        if lr is not None:
            self.lr = lr

        if self.use_param_shift:
            # Usa optimizer Adam do PennyLane com parameter-shift (sem trocar a loss)
            opt = qml.AdamOptimizer(stepsize=self.lr)
            for epoch in range(n_epochs):
                self.params, c = opt.step_and_cost(lambda p: self.cost(p, X, y), self.params)
                if epoch % 20 == 0:
                    print(f"[V5/param-shift] Época {epoch}: loss={c:.4f}")
        else:
            # Gradiente numérico (v5 "puro")
            eps = 1e-3
            for epoch in range(n_epochs):
                # custo atual
                c = self.cost(self.params, X, y)
                # gradiente numérico
                grad = np.zeros_like(self.params)
                for i in range(len(self.params)):
                    p_plus = self.params.copy(); p_plus[i] += eps
                    p_minus = self.params.copy(); p_minus[i] -= eps
                    c_plus = self.cost(p_plus, X, y)
                    c_minus = self.cost(p_minus, X, y)
                    grad[i] = (c_plus - c_minus) / (2 * eps)
                # atualização
                self.params -= (self.lr * grad)
                if epoch % 20 == 0:
                    print(f"[V5] Época {epoch}: loss={c:.4f}")
        return self

    def predict(self, X):
        preds = [np.sign(self._qnode(x, self.params)) for x in X]
        return np.array(preds)


# ------------------------------------------------------------------
# QSVM Variacional – versão v6 (MSE + RY/RZ + Adam)
# ------------------------------------------------------------------

class VariationalQuantumSVM_V6:
    def __init__(self, n_qubits=4, n_layers=2, device_type=None, lr: float = 0.05):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        if device_type is None:
            device_type = get_best_device(n_qubits)
        self.dev = qml.device(device_type, wires=n_qubits)
        self.params = None

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def qnode(x, params):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            for L in range(self.n_layers):
                for i in range(self.n_qubits):
                    base = L * self.n_qubits * 2 + i * 2
                    qml.RY(params[base + 0], wires=i)
                    qml.RZ(params[base + 1], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        self._qnode = qnode

    def _mse(self, y, yhat):
        return (y - yhat) ** 2

    def cost(self, params, X, y):
        preds = np.array([self._qnode(x, params) for x in X])
        return float(np.mean(self._mse(y, preds)))

    def fit(self, X, y, n_epochs=50, lr=None):
        n_params = self.n_layers * self.n_qubits * 2
        self.params = np.random.normal(0, 0.1, n_params)
        lr = self.lr if lr is None else lr
        opt = qml.AdamOptimizer(stepsize=lr)
        for epoch in range(n_epochs):
            self.params, c = opt.step_and_cost(lambda p: self.cost(p, X, y), self.params)
            if epoch % 10 == 0:
                print(f"[V6] Época {epoch}: loss={c:.4f}")
        return self

    def predict(self, X):
        preds = [np.sign(self._qnode(x, self.params)) for x in X]
        return np.array(preds)


# ------------------------------------------------------------------
# QSVM Variacional – Aprimorado (v6) com AngleEmbedding + entanglement circular + Hinge
# ------------------------------------------------------------------

class VariationalQuantumSVM_Enhanced:
    def __init__(self, n_qubits=4, n_layers=2, device_type=None, lr: float = 0.05):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        if device_type is None:
            device_type = get_best_device(n_qubits)
        self.dev = qml.device(device_type, wires=n_qubits)
        self.params = None

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def qnode(x, params):
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation='Y')
            idx = 0
            for L in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[idx], wires=i); idx += 1
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))
        self._qnode = qnode

    def _hinge(self, y, yhat):  # mantém métrica de classificação
        return np.maximum(0.0, 1.0 - y * yhat)

    def cost(self, params, X, y):
        preds = np.array([self._qnode(x, params) for x in X])
        return float(np.mean(self._hinge(y, preds)))

    def fit(self, X, y, n_epochs=50, lr=None):
        n_params = self.n_layers * self.n_qubits
        self.params = np.random.normal(0, 0.1, n_params)
        lr = self.lr if lr is None else lr
        opt = qml.AdamOptimizer(stepsize=lr)
        for epoch in range(n_epochs):
            self.params, c = opt.step_and_cost(lambda p: self.cost(p, X, y), self.params)
            if epoch % 10 == 0:
                print(f"[Enhanced] Época {epoch}: loss={c:.4f}")
        return self

    def predict(self, X):
        preds = [np.sign(self._qnode(x, self.params)) for x in X]
        return np.array(preds)


# ------------------------------------------------------------------
# Avaliação
# ------------------------------------------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test, name: str, precomputed=False):
    try:
        if precomputed:
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n=== {name} ===")
        print(f"Acurácia: {acc:.4f}")
        y_test_names = ['Setosa' if l == -1 else 'Versicolor' for l in y_test]
        y_pred_names = ['Setosa' if l == -1 else 'Versicolor' for l in y_pred]
        print(classification_report(y_test_names, y_pred_names, zero_division=0))
        return acc
    except Exception as e:
        print(f"[ERRO] {name}: {e}")
        return 0.0


def evaluate_precomputed_kernel(K_train, K_test, y_train, y_test, name: str):
    svm = SVC(kernel='precomputed', C=1.0)
    svm.fit(K_train, y_train)
    return evaluate_model(svm, K_train, K_test, y_train, y_test, f"SVM (kernel {name})", precomputed=True)


# ------------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------------

def main(random_state: int = 42, test_size: float = 0.3, max_qubits: int = 4):
    set_seed(random_state)

    # Dados
    X, y, = load_iris_data()
    Xq = normalize_data_for_quantum(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(Xq, y, test_size=test_size, random_state=random_state, stratify=y)

    n_qubits = _infer_qubits(X.shape[1], max_qubits=max_qubits)
    device_type = get_best_device(n_qubits)

    results = {}

    # 1) Clássicos
    print("\n[1] SVMs Clássicos")
    svm_lin = SVC(kernel='linear', C=1.0)
    results['SVM Linear'] = evaluate_model(svm_lin, X_tr, X_te, y_tr, y_te, "SVM Linear Clássico")

    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
    results['SVM RBF'] = evaluate_model(svm_rbf, X_tr, X_te, y_tr, y_te, "SVM RBF Clássico")

    # 2) Kernel Q híbrido (usa X normalizado para ângulo)
    print("\n[2] QSVM Híbrido (Kernel Quântico)")
    hqk = HybridQuantumKernel(n_qubits=n_qubits, device_type=device_type)
    Ktr = hqk.compute_matrix(Xq_tr, Xq_tr)
    Kte = hqk.compute_matrix(Xq_te, Xq_tr)
    results['QSVM Híbrido'] = evaluate_precomputed_kernel(Ktr, Kte, yq_tr, yq_te, "quântico (fidelidade)")

    # 3) Fully Quantum (restaurado)
    print("\n[3] QSVM Completamente Quântico")
    fqs = FullyQuantumSVM(n_qubits=n_qubits, C=1.0, device_type=device_type)
    results['QSVM Completo'] = evaluate_model(fqs, Xq_tr, Xq_te, yq_tr, yq_te, "QSVM Completamente Quântico")

    # 4) Variacional v5 (original) – hinge + grad numérico (lento) OU param-shift opcional
    print("\n[4] QSVM Variacional v5 (original)")
    v5 = VariationalQuantumSVM_V5(n_qubits=n_qubits, n_layers=2, device_type=device_type,
                                  use_param_shift=False, lr=0.005)
    v5.fit(Xq_tr, yq_tr, n_epochs=100)  # espelha v5
    results['QSVM Variacional (v5)'] = evaluate_model(v5, Xq_tr, Xq_te, yq_tr, yq_te,
                                                      "QSVM Variacional v5",)

    # 5) Variacional v6 (MSE + Adam)
    print("\n[5] QSVM Variacional v6 (MSE)")
    v6 = VariationalQuantumSVM_V6(n_qubits=n_qubits, n_layers=2, device_type=device_type, lr=0.05)
    v6.fit(Xq_tr, yq_tr, n_epochs=50)
    results['QSVM Variacional (v6)'] = evaluate_model(v6, Xq_tr, Xq_te, yq_tr, yq_te,
                                                      "QSVM Variacional v6 (MSE)")

    # 6) Variacional Aprimorado (AngleEmbedding + hinge)
    print("\n[6] QSVM Variacional Aprimorado")
    enh = VariationalQuantumSVM_Enhanced(n_qubits=n_qubits, n_layers=2, device_type=device_type, lr=0.05)
    enh.fit(Xq_tr, yq_tr, n_epochs=50)
    results['QSVM Var. Aprimorado'] = evaluate_model(enh, Xq_tr, Xq_te, yq_tr, yq_te,
                                                     "QSVM Variacional Aprimorado (hinge)")

    # Resumo
    print("\n=== RESUMO FINAL ===")
    for k, v in results.items():
        print(f"- {k:>26}: {v:.4f}")

    # Visual
    plt.figure(figsize=(12, 5))
    names = list(results.keys())
    vals = list(results.values())
    bars = plt.bar(names, vals)
    plt.ylabel("Acurácia")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=30, ha="right")
    for b, acc in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()+0.01, f"{acc:.3f}", ha="center", va="bottom")
    plt.title("Comparação de acurácia – v7 (Iris binário)")
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    main()
