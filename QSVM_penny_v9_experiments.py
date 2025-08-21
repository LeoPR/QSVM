#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSVM_penny_v9_experiments.py
------------------------------------------------------------------
Extensão do v8 com:
- Early Stopping opcional (variacionais) com patience/min_delta
- Métricas extras: macro-F1 e balanced accuracy (por fold)
- Relatório HTML consolidado (tabelas + gráficos) salvo em --report
- Mantém compatibilidade (run sequencial) e modo paralelo conservador

Execemplos:
  # Sequencial + salvar figuras e relatório
  python QSVM_penny_v9_experiments.py --kfold 3 --repeats 2 \
    --out qsvm_runs.csv --outdir qsvm_figs --report qsvm_figs/report.html \
    --early --patience 12 --min_delta 1e-4

  # Paralelo conservador (SVMs + Híbrido em paralelo; quânticos sequenciais)
  python QSVM_penny_v9_experiments.py --kfold 3 --repeats 2 \
    --out qsvm_runs_par.csv --outdir qsvm_figs --report qsvm_figs/report.html \
    --parallel --workers 4 --parallel_scope safe --early
"""

import warnings, json, time, argparse, os, base64, io, multiprocessing as mp
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.svm import SVC

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
    devices_priority = [
        ("lightning.gpu", "GPU Lightning"),
        ("lightning.qubit", "CPU Lightning"),
        ("default.qubit", "Default CPU"),
    ]
    for device_name, description in devices_priority:
        try:
            _ = qml.device(device_name, wires=min(2, wires or 1))
            return device_name
        except Exception:
            continue
    return "default.qubit"


def load_iris_binary():
    iris = load_iris()
    X, y = iris.data, iris.target
    mask = y != 2  # Setosa vs Versicolor
    X = X[mask]; y = y[mask]
    y = np.where(y == 0, -1, 1)  # {-1,+1}
    X_std = StandardScaler().fit_transform(X)
    X_ang = MinMaxScaler(feature_range=(0, 2*np.pi)).fit_transform(X_std)
    return X_std, X_ang, y


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(text: str, maxlen: int = 120):
    keep = "".join(c if c.isalnum() or c in "._-+" else "_" for c in text)
    return (keep[:maxlen]).rstrip("_")


# ------------------------------------------------------------------
# Kernels Clássicos e Híbridos
# ------------------------------------------------------------------

def rbf_kernel_classical(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def compute_kernel_matrix(X1, X2, kernel_func, **kwargs):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j], **kwargs)
    return K


class HybridQuantumKernel:
    def __init__(self, n_qubits: int, device_type: str):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_type, wires=n_qubits)

        def data_encoding_circuit(x):
            for i in range(min(len(x), n_qubits)):
                qml.RY(x[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if i < len(x) - 1:
                    qml.RZ(x[i] * x[i + 1], wires=i + 1)
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
        return probs[0]

    def compute_matrix(self, X1, X2):
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                K[i, j] = self.k(X1[i], X2[j])
        return K


# ------------------------------------------------------------------
# Fully Quantum SVM (didático)
# ------------------------------------------------------------------

class FullyQuantumSVM:
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
        return inner()[0]

    def _kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    K[i, j] = self._inner_product_prob0(X[i], X[j])
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
# Variacional v5 (original) com early stopping opcional
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
        self.history = []

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def qnode(x, params):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
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

    def _hinge(self, y, yhat):
        return np.maximum(0.0, 1.0 - y * yhat)

    def cost(self, params, X, y):
        preds = np.array([self._qnode(x, params) for x in X])
        return float(np.mean(self._hinge(y, preds)))

    def fit(self, X, y, n_epochs=100, lr=None, verbose=False,
            early_stopping=False, patience=10, min_delta=1e-4):
        n_params = self.n_layers * self.n_qubits * 3
        self.params = np.random.normal(0, 0.1, n_params)
        if lr is not None:
            self.lr = lr

        self.history = []
        best_loss = float("inf")
        best_params = self.params.copy()
        no_improve = 0

        def _maybe_es(loss):
            nonlocal best_loss, best_params, no_improve
            improved = (best_loss - loss) > float(min_delta)
            if improved:
                best_loss = loss
                best_params = self.params.copy()
                no_improve = 0
            else:
                no_improve += 1
            return improved

        if self.use_param_shift:
            opt = qml.AdamOptimizer(stepsize=self.lr)
            for epoch in range(n_epochs):
                self.params, c = opt.step_and_cost(lambda p: self.cost(p, X, y), self.params)
                self.history.append(c)
                if early_stopping and _maybe_es(c) or early_stopping and no_improve >= patience:
                    break
        else:
            eps = 1e-3
            for epoch in range(n_epochs):
                c = self.cost(self.params, X, y)
                grad = np.zeros_like(self.params)
                for i in range(len(self.params)):
                    p_plus = self.params.copy(); p_plus[i] += eps
                    p_minus = self.params.copy(); p_minus[i] -= eps
                    c_plus = self.cost(p_plus, X, y)
                    c_minus = self.cost(p_minus, X, y)
                    grad[i] = (c_plus - c_minus) / (2 * eps)
                self.params -= (self.lr * grad)
                self.history.append(c)
                if early_stopping and _maybe_es(c) or early_stopping and no_improve >= patience:
                    break

        if early_stopping and best_loss < float("inf"):
            self.params = best_params
        return self

    def predict(self, X):
        preds = [np.sign(self._qnode(x, self.params)) for x in X]
        return np.array(preds)


# ------------------------------------------------------------------
# Variacional v6 flex (com RX opcional, perdas e entangler) + ES
# ------------------------------------------------------------------

class VariationalQuantumSVM_V6Flex:
    def __init__(self, n_qubits=4, n_layers=2, device_type=None, lr: float = 0.05,
                 use_rx: bool = False, loss: str = "mse", entangler: str = "line"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.use_rx = use_rx
        self.loss = loss.lower()
        self.entangler = entangler  # 'line' ou 'circular'
        if device_type is None:
            device_type = get_best_device(n_qubits)
        self.dev = qml.device(device_type, wires=n_qubits)
        self.params = None
        self.history = []

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def qnode(x, params):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)
            idx = 0
            for L in range(self.n_layers):
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

    def fit(self, X, y, n_epochs=50, lr=None, verbose=False,
            early_stopping=False, patience=10, min_delta=1e-4):
        per_qubit = 2 + (1 if self.use_rx else 0)
        n_params = self.n_layers * self.n_qubits * per_qubit
        self.params = np.random.normal(0, 0.1, n_params)
        lr = self.lr if lr is None else lr
        opt = qml.AdamOptimizer(stepsize=lr)

        self.history = []
        best_loss = float("inf")
        best_params = self.params.copy()
        no_improve = 0

        def _maybe_es(loss):
            nonlocal best_loss, best_params, no_improve
            improved = (best_loss - loss) > float(min_delta)
            if improved:
                best_loss = loss
                best_params = self.params.copy()
                no_improve = 0
            else:
                no_improve += 1
            return improved

        for epoch in range(n_epochs):
            self.params, c = opt.step_and_cost(lambda p: self.cost(p, X, y), self.params)
            self.history.append(c)
            if early_stopping and _maybe_es(c) or early_stopping and no_improve >= patience:
                break

        if early_stopping and best_loss < float("inf"):
            self.params = best_params
        return self

    def predict(self, X):
        preds = [np.sign(self._qnode(x, self.params)) for x in X]
        return np.array(preds)


# ------------------------------------------------------------------
# Variacional Aprimorado (AngleEmbedding + hinge) + ES
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
        self.history = []

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

    def _hinge(self, y, yhat):
        return np.maximum(0.0, 1.0 - y * yhat)

    def cost(self, params, X, y):
        preds = np.array([self._qnode(x, params) for x in X])
        return float(np.mean(self._hinge(y, preds)))

    def fit(self, X, y, n_epochs=50, lr=None, verbose=False,
            early_stopping=False, patience=10, min_delta=1e-4):
        n_params = self.n_layers * self.n_qubits
        self.params = np.random.normal(0, 0.1, n_params)
        lr = self.lr if lr is None else lr
        opt = qml.AdamOptimizer(stepsize=lr)

        self.history = []
        best_loss = float("inf")
        best_params = self.params.copy()
        no_improve = 0

        def _maybe_es(loss):
            nonlocal best_loss, best_params, no_improve
            improved = (best_loss - loss) > float(min_delta)
            if improved:
                best_loss = loss
                best_params = self.params.copy()
                no_improve = 0
            else:
                no_improve += 1
            return improved

        for epoch in range(n_epochs):
            self.params, c = opt.step_and_cost(lambda p: self.cost(p, X, y), self.params)
            self.history.append(c)
            if early_stopping and _maybe_es(c) or early_stopping and no_improve >= patience:
                break

        if early_stopping and best_loss < float("inf"):
            self.params = best_params
        return self

    def predict(self, X):
        preds = [np.sign(self._qnode(x, self.params)) for x in X]
        return np.array(preds)


# ------------------------------------------------------------------
# Experiment Runner
# ------------------------------------------------------------------

class ExperimentRunner:
    def __init__(self, kfold=2, repeats=1, out_csv="qsvm_v9_runs.csv",
                 random_seeds=None, outdir="qsvm_figs",
                 early=False, patience=10, min_delta=1e-4):
        self.kfold = kfold
        self.repeats = repeats
        self.out_csv = out_csv
        self.random_seeds = random_seeds or [42]
        self.outdir = outdir
        self.early = early
        self.patience = patience
        self.min_delta = min_delta
        ensure_dir(self.outdir)

    def _make_model(self, spec: dict):
        typ = spec["type"]
        n_qubits = spec.get("n_qubits", 4)
        n_layers = spec.get("n_layers", 2)
        device_type = get_best_device(n_qubits)

        if typ == "svm_linear":
            return SVC(kernel="linear", C=spec.get("C", 1.0)), False
        if typ == "svm_rbf":
            return SVC(kernel="rbf", C=spec.get("C", 1.0), gamma=spec.get("gamma", "scale")), False
        if typ == "hybrid_kernel":
            return ("hybrid_kernel", HybridQuantumKernel(n_qubits=n_qubits, device_type=device_type)), True
        if typ == "fully_quantum":
            return FullyQuantumSVM(n_qubits=n_qubits, C=spec.get("C", 1.0), device_type=device_type), False
        if typ == "var_v5":
            return VariationalQuantumSVM_V5(n_qubits=n_qubits, n_layers=n_layers, device_type=device_type,
                                            use_param_shift=spec.get("use_param_shift", False),
                                            lr=spec.get("lr", 0.005)), False
        if typ == "var_v6flex":
            return VariationalQuantumSVM_V6Flex(n_qubits=n_qubits, n_layers=n_layers, device_type=device_type,
                                                lr=spec.get("lr", 0.05),
                                                use_rx=spec.get("use_rx", False),
                                                loss=spec.get("loss", "mse"),
                                                entangler=spec.get("entangler", "line")), False
        if typ == "var_enhanced":
            return VariationalQuantumSVM_Enhanced(n_qubits=n_qubits, n_layers=n_layers, device_type=device_type,
                                                  lr=spec.get("lr", 0.05)), False
        raise ValueError(f"Tipo desconhecido: {typ}")

    def _fit_predict(self, model_entry, Xtr, Xte, ytr, yte, spec, figtag=None):
        # model_entry pode ser ("hybrid_kernel", kernel_obj) ou um estimador padrão
        start = time.perf_counter()
        history = None
        if isinstance(model_entry, tuple) and model_entry[0] == "hybrid_kernel":
            kernel = model_entry[1]
            Ktr = kernel.compute_matrix(Xtr, Xtr)
            Kte = kernel.compute_matrix(Xte, Xtr)
            clf = SVC(kernel="precomputed", C=spec.get("C", 1.0))
            clf.fit(Ktr, ytr)
            yhat = clf.predict(Kte)
        else:
            model = model_entry
            n_epochs = spec.get("n_epochs", 50)
            kwargs = {"n_epochs": n_epochs, "lr": spec.get("lr", None), "verbose": False}
            if hasattr(model, "fit"):
                # Early stopping global do runner (não obriga specs)
                if self.early:
                    kwargs.update({"early_stopping": True,
                                   "patience": spec.get("patience", self.patience),
                                   "min_delta": spec.get("min_delta", self.min_delta)})
                try:
                    model.fit(Xtr, ytr, **kwargs)
                except TypeError:
                    # Modelos clássicos / fully quantum (sem kwargs extras)
                    model.fit(Xtr, ytr)
            yhat = model.predict(Xte)
            history = getattr(model, "history", None)
        end = time.perf_counter()
        dt = end - start

        # Se houver histórico e tag/dir, salvar curva de custo
        if history is not None:
            try:
                self._save_history_plot(history, figtag or "exp", spec)
            except Exception as e:
                print(f"[WARN] Falha ao salvar curva de custo: {e}")

        return yhat, dt

    def _save_history_plot(self, history, figtag, spec):
        fname = safe_filename(f"{figtag}__{spec['type']}__{spec}", 120) + ".png"
        path = os.path.join(self.outdir, fname)
        plt.figure(figsize=(7,4))
        plt.plot(history)
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title(f"Treinamento: {spec['type']}")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()

    def _save_topn_plot(self, agg_df, topn_plot: int = 10, suffix="topN"):
        top = agg_df.head(topn_plot).copy()
        labels = [f"{t}\n{p}" for t,p in zip(top["type"], top["params"])]
        vals = top["accuracy"].values
        plt.figure(figsize=(12, 5))
        bars = plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=25, ha="right")
        plt.ylabel("Acurácia média")
        plt.ylim(0, 1.1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        plt.title("Top-N configurações – média de acurácia")
        plt.tight_layout()
        path = os.path.join(self.outdir, f"{suffix}_topN.png")
        plt.savefig(path, dpi=160)
        plt.close()

    def _save_per_type_bar(self, df, suffix="per_type_acc"):
        agg = df.dropna(subset=["accuracy"]).groupby("type", as_index=False)["accuracy"].mean()
        plt.figure(figsize=(9,5))
        plt.bar(agg["type"], agg["accuracy"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Acurácia média")
        plt.ylim(0, 1.1)
        plt.title("Acurácia média por tipo de modelo")
        plt.tight_layout()
        path = os.path.join(self.outdir, f"{suffix}.png")
        plt.savefig(path, dpi=160)
        plt.close()

    def _save_metric_boxplots(self, df, suffix="metrics_box"):
        # Boxplots por tipo para accuracy, f1_macro, balanced_accuracy
        metrics = ["accuracy", "f1_macro", "balanced_accuracy"]
        for m in metrics:
            try:
                plt.figure(figsize=(10,5))
                data = [df.loc[df["type"]==t, m].dropna().values for t in sorted(df["type"].unique())]
                plt.boxplot(data, labels=sorted(df["type"].unique()))
                plt.xticks(rotation=25, ha="right")
                plt.ylabel(m)
                plt.ylim(0, 1.1)
                plt.title(f"Distribuição por tipo: {m}")
                plt.tight_layout()
                path = os.path.join(self.outdir, f"{suffix}_{m}.png")
                plt.savefig(path, dpi=160)
                plt.close()
            except Exception as e:
                print(f"[WARN] Falha boxplot {m}: {e}")

    def _generate_report(self, df, agg, report_path, topn_path):
        # Gera HTML simples incorporando imagens do outdir
        try:
            rows = agg.head(15).to_html(index=False)
        except Exception:
            rows = "<p>Falha ao gerar tabela.</p>"

        # Lista de imagens na pasta
        imgs = [f for f in os.listdir(self.outdir) if f.endswith(".png")]
        img_tags = "\n".join([f'<div><img src="{f}" style="max-width:100%;"></div>' for f in imgs if os.path.exists(os.path.join(self.outdir, f))])

        html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <title>Relatório QSVM v9</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1,h2 {{ margin-bottom: 0.4rem; }}
    .small {{ color: #555; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; }}
    th {{ background: #f2f2f2; }}
    img {{ margin: 10px 0; }}
    code {{ background: #f7f7f7; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Relatório de Experimentos – QSVM v9</h1>
  <p class="small">CSV base: <code>{os.path.basename(self.out_csv)}</code> | Pastas de figuras: <code>{self.outdir}</code></p>

  <h2>Resumo Top Configs (média por tipo+parâmetros)</h2>
  {rows}

  <h2>Gráficos Gerados</h2>
  {img_tags}

  <hr/>
  <p class="small">Gerado automaticamente pelo QSVM_penny_v9_experiments.py</p>
</body>
</html>"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Relatório HTML salvo em: {report_path}")

    def run(self, grids: list, topn_plot: int = 10):
        # Sequencial (compatível com v7/v8)
        X_std, X_ang, y = load_iris_binary()
        records = []
        for rep in range(self.repeats):
            seed = self.random_seeds[rep % len(self.random_seeds)]
            set_seed(seed)
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=seed)

            for fold, (idx_tr, idx_te) in enumerate(kf.split(X_std, y), start=1):
                Xtr_std, Xte_std = X_std[idx_tr], X_std[idx_te]
                Xtr_ang, Xte_ang = X_ang[idx_tr], X_ang[idx_te]
                ytr, yte = y[idx_tr], y[idx_te]

                for spec in grids:
                    spec = dict(spec)  # copy
                    typ = spec["type"]
                    if typ in ("svm_linear", "svm_rbf"):
                        Xtr, Xte = Xtr_std, Xte_std
                    else:
                        Xtr, Xte = Xtr_ang, Xte_ang

                    try:
                        model_entry, is_kernel = self._make_model(spec)
                        figtag = f"rep{rep+1}_fold{fold}"
                        yhat, dt = self._fit_predict(model_entry, Xtr, Xte, ytr, yte, spec, figtag=figtag)
                        acc = float(accuracy_score(yte, yhat))
                        f1m = float(f1_score(yte, yhat, average="macro"))
                        bal = float(balanced_accuracy_score(yte, yhat))
                        rec = {
                            "repeat": rep+1,
                            "fold": fold,
                            "seed": seed,
                            "type": typ,
                            "params": json.dumps({k:v for k,v in spec.items() if k not in ("type",)}),
                            "is_kernel": bool(is_kernel) if isinstance((model_entry if isinstance(model_entry, tuple) else (None,)), tuple) else False,
                            "train_time_s": dt,
                            "accuracy": acc,
                            "f1_macro": f1m,
                            "balanced_accuracy": bal,
                        }
                        records.append(rec)
                        print(f"[OK] rep{rep+1}/fold{fold} {typ} acc={acc:.3f} f1={f1m:.3f} bal={bal:.3f} time={dt:.2f}s")
                    except Exception as e:
                        rec = {
                            "repeat": rep+1,
                            "fold": fold,
                            "seed": seed,
                            "type": typ,
                            "params": json.dumps({k:v for k,v in spec.items() if k not in ("type",)}),
                            "is_kernel": None,
                            "train_time_s": None,
                            "accuracy": None,
                            "f1_macro": None,
                            "balanced_accuracy": None,
                            "error": str(e),
                        }
                        records.append(rec)
                        print(f"[ERR] rep{rep+1}/fold{fold} {typ}: {e}")

        df = pd.DataFrame.from_records(records)
        df.to_csv(self.out_csv, index=False)
        print(f"\nResultados salvos em: {self.out_csv}")

        # Resumos e gráficos
        ok_df = df.dropna(subset=["accuracy"])
        agg = ok_df.groupby(["type","params"], as_index=False)[["accuracy","f1_macro","balanced_accuracy"]].mean()
        agg = agg.sort_values(["accuracy","f1_macro"], ascending=False).reset_index(drop=True)
        print("\n=== TOP setups (médias) ===")
        print(agg.head(topn_plot))

        # Plots de resumo
        self._save_topn_plot(agg, topn_plot=topn_plot, suffix="sequential")
        self._save_per_type_bar(ok_df, suffix="per_type_acc_sequential")
        self._save_metric_boxplots(ok_df, suffix="metrics_box_sequential")

        return df

    # -----------------------
    # Execução paralela (nova)
    # -----------------------
    def _evaluate_single(self, payload):
        # Executa em outro processo; re-cria devices no processo filho
        (spec, Xtr, Xte, ytr, yte, rep, fold) = payload
        seed = int((rep*1000 + fold*100 + hash(json.dumps(spec)) % 97) % 2**31)
        set_seed(seed)
        try:
            typ = spec["type"]
            n_qubits = spec.get("n_qubits", 4)
            n_layers = spec.get("n_layers", 2)
            device_type = get_best_device(n_qubits)

            if typ == "svm_linear":
                model = SVC(kernel="linear", C=spec.get("C", 1.0))
                model.fit(Xtr, ytr)
                yhat = model.predict(Xte)
                is_kernel = False
            elif typ == "svm_rbf":
                model = SVC(kernel="rbf", C=spec.get("C", 1.0), gamma=spec.get("gamma", "scale"))
                model.fit(Xtr, ytr)
                yhat = model.predict(Xte)
                is_kernel = False
            elif typ == "hybrid_kernel":
                kernel = HybridQuantumKernel(n_qubits=n_qubits, device_type=device_type)
                Ktr = kernel.compute_matrix(Xtr, Xtr)
                Kte = kernel.compute_matrix(Xte, Xtr)
                clf = SVC(kernel="precomputed", C=spec.get("C", 1.0))
                clf.fit(Ktr, ytr)
                yhat = clf.predict(Kte)
                is_kernel = True
            else:
                raise RuntimeError(f"Tipo {typ} não permitido no modo paralelo 'safe'")

            dt = 0.0  # opcional medir tempo individual
            acc = float(accuracy_score(yte, yhat))
            f1m = float(f1_score(yte, yhat, average="macro"))
            bal = float(balanced_accuracy_score(yte, yhat))
            rec = {
                "repeat": rep,
                "fold": fold,
                "seed": seed,
                "type": typ,
                "params": json.dumps({k:v for k,v in spec.items() if k not in ("type",)}),
                "is_kernel": bool(is_kernel),
                "train_time_s": dt,
                "accuracy": acc,
                "f1_macro": f1m,
                "balanced_accuracy": bal,
            }
            return rec
        except Exception as e:
            return {
                "repeat": rep,
                "fold": fold,
                "seed": seed,
                "type": spec.get("type"),
                "params": json.dumps({k:v for k,v in spec.items() if k not in ("type",)}),
                "is_kernel": None,
                "train_time_s": None,
                "accuracy": None,
                "f1_macro": None,
                "balanced_accuracy": None,
                "error": str(e),
            }

    def run_parallel(self, grids: list, topn_plot: int = 10, workers: int = 2, scope: str = "safe"):
        """
        Paralelismo conservador:
          - scope='safe': paraleliza apenas SVMs clássicos e kernel híbrido
          - scope='all': (experimental) tentaria paralelizar tudo; **NÃO RECOMENDADO**
                         devido a possíveis conflitos de device/backend do PennyLane.
        """
        X_std, X_ang, y = load_iris_binary()
        records = []

        for rep in range(self.repeats):
            seed = self.random_seeds[rep % len(self.random_seeds)]
            set_seed(seed)
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=seed)

            for fold, (idx_tr, idx_te) in enumerate(kf.split(X_std, y), start=1):
                Xtr_std, Xte_std = X_std[idx_tr], X_std[idx_te]
                Xtr_ang, Xte_ang = X_ang[idx_tr], X_ang[idx_te]
                ytr, yte = y[idx_tr], y[idx_te]

                # montar payloads
                payloads = []
                for spec in grids:
                    typ = spec["type"]
                    if typ in ("svm_linear", "svm_rbf"):
                        Xtr, Xte = Xtr_std, Xte_std
                    else:
                        Xtr, Xte = Xtr_ang, Xte_ang

                    if scope == "safe" and typ not in ("svm_linear","svm_rbf","hybrid_kernel"):
                        # Executa sequencialmente fora do pool
                        try:
                            model_entry, is_kernel = self._make_model(spec)
                            figtag = f"rep{rep+1}_fold{fold}"
                            yhat, dt = self._fit_predict(model_entry, Xtr, Xte, ytr, yte, spec, figtag=figtag)
                            acc = float(accuracy_score(yte, yhat))
                            f1m = float(f1_score(yte, yhat, average="macro"))
                            bal = float(balanced_accuracy_score(yte, yhat))
                            records.append({
                                "repeat": rep+1, "fold": fold, "seed": seed, "type": typ,
                                "params": json.dumps({k:v for k,v in spec.items() if k not in ("type",)}),
                                "is_kernel": bool(is_kernel), "train_time_s": dt,
                                "accuracy": acc, "f1_macro": f1m, "balanced_accuracy": bal
                            })
                            print(f"[OK-seq] rep{rep+1}/fold{fold} {typ} acc={acc:.3f} f1={f1m:.3f} bal={bal:.3f} time={dt:.2f}s")
                        except Exception as e:
                            records.append({
                                "repeat": rep+1, "fold": fold, "seed": seed, "type": typ,
                                "params": json.dumps({k:v for k,v in spec.items() if k not in ("type",)}),
                                "is_kernel": None, "train_time_s": None,
                                "accuracy": None, "f1_macro": None, "balanced_accuracy": None,
                                "error": str(e)
                            })
                            print(f"[ERR-seq] rep{rep+1}/fold{fold} {typ}: {e}")
                    else:
                        payloads.append((spec, Xtr, Xte, ytr, yte, rep+1, fold))

                if payloads:
                    with mp.get_context("spawn").Pool(processes=workers) as pool:
                        for rec in pool.imap_unordered(self._evaluate_single, payloads):
                            records.append(rec)
                            stat = "OK" if rec.get("accuracy") is not None else "ERR"
                            print(f"[{stat}-par] rep{rec['repeat']}/fold{rec['fold']} {rec['type']} acc={rec.get('accuracy')}")

        df = pd.DataFrame.from_records(records)
        df.to_csv(self.out_csv, index=False)
        print(f"\nResultados salvos em: {self.out_csv}")

        # Resumo por setup (médias)
        ok_df = df.dropna(subset=["accuracy"])
        agg = ok_df.groupby(["type","params"], as_index=False)[["accuracy","f1_macro","balanced_accuracy"]].mean()
        agg = agg.sort_values(["accuracy","f1_macro"], ascending=False).reset_index(drop=True)

        # Plots de resumo
        self._save_topn_plot(agg, topn_plot=topn_plot, suffix="parallel")
        self._save_per_type_bar(ok_df, suffix="per_type_acc_parallel")
        self._save_metric_boxplots(ok_df, suffix="metrics_box_parallel")

        return df


# ------------------------------------------------------------------
# Grade padrão
# ------------------------------------------------------------------

def default_grids():
    grids = []
    # Clássicos
    grids += [
        {"type":"svm_linear", "C":1.0},
        {"type":"svm_rbf", "C":1.0, "gamma":"scale"},
    ]
    # Híbrido (kernel quântico) + SVC precomputed
    grids += [
        {"type":"hybrid_kernel", "n_qubits":4, "C":1.0},
    ]
    # Fully quantum
    grids += [
        {"type":"fully_quantum", "n_qubits":4, "C":1.0},
    ]
    # Variacional v5 (original)
    for use_ps in (False, True):
        for nL in (1,2):
            grids += [
                {"type":"var_v5", "n_qubits":4, "n_layers":nL, "n_epochs":100, "lr":0.005 if not use_ps else 0.02, "use_param_shift":use_ps},
            ]
    # Variacional v6 flex (perdas/ansatz/entangler)
    for loss in ("mse","hinge"):
        for use_rx in (False, True):
            for ent in ("line","circular"):
                grids += [
                    {"type":"var_v6flex","n_qubits":4,"n_layers":2,"n_epochs":80,"lr":0.02,
                     "use_rx":use_rx,"loss":loss,"entangler":ent},
                ]
    # Enhanced (hinge + AngleEmbedding)
    grids += [
        {"type":"var_enhanced","n_qubits":4,"n_layers":2,"n_epochs":80,"lr":0.02},
    ]
    return grids


# ------------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfold", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=str, default="qsvm_v9_runs.csv")
    ap.add_argument("--outdir", type=str, default="qsvm_figs")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--parallel", action="store_true", help="Executa com run_parallel (conservador)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--parallel_scope", choices=["safe","all"], default="safe")
    ap.add_argument("--report", type=str, default=None, help="Caminho do relatório HTML (opcional)")
    ap.add_argument("--early", action="store_true", help="Liga early stopping global para variacionais")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    report_path = args.report or os.path.join(args.outdir, "report.html")

    runner = ExperimentRunner(
        kfold=args.kfold, repeats=args.repeats, out_csv=args.out,
        random_seeds=[42, 1337, 1234, 2025], outdir=args.outdir,
        early=args.early, patience=args.patience, min_delta=args.min_delta
    )
    grids = default_grids()

    if args.parallel:
        print("== Modo paralelo 'conservador' ==")
        print("* safe: paraleliza somente SVMs clássicos e kernel híbrido; variacionais e fully-quantum seguem sequenciais.")
        if args.parallel_scope == "all":
            print("!! 'all' é experimental e não recomendado (pode conflitar com devices do PennyLane).")
        df = runner.run_parallel(grids, topn_plot=args.topn, workers=args.workers, scope=args.parallel_scope)
    else:
        df = runner.run(grids, topn_plot=args.topn)

    # Salva resumo por configuração (médias)
    try:
        ok_df = df.dropna(subset=["accuracy"])
        agg = ok_df.groupby(["type","params"], as_index=False)[["accuracy","f1_macro","balanced_accuracy"]].mean()
        agg_path = os.path.join(args.outdir, "summary_means.csv")
        agg.to_csv(agg_path, index=False)
        print(f"Resumo de médias salvo em: {agg_path}")

        # Gera relatório HTML incluindo gráficos salvos
        topn_img = os.path.join(args.outdir, ("parallel_topN.png" if args.parallel else "sequential_topN.png"))
        runner._generate_report(ok_df, agg, report_path, topn_img)
    except Exception as e:
        print(f"Falha ao gerar relatório: {e}")


if __name__ == "__main__":
    main()
