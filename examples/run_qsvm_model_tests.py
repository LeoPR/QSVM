#!/usr/bin/env python3
"""
Run quick tests for all models found in qsvm.models and save organized outputs.

- Detects model classes in qsvm.models (or uses __all__ if available).
- Tries to instantiate each model with sensible defaults (overrides provided
  for known model names).
- Runs quick training/prediction on Iris (or binary subset for VFQ).
- Saves per-model report and a global CSV + comparison bar plots.

Also:
- Maps binary predictions between {-1,+1} and {0,1} when needed.
- Saves predictions.csv with features, y_true, y_pred.
- Saves small samples of correct and wrong predictions per model.
- Generates a sorted accuracy plot and top/bottom summary.
"""
import os
import time
import json
import argparse
import inspect
from datetime import datetime
import warnings
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

try:
    import qsvm.models as qmodels
except Exception as e:
    print("ERRO: não foi possível importar qsvm.models:", e)
    raise

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SKIP_NAMES = {"MultiOutputWrapper", "BaseModel"}

MODEL_INIT_OVERRIDES = {
    "ClassicalSVM": {"kernel": "linear", "C": 1.0, "probability": True},
    "QuantumKernelSVM": {"n_qubits": None, "feature_map_layers": 1, "backend": "default.qubit",
                         "device_kwargs": {}, "svc_kwargs": {"kernel": "precomputed", "C": 1.0, "class_weight": "balanced"}},
    "VariationalFullyQuantum": {"n_qubits": 2, "n_layers": 1, "device_type": "default.qubit",
                                "lr": 0.2, "epochs": 5, "batch_size": None},
    "HybridModel": {"n_qubits": 2, "backend": "default.qubit", "device_kwargs": {}},
    "HybridQuantumKernel": {"n_qubits": 2, "device_type": "default.qubit", "C": 1.0},
    "FullyQuantumSVM": {"n_qubits": 2, "C": 1.0, "device_type": "default.qubit"},
    "VariationalQuantumSVM_V6Flex": {"n_qubits": 2, "n_layers": 1, "device_type": "default.qubit",
                                     "lr": 0.05, "use_rx": False, "loss": "mse", "entangler": "line"},
}

def discover_model_classes(module):
    classes = {}
    if hasattr(module, "__all__") and isinstance(module.__all__, (list, tuple)):
        for nm in module.__all__:
            if hasattr(module, nm):
                attr = getattr(module, nm)
                if inspect.isclass(attr):
                    classes[nm] = attr
    else:
        for nm in dir(module):
            if nm.startswith("_"):
                continue
            attr = getattr(module, nm)
            if inspect.isclass(attr) and getattr(attr, "__module__", "").startswith("qsvm"):
                classes[nm] = attr
    for s in SKIP_NAMES:
        classes.pop(s, None)
    return classes

def safe_instantiate(cls, override_kwargs=None):
    override_kwargs = override_kwargs or {}
    try:
        return cls(**override_kwargs)
    except TypeError:
        try:
            return cls()
        except Exception as e:
            raise RuntimeError(f"Não consegui instanciar {cls} com kwargs={override_kwargs}: {e}")

def is_pennylane_required(cls):
    src = ""
    try:
        src = inspect.getsource(cls)
    except Exception:
        pass
    name = cls.__name__.lower()
    return ("pennylane" in src) or ("quantum" in name) or ("vfq" in name) or ("hybrid" in name) or ("qkernel" in name)

def prepare_dataset(binary_for_vfq=False, quick=False, seed=42):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    if binary_for_vfq:
        sel = np.isin(y, [0, 1])
        X = X[sel]
        y = y[sel]
    rng = np.random.RandomState(seed)
    if quick:
        n_keep = min(60, X.shape[0])
        idx = rng.choice(np.arange(X.shape[0]), size=n_keep, replace=False)
        X = X[idx]
        y = y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def stratified_subsample(X, y, n_total, seed=42):
    rng = np.random.RandomState(seed)
    classes, _ = np.unique(y, return_counts=True)
    n_classes = len(classes)
    per_class = max(1, n_total // n_classes)
    chosen = []
    for cl in classes:
        idxs = np.where(y == cl)[0]
        k = min(per_class, len(idxs))
        if k > 0:
            pick = rng.choice(idxs, size=k, replace=False)
            chosen.extend(list(pick))
    remaining = list(set(range(len(y))) - set(chosen))
    rng.shuffle(remaining)
    while len(chosen) < min(n_total, len(y)) and remaining:
        chosen.append(remaining.pop())
    return sorted(chosen)

def remap_binary_if_needed(y_true, y_pred):
    yt = set(np.unique(y_true).tolist())
    yp = set(np.unique(y_pred).tolist())
    # map {-1,1} -> {0,1}
    if yt <= {0, 1} and yp <= {-1, 1}:
        y_pred = ((y_pred + 1) // 2).astype(int)
    # map {0,1} -> {-1,1}
    elif yt <= {-1, 1} and yp <= {0, 1}:
        y_pred = (2 * y_pred - 1).astype(int)
    return y_true, y_pred

def evaluate_model(name, cls, out_base, quick=False, seed=42):
    info = {"model": name, "status": "ok", "accuracy": None, "time_s": None, "error": None}
    binary = (name.lower().startswith("variational") or "vfq" in name.lower())
    if is_pennylane_required(cls):
        try:
            import pennylane  # noqa: F401
        except Exception:
            info["status"] = "skipped"
            info["error"] = "pennylane_not_installed"
            print(f"[SKIP] {name}: pennylane ausente.")
            return info

    X_train, X_test, y_train, y_test, scaler = prepare_dataset(binary_for_vfq=binary, quick=quick, seed=seed)

    overrides = dict(MODEL_INIT_OVERRIDES.get(name, {}))
    if "n_qubits" in overrides and (overrides["n_qubits"] is None):
        overrides["n_qubits"] = max(1, X_train.shape[1])

    train_idx = None
    test_idx = None
    if name == "QuantumKernelSVM" or name.lower().startswith("qkernel"):
        tr_n = min(40, X_train.shape[0])
        te_n = min(20, X_test.shape[0])
        train_idx = stratified_subsample(X_train, y_train, n_total=tr_n, seed=seed)
        test_idx = stratified_subsample(X_test, y_test, n_total=te_n, seed=seed+1)

    try:
        model = safe_instantiate(cls, overrides)
    except Exception as e:
        info["status"] = "fail_instantiate"
        info["error"] = str(e)
        print(f"[ERROR] Instantiation failed for {name}: {e}")
        return info

    t0 = time.time()
    try:
        Xtr = X_train[train_idx] if train_idx is not None else X_train
        ytr = y_train[train_idx] if train_idx is not None else y_train
        Xte = X_test[test_idx] if test_idx is not None else X_test
        yte = y_test[test_idx] if test_idx is not None else y_test

        Xtr = np.asarray(Xtr); Xte = np.asarray(Xte)
        ytr = np.asarray(ytr); yte = np.asarray(yte)

        # Fit
        if hasattr(model, "fit"):
            try:
                model.fit(Xtr, ytr)
            except TypeError:
                model.fit(Xtr, ytr)

        # Predict
        y_pred = model.predict(Xte)
        y_pred = np.asarray(y_pred)

        if y_pred.ndim == 2:
            if yte.ndim == 2:
                y_pred_labels = np.argmax(y_pred, axis=1)
                y_test_labels = np.argmax(yte, axis=1)
            else:
                y_pred_labels = np.argmax(y_pred, axis=1)
                y_test_labels = yte
        else:
            y_pred_labels = y_pred
            y_test_labels = yte

        # Mapear binário se necessário
        y_test_labels, y_pred_labels = remap_binary_if_needed(y_test_labels, y_pred_labels)

        unique_pred = np.unique(y_pred_labels)
        unique_true = np.unique(y_test_labels)
        missing_pred = set(unique_true) - set(unique_pred)
        if missing_pred:
            print(f"[WARN] {name}: predicted classes {sorted(unique_pred)} but true classes {sorted(unique_true)}; missing predictions for classes {sorted(missing_pred)}")

        acc = float(accuracy_score(y_test_labels, y_pred_labels))
        info["accuracy"] = acc

        # Save artifacts per model
        model_out = os.path.join(out_base, name)
        os.makedirs(model_out, exist_ok=True)

        # y_test / y_pred
        np.save(os.path.join(model_out, "y_test.npy"), y_test_labels)
        np.save(os.path.join(model_out, "y_pred.npy"), y_pred_labels)

        # predictions.csv (inclui features)
        header = ["row", "y_true", "y_pred"] + [f"f{i}" for i in range(Xte.shape[1])]
        rows = np.column_stack([np.arange(len(y_test_labels)), y_test_labels, y_pred_labels, Xte])
        np.savetxt(os.path.join(model_out, "predictions.csv"), rows, delimiter=",", fmt="%.10g", header=",".join(header), comments="")

        # amostras corretas/erradas
        correct_idx = np.where(y_test_labels == y_pred_labels)[0]
        wrong_idx = np.where(y_test_labels != y_pred_labels)[0]

        def save_subset(idxs, fname):
            k = min(5, len(idxs))
            if k == 0:
                return
            sel = idxs[:k]
            sub = np.column_stack([sel, y_test_labels[sel], y_pred_labels[sel], Xte[sel]])
            np.savetxt(os.path.join(model_out, fname), sub, delimiter=",", fmt="%.10g", header=",".join(header), comments="")

        save_subset(correct_idx, "examples_correct_sample.csv")
        save_subset(wrong_idx, "examples_wrong_sample.csv")

        cm = confusion_matrix(y_test_labels, y_pred_labels)
        with open(os.path.join(model_out, "report.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Status: ok\n")
            f.write(f"Accuracy: {acc:.6f}\n\n")
            f.write("Confusion matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification report:\n")
            f.write(classification_report(y_test_labels, y_pred_labels, zero_division=0))
        with open(os.path.join(model_out, "init_overrides.json"), "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2)
    except Exception as e:
        info["status"] = "fail_runtime"
        info["error"] = str(e)
        print(f"[ERROR] Runtime for {name}: {e}")
    finally:
        info["time_s"] = time.time() - t0
    return info

def main():
    parser = argparse.ArgumentParser(description="Run quick tests for models in qsvm.models")
    parser.add_argument("--outdir", default="examples/outputs/qsvm_model_tests", help="base output directory")
    parser.add_argument("--quick", action="store_true", help="use smaller subsets to run faster (recommended)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()

    classes = discover_model_classes(qmodels)
    if not classes:
        print("Nenhum modelo detectado em qsvm.models")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(args.outdir, timestamp)
    os.makedirs(out_base, exist_ok=True)

    results = []
    print(f"Detectados {len(classes)} classes: {list(classes.keys())}")
    for name, cls in classes.items():
        print(f"\n==> Testando modelo: {name}")
        res = evaluate_model(name, cls, out_base, quick=args.quick, seed=args.seed)
        print(f"  -> {name} status={res['status']} acc={res.get('accuracy')} time={res.get('time_s'):.2f}s error={res.get('error')}")
        results.append(res)

    # save CSV/JSON summary
    csv_path = os.path.join(out_base, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["model", "status", "accuracy", "time_s", "error"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    with open(os.path.join(out_base, "summary.json"), "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    # Plot accuracies (unsorted)
    labels = [r["model"] for r in results]
    accs = [r["accuracy"] if r["status"] == "ok" and r["accuracy"] is not None else 0.0 for r in results]
    colors = ["tab:blue" if r["status"] == "ok" and r["accuracy"] is not None else "tab:red" for r in results]
    plt.figure(figsize=(max(6, len(labels)*0.8), 4))
    x = np.arange(len(labels))
    plt.bar(x, accs, color=colors)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy (or 0 if skipped/failed)")
    plt.title("qsvm models quick test results")
    plt.tight_layout()
    plot_path = os.path.join(out_base, "models_accuracy.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Sorted accuracy plot + top/bottom txt
    ok = [(r["model"], r["accuracy"]) for r in results if r["status"] == "ok" and r["accuracy"] is not None]
    ok_sorted = sorted(ok, key=lambda x: x[1], reverse=True)
    if ok_sorted:
        labels_s, accs_s = zip(*ok_sorted)
        plt.figure(figsize=(max(6, len(labels_s)*0.8), 4))
        x = np.arange(len(labels_s))
        plt.bar(x, accs_s, color="tab:blue")
        plt.xticks(x, labels_s, rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("qsvm models accuracy (sorted)")
        plt.tight_layout()
        plot_sorted_path = os.path.join(out_base, "models_accuracy_sorted.png")
        plt.savefig(plot_sorted_path, dpi=150)
        plt.close()

        topn = min(3, len(ok_sorted))
        with open(os.path.join(out_base, "top_bottom.txt"), "w", encoding="utf-8") as f:
            f.write("Top modelos por acurácia:\n")
            for i in range(topn):
                f.write(f"{i+1}. {ok_sorted[i][0]}: {ok_sorted[i][1]:.4f}\n")
            f.write("\nPiores modelos por acurácia:\n")
            for i in range(topn):
                m = ok_sorted[-(i+1)]
                f.write(f"{i+1}. {m[0]}: {m[1]:.4f}\n")

    print(f"\nResumo salvo em: {out_base}")
    print(f" - CSV: {csv_path}")
    print(f" - Plot: {plot_path}")

if __name__ == "__main__":
    main()