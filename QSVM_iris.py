import os
import json
import argparse
import inspect
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import qsvm.models as qmodels
except Exception as e:
    print("ERRO: não foi possível importar qsvm.models:", e)
    raise

SKIP_NAMES = {"MultiOutputWrapper", "BaseModel"}

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

def prepare_iris_dataset(test_size=0.3, standardize=False, seed=42):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

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

def save_report(y_true, y_pred, model_name, out_dir, X_test):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write(report)

    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    # Salva predictions.csv: row, y_true, y_pred, features
    header = ["row", "y_true", "y_pred"] + [f"f{i}" for i in range(X_test.shape[1])]
    rows = np.column_stack([np.arange(len(y_true)), y_true, y_pred, X_test])
    np.savetxt(os.path.join(out_dir, "predictions.csv"), rows, delimiter=",", fmt="%.10g", header=",".join(header), comments="")

    # Exemplos corretos/errados
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]
    def save_subset(idxs, fname):
        k = min(5, len(idxs))
        if k == 0: return
        sel = idxs[:k]
        sub = np.column_stack([sel, y_true[sel], y_pred[sel], X_test[sel]])
        np.savetxt(os.path.join(out_dir, fname), sub, delimiter=",", fmt="%.10g", header=",".join(header), comments="")
    save_subset(correct_idx, "examples_correct_sample.csv")
    save_subset(wrong_idx, "examples_wrong_sample.csv")

def main():
    parser = argparse.ArgumentParser(description="Experimento IRIS com todos modelos do pacote qsvm.models.")
    parser.add_argument("--outdir", type=str, default="./examples/outputs/iris",
                        help="Diretório base de saída para salvar artefatos.")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true", help="Usar StandardScaler nos features.")
    parser.add_argument("--model", type=str, default=None, help="Nome do modelo específico a rodar.")
    parser.add_argument("--quick", action="store_true", help="Usar subset menor para rodar mais rápido.")
    args = parser.parse_args()

    # Descobre modelos
    classes = discover_model_classes(qmodels)
    if not classes:
        print("Nenhum modelo detectado em qsvm.models")
        return

    if args.model and args.model not in classes:
        print(f"Modelo {args.model} não encontrado. Escolha entre: {list(classes.keys())}")
        return

    # Prepara dados
    X_train, X_test, y_train, y_test = prepare_iris_dataset(
        test_size=args.test_size,
        standardize=args.standardize,
        seed=args.seed
    )

    if args.quick:
        # Subset para teste rápido
        rng = np.random.RandomState(args.seed)
        idx_tr = rng.choice(len(X_train), min(40, len(X_train)), replace=False)
        idx_te = rng.choice(len(X_test), min(20, len(X_test)), replace=False)
        X_train = X_train[idx_tr]
        y_train = y_train[idx_tr]
        X_test = X_test[idx_te]
        y_test = y_test[idx_te]

    # Diretório de saída com timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(args.outdir, timestamp)
    os.makedirs(out_base, exist_ok=True)

    # Modelos a rodar
    to_run = [args.model] if args.model else list(classes.keys())
    print(f"Rodando modelos: {to_run}")

    results = []
    for name in to_run:
        print(f"\n==> Testando modelo: {name}")
        cls = classes[name]
        model_out = os.path.join(out_base, name)
        os.makedirs(model_out, exist_ok=True)

        # Instanciar modelo (parâmetros padrão ou ajustes)
        try:
            if name == "ClassicalSVM":
                model = cls(kernel="linear", C=1.0, probability=True)
            elif name == "QuantumKernelSVM":
                model = cls(n_qubits=X_train.shape[1], feature_map_layers=1, backend="default.qubit")
            elif name == "VariationalFullyQuantum":
                model = cls(n_qubits=X_train.shape[1], n_layers=1, device_type="default.qubit", lr=0.2, epochs=5)
            elif name == "HybridModel":
                model = cls(n_qubits=X_train.shape[1], backend="default.qubit")
            elif name == "HybridQuantumKernel":
                model = cls(n_qubits=X_train.shape[1], device_type="default.qubit", C=1.0)
            elif name == "FullyQuantumSVM":
                model = cls(n_qubits=X_train.shape[1], C=1.0, device_type="default.qubit")
            elif name == "VariationalQuantumSVM_V6Flex":
                model = cls(n_qubits=X_train.shape[1], n_layers=1, device_type="default.qubit", lr=0.05, entangler="line")
            else:
                model = cls()
        except Exception as e:
            print(f"[ERROR] Falha ao instanciar {name}: {e}")
            results.append({"model": name, "status": "fail_instantiate", "accuracy": None, "error": str(e)})
            continue

        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_true, y_pred = remap_binary_if_needed(y_test, y_pred)
            unique_pred = np.unique(y_pred)
            unique_true = np.unique(y_true)
            missing_pred = set(unique_true) - set(unique_pred)
            if missing_pred:
                print(f"[WARN] {name}: predicted classes {sorted(unique_pred)} but true classes {sorted(unique_true)}; missing predictions for classes {sorted(missing_pred)}")
            acc = float(accuracy_score(y_true, y_pred))
            save_report(y_true, y_pred, name, model_out, X_test)
            results.append({"model": name, "status": "ok", "accuracy": acc, "error": None, "time_s": time.time()-t0})
            print(f"  -> {name} status=ok acc={acc:.4f} time={time.time()-t0:.2f}s")
        except Exception as e:
            print(f"[ERROR] Runtime for {name}: {e}")
            results.append({"model": name, "status": "fail_runtime", "accuracy": None, "error": str(e), "time_s": time.time()-t0})
            continue

    # CSV/JSON summary
    csv_path = os.path.join(out_base, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write("model,status,accuracy,time_s,error\n")
        for r in results:
            cf.write(f"{r['model']},{r['status']},{r.get('accuracy', '')},{r.get('time_s', '')},{r.get('error', '')}\n")
    with open(os.path.join(out_base, "summary.json"), "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    # Gráfico de acurácia comparativa
    labels = [r["model"] for r in results]
    accs = [r["accuracy"] if r["status"] == "ok" and r["accuracy"] is not None else 0.0 for r in results]
    colors = ["tab:blue" if r["status"] == "ok" and r["accuracy"] is not None else "tab:red" for r in results]
    plt.figure(figsize=(max(6, len(labels)*0.8), 4))
    x = np.arange(len(labels))
    plt.bar(x, accs, color=colors)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy (or 0 if skipped/failed)")
    plt.title("qsvm.models IRIS test results")
    plt.tight_layout()
    plot_path = os.path.join(out_base, "models_accuracy.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nResumo salvo em: {out_base}")
    print(f" - CSV: {csv_path}")
    print(f" - Plot: {plot_path}")

if __name__ == "__main__":
    main()