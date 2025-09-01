import os
import json
import argparse
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from qsvm.models import ClassicalSVM


def build_outdir(base_dir: str, params: dict) -> str:
    os.makedirs(base_dir, exist_ok=True)
    # pasta com hiperparâmetros para facilitar comparação
    parts = [
        f"kernel-{params.get('kernel')}",
        f"C-{params.get('C')}",
        f"gamma-{params.get('gamma')}",
    ]
    if params.get("kernel") == "poly":
        parts.append(f"deg-{params.get('degree')}")
    if params.get("standardize"):
        parts.append("std-true")
    else:
        parts.append("std-false")
    outdir = os.path.join(base_dir, "_".join(str(p) for p in parts))
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_report(y_true, y_pred, out_dir):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write(report)

    # também salvar cm como .npy e .csv
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    print(f"[OK] Relatório salvo em {out_dir} (accuracy={acc:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Experimento IRIS com qsvm.models (SVM clássico por enquanto).")
    parser.add_argument("--outdir", type=str, default="./examples/outputs/iris",
                        help="Diretório base de saída para salvar artefatos.")
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale", help='"scale", "auto" ou valor float (ex: 0.1)')
    parser.add_argument("--degree", type=int, default=3, help="Grau do kernel poly (se usado).")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true", help="Usar StandardScaler nos features.")
    parser.add_argument("--save_model", action="store_true", help="Salvar o modelo treinado (.joblib).")
    args = parser.parse_args()

    # Parse gamma float se necessário
    gamma = args.gamma
    try:
        if gamma not in ("scale", "auto"):
            gamma = float(gamma)
    except Exception:
        raise ValueError('--gamma deve ser "scale", "auto" ou um float (ex: 0.1)')

    # Carrega IRIS
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Standardização opcional
    scaler = None
    if args.standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Define hiperparâmetros do SVM
    model_params = {
        "kernel": args.kernel,
        "C": args.C,
        "gamma": gamma,
    }
    if args.kernel == "poly":
        model_params["degree"] = args.degree

    # Instancia e treina
    svm = ClassicalSVM(**model_params)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Preparar pasta de saída organizada
    out_params = {
        **model_params,
        "standardize": args.standardize,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    out_dir = build_outdir(args.outdir, out_params)

    # Salvar artefatos
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)

    # Salvar metadados/params
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(out_params, f, indent=2, ensure_ascii=False)

    # Salvar relatório
    save_report(y_test, y_pred, out_dir)

    # Salvar modelo opcionalmente
    if args.save_model:
        try:
            model_path = os.path.join(out_dir, "svm.joblib")
            svm.save(model_path)
            print(f"[OK] Modelo salvo em: {model_path}")
            if scaler is not None:
                # salvar scaler também
                import joblib
                joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
        except Exception as e:
            print(f"[WARN] Não foi possível salvar o modelo: {e}")

    print("[DONE] Execução concluída.")


if __name__ == "__main__":
    main()