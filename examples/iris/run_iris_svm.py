#!/usr/bin/env python3
"""
Exemplo minimalista: treina SVMs (linear, poly, rbf) no dataset Iris,
salva relatórios, matrizes de confusão e gráficos (comparativo de acurácias + 3D train/test).
Execução: python -m examples.iris.run_iris_svm
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# config local
from examples.iris.config import OUTPUTS_ROOT, RANDOM_SEED, TEST_SIZE

KERNELS = ["linear", "poly", "rbf"]

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def plot_confusion(cm, labels, outpath):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_3d_scatter(Xp, y, train_idx, outpath, title="3D PCA - train/test"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection="3d")
    markers = {True: "o", False: "s"}
    labels = np.unique(y)
    for lbl in labels:
        sel = (y == lbl)
        # train points of this label
        sel_train = sel & train_idx
        sel_test = sel & (~train_idx)
        if sel_train.any():
            ax.scatter(Xp[sel_train,0], Xp[sel_train,1], Xp[sel_train,2],
                       marker=markers[True], s=30, label=f"train:{lbl}", alpha=0.8)
        if sel_test.any():
            ax.scatter(Xp[sel_test,0], Xp[sel_test,1], Xp[sel_test,2],
                       marker=markers[False], s=40, label=f"test:{lbl}", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(loc="upper right", fontsize="small", ncol=1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    # carregar Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names.tolist()

    # split + scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # preparar PCA (fit no dataset completo para visual consistente)
    pca = PCA(n_components=3).fit(np.vstack([X_train_s, X_test_s]))
    X_all_s = np.vstack([X_train_s, X_test_s])
    Xp = pca.transform(X_all_s)
    # índice booleano para treino/teste no array concatenado
    n_train = X_train_s.shape[0]
    train_idx = np.zeros(X_all_s.shape[0], dtype=bool)
    train_idx[:n_train] = True
    y_all = np.concatenate([y_train, y_test])

    results = []
    for kernel in KERNELS:
        out_dir = _ensure_dir(os.path.join(OUTPUTS_ROOT, kernel))
        # instanciar SVC (poly com degree=3)
        if kernel == "poly":
            clf = SVC(kernel=kernel, degree=3, probability=False, random_state=RANDOM_SEED)
        else:
            clf = SVC(kernel=kernel, probability=False, random_state=RANDOM_SEED)
        # treinar usando apenas treino
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc = float(accuracy_score(y_test, y_pred))
        rep = classification_report(y_test, y_pred, target_names=target_names)
        cm = confusion_matrix(y_test, y_pred)

        # salvar relatório texto
        _save_text(os.path.join(out_dir, "report.txt"),
                   f"kernel: {kernel}\naccuracy: {acc:.6f}\n\nclassification_report:\n{rep}\n")
        # salvar conf matrix
        plot_confusion(cm, target_names, os.path.join(out_dir, "confusion_matrix.png"))
        # salvar 3D (usando true labels, indicando train/test)
        plot_3d_scatter(Xp, y_all, train_idx, os.path.join(out_dir, "train_test_3d.png"),
                        title=f"Iris PCA 3D - kernel={kernel}")

        # salvar previsões (opcional, pequeno CSV)
        preds_path = os.path.join(out_dir, "predictions.csv")
        header = "split,row_index,y_true,y_pred"
        rows = []
        for i in range(len(y_test)):
            rows.append(f"test,{i},{int(y_test[i])},{int(y_pred[i])}")
        with open(preds_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("\n".join(rows))

        results.append({"kernel": kernel, "accuracy": acc, "out_dir": out_dir})
        print(f"[OK] kernel={kernel} acc={acc:.4f} -> outputs in {out_dir}")

    # resumo comparativo
    out_acc_csv = os.path.join(OUTPUTS_ROOT, "accuracies.csv")
    with open(out_acc_csv, "w", encoding="utf-8") as f:
        f.write("kernel,accuracy\n")
        for r in results:
            f.write(f"{r['kernel']},{r['accuracy']:.6f}\n")

    # barplot
    plt.figure(figsize=(5,3))
    kernels = [r["kernel"] for r in results]
    accs = [r["accuracy"] for r in results]
    plt.bar(kernels, accs, color=["tab:blue","tab:orange","tab:green"])
    plt.ylim(0,1)
    plt.ylabel("accuracy")
    plt.title("Iris SVM: kernel comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_ROOT, "accuracies.png"), dpi=150)
    plt.close()

    # resumo JSON
    with open(os.path.join(OUTPUTS_ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nResumo salvo em:", OUTPUTS_ROOT)
    for r in results:
        print(f" - {r['kernel']}: {r['accuracy']:.4f}")

if __name__ == "__main__":
    main()