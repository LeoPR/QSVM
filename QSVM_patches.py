#!/usr/bin/env python3
"""
QSVM patch dataset test flow (robust quick-mode for patches)

- Quick subset garante pelo menos 2 classes e 2 amostras por classe.
- Para modelos quânticos/variacionais/hybrid reduzimos dimensionalidade via PCA
  para evitar vetores exponenciais e erros de alocação.
- Gera "flow" visual: small, large, patch small, patch large, recon small.
- Salva report.txt, confusion_matrix, predictions.csv, exemplos e imagens de flow.
"""
import os
import json
import argparse
import inspect
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import qsvm.models as qmodels
except Exception as e:
    print("ERRO: não foi possível importar qsvm.models:", e)
    raise

SKIP_NAMES = {"MultiOutputWrapper", "BaseModel"}

def discover_model_classes(module):
    """
    Descobre classes exportadas em qsvm.models de forma robusta,
    usando __all__ quando disponível ou varredura por atributos.
    """
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

# == Dummy dataset loader: substitua por seu loader real ==
def load_patches_dataset(classes=None, n_per_class=2, seed=42):
    """
    Dummy loader for development. Replace with the real loader.
    Returns:
      X_small: (N, Hs, Ws)
      X_large: (N, Hl, Wl)
      y: (N,)
    """
    rng = np.random.RandomState(seed)
    all_classes = np.arange(10)
    selected = classes if classes is not None and len(classes) >= 2 else list(rng.choice(all_classes, 2, replace=False))
    X_small, X_large, y = [], [], []
    for cl in selected:
        for i in range(n_per_class):
            small = rng.randint(0, 255, (8, 8), dtype=np.uint8)
            large = rng.randint(0, 255, (32, 32), dtype=np.uint8)
            X_small.append(small)
            X_large.append(large)
            y.append(cl)
    return np.array(X_small), np.array(X_large), np.array(y)

def extract_patch(img, patch_size, stride, patch_idx):
    """
    Extrai o patch de índice patch_idx de uma imagem 2D seguindo varredura
    linha-por-linha com dado stride.
    """
    h, w = img.shape
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
    if patch_idx < 0 or patch_idx >= len(patches):
        return None
    return patches[patch_idx]

def reassemble_from_patches_from_image(img, patch_size, stride):
    """
    Reconstrói imagem a partir dos próprios patches da imagem (média nas sobreposições).
    """
    h, w = img.shape
    out = np.zeros((h, w), dtype=float)
    count = np.zeros((h, w), dtype=float)
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            out[i:i+patch_size, j:j+patch_size] += img[i:i+patch_size, j:j+patch_size]
            count[i:i+patch_size, j:j+patch_size] += 1
    out /= np.maximum(count, 1)
    return out

def plot_flow(X_small, X_large, y, idx, save_path, patch_idx=None):
    """
    Gera figura com: pequena, grande, patch pequeno, patch grande, reconstrução pequena.
    """
    small = X_small[idx]
    large = X_large[idx]
    label = y[idx]

    hs, ws = small.shape
    num_patches = (hs - 2 + 1) * (ws - 2 + 1)
    if patch_idx is None:
        patch_idx = num_patches // 2

    patch_small = extract_patch(small, 2, 1, patch_idx)
    patch_large = extract_patch(large, 4, 2, patch_idx)
    recon_small = reassemble_from_patches_from_image(small, 2, 1)

    fig, axs = plt.subplots(1, 5, figsize=(14, 3))
    axs[0].imshow(small, cmap="gray"); axs[0].set_title("Small")
    axs[1].imshow(large, cmap="gray"); axs[1].set_title("Large")
    axs[2].imshow(patch_small, cmap="gray"); axs[2].set_title("Patch Small (2x2)")
    axs[3].imshow(patch_large, cmap="gray"); axs[3].set_title("Patch Large (4x4)")
    axs[4].imshow(recon_small, cmap="gray"); axs[4].set_title("Recon Small")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Label: {label} - Sample {idx}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def is_quantum_like_model(name_or_cls):
    nm = name_or_cls if isinstance(name_or_cls, str) else getattr(name_or_cls, "__name__", "").lower()
    nm = nm.lower()
    return any(token in nm for token in ("quantum", "variational", "hybrid", "qkernel", "vfq"))

# sensible defaults per model to keep runs short and stable
MODEL_INIT_OVERRIDES = {
    "ClassicalSVM": {"kernel": "linear", "C": 1.0, "probability": True},
    "QuantumKernelSVM": {"n_qubits": None, "feature_map_layers": 1, "backend": "default.qubit", "svc_kwargs": {"kernel": "precomputed", "C": 1.0}},
    "VariationalFullyQuantum": {"n_qubits": None, "n_layers": 1, "device_type": "default.qubit", "lr": 0.1, "epochs": 5},
    "HybridModel": {"n_qubits": None, "backend": "default.qubit"},
    "HybridQuantumKernel": {"n_qubits": None, "device_type": "default.qubit", "C": 1.0},
    "FullyQuantumSVM": {"n_qubits": None, "C": 1.0, "device_type": "default.qubit"},
    "VariationalQuantumSVM_V6Flex": {"n_qubits": None, "n_layers": 1, "device_type": "default.qubit", "lr": 0.05, "use_rx": False, "entangler": "line"},
}

def remap_binary_if_needed(y_true, y_pred):
    yt = set(np.unique(y_true).tolist())
    yp = set(np.unique(y_pred).tolist())
    if yt <= {0, 1} and yp <= {-1, 1}:
        y_pred = ((y_pred + 1) // 2).astype(int)
    elif yt <= {-1, 1} and yp <= {0, 1}:
        y_pred = (2 * y_pred - 1).astype(int)
    return y_true, y_pred

def main():
    parser = argparse.ArgumentParser(description="QSVM patch dataset test flow")
    parser.add_argument("--outdir", type=str, default="./examples/outputs/patches", help="Diretório de saída")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Classes a usar (ex: 1 8)")
    parser.add_argument("--n_per_class", type=int, default=2, help="Amostras por classe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Subset rápido (duas classes, duas amostras)")
    parser.add_argument("--model", type=str, default=None, help="Modelo específico a rodar")
    parser.add_argument("--show_flow_samples", type=int, default=2, help="Quantas amostras salvar o flow por modelo")
    args = parser.parse_args()

    # Quick = garante 2 classes e 2 amostras
    if args.quick:
        args.n_per_class = 2
        args.classes = None  # aleatório escolhido pelo loader

    # Carrega dataset (troque pelo loader real)
    X_small, X_large, y = load_patches_dataset(classes=args.classes, n_per_class=args.n_per_class, seed=args.seed)
    if len(X_small) == 0:
        print("Dataset vazio. Cheque o loader.")
        return

    # Flatten small images para alimentar modelos clássicos -> (N, Hs*Ws)
    X_flat = X_small.reshape(len(X_small), -1)
    y = np.array(y)

    # Saída
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(args.outdir, timestamp)
    os.makedirs(out_base, exist_ok=True)

    classes_mod = discover_model_classes(qmodels)
    to_run = [args.model] if args.model else list(classes_mod.keys())
    print(f"Rodando modelos: {to_run}")

    results = []
    for name in to_run:
        print(f"\n==> Testando modelo: {name}")
        cls = classes_mod.get(name)
        if cls is None:
            print(f"[WARN] Modelo {name} não encontrado em qsvm.models, pulando.")
            continue
        model_out = os.path.join(out_base, name)
        os.makedirs(model_out, exist_ok=True)

        # Decide features to feed model: reduce for quantum-like models
        X_for_model = X_flat.copy()
        pca_obj = None
        if is_quantum_like_model(name):
            # choose small number of components safe for pennylane/backends
            max_q = min(4, X_flat.shape[1])
            if max_q < 1:
                max_q = 1
            try:
                pca_obj = PCA(n_components=max_q, random_state=args.seed)
                X_for_model = pca_obj.fit_transform(X_flat)
                print(f"  -> {name}: usando PCA -> {max_q} componentes para modelos quânticos.")
            except Exception as e:
                print(f"  -> Aviso PCA falhou para {name}: {e}; usando features originais (cuidado com tamanho).")

        # Instantiate with safe overrides
        overrides = dict(MODEL_INIT_OVERRIDES.get(name, {}))
        # If override n_qubits is None, set to features count used
        if "n_qubits" in overrides and overrides["n_qubits"] is None:
            overrides["n_qubits"] = max(1, min(6, X_for_model.shape[1]))  # cap to avoid huge circuits

        try:
            # Try instantiating with overrides, fallback to default constructor
            try:
                model = cls(**overrides)
            except TypeError:
                model = cls()
        except Exception as e:
            print(f"[ERROR] Falha ao instanciar {name}: {e}")
            results.append({"model": name, "status": "fail_instantiate", "accuracy": None, "error": str(e)})
            continue

        t0 = time.time()
        try:
            # Fit/predict on small dataset (quick). Use X_for_model for training.
            model.fit(X_for_model, y)
            y_pred = model.predict(X_for_model)
            # If we used PCA, predictions map to same labels (no inverse transform needed)
            y_true, y_pred = remap_binary_if_needed(y, np.asarray(y_pred))

            # Warning about missing predicted classes
            unique_pred = np.unique(y_pred)
            unique_true = np.unique(y_true)
            missing_pred = set(unique_true) - set(unique_pred)
            if missing_pred:
                print(f"[WARN] {name}: predicted classes {sorted(unique_pred)} but true classes {sorted(unique_true)}; missing predictions for classes {sorted(missing_pred)}")

            acc = float(accuracy_score(y_true, y_pred))
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, zero_division=0)

            with open(os.path.join(model_out, "report.txt"), "w", encoding="utf-8") as f:
                f.write(f"Model: {name}\n")
                f.write(f"Accuracy: {acc:.6f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + "\n\n")
                f.write(report)
            np.savetxt(os.path.join(model_out, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
            np.save(os.path.join(model_out, "confusion_matrix.npy"), cm)

            # Save predictions.csv (features = flattened small)
            header = ["row", "y_true", "y_pred"] + [f"f{i}" for i in range(X_flat.shape[1])]
            rows = np.column_stack([np.arange(len(y_true)), y_true, y_pred, X_flat])
            np.savetxt(os.path.join(model_out, "predictions.csv"), rows, delimiter=",", fmt="%.10g", header=",".join(header), comments="")

            # Samples correct / wrong
            correct_idx = np.where(y_true == y_pred)[0]
            wrong_idx = np.where(y_true != y_pred)[0]
            def save_subset(idxs, fname):
                k = min(2, len(idxs))
                if k == 0: return
                sel = idxs[:k]
                sub = np.column_stack([sel, y_true[sel], y_pred[sel], X_flat[sel]])
                np.savetxt(os.path.join(model_out, fname), sub, delimiter=",", fmt="%.10g",
                           header="row,y_true,y_pred," + ",".join([f"f{i}" for i in range(X_flat.shape[1])]), comments="")
            save_subset(correct_idx, "examples_correct_sample.csv")
            save_subset(wrong_idx, "examples_wrong_sample.csv")

            # Flow images: save first N flows
            n_flow = min(args.show_flow_samples, len(X_small))
            for i in range(n_flow):
                flow_path = os.path.join(model_out, f"flow_sample_{i}.png")
                plot_flow(X_small, X_large, y, i, flow_path)

            results.append({"model": name, "status": "ok", "accuracy": acc, "error": None, "time_s": time.time()-t0})
            print(f"  -> {name} status=ok acc={acc:.4f} time={time.time()-t0:.2f}s")
        except Exception as e:
            print(f"[ERROR] Runtime for {name}: {e}")
            results.append({"model": name, "status": "fail_runtime", "accuracy": None, "error": str(e), "time_s": time.time()-t0})
            continue

    # CSV/JSON summary and accuracy plot
    csv_path = os.path.join(out_base, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write("model,status,accuracy,time_s,error\n")
        for r in results:
            cf.write(f"{r['model']},{r['status']},{r.get('accuracy','')},{r.get('time_s','')},{r.get('error','')}\n")
    with open(os.path.join(out_base, "summary.json"), "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    labels = [r["model"] for r in results]
    accs = [r["accuracy"] if r["status"] == "ok" and r["accuracy"] is not None else 0.0 for r in results]
    colors = ["tab:blue" if r["status"] == "ok" and r["accuracy"] is not None else "tab:red" for r in results]
    plt.figure(figsize=(max(6, len(labels)*0.8), 4))
    x = np.arange(len(labels))
    plt.bar(x, accs, color=colors)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy (or 0 if skipped/failed)")
    plt.title("qsvm.models PATCHES test results")
    plt.tight_layout()
    plot_path = os.path.join(out_base, "models_accuracy.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nResumo salvo em: {out_base}")
    print(f" - CSV: {csv_path}")
    print(f" - Plot: {plot_path}")

if __name__ == "__main__":
    main()