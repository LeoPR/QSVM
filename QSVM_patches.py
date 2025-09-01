#!/usr/bin/env python3
"""
QSVM_patches.py

Similar flow to QSVM_iris but using MNIST and the local patchkit to produce patches.
- Loads MNIST (torchvision)
- Uses patchkit.SuperResPatchDataset + OptimizedPatchExtractor and select_informative_patch
  to extract paired (small_patch, large_patch) for selected images/classes.
- Builds feature vectors from small patches (flatten or optionally PCA-reduced)
- Trains a classical SVM (and reports accuracy) using the same overall "exercise" as QSVM_iris.

Usage examples:
  # quick random run (picks random images and best patch per image)
  python QSVM_patches.py --quick --n_samples 40

  # select specific classes and samples per class (deterministic with seed)
  python QSVM_patches.py --classes 1 8 --n_per_class 10 --seed 0

Notes:
- Requires the repository's patchkit package (the script imports `patchkit`).
- ProcessedDataset caches processed images (zstd) in cache_dir; first run may take time.
"""

import argparse
import os
import sys
import numpy as np
import torch
from torchvision import transforms, datasets

# patchkit imports (from the repository folder)
try:
    from patchkit import SuperResPatchDataset, ProcessedDataset, PatchExtractor, select_informative_patch
except Exception as e:
    print("Erro ao importar patchkit. Certifique-se de que o diretório do repositório está no PYTHONPATH.")
    print("Import error:", e)
    raise

# sklearn for training classical models
try:
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except Exception as e:
    print("Falha ao importar scikit-learn. Instale scikit-learn para prosseguir.")
    raise

from PIL import Image

def build_superres_dataset(mnist_root, train, cache_dir,
                           low_size, high_size,
                           small_patch_size, large_patch_size,
                           stride, scale_factor,
                           classes=None, n_per_class=2, n_samples=None,
                           quick=False, seed=42, max_prefetch=128):
    """
    Build a dataset of paired patches using patchkit.SuperResPatchDataset.
    Returns:
      X_small: numpy array (N, Hs, Ws) uint8
      X_large: numpy array (N, Hl, Wl) uint8
      y:       numpy array (N,) int labels

    Behavior:
      - quick=True: sample n_samples random images and pick one informative patch per image.
      - quick=False: select images by classes (classes list) and n_per_class each.
    """
    rng = np.random.RandomState(seed)

    # Load MNIST original dataset (PIL images)
    original_ds = datasets.MNIST(root=mnist_root, train=train, download=True)

    # Create SuperResPatchDataset which internally creates processed low/high datasets and extractors
    low_conf = {'target_size': tuple(low_size)}
    high_conf = {'target_size': tuple(high_size)}

    ds = SuperResPatchDataset(
        original_ds,
        low_res_config=low_conf,
        high_res_config=high_conf,
        small_patch_size=tuple(small_patch_size),
        large_patch_size=tuple(large_patch_size),
        stride=stride,
        scale_factor=scale_factor,
        cache_dir=cache_dir,
        cache_rebuild=False,
        max_memory_cache=256
    )

    # helper: convert processed tensor (0..1 float) to uint8 HxW numpy
    def tensor_to_uint8_img(tensor):
        t = tensor.detach().cpu()
        # expected [C,H,W] or [H,W]
        if t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() == 3 and t.shape[0] > 1:
            # combine channels by mean
            t = t.mean(dim=0)
        arr = (t.clamp(0, 1).mul(255.0).round().to(torch.uint8).numpy())
        return arr

    # Determine which image indices to use
    num_images = len(ds.low_res_ds)  # same as original_ds
    labels_all = ds.low_res_ds.labels.numpy()  # tensor of labels

    selected_img_indices = []

    if quick:
        if n_samples is None:
            n_samples = 32
        # sample random image indices
        selected_img_indices = list(rng.choice(np.arange(num_images), size=min(n_samples, num_images), replace=False))
    else:
        # use classes + n_per_class
        if classes is None or len(classes) == 0:
            raise ValueError("Quando --quick não for usado, indique --classes X Y ...")
        for cl in classes:
            idxs = np.where(labels_all == cl)[0]
            if len(idxs) == 0:
                raise ValueError(f"Classe {cl} não encontrada no dataset.")
            k = min(n_per_class, len(idxs))
            picks = rng.choice(idxs, size=k, replace=False)
            selected_img_indices.extend(picks.tolist())

    selected_img_indices = sorted(selected_img_indices)

    X_small = []
    X_large = []
    y_list = []

    to_pil = transforms.ToPILImage()

    # If many images, optionally prefetch patches for speed
    for img_idx in selected_img_indices:
        # get low/high processed tensors and convert to PIL for patch extractor methods
        low_tensor = ds.low_res_ds.data[img_idx]
        high_tensor = ds.high_res_ds.data[img_idx]

        low_pil = to_pil(low_tensor.squeeze())
        high_pil = to_pil(high_tensor.squeeze())

        # get all small patches (tensor uint8 [L, H, W] or [L,C,H,W])
        small_patches = ds.small_patch_extractor.process(low_pil, img_idx)
        large_patches = ds.large_patch_extractor.process(high_pil, img_idx)

        # choose best patch index using select_informative_patch
        best_idx, scores = select_informative_patch(small_patches, num_candidates=5)
        # Best patch is small_patches[best_idx], large_patches[best_idx] correspondingly

        small_patch = small_patches[best_idx]
        large_patch = large_patches[best_idx]

        # convert patches (torch uint8) to numpy HxW uint8
        # small_patch may be [H,W] or [C,H,W]
        if isinstance(small_patch, torch.Tensor):
            sp = small_patch
            if sp.dim() == 3 and sp.shape[0] == 1:
                sp = sp.squeeze(0)
            if sp.dim() == 3 and sp.shape[0] > 1:
                sp = sp.mean(dim=0)
            sp_np = sp.cpu().numpy().astype(np.uint8)
        else:
            # fallback if patch is PIL
            sp_np = np.array(small_patch).astype(np.uint8)
            if sp_np.ndim == 3:
                sp_np = sp_np.mean(axis=2).astype(np.uint8)

        if isinstance(large_patch, torch.Tensor):
            lp = large_patch
            if lp.dim() == 3 and lp.shape[0] == 1:
                lp = lp.squeeze(0)
            if lp.dim() == 3 and lp.shape[0] > 1:
                lp = lp.mean(dim=0)
            lp_np = lp.cpu().numpy().astype(np.uint8)
        else:
            lp_np = np.array(large_patch).astype(np.uint8)
            if lp_np.ndim == 3:
                lp_np = lp_np.mean(axis=2).astype(np.uint8)

        X_small.append(sp_np)
        X_large.append(lp_np)
        y_list.append(int(labels_all[img_idx]))

    X_small = np.stack(X_small, axis=0)
    X_large = np.stack(X_large, axis=0)
    y = np.array(y_list, dtype=int)

    return X_small, X_large, y


def flatten_patches(X):
    # X: (N, H, W) uint8 -> (N, H*W) float64
    N, H, W = X.shape
    return X.reshape(N, H * W).astype(np.float32)


def train_and_evaluate(X, y, test_size=0.3, pca_components=None, random_state=0):
    """
    Simple classical pipeline:
      - split
      - scale
      - optionally PCA (for quantum alignment or dimensionality reduction)
      - train SVM (RBF) and report metrics
    """
    Xf = flatten_patches(X)
    X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=test_size, stratify=y, random_state=random_state)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)
        print(f"PCA applied: reduced to {pca_components} components")

    clf = SVC(kernel='rbf', gamma='scale', C=1.0, probability=False, random_state=random_state)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return clf, scaler


def parse_args():
    parser = argparse.ArgumentParser(description="QSVM_patches - MNIST patches pipeline")
    parser.add_argument("--mnist-root", type=str, default="./data", help="MNIST root directory")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="cache base directory used by patchkit")
    parser.add_argument("--quick", action="store_true", help="Quick random sampling flow (no class selection)")
    parser.add_argument("--n-samples", type=int, default=32, help="when --quick, number of images to sample")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="classes to select (when not --quick)")
    parser.add_argument("--n-per-class", type=int, default=5, help="samples per class when not --quick")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--low-size", type=int, nargs=2, default=[8, 8], help="low res target size H W")
    parser.add_argument("--high-size", type=int, nargs=2, default=[32, 32], help="high res target size H W")
    parser.add_argument("--small-patch", type=int, nargs=2, default=[4, 4], help="small patch H W")
    parser.add_argument("--large-patch", type=int, nargs=2, default=[16, 16], help="large patch H W (should be small*scale)")
    parser.add_argument("--stride", type=int, default=4, help="patch stride on small image")
    parser.add_argument("--scale-factor", type=int, default=4, help="scale factor between small and large patches")
    parser.add_argument("--pca", type=int, default=None, help="apply PCA to reduce dimensionality (optional)")
    parser.add_argument("--test-size", type=float, default=0.3, help="test fraction")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Construindo dataset de patches usando patchkit (pode demorar na primeira execução, cache será criado).")
    if args.quick:
        Xs, Xl, y = build_superres_dataset(
            mnist_root=args.mnist_root,
            train=True,
            cache_dir=args.cache_dir,
            low_size=args.low_size,
            high_size=args.high_size,
            small_patch_size=args.small_patch,
            large_patch_size=args.large_patch,
            stride=args.stride,
            scale_factor=args.scale_factor,
            quick=True,
            n_samples=args.n_samples,
            seed=args.seed
        )
    else:
        Xs, Xl, y = build_superres_dataset(
            mnist_root=args.mnist_root,
            train=True,
            cache_dir=args.cache_dir,
            low_size=args.low_size,
            high_size=args.high_size,
            small_patch_size=args.small_patch,
            large_patch_size=args.large_patch,
            stride=args.stride,
            scale_factor=args.scale_factor,
            quick=False,
            classes=args.classes,
            n_per_class=args.n_per_class,
            seed=args.seed
        )

    print(f"Dataset criado: {Xs.shape[0]} samples; small patch shape {Xs.shape[1:]} ; large patch shape {Xl.shape[1:]}")
    print("Treinando classificador clássico (SVM RBF) usando patches pequenos (flatten).")

    clf, scaler = train_and_evaluate(Xs, y, test_size=args.test_size, pca_components=args.pca, random_state=args.seed)

    print("Pronto. Você pode usar 'clf' e 'scaler' salvos para previsões ou adaptar para o pipeline QSVM/quantum.")
    # Optionally save model artifacts
    try:
        import joblib
        out_dir = "./models_patches"
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump({'clf': clf, 'scaler': scaler}, os.path.join(out_dir, "svm_patches.joblib"))
        print(f"Modelo salvo em {out_dir}/svm_patches.joblib")
    except Exception:
        print("joblib não disponível; pulando salvamento do modelo.")

if __name__ == "__main__":
    main()