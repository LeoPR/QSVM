#!/usr/bin/env python3
"""
MNIST patches showcase integrado com patchkit.

Grid: 6 linhas (condições) x 7 colunas:
Colunas:
[Original 28x28, Original filtrado 28x28, Reduced 14x14 (exib. 28) based on filtered,
 Patch 4x4 (orig) up (from filtered), Patch 2x2 (red) up (from filtered),
 Reconstr. 4x4/2 -> 28x28 (from filtered), Reconstr. 2x2/1 -> 14x14 (exib. 28) (from filtered)]

idx_rel_base é calculado apenas a partir da primeira condição (argmax dos scores sobre patches_14 da
versão filtrada) e é fixo para todas as linhas.

Observação: a coluna "Original 28x28" é apenas referência visual; as extrações e reconstruções vêm
da "Original filtrado 28x28" (coluna 2) e do resized dessa versão.
"""
import os
import io
import json
import math
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

from patchkit import image_utils as iu
from patchkit.image_metrics import ImageMetrics

# importa config centralizado (apenas alteração pontual)
from examples.patchs.config import OUTPUTS_ROOT
# utilitário compartilhado: normalize_patches_from_extractor
from examples.patchs.utils import normalize_patches_from_extractor

# Preferir extractor e filtro do patchkit
try:
    from patchkit.patches import OptimizedPatchExtractor, filter_active_patches
    _HAS_EXTRACTOR = True
except Exception:
    OptimizedPatchExtractor = None
    _HAS_EXTRACTOR = False
    from patchkit.patches import filter_active_patches  # type: ignore


# -------------------------
# Helpers
# -------------------------

def extract_patches_tensor(image: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """Extrai patches a partir de tensor [C,H,W] usando unfold. Retorna [N,C,ps,ps] float [0,1]."""
    if image.ndim != 3:
        raise ValueError("Esperado tensor 3D (C,H,W)")
    C, H, W = image.shape
    if patch_size > H or patch_size > W:
        raise ValueError("Patch maior que imagem")
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    n_rows, n_cols = patches.shape[1], patches.shape[2]
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(n_rows * n_cols, C, patch_size, patch_size)
    return patches


def combine_patches_tensor(patches: torch.Tensor, original_size: Tuple[int, int], patch_size: int, stride: int,
                           normalize: bool = True) -> torch.Tensor:
    """Recombina patches [N,C,ps,ps] -> [C,H,W] via F.fold. Espera patches float [0,1]."""
    if patches.ndim != 4:
        raise ValueError("Esperado patches [N,C,ps,ps]")
    N, C, ps, _ = patches.shape
    H, W = original_size
    n_rows = math.floor((H - patch_size) / stride) + 1
    n_cols = math.floor((W - patch_size) / stride) + 1
    if n_rows * n_cols != N:
        raise ValueError("Dimensões de patches não batem com tamanho original")
    x = patches.permute(1, 2, 3, 0).reshape(1, C * patch_size * patch_size, N).contiguous()
    out = F.fold(x, output_size=(H, W), kernel_size=patch_size, stride=stride)
    if normalize and stride < patch_size:
        ones = torch.ones_like(x)
        norm = F.fold(ones, output_size=(H, W), kernel_size=patch_size, stride=stride)
        out = out / norm.clamp_min(1e-8)
    return out.squeeze(0)


def to_numpy(img_t: torch.Tensor) -> np.ndarray:
    """Converte tensor [C,H,W] (C=1) para numpy [H,W] float [0,1]."""
    if img_t.ndim == 3 and img_t.shape[0] == 1:
        arr = img_t[0].detach().cpu().numpy()
    elif img_t.ndim == 2:
        arr = img_t.detach().cpu().numpy()
    else:
        raise ValueError("Apenas imagens 1-canal suportadas neste exemplo")
    return np.clip(arr.astype(np.float64), 0.0, 1.0)


def save_combined_grid(path: str, grid_tiles: List[List[np.ndarray]], row_titles: List[str], col_titles: List[str],
                       tile_size: Optional[Tuple[int, int]] = None, cmap: str = "viridis"):
    """Salva grade m x n de tiles. Cada tile é [H,W] float [0,1]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = len(grid_tiles)
    n = len(grid_tiles[0]) if m > 0 else 0
    fig, axes = plt.subplots(m, n, figsize=(n * 2.2, m * 2.4), constrained_layout=True)
    # garantir formato de axes consistente
    if m == 1:
        axes = [axes]
    for i in range(m):
        for j in range(n):
            ax = axes[i][j] if m > 1 else axes[j]
            tile = grid_tiles[i][j]
            tile = np.clip(tile.astype(np.float32), 0.0, 1.0)
            if tile_size is not None:
                pil = iu.to_pil(tile, mode='L')
                pil = pil.resize((int(tile_size[0]), int(tile_size[1])), resample=iu.Image.NEAREST)
                arr = np.asarray(pil).astype(np.float32) / 255.0
                ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                ax.imshow(tile, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.axis("off")
            if i == 0 and col_titles and j < len(col_titles):
                ax.set_title(col_titles[j], fontsize=9)
        # título da linha como ylabel na primeira coluna
        if row_titles and i < len(row_titles):
            ax0 = axes[i][0] if m > 1 else axes[0]
            ax0.set_ylabel(row_titles[i], fontsize=9, rotation=0, labelpad=40, va='center')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def overlay_patch_border(tile: np.ndarray, top: int, left: int, ph: int, pw: int, thickness: int = 1,
                        value: float = 1.0) -> np.ndarray:
    """Desenha borda em tile [H,W] float [0,1]."""
    out = tile.copy()
    H, W = out.shape
    for t in range(thickness):
        r0 = max(0, top - t)
        r1 = min(H, top + ph + t)
        c0 = max(0, left - t)
        c1 = min(W, left + pw + t)
        if r0 < H and c0 < c1:
            out[r0, c0:c1] = value
        if r1 - 1 >= 0 and c0 < c1:
            out[r1 - 1, c0:c1] = value
        if c0 < W and r0 < r1:
            out[r0:r1, c0] = value
        if c1 - 1 >= 0 and r0 < r1:
            out[r0:r1, c1 - 1] = value
    return out


def compress_pil_to_jpeg(pil_img: Image.Image, quality: int) -> Image.Image:
    """Compress PIL image to JPEG quality in-memory and return reopened PIL image (L mode)."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    img = Image.open(buf).convert('L')
    return img


# -------------------------
# Pipeline principal
# -------------------------

def main(classes=(1, 8), samples_per_class=2, out_root=OUTPUTS_ROOT):
    device = torch.device("cpu")

    conditions = [
        {"name": "NEAREST", "resize_alg": iu.Image.NEAREST, "jpeg": None},
        {"name": "BILINEAR", "resize_alg": iu.Image.BILINEAR, "jpeg": None},
        {"name": "BICUBIC", "resize_alg": iu.Image.BICUBIC, "jpeg": None},
        {"name": "LANCZOS", "resize_alg": iu.Image.LANCZOS, "jpeg": None},
        {"name": "BILINEAR + JPEG90", "resize_alg": iu.Image.BILINEAR, "jpeg": 90},
        {"name": "BILINEAR + JPEG50", "resize_alg": iu.Image.BILINEAR, "jpeg": 50},
    ]

    # dataset MNIST
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())

    # selecionar índices por classe
    targets = ds.targets.numpy()
    selected: Dict[int, list] = {int(c): [] for c in classes}
    for idx in range(len(ds)):
        y = int(targets[idx])
        if y in selected and len(selected[y]) < samples_per_class:
            selected[y].append(idx)
        if all(len(v) >= samples_per_class for v in selected.values()):
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"mnist_showcase_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"classes": list(map(int, classes)), "samples_per_class": int(samples_per_class), "samples": []}

    col_titles = (
        "Original 28x28",
        "Original filtrado 28x28",
        "Reduced 14x14 (exib. 28) (from filtered)",
        "Patch 4x4 (orig) up",
        "Patch 2x2 (red) up",
        "Reconstr. 4x4/2 -> 28x28",
        "Reconstr. 2x2/1 -> 14x14 (exib. 28)"
    )

    # itera amostras
    for cls in classes:
        for ds_idx in selected[int(cls)]:
            img_t, label = ds[ds_idx]  # img_t: [1,28,28]
            assert int(label) == int(cls)
            img_t = img_t.to(device)

            pil_orig = iu.to_pil(img_t)  # usado apenas para exibição (col 1)

            # grid de tiles e titles
            grid_tiles: List[List[np.ndarray]] = []
            row_titles: List[str] = []

            idx_rel_base: Optional[int] = None

            # processar condições (cada linha será uma condição)
            for ci, cond in enumerate(conditions):
                row_titles.append(cond["name"])

                # criar pil_filtered: nesta versão o "filtro" é simulação de artefato (JPEG)
                if cond["jpeg"] is not None:
                    pil_filtered = compress_pil_to_jpeg(pil_orig, cond["jpeg"])
                else:
                    # sem artefato -> cópia idêntica
                    pil_filtered = pil_orig.copy()

                # converter filtered para tensor float [0,1]
                filtered_t = iu.to_tensor(pil_filtered)
                if not torch.is_floating_point(filtered_t):
                    filtered_t = filtered_t.to(torch.float32) / 255.0

                # Extrair patches_28 a partir da imagem filtrada (fonte das extrações e reconstr.)
                patches_28_from_filtered = None
                try:
                    if _HAS_EXTRACTOR:
                        extractor_28 = OptimizedPatchExtractor(patch_size=(4, 4), stride=2,
                                                               cache_dir=os.path.join(out_dir, "cache"),
                                                               image_size=(28, 28))
                        # Forçar processamento a partir da imagem atual (sem index-based cache)
                        raw28 = extractor_28.process(pil_filtered, index=None)
                        patches_28_from_filtered = normalize_patches_from_extractor(raw28)  # [N,1,4,4]
                except Exception:
                    patches_28_from_filtered = None
                if patches_28_from_filtered is None:
                    patches_28_from_filtered = extract_patches_tensor(filtered_t, patch_size=4, stride=2)

                # DEBUG: informar de onde vieram os patches (devem vir da imagem filtrada)
                print(f"[DEBUG] Condição '{cond['name']}' - patches_28_from_filtered: dtype={patches_28_from_filtered.dtype}, shape={patches_28_from_filtered.shape}")

                assert patches_28_from_filtered.shape[0] == 169

                # Resize a partir da imagem filtrada (algoritmo da condição)
                try:
                    resized_t = iu.resize(filtered_t, (14, 14), alg=cond["resize_alg"], backend="pil", return_type="tensor")
                except Exception:
                    resized_t = TF.resize(filtered_t, [14, 14])
                if not torch.is_floating_point(resized_t):
                    resized_t = resized_t.to(torch.float32) / 255.0

                # Extrair patches_14 a partir do resized da versão filtrada
                patches_14 = None
                try:
                    if _HAS_EXTRACTOR:
                        extractor_14 = OptimizedPatchExtractor(patch_size=(2, 2), stride=1,
                                                               cache_dir=os.path.join(out_dir, "cache"),
                                                               image_size=(14, 14))
                        # Forçar processamento a partir do resized atual (sem index-based cache)
                        raw14 = extractor_14.process(iu.to_pil(resized_t), index=None)
                        patches_14 = normalize_patches_from_extractor(raw14)
                except Exception:
                    patches_14 = None
                if patches_14 is None:
                    patches_14 = extract_patches_tensor(resized_t, patch_size=2, stride=1)

                # DEBUG: confirmar origem dos patches_14
                print(f"[DEBUG] Condição '{cond['name']}' - patches_14: dtype={patches_14.dtype}, shape={patches_14.shape}")

                assert patches_14.shape[0] == 169

                # obter scores/indices via filter_active_patches (retorna indices_ordenados, scores)
                try:
                    indices_active, scores = filter_active_patches(patches_14, min_mean=0.1, max_mean=0.9)
                except Exception:
                    # fallback simples: usar variância como score
                    pf = patches_14.float()
                    if pf.max() > 1.0:
                        pf = pf / 255.0
                    pf_cpu = pf.detach().cpu()
                    L = pf_cpu.shape[0]
                    scores = torch.zeros(L)
                    for i in range(L):
                        p_s = pf_cpu[i].mean(dim=0) if pf_cpu[i].dim() == 3 else pf_cpu[i]
                        scores[i] = float(torch.var(p_s).item())
                    indices_active = torch.argsort(scores, descending=True)

                # definir idx_rel_base na primeira condição (argmax dos scores)
                if ci == 0:
                    if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                        idx_rel_base = int(torch.argmax(scores).item())
                    else:
                        idx_rel_base = (13 // 2) * 13 + (13 // 2)

                # usar idx_rel_base fixo para todas as linhas
                idx_rel = int(idx_rel_base)

                # calcular posição do patch na grade 13x13
                row_idx = idx_rel // 13
                col_idx = idx_rel % 13

                # coordenadas para original 28x28 (patch 4x4, stride 2)
                orig_top = int(row_idx * 2)
                orig_left = int(col_idx * 2)
                orig_ph = 4
                orig_pw = 4

                # coordenadas para reduced 14x14 exibido em 28x28 (upsample x2)
                red_top = int(row_idx * 1 * 2)
                red_left = int(col_idx * 1 * 2)
                red_ph = 2 * 2
                red_pw = 2 * 2

                # preparar tiles:
                orig_tile = to_numpy(img_t)  # original sem filtro (col 1)
                filtered_tile = np.asarray(pil_filtered.resize((28, 28), resample=iu.Image.NEAREST)).astype(np.float32) / 255.0  # col 2

                # extrair patch_orig (da versão filtrada) e patch_red (da versão filtrada/resized)
                patch_orig = patches_28_from_filtered[idx_rel, 0]
                patch_red = patches_14[idx_rel, 0]

                # reconstruções baseadas em patches extraídos da versão filtrada
                recon_28 = combine_patches_tensor(patches_28_from_filtered, original_size=(28, 28), patch_size=4, stride=2)
                recon_14 = combine_patches_tensor(patches_14, original_size=(14, 14), patch_size=2, stride=1)

                # tiles dos patches e recon (todos float [0,1])
                reduced_ups = np.asarray(iu.to_pil(resized_t).resize((28, 28), resample=iu.Image.NEAREST)).astype(np.float32) / 255.0
                patch_orig_up = np.asarray(iu.to_pil(patch_orig.detach().cpu().numpy()).resize((28, 28), resample=iu.Image.NEAREST)).astype(np.float32) / 255.0
                patch_red_up = np.asarray(iu.to_pil(patch_red.detach().cpu().numpy()).resize((28, 28), resample=iu.Image.NEAREST)).astype(np.float32) / 255.0
                recon28_tile = to_numpy(recon_28)
                recon14_up = np.asarray(iu.to_pil(recon_14).resize((28, 28), resample=iu.Image.NEAREST)).astype(np.float32) / 255.0

                # overlay: desenhar borda nas duas primeiras colunas (orig e filtered)
                orig_tile_ov = overlay_patch_border(orig_tile, orig_top, orig_left, orig_ph, orig_pw, thickness=1, value=1.0)
                filtered_tile_ov = overlay_patch_border(filtered_tile, orig_top, orig_left, orig_ph, orig_pw, thickness=1, value=1.0)

                row_tiles = [
                    orig_tile_ov,
                    filtered_tile_ov,
                    reduced_ups,
                    patch_orig_up,
                    patch_red_up,
                    recon28_tile,
                    recon14_up,
                ]
                grid_tiles.append(row_tiles)

            # salvar grade 6x7 por amostra
            sample_dir = os.path.join(out_dir, f"class_{int(cls)}_idx_{int(ds_idx)}")
            os.makedirs(sample_dir, exist_ok=True)
            combined_path = os.path.join(sample_dir, "combined_6x7_conditions_filtered_base.png")
            save_combined_grid(combined_path, grid_tiles, row_titles, list(col_titles), tile_size=(28, 28), cmap="viridis")

            # salvar metadata/summary (registro simples)
            sample_rec = {
                "class": int(cls),
                "dataset_index": int(ds_idx),
                "idx_rel_base": int(idx_rel_base) if idx_rel_base is not None else None,
                "patch_grid_rc": [int(idx_rel_base // 13) if idx_rel_base is not None else None,
                                   int(idx_rel_base % 13) if idx_rel_base is not None else None],
                "conditions": [c["name"] for c in conditions],
                "combined_png": os.path.relpath(combined_path, out_dir),
                "note": "col1 original sem filtro; col2 original filtrado. resto baseado em original filtrado"
            }
            summary["samples"].append(sample_rec)

            with open(os.path.join(sample_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(sample_rec, f, indent=2, ensure_ascii=False)

            print(f"[OK] Classe {cls} idx {ds_idx} -> combined grid salvo em {combined_path}")

    # salva resumo geral
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaída salva em: {out_dir}")


if __name__ == "__main__":
    main()