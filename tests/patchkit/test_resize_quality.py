import pytest
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from patchkit import ProcessedDataset

def has_skimage():
    try:
        from skimage.metrics import structural_similarity, peak_signal_noise_ratio  # noqa: F401
        return True
    except Exception:
        return False

@pytest.mark.skipif(not has_skimage(), reason="scikit-image is required for SSIM/PSNR metrics")
def test_resize_quality_comparison(tmp_path, tiny_synthetic):
    """
    Downsample (28->14) with different PIL algorithms, upsample back to 28 and
    compare reconstruction quality against original using SSIM and MSE.
    Relaxed tolerances because images are small (28x28) and algorithms produce
    very similar numeric results at that scale.
    """
    from skimage.metrics import structural_similarity  # local import guarded by skip
    algs = [
        ("NEAREST", Image.NEAREST),
        ("BILINEAR", Image.BILINEAR),
        ("BICUBIC", Image.BICUBIC),
        ("LANCZOS", Image.LANCZOS),
    ]

    # use a small batch (3) to reduce flakiness by averaging
    ds = tiny_synthetic(n=3, size=(28, 28), pattern="gradient")
    to_pil = transforms.ToPILImage()

    results = {}
    for name, alg in algs:
        processed = ProcessedDataset(
            ds,
            target_size=(14, 14),
            resize_alg=alg,
            cache_dir=str(tmp_path / f"cache_{name}"),
            cache_rebuild=True
        )

        ssim_list = []
        mse_list = []
        for i in range(len(ds)):
            # original
            orig_tensor, _ = ds[i]
            orig = orig_tensor.squeeze().numpy()
            # processed.data shape: (N, C, H, W) and order is same as ds
            out = processed.data[i]
            pil = to_pil(out)
            up = pil.resize((orig.shape[1], orig.shape[0]), resample=Image.BICUBIC)
            recon = np.array(up).astype(np.float32)
            if recon.max() > 1.0:
                recon = recon / 255.0
            if recon.ndim == 3:
                recon = recon[..., 0]
            ssim = structural_similarity(orig, recon, data_range=orig.max() - orig.min())
            mse = float(((orig - recon) ** 2).mean())
            ssim_list.append(float(ssim))
            mse_list.append(mse)

        results[name] = {"ssim_mean": float(np.mean(ssim_list)), "mse_mean": float(np.mean(mse_list))}

    # Basic sanity checks
    assert set(results.keys()) == {a[0] for a in algs}

    # Tolerance for tiny images
    TOL = 1e-4

    # NEAREST should be among the worst; LANCZOS among the best (allow tiny tolerance)
    s_nearest = results["NEAREST"]["ssim_mean"]
    s_lanczos = results["LANCZOS"]["ssim_mean"]
    assert s_lanczos + TOL >= s_nearest, f"LANCZOS SSIM ({s_lanczos:.6f}) should be >= NEAREST SSIM ({s_nearest:.6f}) within tol={TOL}"

    # Expect non-decreasing-ish order, allow small decreases up to TOL
    order = ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"]
    prev = -1.0
    for k in order:
        cur = results[k]["ssim_mean"]
        assert cur + TOL >= prev, (
            f"SSIM order violated at {k}: cur={cur:.6f}, prev={prev:.6f}, tol={TOL}. Full results: {results}"
        )
        prev = cur