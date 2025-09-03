import numpy as np

def has_skimage():
    try:
        from skimage.metrics import structural_similarity, peak_signal_noise_ratio  # noqa: F401
        return True
    except Exception:
        return False

def compute_ssim(a: np.ndarray, b: np.ndarray):
    """
    Wrapper for SSIM. Inputs: float arrays in [0,1], 2D (H,W).
    Returns float SSIM. Raises ImportError if skimage missing.
    """
    if not has_skimage():
        raise ImportError("scikit-image required for compute_ssim")
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=a.max() - a.min()))

def compute_psnr(a: np.ndarray, b: np.ndarray):
    """
    Wrapper for PSNR. Inputs: float arrays in [0,1], 2D (H,W).
    Returns float PSNR. Raises ImportError if skimage missing.
    """
    if not has_skimage():
        raise ImportError("scikit-image required for compute_psnr")
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(a, b, data_range=a.max() - a.min()))

def detect_jpeg_blocking(img: np.ndarray, block_size: int = 8, factor: float = 1.5):
    """
    Heuristic detector for JPEG blocking artifacts.
    - img: 2D float array in [0,1]
    - block_size: typically 8 for JPEG
    - factor: threshold multiplier comparing boundary discontinuity to interior gradients

    Returns a float score and boolean (score, detected). Higher score implies more blockiness.
    The boolean is True if boundary_mean > factor * interior_mean.
    """
    assert img.ndim == 2, "detect_jpeg_blocking expects grayscale 2D image"

    H, W = img.shape
    if H < block_size or W < block_size:
        return 0.0, False

    # compute absolute differences between adjacent columns and rows
    col_diff = np.abs(img[:, 1:] - img[:, :-1])  # shape (H, W-1)
    row_diff = np.abs(img[1:, :] - img[:-1, :])  # shape (H-1, W)

    # boundary positions (internal boundaries only)
    col_boundaries = [c for c in range(block_size, W, block_size) if c < W]
    row_boundaries = [r for r in range(block_size, H, block_size) if r < H]

    if not col_boundaries and not row_boundaries:
        return 0.0, False

    # mean difference at boundaries
    b_vals = []
    for c in col_boundaries:
        # difference between column c-1 and c across rows
        b_vals.append(col_diff[:, c-1].mean())
    for r in row_boundaries:
        b_vals.append(row_diff[r-1, :].mean())
    if len(b_vals) == 0:
        return 0.0, False
    boundary_mean = float(np.mean(b_vals))

    # interior mean diff (sample non-boundary diffs)
    # mask out boundary columns/rows from diffs
    col_mask = np.ones(W-1, dtype=bool)
    for c in col_boundaries:
        if c-1 >= 0 and c-1 < W-1:
            col_mask[c-1] = False
    row_mask = np.ones(H-1, dtype=bool)
    for r in row_boundaries:
        if r-1 >= 0 and r-1 < H-1:
            row_mask[r-1] = False

    interior_col_vals = col_diff[:, col_mask].ravel() if col_mask.any() else np.array([])
    interior_row_vals = row_diff[row_mask, :].ravel() if row_mask.any() else np.array([])

    interior_vals = np.concatenate([v for v in [interior_col_vals, interior_row_vals] if v.size > 0]) \
        if (interior_col_vals.size > 0 or interior_row_vals.size > 0) else np.array([])

    interior_mean = float(interior_vals.mean()) if interior_vals.size > 0 else 1e-6

    score = boundary_mean / (interior_mean + 1e-12)
    detected = score > factor
    return float(score), bool(detected)