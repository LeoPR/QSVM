import torch
import torch.nn.functional as F

def select_informative_patch(patches, num_candidates=5):
    """
    Select an informative patch index from patches.
    patches: [L,H,W] uint8 or [L,C,H,W] uint8
    Returns: (best_idx:int, scores:torch.Tensor)
    """
    pf = patches.float() / 255.0
    L = pf.shape[0]
    scores = torch.zeros(L)
    for i in range(L):
        p = pf[i]
        # if multi-channel, convert to single channel by mean
        if p.dim() == 3:
            p_s = p.mean(dim=0)
        else:
            p_s = p
        var = torch.var(p_s).item()
        hist = torch.histc(p_s, bins=16, min=0, max=1)
        hist = hist / (hist.sum() + 1e-8)
        hist_nz = hist[hist > 0]
        entropy = -(hist_nz * torch.log(hist_nz)).sum().item() if hist_nz.numel() > 0 else 0.0
        dx = torch.abs(p_s[1:, :] - p_s[:-1, :]).mean().item() if p_s.shape[0] > 1 else var
        dy = torch.abs(p_s[:, 1:] - p_s[:, :-1]).mean().item() if p_s.shape[1] > 1 else var
        grad = 0.5 * (dx + dy)
        mean_val = p_s.mean().item()
        range_score = 4 * mean_val * (1 - mean_val)
        contrast = p_s.max().item() - p_s.min().item()
        scores[i] = 0.25 * var + 0.2 * (entropy / 3.0) + 0.2 * grad + 0.2 * range_score + 0.15 * contrast

    top_k = min(num_candidates, L)
    top_indices = torch.topk(scores, top_k).indices
    best_idx = top_indices[0].item()
    for idx in top_indices:
        m = (pf[idx].mean() if pf[idx].dim() == 2 else pf[idx].mean())
        if 0.3 <= m <= 0.7:
            best_idx = idx.item()
            break
    return best_idx, scores