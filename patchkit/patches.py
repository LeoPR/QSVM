import os
import io
import tempfile
import hashlib
import torch
import zstandard as zstd
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from collections import OrderedDict
from Logger import Logger

def filter_active_patches(patches: torch.Tensor, min_mean: float = 0.1, max_mean: float = 0.9):
    """
    Filtra patches "ativos" e calcula score de informatividade para cada patch.

    Entrada:
      - patches: torch.Tensor [L, H, W] ou [L, C, H, W], dtype uint8 (0..255) ou float em [0,1].
      - min_mean, max_mean: limites para considerar um patch "ativo" (média do patch no intervalo).

    Saída (sempre):
      - indices_ordenados: torch.LongTensor (K,) com índices dos patches ativos, ordenados do melhor para o pior
                          segundo o score (melhor primeiro). Se nenhum patch for ativo, retorna tensor longo vazio.
      - scores: torch.FloatTensor (L,) com o score de informatividade de cada patch (mesma heurística do utils).

    Observações:
      - A função não tenta suportar múltiplos formatos mágicamente; espera um tensor conforme descrito.
      - Caso queira os patches ativos, use: patches[indices_ordenados]
      - Se quiser um fallback quando indices_ordenados estiver vazio (ex.: índice central), trate no chamador.
    """
    # validação básica
    if not isinstance(patches, torch.Tensor):
        raise TypeError("filter_active_patches espera um torch.Tensor com shape [L,H,W] ou [L,C,H,W]")

    if patches.numel() == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)

    # normalizar para float em [0,1] se necessário
    p = patches.float()
    try:
        pmax = float(p.max().item())
    except Exception:
        pmax = 1.0
    if pmax > 1.0:
        p = p / 255.0

    # calcular média por patch (reduz canais se houver)
    means = p.mean(dim=(-2, -1))
    if means.dim() > 1:
        means = means.mean(dim=-1)

    sel_mask = (means > float(min_mean)) & (means < float(max_mean))
    active_indices = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    if active_indices.numel() == 0:
        # produzir tensor vazio do tipo long
        active_indices = torch.tensor([], dtype=torch.long)

    # calcular scores para todos os patches usando heurística previamente definida
    L = p.shape[0]
    scores = torch.zeros(L, dtype=torch.float32)

    # usar CPU para operações de histograma / estatísticas (mais estável e compatível)
    p_cpu = p.detach().cpu()

    for i in range(L):
        patch_i = p_cpu[i]
        # converter para single-channel para análise
        if patch_i.dim() == 3:
            p_s = patch_i.mean(dim=0)
        else:
            p_s = patch_i
        # variância
        var = float(torch.var(p_s).item())
        # histograma (16 bins em [0,1])
        hist = torch.histc(p_s, bins=16, min=0.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)
        hist_nz = hist[hist > 0]
        entropy = float((-(hist_nz * torch.log(hist_nz)).sum().item()) if hist_nz.numel() > 0 else 0.0)
        # gradientes médios
        dx = float(torch.abs(p_s[1:, :] - p_s[:-1, :]).mean().item()) if p_s.shape[0] > 1 else var
        dy = float(torch.abs(p_s[:, 1:] - p_s[:, :-1]).mean().item()) if p_s.shape[1] > 1 else var
        grad = 0.5 * (dx + dy)
        mean_val = float(p_s.mean().item())
        range_score = 4 * mean_val * (1 - mean_val)
        contrast = float(p_s.max().item() - p_s.min().item())
        # combinação heurística de métricas (mesma que utils.select_informative_patch)
        score = 0.25 * var + 0.2 * (entropy / 3.0) + 0.2 * grad + 0.2 * range_score + 0.15 * contrast
        scores[i] = float(score)

    # ordenar os índices ativos por score desc (melhor primeiro)
    if active_indices.numel() > 0:
        # extrair scores dos ativos e ordenar
        active_scores = scores[active_indices].cpu()
        # obter ordem descendente
        order_rel = torch.argsort(active_scores, descending=True)
        ordered_active = active_indices[order_rel].long()
    else:
        ordered_active = torch.tensor([], dtype=torch.long)

    return ordered_active, scores


class OptimizedPatchExtractor:
    def __init__(self, patch_size, stride, cache_dir, image_size, max_memory_cache=100):
        if cache_dir is None:
            # respeita variável de ambiente pra facilitar CI/local dev
            cache_dir = os.environ.get("QSVM_CACHE_DIR",
                                       os.path.join(tempfile.gettempdir(), "qsvm_cache"))
        self.cache_dir = os.path.join(cache_dir, "patches_optimized")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.patch_size = patch_size  # (h,w)
        self.stride = stride
        self.image_size = image_size  # (h,w)
        self.max_memory_cache = max_memory_cache

        self.num_patches_h = (image_size[0] - patch_size[0]) // stride + 1
        self.num_patches_w = (image_size[1] - patch_size[1]) // stride + 1
        self.num_patches_per_image = self.num_patches_h * self.num_patches_w

        self.cache_dir = os.path.join(cache_dir, "patches_optimized")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.memory_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_path(self, image_index):
        config_str = (f"img_{image_index}_size_{self.image_size[0]}x{self.image_size[1]}_"
                      f"patch_{self.patch_size[0]}x{self.patch_size[1]}_stride_{self.stride}")
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{config_str}_{config_hash}.pt.zst")

    def _extract_all_patches(self, image):
        tensor = TF.pil_to_tensor(image)  # uint8 [C,H,W]
        C, H, W = tensor.shape
        tensor = tensor.unsqueeze(0).float()  # [1,C,H,W] in 0-255
        ph, pw = self.patch_size
        patches_unf = F.unfold(tensor, kernel_size=(ph, pw), stride=self.stride)  # [1, C*ph*pw, L]
        patches_flat = patches_unf.squeeze(0).transpose(0, 1)  # [L, C*ph*pw]
        patches = patches_flat.view(-1, C, ph, pw)
        patches = patches.round().to(torch.uint8)
        if C == 1:
            return patches.squeeze(1)
        return patches

    def _save_compressed_patches(self, patches, cache_path):
        buffer = io.BytesIO()
        torch.save(patches, buffer)
        buffer.seek(0)
        with open(cache_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=3, threads=-1)
            compressed = cctx.compress(buffer.read())
            f.write(compressed)

    def _load_compressed_patches(self, cache_path):
        try:
            with open(cache_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(f.read())
            buffer = io.BytesIO(decompressed)
            patches = torch.load(buffer, map_location="cpu")
            if patches.shape[0] != self.num_patches_per_image:
                Logger.warning("Invalid patch count in cache")
                return None
            return patches
        except Exception as e:
            Logger.warning(f"Failed to load patches: {e}")
            return None

    def _update_memory_cache(self, image_index, patches):
        if image_index in self.memory_cache:
            del self.memory_cache[image_index]
        self.memory_cache[image_index] = patches
        if len(self.memory_cache) > self.max_memory_cache:
            evicted = next(iter(self.memory_cache))
            del self.memory_cache[evicted]

    def process(self, image, index=None):
        if index is None:
            return self._extract_all_patches(image)

        if index in self.memory_cache:
            self.cache_hits += 1
            patches = self.memory_cache[index]
            # move to end
            del self.memory_cache[index]
            self.memory_cache[index] = patches
            return patches

        self.cache_misses += 1
        cache_path = self._get_cache_path(index)
        if os.path.exists(cache_path):
            patches = self._load_compressed_patches(cache_path)
            if patches is not None:
                self._update_memory_cache(index, patches)
                return patches

        patches = self._extract_all_patches(image)
        self._save_compressed_patches(patches, cache_path)
        self._update_memory_cache(index, patches)
        return patches

    def get_patch(self, image, index, patch_idx):

        if patch_idx < 0:
            raise IndexError(f"Patch index {patch_idx} cannot be negative")
        if patch_idx >= self.num_patches_per_image:
            raise IndexError(f"Patch index {patch_idx} out of bounds (max: {self.num_patches_per_image - 1})")

        all_patches = self.process(image, index)
        return all_patches[patch_idx]

    def clear_memory_cache(self):
        self.memory_cache.clear()

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'max_cache_size': self.max_memory_cache
        }

    def validate_and_clean_cache(self):
        invalid = 0
        files = os.listdir(self.cache_dir)
        for f in files:
            if f.endswith('.pt.zst'):
                p = os.path.join(self.cache_dir, f)
                try:
                    patches = self._load_compressed_patches(p)
                    if patches is None or patches.shape[0] != self.num_patches_per_image:
                        os.remove(p)
                        invalid += 1
                except Exception:
                    os.remove(p)
                    invalid += 1
        return invalid

    def reconstruct_image(self, patches, device=torch.device("cpu")):
        """
        Reconstruct an image from a set of patches using F.fold (vectorized).

        Accepts:
            patches: Tensor [L, H, W] (uint8) or [L, C, H, W] (uint8)
        Returns:
            - single-channel: Tensor [recon_h, recon_w] uint8
            - multi-channel: Tensor [C, recon_h, recon_w] uint8
        """
        num_h = self.num_patches_h
        num_w = self.num_patches_w
        patch_h, patch_w = self.patch_size

        recon_h = patch_h + (num_h - 1) * self.stride
        recon_w = patch_w + (num_w - 1) * self.stride

        # Detect format of patches and prepare tensor [L, C, ph, pw]
        if patches.dim() == 3:
            # [L, H, W] -> [L, 1, H, W]
            patches_f = patches.unsqueeze(1).float() / 255.0
        elif patches.dim() == 4:
            # [L, C, H, W]
            patches_f = patches.float() / 255.0
        else:
            raise ValueError("Unsupported patches shape for reconstruction")

        L, C, ph, pw = patches_f.shape
        # Move to requested device (CPU by default). fold will sum contributions.
        patches_f = patches_f.to(device)

        # Prepare for fold: need [1, C*ph*pw, L]
        patches_flat = patches_f.view(L, -1).transpose(0, 1).unsqueeze(0)  # [1, C*ph*pw, L]

        # Sum contributions (reconstruction) using fold
        recon_sum = F.fold(patches_flat, output_size=(recon_h, recon_w),
                           kernel_size=(ph, pw), stride=self.stride)  # [1, C, H, W]

        # Create weight matrix (how many times cada pixel foi somado) usando ones
        ones = torch.ones_like(patches_f, dtype=patches_f.dtype, device=device)
        ones_flat = ones.view(L, -1).transpose(0, 1).unsqueeze(0)  # [1, C*ph*pw, L]
        weight = F.fold(ones_flat, output_size=(recon_h, recon_w),
                        kernel_size=(ph, pw), stride=self.stride)  # [1, C, H, W]

        # Avoid division by zero
        recon = recon_sum / weight.clamp(min=1e-6)

        # Convert back to uint8
        recon = (recon * 255.0).round().to(torch.uint8).squeeze(0)  # [C, H, W] or [H, W] if C==1

        if recon.shape[0] == 1:
            return recon.squeeze(0)  # [H, W]
        return recon  # [C, H, W]


PatchExtractor = OptimizedPatchExtractor