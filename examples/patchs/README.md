```markdown
# MNIST patches showcase (examples/patch)

Este exemplo gera uma pequena amostra do MNIST demonstrando extração de patches,
reconstrução e métricas, integrado ao framework `patchkit` do repositório.

Características
- Seleciona por padrão as classes (dígitos) 1 e 8, 2 amostras por classe.
- Gera:
  - representação original (28x28) com patches 4x4 e stride 2 (13x13 = 169 patches)
  - representação reduzida (14x14) com patches 2x2 e stride 1 (13x13 = 169 patches)
- Seleciona um patch "relevante" (não todo preto/nem todo branco) no mesmo índice linear nas grades 13x13.
- Reconstrói imagens a partir dos patches e computa métricas PSNR/MSE/SSIM via `patchkit.image_metrics.ImageMetrics`.
- Salva arquivos em `examples/patch/outputs/mnist_showcase_YYYYMMDD_HHMMSS/` por amostra:
  - `original.png`, `reduced_14.png`
  - `reconstructed_original.png`, `reconstructed_reduced_14.png`
  - `patch_original.png`, `patch_reduced.png`
  - `line_original_reduced_patches.png` (linha com 4 imagens)
  - `line_reconstructed.png` (linha com 2 imagens)
  - `metrics.json` (métricas da amostra)
- Gera um `summary.json` com todas as amostras.

Integração com patchkit
- Usa `patchkit.image_utils` (to_pil/resize/to_tensor) para compatibilidade de conversões e redimensionamentos.
- Tenta usar `patchkit.patches.OptimizedPatchExtractor` para extrair patches (se disponível) e converte o resultado para o formato usado internamente; caso contrário, usa extras locais via `torch.unfold`.
- Usa `patchkit.image_metrics.ImageMetrics` para PSNR/SSIM/PSNR.

Como usar
- Colar `mnist_patches_showcase.py` em `examples/patch/`.
- Rodar:
  - python examples/patch/mnist_patches_showcase.py
- Verificar a pasta `examples/patch/outputs/mnist_showcase_YYYYMMDD_HHMMSS/`.

Observações
- O script evita import direto `from PIL import Image` no topo; usa `patchkit.image_utils` para centralizar a dependência do PIL.
- SSIM depende de scikit-image; se ausente, `ssim` no resumo ficará `null`.
- Teste aos poucos: rode o script, verifique outputs; se quiser que eu ajuste para usar estritamente `OptimizedPatchExtractor` (por exemplo, para usar cache do extractor), eu posso adaptar caso queira testar esse comportamento.
```