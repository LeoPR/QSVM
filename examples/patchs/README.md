# Exemplos de patches (examples/patchs)

Este diretório reúne exemplos focados em extração e uso de patches no MNIST, todos padronizados para salvar saídas em pastas relativas. Os diretórios de saída/caches são centralizados em `examples/patchs/config.py` (constante `OUTPUTS_ROOT`, padrão: `examples/patchs/outputs`).

## Scripts disponíveis

- `teste_cache.py`
  - Pipeline mínimo de cache em camadas (quantização → artefato → patches), útil como smoke test.
  - Verifica integração de cache e extração de patches com/fallback sem extractor otimizado.

- `teste_quantizacao.py`
  - Aplica diferentes métodos de quantização (`uniform`, `otsu`, `adaptive`, opcional `kmeans`) e extrai patches ativos.

- `teste_process_data.py`
  - Usa `ProcessedDataset` + `OptimizedPatchExtractor` para processar amostras selecionadas e salvar patches ativos.

- `teste_extract_binarized_dataset.py`
  - Gera versões binarizadas redimensionadas e salva patches ativos por amostra, organizados por configuração.

- `teste_mnist_patches_showcase.py`
  - Showcase completo (grade 6x7) comparando condições de resize/artefato, com reconstruções a partir de patches.

## Como executar (como módulo)

Execute a partir da raiz do repositório (com seu ambiente ativo):

```bash
python -m examples.patchs.teste_cache
python -m examples.patchs.teste_quantizacao
python -m examples.patchs.teste_process_data
python -m examples.patchs.teste_extract_binarized_dataset
python -m examples.patchs.teste_mnist_patches_showcase
```

Observação: usar `-m` garante que os imports relativos funcionem e que as saídas sejam escritas nas pastas relativas configuradas.

## Saídas (padronização)

- Base: `OUTPUTS_ROOT` em `examples/patchs/config.py` (padrão: `examples/patchs/outputs`).
- Por script (subpastas geradas):

  - `teste_cache.py`
    - `outputs/cache_quan_test/`
    - `outputs/cache_art_test/`
    - `outputs/cache_patch_test/`

  - `teste_quantizacao.py`
    - `outputs/mnist/class_{label}/{método}_ps{h}x{w}_str{stride}/`
    - Arquivos: `sample_{i}_original.png`, `sample_{i}_quant.png`, `patch_*.png`

  - `teste_process_data.py`
    - `outputs/outputs/mnist/class_{label}/sample_{i}/`
    - Caches: `outputs/outputs/cache_processed/`, `outputs/outputs/cache_patches/`
    - Arquivos: `processed.png`, `patches_active/patch_*.png`

  - `teste_extract_binarized_dataset.py`
    - `outputs/binarized_datasets/{dataset}_{WxH}_q{levels}_{method}/class_{label}/sample_{i}/`
    - Cache global: `outputs/binarized_datasets/cache_processed/`
    - Cache por amostra: `.../patches_active/cache_patches/`
    - Arquivos: `processed.png`, `patches_active/patch_*.png`

  - `teste_mnist_patches_showcase.py`
    - `outputs/mnist_showcase_YYYYMMDD_HHMMSS/class_{label}_idx_{orig_idx}/`
    - Arquivos: `combined_6x7_conditions_filtered_base.png`, `metrics.json`
    - Resumo: `outputs/mnist_showcase_YYYYMMDD_HHMMSS/summary.json`

Se preferir, ajuste `OUTPUTS_ROOT` no `config.py` para mudar o diretório base das saídas sem alterar os scripts.

## Dependências

- Obrigatórias para os exemplos de patches: `torch`, `torchvision`, `Pillow`, `numpy`
- Recomendadas/optativas:
  - `matplotlib` (visualizações no showcase)
  - `scikit-image` (SSIM/métricas)
  - `scikit-learn` (quantização `kmeans`)
  - `zstandard` (compressão de caches, se disponível)

Instale conforme necessário para o(s) script(s) que for executar.

## Dicas

- Execute os scripts em etapas (na ordem acima) para validar o ambiente.
- Caso use Git, mantenha `outputs` fora de versionamento; os caminhos já são relativos e podem ser limpos facilmente.