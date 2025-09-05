# 🧪 Plano de Testes e Ideias para patchkit

Este documento consolida ideias de testes, o que já foi feito, e um plano de migração/limpeza para manter o repositório organizado. Está escrito para ser prático: copie/cole trechos e rode comandos locais conforme indicado.

Última atualização: 2025-09-05

---

## Estado atual — o que já foi feito (resumo)
- `to_pil` e utilitários de conversão foram centralizados em `patchkit/image_utils.py`.
  - Corrigidos problemas de comportamento: comparação indevida de tensores (`.max()`), alinhamento em `interpolate`, e tratamento de canais incomuns (ex.: HxWx2 → RGB).
- Os testes foram atualizados para importar `to_pil` a partir de `patchkit.image_utils`.
- A cópia antiga `tests/patchkit/image_utils.py` foi removida (migração concluída).
- Suites de testes executadas localmente: `pytest tests/patchkit/` → 82 passed.

Checklist (concluído)
- [x] Criar/ajustar `patchkit/image_utils.py`.
- [x] Atualizar `tests/patchkit/test_processed.py` (e outros) para `from patchkit.image_utils import to_pil`.
- [x] Executar a suíte parcial/inteira de testes na pasta `tests/patchkit` — OK (82 passed).

---

## Objetivo deste README
1. Registrar ideias de testes adicionais (priorizadas).
2. Orientar limpeza segura de arquivos auxiliares nos testes.
3. Fornecer passos práticos para continuar a migração, verificação e limpeza incremental.

---

## Prioridade de testes sugeridos (ordem prática)
- Alta prioridade (implementar primeiro, vale testar incrementalmente)
  1. Comparação de algoritmos de resize (qualidade: SSIM/MSE) — já há um teste pequeno; ampliar para casos maiores.
  2. Teste de compressão JPEG (várias qualidades) e detecção de blocking.
  3. Testes de resizing extremo (upsampling/downsampling) e mudança de aspect ratio.

- Média prioridade
  4. Quantização avançada (dither, paletas).
  5. Métricas perceptuais (SSIM/PSNR/LPIPS — LPIPS opcional).
  6. Integração com datasets reais (CIFAR, subset ImageNet).

- Baixa prioridade
  7. Benchmarks de performance/memória.
  8. Testes adaptativos (seleção dinâmica de algoritmo).
  9. Testes cross-platform/regressão.

---

## Estrutura recomendada de testes (onde colocar)
- tests/patchkit/
  - test_image_utils.py         # validações do `patchkit.image_utils`
  - test_processed.py           # integra ProcessedDataset e caching
  - test_quantizer.py           # quantização
  - test_extractor.py           # extractor otimizado
  - test_resize_quality.py      # comparação de algoritmos (SSIM)
  - test_superres_dataset.py    # cenários de super-resolution
  - optional_deps.py            # checagens centralizadas de deps (já existe)

---

## Limpeza segura (arquivos que podem ser apagados agora)
- `tests/patchkit/additional_tests_ideas.md`
  - Recomendação: renomear para `README.md` (você solicitou isso). O conteúdo abaixo já foi convertido para este README.
- Outros arquivos a considerar (verificar referências antes de apagar):
  - `tests/patchkit/test_utils.py` — contém utilitários úteis (compute_ssim, detect_jpeg_blocking). Só remova se:
    - Você mover as funções para `patchkit` (ex.: `patchkit.testing_utils`) ou,
    - Confirmar que nenhum teste ou script local depende delas.
  - Antes de apagar qualquer arquivo, rode:
    - PowerShell:
      - Select-String -Path .\**\*.py -Pattern "nome_do_arquivo" -List
    - Bash:
      - grep -R "nome_do_arquivo" .

---

## Passo a passo recomendado para a migração/limpeza (seguro / incremental)
1. Confirme que `patchkit/image_utils.py` está com a versão final (a que passou nos testes).
   - Se não, cole a versão que você já testou.
2. Substitua `tests/patchkit/additional_tests_ideas.md` por este `README.md`.
   - Comando (PowerShell): Rename-Item .\tests\patchkit\additional_tests_ideas.md .\tests\patchkit\README.md
   - Ou apague e cole novo arquivo.
3. Atualize todos os testes que referenciam a cópia antiga do utilitário (caso reste algum):
   - Procurar:
     - PowerShell: Select-String -Path tests\**\*.py -Pattern "image_utils" -List
     - Bash: grep -R "image_utils" tests || true
   - Substituir por:
     - from patchkit.image_utils import to_pil
4. Rodar suites parciais (faça incremental):
   - pytest tests/patchkit/test_image_utils.py -q
   - pytest tests/patchkit/test_processed.py -q
   - pytest tests/patchkit/test_quantizer.py -q
   - pytest tests/patchkit -q
5. Se algum teste falhar por diferenças numéricas, ajuste `patchkit/image_utils.py` para reproduzir exatamente o comportamento anterior (ex.: arredondamento/clip/ordem de canais). Teste incrementalmente até estar OK.
6. Opcional: consolidar `compute_ssim/psnr` e `detect_jpeg_blocking` em `patchkit/testing_utils.py` se quiser usar essas funções também fora dos testes.

---

## Cuidados/risks e dicas
- Evite monkeypatch global de `builtins.__import__` nos testes (podem quebrar o ambiente). Use `monkeypatch.setitem(sys.modules, 'nome_modulo', None)` para simular ausência de dependência.
- Pequenas diferenças de arredondamento podem afetar testes sensíveis (SSIM). Se ocorrerem, alinhe `to_pil`/`to_tensor` no pacote para reproduzir o comportamento esperado pelos testes.
- Garanta que `patchkit` é um package (tem `__init__.py`) para que `from patchkit.image_utils import to_pil` funcione no pytest/CI.

---

## Comandos úteis
- Rodar todos os testes na pasta patchkit:
  - pytest tests/patchkit -v
- Procurar referências a `image_utils`:
  - PowerShell: Select-String -Path .\**\*.py -Pattern "image_utils" -List
  - Bash: grep -R "image_utils" tests || true
- Renomear arquivo:
  - PowerShell: Rename-Item .\tests\patchkit\additional_tests_ideas.md .\tests\patchkit\README.md
  - Bash: mv tests/patchkit/additional_tests_ideas.md tests/patchkit/README.md

---

## Próximos passos sugeridos (curto prazo)
- [ ] Substituir `additional_tests_ideas.md` pelo README desta versão (você já planeja renomear).
- [ ] Revisar `test_utils.py`: decidir se mover para `patchkit` (como `patchkit/testing_utils.py`) ou mantê-lo nos testes.
- [ ] Revisar testes que reimplementam `has_skimage()` e apontá-los para `tests/patchkit/optional_deps.py` para evitar duplicação.
- [ ] Substituir monkeypatch perigosos (se existirem) por manipulação de `sys.modules` nos testes de fallback.

---

Se concordar com o conteúdo, cole este arquivo como `tests/patchkit/README.md` (ou substitua o `additional_tests_ideas.md`). Se quiser, eu gero o conteúdo do arquivo `patchkit/testing_utils.py` para consolidar `compute_ssim`/`detect_jpeg_blocking` e a alteração de testes que usam essas funções — mas faço isso só quando você aprovar para eu não aplicar mudanças em massa de uma vez.