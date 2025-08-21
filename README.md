# QSVM PennyLane – v7, v8 e v10 (Variational Fully Quantum)
 
###### Explicações geradas a partir dos códigos, usado o  Claude.ai para ajudar nas no texto

Este repositório reúne três marcos da evolução do seu pipeline de **Quantum Support Vector Machines (QSVM)** com **PennyLane**:

- **v7_flores** – base consolidada dos modelos (clássicos, kernel quântico, variacionais e o *fullyquantum* original).
- **v8_flores** – framework de **experimentos** (grade de configurações, K-fold, CSV de resultados, gráficos).
- **v10_experiments** – inclui o novo **Variational Fully Quantum (VFQ)** com **shots finitos** e decisão 100% dentro do circuito.

> Observação: este README *não* documenta o v9; foco somente em v7, v8 e v10 conforme pedido.

---

## 🔹 v7_flores — Base consolidada
**Arquivo:** `QSVM_penny_v7_flores.py`

### O que tem
- **SVMs clássicos** (Linear, RBF) como *baseline*.
- **Kernel quântico híbrido**: estados preparados no circuito e kernel calculado (geralmente usando *overlaps*), mas o treino/decisor é **clássico (SVM)**.
- **Variacionais** (VQC-style): embedding (geralmente `RY`) + ansatz parametrizado (`RY/RZ/RX` + `CNOTs`), treino via gradiente (parameter-shift) e decisão por ⟨Z⟩.
- **“Fullyquantum” original**: circuito para kernel via **SWAP test** (ou overlap) e uso em SVM; “fully” aqui é do kernel, mas ainda há solver clássico.

### Quando usar
- Para ter um **ponto de partida** com todos os sabores (clássico, híbrido, variacional).
- Para comparar rapidamente **RX on/off** e diferentes entanglers.

---

## 🔹 v8_flores — Framework de experimentos
**Arquivo:** `QSVM_penny_v8_flores.py`

### O que tem
- **Execução em lote** de experimentos: grade de configs (loss, RX, entangler, camadas, etc.).
- **K-fold + repetições** com *seeds* diferentes.
- **CSV de resultados** (uma linha por fold) + **gráficos** (Top-N, por tipo, boxplots, curvas de *loss* quando disponível).
- **Paralelismo conservador**: apenas SVMs clássicos e kernel híbrido rodam em *pool* de processos; variacionais e fully-quantum seguem sequenciais para não conflitar com os *devices* do PennyLane e manter aderência a hardware real.

### Exemplos
```bash
# Sequencial
python QSVM_penny_v8_flores.py --kfold 3 --repeats 2 --out runs_v8.csv --outdir figs_v8

# Paralelo conservador
python QSVM_penny_v8_flores.py --kfold 3 --repeats 2 --out runs_v8_par.csv \
  --outdir figs_v8 --parallel --workers 4 --parallel_scope safe
```

---

## 🔹 v10_experiments — Variational Fully Quantum (VFQ)
**Arquivo:** `QSVM_penny_v10_experiments.py`

### O que é o VFQ?
Um **classificador variacional 100% quântico na decisão**:
- O circuito recebe os dados via **AngleEmbedding(Y)** e passa por um **ansatz** parametrizado (camadas com `RY/RZ` e `RX` opcional + **entanglement circular**).
- A **predição** é feita diretamente por uma **medição quântica**: usamos a expectativa ⟨Z(0)⟩ (com **shots finitos**), transformada em probabilidade `p = (1+⟨Z⟩)/2` ou limiar de sinal.
- O **treino** minimiza uma *loss* construída **a partir das medidas** do circuito:
  - `hinge` (para rótulos ±1, estilo SVM)
  - `bce` (binary cross-entropy com `p`)
- **Sem classificador clássico** na ponta — a decisão vem do circuito.

> Isso aproxima o comportamento de execução em hardware real: variâncias por **shots finitos** e otimização variacional com **parameter-shift**.

### Outras variantes incluídas no v10
- **SVMs clássicos** (Linear, RBF).
- **Kernel quântico híbrido** (overlap entre estados + SVM clássico).
- **Variacional analítico (v6flex)** como *baseline* (sem shots, expval analítico; útil para depurar e comparar).

### Como rodar
```bash
# Sequencial com relatório e early stopping
python QSVM_penny_v10_experiments.py --kfold 3 --repeats 2 \
  --out qsvm_v10_runs.csv --outdir qsvm_figs --report qsvm_figs/report.html \
  --early --patience 12 --min_delta 1e-4

# Paralelo conservador (SVMs + kernel híbrido no pool)
python QSVM_penny_v10_experiments.py --kfold 3 --repeats 2 \
  --out qsvm_v10_par.csv --outdir qsvm_figs --report qsvm_figs/report.html \
  --parallel --workers 4 --parallel_scope safe

# Ajustar shots do VFQ para simular mais/menos ruído estatístico
python QSVM_penny_v10_experiments.py --kfold 3 --repeats 1 \
  --out qsvm_v10_vfq.csv --outdir figs --report figs/report.html \
  --early --shots 2048 --topn 10
```

### Saídas
- **CSV bruto** (`--out`) com `accuracy`, `f1_macro`, `balanced_accuracy`, `train_time_s`.
- **Figuras** em `--outdir`: Top-N, barras por tipo, boxplots, **curvas de loss** dos variacionais (inclui VFQ).
- **Resumo** por configuração: `summary_means.csv`.
- **Relatório HTML** com tabela e as imagens geradas.

---

## 🔹 Tipos, prós e contras (visão geral)
| Tipo | Como funciona | Prós | Contras |
|---|---|---|---|
| **SVM Clássico (Linear/RBF)** | Treino e decisão 100% clássicos | Estável, rápido, baseline forte | Não usa recursos quânticos |
| **Kernel Quântico Híbrido** | Kernel via overlap/estado quântico + SVM clássico | Capta estruturas não lineares no espaço quântico | Ainda depende de solver clássico; custo quadrático no nº de amostras |
| **Variacional Analítico (v6flex)** | Embedding + ansatz; expval analítico; loss (MSE/hinge) | Treino suave (sem ruído de shots), bom para *debug* | Menos realista p/ hardware; pode superestimar |
| **Variational Fully Quantum (VFQ)** | Igual ao anterior, mas com **shots finitos** e decisão no circuito (hinge/BCE) | Mais realista; decisão 100% quântica; compatível com QCs reais | Treino mais ruidoso; requer mais épocas/shots; sensível a *barren plateaus* |

---

## 🔹 “Fully quantum” — por que alguns não são totalmente?
- **Kernel quântico híbrido**: o kernel é quântico, mas o **classificador** (SVM) é **clássico**.
- **Variacionais analíticos**: usam **expval exato** (sem shots). Isso simplifica a simulação, mas não espelha a estatística de contagem dos dispositivos reais.
- **VFQ (v10)**: a **decisão é quântica** e o treinamento usa medidas com **shots finitos**. Ainda existe um **otimizador clássico** (Adam), como em VQCs padrão — isso é normal e aceito na prática atual.

---

## 🔹 Normalização com π — faz sentido?
- Em v7/v8/v10, os dados passam por **preprocessamento**:
  - `StandardScaler` (z-score) → centraliza e normaliza a variância.
  - `MinMaxScaler` para mapear os recursos para **[0, 2π]** quando usados como **ângulos de rotação** (`AngleEmbedding`/`RY`).  
- **Por quê 2π?** Rotacionar por um ângulo fora de `[0, 2π]` equivale modularmente a um ângulo dentro; mapear para esse intervalo **estabiliza** o embedding e evita saturação atípica.
- **É sempre ideal?** Não necessariamente. Algumas tarefas funcionam melhor com `[-π, π]` . A recomendação é **experimentar** (o framework v8/v10 facilita).

---

## 🔹 Melhorias sugeridas
- **Grid externo** (JSON/YAML) para configs (`--grid`): fácil de acrescentar.
- **Early stopping por validação intra-fold** (já há uma versão global; dá para refinar).
- **Penalizações** (L2/L1) nos parâmetros variacionais (regularização).
- **Ansätze hardware-efficient** específicos para topologia IBM/rigetti.
- **Observables alternativos** (paridades, *majority vote* em vários qubits).
- **Calibração de probabilidades** pós-medida para BCE.
- **Gestão de *shots* adaptativa** (aumentar shots conforme treino estabiliza).

---

## 🔹 Referências úteis
- **Havlíček, V. et al. (2019)**, *Supervised learning with quantum-enhanced feature spaces*, **Nature 567**, 209–212.  
- **Schuld, M. & Killoran, N. (2019)**, *Quantum Machine Learning in Feature Hilbert Spaces*, **PRL 122**, 040504.  
- **Benedetti, M. et al. (2019)**, *Parameterized quantum circuits as machine learning models*, **Quantum Sci. Technol. 4(4)**.  
- **Schuld, M. (2021)**, *Supervised quantum machine learning models are kernel methods*, **arXiv:2101.11020**.  
- **PennyLane Demos & Docs** – VQCs, kernels e parameter-shift.

---

## 🔹 Resumo rápido
- **v5**: apenas referencia histórica código inicial (clássico, híbrido, variacional e kernel “fully”).
- **v7**: todos os sabores (clássico, híbrido, variacional e kernel “fully”).
- **v8**: framework de experimentos (K-fold, CSV, gráficos, paralelismo seguro).
- **v10**: **Variational Fully Quantum** com **shots finitos** e decisão no circuito, mantendo toda a instrumentação do framework.

