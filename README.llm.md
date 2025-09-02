# README.llm.md - Checkpoint do Projeto QSVM

## 📁 **Status Atual do Projeto**

### ✅ **O que está funcionando:**
- **QSVM_iris.py**: Pipeline completo testando todos os modelos QSVM no dataset Iris
- **patchkit/**: Sistema robusto de processamento de patches com cache otimizado
- **qsvm/models/**: Framework com 7+ modelos (clássicos + quânticos) + MultiOutputWrapper
- **Testes**: Suite básica funcionando (`pytest tests/`)

### ⚠️ **Problemas identificados no QSVM_iris.py:**
- Alguns modelos quânticos dão warnings/falhas
- **Correção sugerida** (testar primeiro):
  ```python
  # Ajustar parâmetros em QSVM_iris.py ~linha 75
  elif name == "VariationalFullyQuantum":
      model = cls(n_qubits=min(4, X_train.shape[1]), n_layers=1, 
                 lr=0.1, epochs=20)  # era lr=0.2, epochs=5
  elif name == "VariationalQuantumSVM_V6Flex":
      model = cls(n_qubits=min(4, X_train.shape[1]), n_layers=1, 
                 lr=0.01, entangler="line")  # era lr=0.05
  ```

## 🎯 **Objetivo Principal: QSVM_patches.py**

### **Meta:** Adaptar estrutura do QSVM_iris.py para funcionar com patches do MNIST

### **Pipeline desejado:**
1. **Dataset:** MNIST → Imagens 28×28 → downsampling → 14×14
2. **Patches:** 
   - Pequenos: 2×2 (stride=1) da imagem 14×14 
   - Grandes: 4×4 (stride=2) da imagem 28×28
   - **Ambos geram 169 patches** (correspondência 1:1)
3. **Quantização:** Binária (preto/branco) usando `quantization_levels=2`
4. **Treinamento:** Patch pequeno → Patch grande (super-resolução)

### **Configurações específicas:**
- **Classes selecionáveis:** ex: `--classes 1 8 --n-per-class 2` (total: 4 imagens)
- **Saídas:** Duas abordagens possíveis:
  - **MultiOutput:** 16 saídas (pixels do patch 4×4)  
  - **Single:** 1 saída + índice do pixel na entrada (dataset 16× maior)
- **Velocidade:** Mínimo de dados por ser ultra-lento

## 📋 **Plano de Implementação**

### **Fase 1: Corrigir QSVM_iris.py**
- [ ] Aplicar correções de parâmetros sugeridas
- [ ] Testar execução: `python QSVM_iris.py --quick`
- [ ] Verificar se warnings diminuíram

### **Fase 2: Criar função prepare_patches_dataset**
```python
def prepare_patches_dataset(classes=[1, 8], n_per_class=2, 
                           low_size=(14,14), high_size=(28,28),
                           small_patch=(2,2), large_patch=(4,4),
                           quantization_levels=2, seed=42):
    # Retorna: X_train, X_test, y_train, y_test
    # X: patches pequenos flattened
    # y: patches grandes flattened OU com MultiOutputWrapper
```

### **Fase 3: Adaptar QSVM_patches.py**
- [ ] Copiar estrutura do QSVM_iris.py
- [ ] Integrar prepare_patches_dataset
- [ ] Usar MultiOutputWrapper nos modelos
- [ ] Adicionar args: `--classes`, `--n-per-class`, `--quantization`

### **Fase 4: Visualizações específicas**
- [ ] Função `save_reconstruction_report()`:
  - Imagens originais (28×28 e 14×14)
  - Patches exemplo (2×2, 4×4, predito)  
  - Reconstrução completa
  - Métricas: MSE, SSIM
  - Diff visual

## 🔧 **Detalhes Técnicos**

### **Não mexer no core:**
- `patchkit/` e `qsvm/` já suportam tudo necessário
- `MultiOutputWrapper` já existe em `qsvm/base.py`

### **Resize algorithm:** `Image.LANCZOS` ou `Image.BICUBIC`

### **Uso do MultiOutputWrapper:**
```python
from qsvm.base import MultiOutputWrapper
model = MultiOutputWrapper(base_model, mode="multioutput")
```

### **Exemplo de uso esperado:**
```bash
# Teste rápido: 2 classes, 2 imagens cada, patches binários
python QSVM_patches.py --quick --classes 1 8 --n-per-class 2 --quantization 2

# Modelo específico
python QSVM_patches.py --model ClassicalSVM --classes 0 1 --n-per-class 3
```

## ❓ **Decisões pendentes:**
1. **MultiOutput (16 saídas) vs Single Output + índice?**
2. **Classes iniciais sugeridas:** 0, 1 (mais distintas)
3. **Métricas além de accuracy:** PSNR, SSIM?

## 🚀 **Próximo passo:**
Testar correções no QSVM_iris.py, depois criar prepare_patches_dataset isoladamente.

---
**Como usar este README:** Cole este conteúdo em um novo chat para retomar o contexto completo do projeto.