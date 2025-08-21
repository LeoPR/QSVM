import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from pennylane import numpy as pnp
import warnings

warnings.filterwarnings('ignore')


def get_best_device(wires):
    """Retorna o melhor dispositivo disponível"""
    devices_priority = [
        ("lightning.gpu", "GPU Lightning"),
        ("lightning.qubit", "CPU Lightning"),
        ("default.qubit", "Default CPU")
    ]

    for device_name, description in devices_priority:
        try:
            test_dev = qml.device(device_name, wires=2)
            print(f"✅ Usando: {description}")
            return device_name
        except:
            continue

    return "default.qubit"  # Fallback final


# =============================================
# CARREGAMENTO DO DATASET IRIS
# =============================================

def load_iris_data():
    """Carrega e prepara o dataset Iris clássico"""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Usar apenas 2 classes para classificação binária (Setosa vs Versicolor)
    mask = y != 2  # Remove Virginica
    X = X[mask]
    y = y[mask]

    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Converter labels para -1 e +1 (padrão SVM)
    y = np.where(y == 0, -1, 1)

    print(f"Classes após conversão: {np.unique(y)} (Setosa=-1, Versicolor=+1)")

    print("Dataset Iris carregado:")
    print(f"- Features: {iris.feature_names}")
    print(f"- Classes: Setosa vs Versicolor")
    print(f"- Samples: {len(X)}")
    print(f"- Features por sample: {X.shape[1]}")

    return X, y, scaler, iris.feature_names


# =============================================
# KERNELS CLÁSSICOS
# =============================================

def rbf_kernel_classical(x1, x2, gamma=1.0):
    """Kernel RBF clássico"""
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def linear_kernel_classical(x1, x2):
    """Kernel linear clássico"""
    return np.dot(x1, x2)


def polynomial_kernel_classical(x1, x2, degree=2, coeff=1.0):
    """Kernel polinomial clássico"""
    return (coeff + np.dot(x1, x2)) ** degree


def compute_classical_kernel_matrix(X1, X2, kernel_func, **kwargs):
    """Computa matriz de kernel clássica"""
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j], **kwargs)

    return K


# =============================================
# QUANTUM SVM HÍBRIDO (KERNEL QUÂNTICO)
# =============================================

# Configuração do dispositivo quântico

n_qubits = 2

device_type = get_best_device(n_qubits)

dev = qml.device(device_type, wires=n_qubits)


def data_encoding_circuit(x):
    """Circuito de codificação de dados - ZZFeatureMap inspirado"""
    # Primeira camada: codificação linear
    for i in range(min(len(x), n_qubits)):
        qml.RY(x[i], wires=i)

    # Segunda camada: interações não-lineares
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        if i < len(x) - 1:
            qml.RZ(x[i] * x[i + 1], wires=i + 1)

    # Terceira camada: mais rotações
    for i in range(min(len(x), n_qubits)):
        qml.RY(x[i], wires=i)


@qml.qnode(dev)
def quantum_kernel_circuit(x1, x2):
    """Circuito para calcular kernel quântico híbrido"""
    # Codificar primeiro vetor
    data_encoding_circuit(x1)

    # Aplicar adjoint do segundo vetor
    qml.adjoint(data_encoding_circuit)(x2)

    # Medir probabilidade no estado |0⟩^n (fidelidade)
    return qml.probs(wires=range(n_qubits))


def quantum_kernel_hybrid(x1, x2):
    """Calcula kernel quântico híbrido baseado na fidelidade"""
    probs = quantum_kernel_circuit(x1, x2)
    return probs[0]  # Probabilidade do estado |0000⟩


def compute_quantum_kernel_matrix_hybrid(X1, X2):
    """Computa matriz de kernel quântico híbrido"""
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            K[i, j] = quantum_kernel_hybrid(X1[i], X2[j])

    return K


# =============================================
# QUANTUM SVM COMPLETAMENTE QUÂNTICO
# =============================================

# Baseado no trabalho de Rebentrost et al. (2014)

class FullyQuantumSVM:
    """
    Implementação do QSVM completamente quântico baseado em Rebentrost et al.
    Usa quantum least squares e quantum matrix inversion
    """

    def __init__(self, n_qubits=5, C=1.0):
        self.n_qubits = n_qubits
        self.C = C  # Parâmetro de regularização
        self.dev = qml.device(device_type, wires=2 * n_qubits + 2)
        self.alpha = None  # Coeficientes do SVM
        self.X_train = None
        self.y_train = None

    def quantum_state_preparation(self, x):
        """Preparação do estado quântico para um vetor de dados - versão melhorada"""
        # Normalizar o vetor
        norm = np.linalg.norm(x)
        if norm > 1e-8:
            x_normalized = x / norm
        else:
            x_normalized = np.ones_like(x) * 1e-8  # Evitar zeros

        # Codificação por amplitude melhorada
        for i in range(min(len(x_normalized), self.n_qubits)):
            # Usar arctan2 para melhor estabilidade numérica
            angle = 2 * np.arctan2(abs(x_normalized[i]), 1)
            if abs(angle) > 1e-8:  # Só aplicar se o ângulo for significativo
                qml.RY(angle, wires=i)

        # X_quantum = 2*np.pi * (X_scaled - X_scaled.min()) / (X_scaled.max() - X_scaled.min())

    def quantum_inner_product(self, x1, x2):
        """Produto interno quântico rápido usando SWAP test"""
        dev_temp = qml.device(device_type, wires=5)  # Usar dispositivo otimizado

        @qml.qnode(dev_temp)
        def inner_product_circuit():
            # Preparar estados |x1⟩ nos qubits 0,1
            for i in range(min(2, len(x1))):
                if abs(x1[i]) > 1e-8:  # Evitar valores muito pequenos
                    qml.RY(2 * np.arctan2(abs(x1[i]), 1), wires=i)

            # Preparar estados |x2⟩ nos qubits 2,3
            for i in range(min(2, len(x2))):
                if abs(x2[i]) > 1e-8:
                    qml.RY(2 * np.arctan2(abs(x2[i]), 1), wires=i + 2)

            # SWAP test com qubit ancilla 4
            qml.Hadamard(wires=4)

            # CSWAP entre qubits correspondentes
            qml.CSWAP(wires=[4, 0, 2])  # control=4, swap entre 0 e 2
            qml.CSWAP(wires=[4, 1, 3])  # control=4, swap entre 1 e 3

            qml.Hadamard(wires=4)

            return qml.probs(wires=4)

        return inner_product_circuit()

    def quantum_kernel_matrix(self, X):
        """Computa matriz de kernel usando circuitos quânticos"""
        n = len(X)
        K = np.zeros((n, n))

        print(f"Computando matriz {n}x{n} do kernel quântico...")

        for i in range(n):
            for j in range(n):
                try:
                    # Usar produto interno quântico como kernel
                    probs = self.quantum_inner_product(X[i], X[j])
                    # Kernel baseado na fidelidade: K = |⟨ψ₁|ψ₂⟩|²
                    K[i, j] = probs[0]  # Probabilidade de medir |0⟩ no ancilla
                except Exception as e:
                    print(f"Erro no cálculo do kernel ({i},{j}): {str(e)}")
                    # Fallback para kernel RBF clássico
                    K[i, j] = np.exp(-0.5 * np.linalg.norm(X[i] - X[j]) ** 2)

            if (i + 1) % 10 == 0:
                print(f"Progresso: {i + 1}/{n} linhas processadas")

        return K

    def quantum_least_squares_solver(self, A, b):
        """
        Solver quântico para sistema linear Ax = b
        Implementação simplificada do quantum linear system algorithm
        """
        # Para esta demonstração, usar solver clássico
        # Em implementação real, usaria HHL algorithm ou variações
        try:
            # Adicionar regularização para estabilidade
            A_reg = A + self.C * np.eye(len(A))
            solution = np.linalg.solve(A_reg, b)
            return solution
        except np.linalg.LinAlgError:
            # Fallback para least squares
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            return solution

    def fit(self, X, y):
        """Treina o QSVM completamente quântico"""
        self.X_train = X.copy()
        self.y_train = y.copy()

        print("Computando matriz de kernel quântica...")
        # Computar matriz de kernel quânticamente
        K = self.quantum_kernel_matrix(X)

        print("Resolvendo sistema linear quântico...")
        # Resolver o problema de otimização como least squares
        # min ||Kα - y||² + λ||α||²

        # Construir matriz do sistema
        n = len(X)
        Y = np.diag(y)  # Matriz diagonal dos labels

        # Sistema: (K + λI)α = y
        self.alpha = self.quantum_least_squares_solver(K, y)

        print(f"Treinamento concluído. Norma dos coeficientes: {np.linalg.norm(self.alpha):.4f}")

    def predict(self, X_test):
        """Faz predições usando o modelo treinado"""
        if self.alpha is None:
            raise ValueError("Modelo não foi treinado!")

        predictions = []

        for x_test in X_test:
            # Calcular soma ponderada dos kernels
            decision_value = 0

            for i, x_train in enumerate(self.X_train):
                # Kernel quântico entre teste e treino
                probs = self.quantum_inner_product(x_test, x_train)
                kernel_val = 2 * probs[0] - 1

                decision_value += self.alpha[i] * self.y_train[i] * kernel_val

            predictions.append(np.sign(decision_value))

        return np.array(predictions)


# =============================================
# QUANTUM VARIATIONAL SVM
# =============================================

class VariationalQuantumSVM:
    """QSVM variacional com ansatz parametrizado"""

    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_type, wires=n_qubits)  # Usar dispositivo otimizado
        self.params = None

        # Criar o QNode uma vez durante a inicialização
        @qml.qnode(self.dev)
        def quantum_classifier_circuit(x, params):
            """Circuito classificador quântico variacional"""
            # Codificação de dados
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)

            # Layers variacionals
            for layer in range(self.n_layers):
                # Rotações parametrizadas
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits * 3 + i * 3], wires=i)
                    qml.RZ(params[layer * self.n_qubits * 3 + i * 3 + 1], wires=i)
                    qml.RX(params[layer * self.n_qubits * 3 + i * 3 + 2], wires=i)

                # Emaranhamento
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])


            # Medição
            return qml.expval(qml.PauliZ(0))

        self.quantum_classifier_circuit = quantum_classifier_circuit

    def cost_function(self, params, X, y):
        """Função de custo para o classificador variacional"""
        predictions = []
        for x in X:
            pred = self.quantum_classifier_circuit(x, params)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Hinge loss
        margins = y * predictions
        hinge_losses = np.maximum(0, 1 - margins)

        return np.mean(hinge_losses)

    def fit(self, X, y, n_epochs=100, lr=0.1):
        """Treina o classificador variacional"""
        n_params = self.n_layers * self.n_qubits * 3
        self.params = np.random.normal(0, 0.1, n_params)

        costs = []

        for epoch in range(n_epochs):
            # Calcular custo atual
            cost = self.cost_function(self.params, X, y)
            costs.append(cost)

            # Gradiente numérico
            grad = np.zeros_like(self.params)
            eps = 1e-3

            for i in range(len(self.params)):
                params_plus = self.params.copy()
                params_plus[i] += eps
                params_minus = self.params.copy()
                params_minus[i] -= eps

                cost_plus = self.cost_function(params_plus, X, y)
                cost_minus = self.cost_function(params_minus, X, y)

                grad[i] = (cost_plus - cost_minus) / (2 * eps)

            # Atualizar parâmetros
            self.params -= lr * grad

            if epoch % 20 == 0:
                print(f"Época {epoch}: Custo = {cost:.4f}")

        return costs

    def predict(self, X):
        """Faz predições"""
        predictions = []
        for x in X:
            pred = self.quantum_classifier_circuit(x, self.params)
            predictions.append(np.sign(pred))

        return np.array(predictions)


# =============================================
# FUNÇÕES DE AVALIAÇÃO
# =============================================

def evaluate_svm(svm_model, X_train, X_test, y_train, y_test, model_name, use_precomputed=False):
    """Avalia um modelo SVM"""
    try:
        if use_precomputed:
            # Para kernels pré-computados
            y_pred = svm_model.predict(X_test)
        else:
            # Para modelos customizados
            if hasattr(svm_model, 'fit'):
                svm_model.fit(X_train, y_train)
            y_pred = svm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n=== Resultados do {model_name} ===")
        print(f"Acurácia: {accuracy:.4f}")

        # Converter labels para nomes das classes para o relatório
        y_test_names = ['Setosa' if label == -1 else 'Versicolor' for label in y_test]
        y_pred_names = ['Setosa' if label == -1 else 'Versicolor' for label in y_pred]

        print("Relatório de classificação:")
        print(classification_report(y_test_names, y_pred_names))

        return accuracy, y_pred

    except Exception as e:
        print(f"\nErro ao avaliar {model_name}: {str(e)}")
        return 0.0, np.zeros_like(y_test)


def evaluate_kernel_svm(K_train, K_test, y_train, y_test, kernel_name):
    """Avalia SVM com kernel pré-computado"""
    try:
        svm = SVC(kernel='precomputed', C=1.0)
        svm.fit(K_train, y_train)
        y_pred = svm.predict(K_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n=== Resultados do Kernel {kernel_name} ===")
        print(f"Acurácia: {accuracy:.4f}")

        # Converter labels para nomes das classes
        y_test_names = ['Setosa' if label == -1 else 'Versicolor' for label in y_test]
        y_pred_names = ['Setosa' if label == -1 else 'Versicolor' for label in y_pred]

        print("Relatório de classificação:")
        print(classification_report(y_test_names, y_pred_names))

        return accuracy, y_pred

    except Exception as e:
        print(f"\nErro ao avaliar kernel {kernel_name}: {str(e)}")
        return 0.0, np.zeros_like(y_test)


# =============================================
# FUNÇÃO PRINCIPAL
# =============================================

def main():
    """Função principal para execução completa"""
    print("=== COMPARAÇÃO COMPLETA: CLÁSSICO vs HÍBRIDO vs COMPLETAMENTE QUÂNTICO ===")
    print("Dataset: Iris (Setosa vs Versicolor)")

    # 1. Carregar dados Iris
    print("\n1. Carregando dataset Iris...")
    X, y, scaler, feature_names = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Conjunto de treino: {len(X_train)} amostras")
    print(f"Conjunto de teste: {len(X_test)} amostras")
    print(f"Distribuição treino: Setosa(-1): {np.sum(y_train == -1)}, Versicolor(+1): {np.sum(y_train == 1)}")
    print(f"Distribuição teste: Setosa(-1): {np.sum(y_test == -1)}, Versicolor(+1): {np.sum(y_test == 1)}")

    results = {}

    # 2. SVMs Clássicos
    print("\n2. Testando SVMs Clássicos...")

    # SVM Linear Clássico
    svm_linear = SVC(kernel='linear', C=1.0)
    acc_linear, _ = evaluate_svm(svm_linear, X_train, X_test, y_train, y_test, "SVM Linear Clássico")
    results['SVM Linear'] = acc_linear

    # SVM RBF Clássico
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
    acc_rbf_sklearn, _ = evaluate_svm(svm_rbf, X_train, X_test, y_train, y_test, "SVM RBF Clássico")
    results['SVM RBF'] = acc_rbf_sklearn

    # Kernel RBF customizado
    print("Computando kernel RBF customizado...")
    K_train_rbf = compute_classical_kernel_matrix(X_train, X_train, rbf_kernel_classical, gamma=0.5)
    K_test_rbf = compute_classical_kernel_matrix(X_test, X_train, rbf_kernel_classical, gamma=0.5)
    acc_rbf_custom, _ = evaluate_kernel_svm(K_train_rbf, K_test_rbf, y_train, y_test, "RBF Customizado")
    results['RBF Customizado'] = acc_rbf_custom

    # 3. QSVM Híbrido (Kernel Quântico)
    print("\n3. Testando QSVM Híbrido (Kernel Quântico)...")
    print("Computando matriz de kernel quântico híbrido...")
    K_train_hybrid = compute_quantum_kernel_matrix_hybrid(X_train, X_train)
    K_test_hybrid = compute_quantum_kernel_matrix_hybrid(X_test, X_train)
    acc_hybrid, _ = evaluate_kernel_svm(K_train_hybrid, K_test_hybrid, y_train, y_test, "Quântico Híbrido")
    results['QSVM Híbrido'] = acc_hybrid

    # 4. QSVM Completamente Quântico
    print("\n4. Testando QSVM Completamente Quântico...")
    try:
        fully_quantum_svm = FullyQuantumSVM(n_qubits=n_qubits, C=1.0)
        acc_fully_quantum, _ = evaluate_svm(fully_quantum_svm, X_train, X_test, y_train, y_test,
                                            "QSVM Completamente Quântico")
        results['QSVM Completo'] = acc_fully_quantum
    except Exception as e:
        print(f"Erro no QSVM Completamente Quântico: {str(e)}")
        results['QSVM Completo'] = 0.0

    # 5. QSVM Variacional
    print("\n5. Testando QSVM Variacional...")
    try:
        variational_svm = VariationalQuantumSVM(n_qubits=n_qubits, n_layers=2)
        print("Treinando classificador variacional...")
        costs = variational_svm.fit(X_train, y_train, n_epochs=100, lr=0.005)  # Reduzido epochs e lr
        acc_variational, _ = evaluate_svm(variational_svm, X_train, X_test, y_train, y_test, "QSVM Variacional",
                                          use_precomputed=False)
        results['QSVM Variacional'] = acc_variational
    except Exception as e:
        print(f"Erro no QSVM Variacional: {str(e)}")
        results['QSVM Variacional'] = 0.0
        costs = [0.5]  # Para evitar erro no plot

    # 6. Resumo dos Resultados
    print("\n=== RESUMO FINAL DOS RESULTADOS ===")
    print("Acurácias obtidas:")
    for model, accuracy in results.items():
        print(f"  {model:<20}: {accuracy:.4f}")

    # 7. Visualização dos Resultados
    plt.figure(figsize=(15, 10))

    # Gráfico de acurácias
    plt.subplot(2, 3, 1)
    models = list(results.keys())
    accuracies = list(results.values())
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown'][:len(models)]

    bars = plt.bar(models, accuracies, color=colors)
    plt.title('Comparação de Acurácias - Dataset Iris')
    plt.ylabel('Acurácia')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    # Evolução do treinamento variacional
    plt.subplot(2, 3, 2)
    if len(costs) > 1:  # Só plotar se tiver dados de treinamento
        plt.plot(costs, 'purple', marker='o', markersize=3)
        plt.title('Treinamento QSVM Variacional')
        plt.xlabel('Época')
        plt.ylabel('Custo (Hinge Loss)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Treinamento\nFalhou', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('QSVM Variacional - Erro')

    # Visualização dos dados originais (2D projection)
    plt.subplot(2, 3, 3)
    colors_data = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors_data, alpha=0.7)
    plt.title('Dataset Iris (Sepal Length vs Width)')
    plt.xlabel('Sepal Length (normalizada)')
    plt.ylabel('Sepal Width (normalizada)')
    plt.grid(True, alpha=0.3)

    # Heatmap da matriz de kernel RBF
    plt.subplot(2, 3, 4)
    plt.imshow(K_train_rbf, cmap='viridis', aspect='auto')
    plt.title('Matriz Kernel RBF')
    plt.colorbar()

    # Heatmap da matriz de kernel quântico
    plt.subplot(2, 3, 5)
    plt.imshow(K_train_hybrid, cmap='plasma', aspect='auto')
    plt.title('Matriz Kernel Quântico')
    plt.colorbar()

    # Comparação direta dos melhores modelos
    plt.subplot(2, 3, 6)
    best_models = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
    names, accs = zip(*best_models)

    colors_top3 = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze em hex
    plt.bar(names, accs, color=colors_top3)
    plt.title('Top 3 Modelos')
    plt.ylabel('Acurácia')
    plt.xticks(rotation=45, ha='right')

    for i, (name, acc) in enumerate(best_models):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return results, X_train, X_test, y_train, y_test


# =============================================
# FUNÇÃO DE EXEMPLO RÁPIDO
# =============================================

def quick_iris_example():
    """Exemplo rápido com Iris"""
    print("=== EXEMPLO RÁPIDO - IRIS ===")

    # Carregar dados
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # SVM Clássico
    svm_classical = SVC(kernel='rbf', C=1.0)
    svm_classical.fit(X_train, y_train)
    y_pred_classical = svm_classical.predict(X_test)
    acc_classical = accuracy_score(y_test, y_pred_classical)

    # QSVM Híbrido
    print("Computando kernel quântico...")
    K_train = compute_quantum_kernel_matrix_hybrid(X_train, X_train)
    K_test = compute_quantum_kernel_matrix_hybrid(X_test, X_train)

    svm_quantum = SVC(kernel='precomputed')
    svm_quantum.fit(K_train, y_train)
    y_pred_quantum = svm_quantum.predict(K_test)
    acc_quantum = accuracy_score(y_test, y_pred_quantum)

    print(f"\nResultados no Iris:")
    print(f"SVM Clássico RBF: {acc_classical:.3f}")
    print(f"QSVM Híbrido:     {acc_quantum:.3f}")

    if acc_quantum > acc_classical:
        print("✓ Vantagem quântica detectada!")
    else:
        print("○ SVM clássico mantém vantagem")


if __name__ == "__main__":
    # Descomente a linha desejada:
    results = main()  # Execução completa
    # quick_iris_example()  # Exemplo rápido