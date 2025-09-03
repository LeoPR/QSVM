"""
Otimizadores Quânticos para VFQ
Implementa QNG (Quantum Natural Gradient) e outras estratégias quânticas
Ainda em teste, precisa de mais estudos, ainda tem pézinho classico
Usando Fubini e nós com shift
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QuantumOptimizers:
    """Coleção de otimizadores mais adequados para circuitos quânticos"""

    @staticmethod
    def quantum_natural_gradient(params, grad, metric_tensor, lr=0.01):
        """
        QNG: considera a geometria do espaço de estados quânticos
        Resolve: params_new = params - lr * F^(-1) @ grad
        onde F é o tensor métrico de Fubini-Study
        """
        # Regularização para estabilidade numérica
        F_reg = metric_tensor + 1e-4 * np.eye(len(metric_tensor))

        try:
            # Inverte o tensor métrico
            F_inv = np.linalg.inv(F_reg)
            # Aplica a transformação natural
            natural_grad = F_inv @ grad
        except np.linalg.LinAlgError:
            # Fallback para gradiente vanilla se inversão falhar
            natural_grad = grad

        return params - lr * natural_grad

    @staticmethod
    def rotosolve(circuit, params, param_idx, X, y):
        """
        Rotosolve: encontra ótimo analítico para cada parâmetro de rotação
        Funciona porque a paisagem de loss é sinusoidal em cada parâmetro
        """
        # Para um parâmetro θ, a loss tem forma: a*cos(θ) + b*sin(θ) + c
        # Precisamos avaliar em 3 pontos para encontrar o mínimo

        theta_vals = [0, np.pi/2, np.pi]
        losses = []

        for theta in theta_vals:
            params_temp = params.copy()
            params_temp[param_idx] = theta
            # Calcula loss para este valor
            pred = circuit(X, params_temp)
            loss = np.mean((pred - y) ** 2)
            losses.append(loss)

        # Resolve sistema para encontrar coeficientes a, b, c
        # Loss(θ) = a*cos(θ) + b*sin(θ) + c
        L0, Lpi2, Lpi = losses

        a = (L0 - Lpi) / 2
        b = Lpi2 - (L0 + Lpi) / 2
        c = (L0 + Lpi) / 2

        # Mínimo analítico
        theta_opt = np.arctan2(-b, -a)

        return theta_opt

    @staticmethod
    def qaoa_schedule(epoch, total_epochs, initial_lr=0.1):
        """
        Learning rate schedule inspirado em QAOA
        Começa com steps grandes e refina gradualmente
        """
        # Annealing schedule
        progress = epoch / total_epochs

        if progress < 0.3:
            # Exploração inicial
            return initial_lr
        elif progress < 0.7:
            # Refinamento médio
            return initial_lr * 0.5
        else:
            # Ajuste fino
            return initial_lr * 0.1


class VariationalFullyQuantum_QNG:
    """VFQ com Quantum Natural Gradient"""

    def __init__(self, n_qubits=4, n_layers=3, optimizer="qng", lr=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.lr = lr
        self.params = None

    def _construct_circuit(self):
        """Circuito com capacidade de calcular tensor métrico"""
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x, params):
            # Data encoding com re-uploading
            for layer in range(self.n_layers):
                # Re-upload dados entre camadas (aumenta expressividade)
                for i in range(self.n_qubits):
                    qml.RY(x[i % len(x)] * (layer + 1), wires=i)

                # Camada variacional
                param_idx = layer * self.n_qubits * 2
                for i in range(self.n_qubits):
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1

                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i+1])

            return qml.expval(qml.PauliZ(0))

        # Circuito auxiliar para calcular elementos do tensor métrico
        @qml.qnode(dev)
        def metric_circuit(x, params, i, j):
            """Calcula elemento (i,j) do tensor métrico de Fubini-Study"""
            # Prepara estado com parâmetros
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(x[q % len(x)] * (layer + 1), wires=q)

                param_idx = layer * self.n_qubits * 2
                for q in range(self.n_qubits):
                    qml.RY(params[param_idx], wires=q)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=q)
                    param_idx += 1

                for q in range(self.n_qubits - 1):
                    qml.CZ(wires=[q, q+1])

            # Aqui calcularia o overlap dos gradientes
            # Simplificado para demonstração
            return qml.state()

        self._circuit = circuit
        self._metric_circuit = metric_circuit
        return circuit

    def _compute_metric_tensor(self, x, params):
        """
        Calcula tensor métrico de Fubini-Study
        Este tensor captura a geometria do espaço de estados quânticos
        Tenho que validar melhor essa matemática, a variância de 1/4 parece fazer sentido de acordo
        com Fubini
        """
        n_params = len(params)
        F = np.zeros((n_params, n_params))

        # Para demonstração, usando aproximação diagonal
        # Em implementação completa, calcularíamos overlaps de gradientes
        for i in range(n_params):
            # Elemento diagonal: variância do gerador
            F[i, i] = 0.25  # Para rotações Pauli, a variância é 1/4

        # Adiciona pequenos elementos off-diagonal para correlações
        for i in range(n_params - 1):
            F[i, i+1] = 0.05
            F[i+1, i] = 0.05

        return F

    def fit(self, X, y, epochs=100):
        """Treinamento com Quantum Natural Gradient"""
        X = np.asarray(X)
        y = np.asarray(y)

        self._construct_circuit()

        # Inicialização
        n_params = self.n_layers * self.n_qubits * 2
        params = pnp.array(np.random.normal(0, 0.1, n_params), requires_grad=True)

        if self.optimizer == "qng":
            print("Usando Quantum Natural Gradient")

            for epoch in range(epochs):
                total_loss = 0

                # Compute gradients e metric tensor
                for x_sample, y_sample in zip(X, y):
                    # Forward pass
                    pred = self._circuit(x_sample, params)
                    loss = (pred - (2*y_sample - 1)) ** 2
                    total_loss += loss

                    # Gradiente via parameter-shift
                    grad = qml.grad(self._circuit, argnum=1)
                    g = grad(x_sample, params)

                    # Tensor métrico
                    F = self._compute_metric_tensor(x_sample, params)

                    # Atualização QNG
                    params = QuantumOptimizers.quantum_natural_gradient(
                        params, g, F, lr=self.lr
                    )

                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}: Loss = {total_loss/len(X):.4f}")

        elif self.optimizer == "rotosolve":
            print("Usando Rotosolve (otimização analítica)")

            for epoch in range(epochs):
                # Otimiza cada parâmetro analiticamente
                for param_idx in range(len(params)):
                    # Encontra ótimo para este parâmetro
                    theta_opt = QuantumOptimizers.rotosolve(
                        self._circuit, params, param_idx, X[0], y[0]
                    )
                    params[param_idx] = theta_opt

                if (epoch + 1) % 20 == 0:
                    # Calcula loss total
                    total_loss = 0
                    for x_sample, y_sample in zip(X, y):
                        pred = self._circuit(x_sample, params)
                        loss = (pred - (2*y_sample - 1)) ** 2
                        total_loss += loss
                    print(f"Epoch {epoch+1}: Loss = {total_loss/len(X):.4f}")

        else:  # Adam padrão
            print("Usando Adam optimizer (baseline)")
            opt = qml.AdamOptimizer(self.lr)

            def cost_fn(params):
                total_loss = 0
                for x_sample, y_sample in zip(X, y):
                    pred = self._circuit(x_sample, params)
                    loss = (pred - (2*y_sample - 1)) ** 2
                    total_loss += loss
                return total_loss / len(X)

            for epoch in range(epochs):
                params, loss = opt.step_and_cost(cost_fn, params)

                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

        self.params = params
        return self

    def predict(self, X):
        """Predição usando circuito otimizado"""
        X = np.asarray(X)
        predictions = []

        for x in X:
            pred = self._circuit(x, self.params)
            predictions.append(pred)

        predictions = np.array(predictions)
        # Converte [-1, 1] para [0, 1]
        return (predictions > 0).astype(int)