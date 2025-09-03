"""
VFQ Ultra Mega Quântico: Explorando propriedades genuinamente quânticas (na minha opinião kkkk)
Implementa técnicas avançadas como:
- Quantum embedding em amplitudes, aqui é a normalização em quantum
- Medição de entanglement
- Error mitigation
- Quantum kernel alignment

Ainda precisa validar tudo, não esta rodando, é mais um esboço matemático.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class VariationalFullyQuantum_Ultra:
    """
    Versão ultra-quântica do VFQ que maximiza uso de propriedades quânticas
    """

    def __init__(self, n_qubits=6, n_layers=4,
                 encoding="amplitude",  # 'angle', 'amplitude', 'iqp'
                 ansatz="hardware_efficient",  # 'hardware_efficient', 'strongly_entangling', 'mera'
                 measurement="quantum_kernel",  # 'single', 'multi', 'quantum_kernel'
                 error_mitigation=True,
                 optimizer="spsa",  # SPSA é mais robusto a ruído
                 lr=0.01):

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.ansatz = ansatz
        self.measurement = measurement
        self.error_mitigation = error_mitigation
        self.optimizer = optimizer
        self.lr = lr
        self.params = None

    def _amplitude_encoding(self, x):
        """
        Codifica dados diretamente em amplitudes quânticas
        Permite encoding exponencial: 2^n amplitudes com n qubits
        Esse é o melhor que consegui até o momento tem ainda o 'angle', 'amplitude', 'iqp'
        o angle já estava interessante.
        o QubitStateVector do penylanne parece adequado
        """
        # Normaliza dados para formar vetor de amplitudes válido
        x_norm = x / np.linalg.norm(x)

        # Padding para 2^n dimensões
        target_dim = 2 ** self.n_qubits
        if len(x_norm) < target_dim:
            x_padded = np.zeros(target_dim)
            x_padded[:len(x_norm)] = x_norm
        else:
            x_padded = x_norm[:target_dim]

        # Prepara estado com amplitudes especificadas
        qml.QubitStateVector(x_padded, wires=range(self.n_qubits))

    def _iqp_encoding(self, x):
        """
        Instantaneous Quantum Polynomial encoding
        Cria correlações não-clássicas difíceis de simular
        """
        # Primeira camada: Hadamards para superposição
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        # Segunda camada: rotações baseadas em dados
        for i in range(self.n_qubits):
            qml.RZ(x[i % len(x)], wires=i)

        # Terceira camada: entangling via fases controladas
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                # Correlações quadráticas
                qml.CPhase(x[i % len(x)] * x[j % len(x)], wires=[i, j])

    def _strongly_entangling_ansatz(self, params):
        """
        Ansatz com entanglement máximo
        Explora todo o espaço de Hilbert disponível
        Esse parece o melhor também, mas nào consigo ajustar adequadamente pros qubits vs parametros.
        tem algo faltando ainda
        """
        params = params.reshape(self.n_layers, self.n_qubits, 3)

        for layer in range(self.n_layers):
            # Camada de rotações single-qubit (explora SU(2))
            for i in range(self.n_qubits):
                qml.Rot(params[layer, i, 0],
                       params[layer, i, 1],
                       params[layer, i, 2], wires=i)

            # Entangling: padrão de conectividade completa
            if layer % 2 == 0:
                # Even layers: ladder pattern
                for i in range(0, self.n_qubits - 1, 2):
                    if i + 1 < self.n_qubits:
                        qml.CNOT(wires=[i, i+1])
                for i in range(1, self.n_qubits - 1, 2):
                    if i + 1 < self.n_qubits:
                        qml.CNOT(wires=[i, i+1])
            else:
                # Odd layers: circular + cross connections
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i+1) % self.n_qubits])
                # Cross-connections para qubits distantes
                if self.n_qubits > 3:
                    for i in range(0, self.n_qubits//2):
                        qml.CZ(wires=[i, i + self.n_qubits//2])

    def _mera_ansatz(self, params):
        """
        MERA (Multi-scale Entanglement Renormalization Ansatz)
        Captura correlações em múltiplas escalas - inspirado em tensor networks
        experimentação do MERA, pode dar uma perspectiva melhor para parametrização.
        """
        params = params.reshape(-1, 3)  # Flatten para facilitar indexação
        idx = 0

        # Disentanglers
        for scale in range(int(np.log2(self.n_qubits))):
            step = 2 ** (scale + 1)
            for start in range(0, self.n_qubits, step):
                if start + step//2 < self.n_qubits:
                    # Two-qubit unitary (simplificado)
                    qml.Rot(params[idx, 0], params[idx, 1], params[idx, 2],
                           wires=start)
                    idx = (idx + 1) % len(params)
                    qml.CNOT(wires=[start, start + step//2])

        # Isometries (tree-like structure)
        for scale in range(int(np.log2(self.n_qubits)) - 1, -1, -1):
            step = 2 ** scale
            for start in range(0, self.n_qubits - step, step * 2):
                qml.CNOT(wires=[start, start + step])
                qml.Rot(params[idx, 0], params[idx, 1], params[idx, 2],
                       wires=start + step)
                idx = (idx + 1) % len(params)

    def _quantum_kernel_measurement(self, params):
        """
        Medição baseada em kernel quântico
        Extrai informação máxima do estado quântico
        Aqui uma melhoria, baseado no Efficient quantum kernel estimation (2023)
        """
        measurements = []


        for basis in ['Z', 'X', 'Y']:
            if basis == 'X':
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
            elif basis == 'Y':
                for i in range(self.n_qubits):
                    qml.RX(-np.pi/2, wires=i)

            # Mede expectativas
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))

            # Desfaz rotação de base
            if basis == 'X':
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
            elif basis == 'Y':
                for i in range(self.n_qubits):
                    # te amo PI, espero que faça sentido kkkk
                    # mesmo esquema, pega metade pra medir distancia
                    qml.RX(np.pi/2, wires=i)

        # 2. Correlações de dois corpos
        for i in range(self.n_qubits - 1):
            measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i+1)))

        # 3. Medição de entanglement via pureza parcial
        # (simplificado - mediria entropia de emaranhamento em implementação completa)
        if self.n_qubits >= 4:
            # Mede correlações de longo alcance
            measurements.append(
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(self.n_qubits-1))
            )
        # isso ajuda a fazer medidas sem "apagar" o emaranhamento

        return measurements

    def _error_mitigation_extrapolation(self, circuit_fn, x, params, noise_levels=[1.0, 1.5, 2.0]):
        """
        Zero-noise extrapolation: roda circuito com diferentes níveis de ruído
        e extrapola para ruído zero
        """
        results = []

        for noise_factor in noise_levels:
            # Simula ruído aumentado repetindo gates (simplified)
            # Em hardware real, isso seria feito ajustando tempo de gate

            # Para demonstração, adiciona rotações pequenas como "ruído"
            if noise_factor > 1.0:
                for i in range(self.n_qubits):
                    qml.RZ(0.01 * (noise_factor - 1.0), wires=i)

            result = circuit_fn(x, params)
            results.append(result)

        # Extrapolação linear para noise_factor = 0
        # (Em produção, usaria Richardson extrapolation)
        coeffs = np.polyfit(noise_levels, results, deg=1)
        zero_noise_result = np.polyval(coeffs, 0)

        return zero_noise_result

    def _construct_circuit(self):
        """Constrói circuito quântico completo"""
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x, params):
            # 1. Encoding de dados
            if self.encoding == "amplitude":
                self._amplitude_encoding(x)
            elif self.encoding == "iqp":
                self._iqp_encoding(x)
            else:  # angle encoding padrão
                for i in range(self.n_qubits):
                    qml.RY(x[i % len(x)], wires=i)

            # 2. Ansatz variacional
            if self.ansatz == "strongly_entangling":
                self._strongly_entangling_ansatz(params)
            elif self.ansatz == "mera":
                self._mera_ansatz(params)
            else:  # hardware_efficient padrão
                params_reshaped = params.reshape(self.n_layers, self.n_qubits, 2)
                for layer in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.RY(params_reshaped[layer, i, 0], wires=i)
                        qml.RZ(params_reshaped[layer, i, 1], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])

            # 3. Medição
            if self.measurement == "quantum_kernel":
                return self._quantum_kernel_measurement(params)
            elif self.measurement == "multi":
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            else:
                return qml.expval(qml.PauliZ(0))

        self._circuit = circuit
        return circuit

    def _spsa_optimizer(self, cost_fn, params, n_iterations=100):
        """
        Simultaneous Perturbation Stochastic Approximation
        Mais robusto a ruído quântico que gradientes exatos
        """
        # Hiperparâmetros SPSA
        a = 0.2
        c = 0.1
        A = n_iterations // 10
        alpha = 0.602
        gamma = 0.101

        for k in range(n_iterations):
            # Ajusta learning rates
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma

            # Perturbação aleatória
            delta = np.random.choice([-1, 1], size=len(params))

            # Avalia função em dois pontos perturbados
            params_plus = params + ck * delta
            params_minus = params - ck * delta

            cost_plus = cost_fn(params_plus)
            cost_minus = cost_fn(params_minus)

            # Estimativa de gradiente
            grad_estimate = (cost_plus - cost_minus) / (2 * ck) * delta

            # Atualização
            params = params - ak * grad_estimate

            if (k + 1) % 20 == 0:
                current_cost = cost_fn(params)
                print(f"SPSA iteration {k+1}: Cost = {current_cost:.4f}")

        return params

    def fit(self, X, y, epochs=100):
        """Treinamento com técnicas quânticas avançadas"""
        X = np.asarray(X)
        y = np.asarray(y)

        self._construct_circuit()

        # Inicialização de parâmetros
        if self.ansatz == "mera":
            n_params = 3 * self.n_qubits * 2  # Simplificado
        else:
            n_params_per_qubit = 3 if self.ansatz == "strongly_entangling" else 2
            n_params = self.n_layers * self.n_qubits * n_params_per_qubit

        params = pnp.array(np.random.normal(0, 0.1, n_params), requires_grad=True)

        def cost_fn(params):
            total_loss = 0
            for x_sample, y_sample in zip(X, y):
                if self.error_mitigation:
                    # Com mitigação de erro
                    pred = self._error_mitigation_extrapolation(
                        self._circuit, x_sample, params
                    )
                else:
                    pred = self._circuit(x_sample, params)

                # Processa predições baseado no tipo de medição
                if isinstance(pred, list):
                    pred = np.mean(pred)  # Simplificado, média parece uma boa

                # Loss
                target = 2 * y_sample - 1  # Mapeia 0,1 para -1,1 ; talvez de pra melhorar esse range pra algo mais "quântico"
                loss = (pred - target) ** 2
                total_loss += loss

            return total_loss / len(X)

        if self.optimizer == "spsa":
            # o spsa é mais quantum, mas precisa de testes pra validar minha implementação.
            print("Treinando com SPSA (robusto a ruído)")
            self.params = self._spsa_optimizer(cost_fn, params, epochs)
        else:
            # o Adam mesmo que feito pelo Penylane, ainda não é fullyquantum
            print("Treinando com otimizador quântico padrão")
            opt = qml.AdamOptimizer(self.lr)

            for epoch in range(epochs):
                params, cost = opt.step_and_cost(cost_fn, params)

                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}: Cost = {cost:.4f}")

            self.params = params

        return self

    def predict(self, X):
        """Predição com circuito quântico"""
        X = np.asarray(X)
        predictions = []

        for x in X:
            if self.error_mitigation:
                pred = self._error_mitigation_extrapolation(
                    self._circuit, x, self.params
                )
            else:
                pred = self._circuit(x, self.params)

            if isinstance(pred, list):
                pred = np.mean(pred)

            predictions.append(pred)

        predictions = np.array(predictions)
        return (predictions > 0).astype(int)

    def get_entanglement_measure(self, x):
        """
        Mede o nível de entanglement gerado pelo circuito
        Necessário para fazer debug
        Útil para entender se o modelo está aproveitando propriedades quânticas

        """
        # Prepara circuito
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def entanglement_circuit(x, params):
            # Aplica circuito
            if self.encoding == "amplitude":
                self._amplitude_encoding(x)
            elif self.encoding == "iqp":
                self._iqp_encoding(x)
            else:
                for i in range(self.n_qubits):
                    qml.RY(x[i % len(x)], wires=i)

            # Aplica ansatz
            if self.ansatz == "strongly_entangling":
                self._strongly_entangling_ansatz(params)
            else:
                params_reshaped = params.reshape(self.n_layers, self.n_qubits, 2)
                for layer in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.RY(params_reshaped[layer, i, 0], wires=i)
                        qml.RZ(params_reshaped[layer, i, 1], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])

            return qml.state()

        # Obtém estado quântico
        state = entanglement_circuit(x, self.params)

        # Calcula medida de entanglement simplificada
        # (Meyer-Wallach measure ou similar em implementação completa)
        probs = np.abs(state) ** 2
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normaliza pela entropia máxima
        max_entropy = np.log(2 ** self.n_qubits)
        entanglement = entropy / max_entropy

        return entanglement