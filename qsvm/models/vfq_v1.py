"""
VariationalFullyQuantum Melhorado V1: Múltiplas Medições
Aproveita informação de todos os qubits, não apenas um
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class VariationalFullyQuantum_V1:
    def __init__(self, n_qubits=4, n_layers=3, backend="default.qubit",
                 measurement_strategy="all_z", lr=0.1, epochs=50):
        """
        measurement_strategy:
        - 'single_z': apenas Z no qubit 0
        - 'all_z': mede Z em todos os qubits
        - 'correlations': inclui medições de correlação ZZ
        - 'full_pauli': amostra diferentes bases de Pauli
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.measurement_strategy = measurement_strategy
        self.lr = lr
        self.epochs = epochs
        self.params = None

    def _construct_circuit(self):
        """Constrói circuito com estratégia de medição expandida"""
        dev = qml.device(self.backend, wires=self.n_qubits)

        if self.measurement_strategy == "single_z":
            # Original - apenas um qubit
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(x, params):
                self._apply_encoding(x)
                self._apply_variational_layers(params)
                return qml.expval(qml.PauliZ(0))

        elif self.measurement_strategy == "all_z":
            # Mede todos os qubits e combina informação
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(x, params):
                self._apply_encoding(x)
                self._apply_variational_layers(params)
                # Retorna vetor de expectativas
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        elif self.measurement_strategy == "correlations":
            # Inclui correlações entre qubits
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(x, params):
                self._apply_encoding(x)
                self._apply_variational_layers(params)
                measurements = []
                # Medições locais
                for i in range(self.n_qubits):
                    measurements.append(qml.expval(qml.PauliZ(i)))
                # Correlações de dois corpos
                for i in range(self.n_qubits - 1):
                    measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
                return measurements

        elif self.measurement_strategy == "full_pauli":
            # Amostra diferentes bases - mais informação quântica
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(x, params, basis_choice):
                self._apply_encoding(x)
                self._apply_variational_layers(params)
                # Rotaciona para diferentes bases antes de medir
                if basis_choice == 1:  # Base X
                    for i in range(self.n_qubits):
                        qml.Hadamard(wires=i)
                elif basis_choice == 2:  # Base Y
                    for i in range(self.n_qubits):
                        qml.RX(-np.pi / 2, wires=i)
                # Sempre mede em Z computacional, mas após rotação
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._qnode = circuit
        return circuit

    def _apply_encoding(self, x):
        """Codificação de dados melhorada com re-uploading"""
        # Primeira camada: angle encoding
        for i in range(self.n_qubits):
            qml.RY(x[i % len(x)], wires=i)

        # Data re-uploading: recodifica dados entre camadas variacionais
        # Isso aumenta expressividade sem custo de qubits extras

    def _apply_variational_layers(self, params):
        """Camadas variacionais com diferentes topologias de entanglement"""
        params = params.reshape(self.n_layers, self.n_qubits, 3)

        for layer in range(self.n_layers):
            # Rotações parametrizadas - explora grupo SU(2) completo
            for i in range(self.n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
                qml.RY(params[layer, i, 2], wires=i)  # Decomposição Euler completa

            # Entanglement pattern: alternating layers
            if layer % 2 == 0:
                # Linear connectivity
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            else:
                # All-to-all connectivity (se poucos qubits)
                if self.n_qubits <= 4:
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CZ(wires=[i, j])
                else:
                    # Ring connectivity para muitos qubits
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

    def _process_measurements(self, measurements):
        """Combina múltiplas medições em uma predição"""
        if self.measurement_strategy == "single_z":
            return measurements  # Escalar

        elif self.measurement_strategy == "all_z":
            # Média ponderada com decay
            weights = np.exp(-np.arange(self.n_qubits) / 2.0)
            weights /= weights.sum()
            return np.dot(measurements, weights)

        elif self.measurement_strategy == "correlations":
            # Combina medições locais e correlações
            n_local = self.n_qubits
            local = measurements[:n_local]
            correlations = measurements[n_local:]

            # Peso maior para correlações (capturam entanglement)
            local_score = np.mean(local)
            correlation_score = np.mean(correlations) if correlations else 0
            return 0.3 * local_score + 0.7 * correlation_score

        elif self.measurement_strategy == "full_pauli":
            # Será processado diferentemente no fit
            return measurements

    def fit(self, X, y):
        """Treinamento com estratégias de medição expandidas"""
        X = np.asarray(X)
        y = np.asarray(y)

        self._construct_circuit()

        # Inicialização de parâmetros melhorada
        n_params = self.n_layers * self.n_qubits * 3
        params = pnp.array(
            np.random.normal(0, 0.1, n_params),
            requires_grad=True
        )

        # Otimizador - podemos variar
        opt = qml.AdamOptimizer(self.lr)

        def loss_fn(params, X_batch, y_batch):
            predictions = []

            for x_sample in X_batch:
                if self.measurement_strategy == "full_pauli":
                    # Amostra diferentes bases e combina
                    measurements = []
                    for basis in range(3):  # Z, X, Y
                        m = self._qnode(x_sample, params, basis)
                        measurements.extend(m)
                    pred = np.mean(measurements)  # Simplificado
                else:
                    m = self._qnode(x_sample, params)
                    pred = self._process_measurements(m)

                predictions.append(pred)

            predictions = pnp.array(predictions)

            # Loss function melhorada com regularização
            # Mapeia [-1,1] para [0,1]
            probs = (1 + predictions) / 2

            # Cross entropy com regularização L2
            eps = 1e-8
            ce_loss = -pnp.mean(
                y_batch * pnp.log(probs + eps) +
                (1 - y_batch) * pnp.log(1 - probs + eps)
            )

            # Regularização para evitar overfitting
            l2_reg = 0.01 * pnp.sum(params ** 2)

            return ce_loss + l2_reg

        # Training loop
        for epoch in range(self.epochs):
            params = opt.step(lambda p: loss_fn(p, X, y), params)

            if (epoch + 1) % 10 == 0:
                loss = float(loss_fn(params, X, y))
                print(f"Epoch {epoch + 1}/{self.epochs}: Loss = {loss:.4f}")

        self.params = params
        return self

    def predict(self, X):
        """Predição usando o circuito treinado"""
        X = np.asarray(X)
        predictions = []

        for x in X:
            if self.measurement_strategy == "full_pauli":
                measurements = []
                for basis in range(3):
                    m = self._qnode(x, self.params, basis)
                    measurements.extend(m)
                pred = np.mean(measurements)
            else:
                m = self._qnode(x, self.params)
                pred = self._process_measurements(m)

            predictions.append(pred)

        predictions = np.array(predictions)
        probs = (1 + predictions) / 2
        return (probs >= 0.5).astype(int)