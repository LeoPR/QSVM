"""
Quantum Kernel Measurement Explicado
Demonstração detalhada de como extrair informação máxima de estados quânticos
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def quantum_kernel_measurement_explained(n_qubits=4):
    """
    Demonstração educacional do quantum kernel measurement
    """

    def _quantum_kernel_measurement(circuit_state):
        """
        OBJETIVO: Extrair o máximo de informação clássica útil de um estado quântico
        sem destruir correlações importantes.

        Por que isso é importante?
        - Um estado de n qubits vive em espaço 2^n dimensional
        - Uma única medição nos dá apenas 1 bit de informação
        - Precisamos ser estratégicos sobre COMO medimos
        """
        measurements = []

        # ============================================================
        # PARTE 1: MEDIÇÕES EM DIFERENTES BASES (TOMOGRAFIA PARCIAL)
        # ============================================================
        print("PARTE 1: Mudança de Base Quântica")
        print("-" * 40)

        # Por que medir em diferentes bases?
        # Imagine um spin quântico apontando na direção X.
        # Se você medir apenas em Z, sempre obtém 50/50 aleatório.
        # Mas se medir em X, obtém informação determinística!

        for basis in ['Z', 'X', 'Y']:
            print(f"\nMedindo na base {basis}:")

            if basis == 'X':
                # TRANSFORMAÇÃO PARA BASE X
                # Hadamard rotaciona: |0⟩ → |+⟩, |1⟩ → |−⟩
                # Isso mapeia autoestados de X para autoestados de Z
                print("  Aplicando Hadamard em todos os qubits")
                print("  Efeito: |0⟩ → (|0⟩+|1⟩)/√2, |1⟩ → (|0⟩-|1⟩)/√2")
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)

            elif basis == 'Y':
                # TRANSFORMAÇÃO PARA BASE Y
                # RX(-π/2) rotaciona o estado para medir Y via Z
                # Mapeia autoestados de Y para autoestados de Z
                print("  Aplicando RX(-π/2) em todos os qubits")
                print("  Efeito: Rotaciona da base Y para base Z")
                for i in range(n_qubits):
                    qml.RX(-np.pi / 2, wires=i)

            # Agora medimos em Z (computacional) mas o estado foi rotacionado
            # Isso efetivamente mede na base original desejada
            print(f"  Extraindo expectativas na base {basis}:")
            for i in range(n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
                print(f"    Qubit {i}: ⟨σ_{basis}⟩")

            # DESFAZER A ROTAÇÃO (importante para próximas medições!)
            if basis == 'X':
                print("  Desfazendo Hadamard (voltando à base computacional)")
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
            elif basis == 'Y':
                print("  Desfazendo RX (voltando à base computacional)")
                for i in range(n_qubits):
                    qml.RX(np.pi / 2, wires=i)

        # Por que medimos em 3 bases?
        # - Base Z: informação sobre populações |0⟩ vs |1⟩
        # - Base X: informação sobre coerências (|0⟩+|1⟩) vs (|0⟩-|1⟩)
        # - Base Y: informação sobre coerências com fase imaginária
        # Juntas, elas formam uma representação completa do estado de 1 qubit

        print("\n" + "=" * 50)
        print("PARTE 2: CORRELAÇÕES QUÂNTICAS (ENTANGLEMENT)")
        print("=" * 50)

        # ============================================================
        # PARTE 2: MEDIÇÕES DE CORRELAÇÃO (DETECTANDO ENTANGLEMENT)
        # ============================================================

        # Correlações de dois corpos: ZZ measurements
        print("\nMedindo correlações de dois corpos (entanglement):")
        for i in range(n_qubits - 1):
            measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
            print(f"  ⟨Z_{i} ⊗ Z_{i + 1}⟩: Correlação entre qubits {i} e {i + 1}")

        # O que significa ⟨Zi ⊗ Zj⟩?
        # - Se = +1: qubits estão correlacionados (ambos 0 ou ambos 1)
        # - Se = -1: qubits estão anti-correlacionados (um 0, outro 1)
        # - Se = 0: qubits não estão correlacionados
        #
        # Para estados separáveis: ⟨Zi ⊗ Zj⟩ = ⟨Zi⟩ × ⟨Zj⟩
        # Para estados emaranhados: ⟨Zi ⊗ Zj⟩ ≠ ⟨Zi⟩ × ⟨Zj⟩

        print("\n" + "=" * 50)
        print("PARTE 3: CORRELAÇÕES DE LONGO ALCANCE")
        print("=" * 50)

        # ============================================================
        # PARTE 3: MEDIÇÃO DE ENTANGLEMENT DE LONGO ALCANCE
        # ============================================================

        if n_qubits >= 4:
            print("\nMedindo correlação de longo alcance:")
            measurements.append(
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(n_qubits - 1))
            )
            print(f"  ⟨Z_0 ⊗ Z_{n_qubits - 1}⟩: Correlação entre primeiro e último qubit")
            print("  Isso detecta entanglement 'global' vs apenas vizinhos próximos")

        # Por que correlações de longo alcance importam?
        # - Estados GHZ têm correlação máxima entre qubits distantes
        # - Estados com apenas entanglement local não mostram isso
        # - É uma assinatura de genuíno entanglement multipartite

        return measurements

    # ============================================================
    # INTERPRETAÇÃO MATEMÁTICA
    # ============================================================

    print("\n" + "=" * 60)
    print("INTERPRETAÇÃO MATEMÁTICA DO QUANTUM KERNEL")
    print("=" * 60)

    print("""
    O que estamos realmente fazendo matematicamente?

    1. REPRESENTAÇÃO DO ESTADO:
       |ψ⟩ = Σ αij...k |ijk...⟩

       Tem 2^n amplitudes complexas, mas medições nos dão apenas
       valores reais (expectativas).

    2. KERNEL QUÂNTICO:
       K(x, x') = |⟨ψ(x)|ψ(x')⟩|²

       Mas não calculamos isso diretamente. Em vez disso,
       extraímos "features" ϕ(x) tal que:
       K(x, x') ≈ ϕ(x) · ϕ(x')

    3. VETOR DE FEATURES:
       ϕ(x) = [⟨Z₀⟩, ⟨Z₁⟩, ...,     # Bases Z
                ⟨X₀⟩, ⟨X₁⟩, ...,     # Bases X  
                ⟨Y₀⟩, ⟨Y₁⟩, ...,     # Bases Y
                ⟨Z₀Z₁⟩, ⟨Z₁Z₂⟩, ..., # Correlações
                ⟨Z₀Zₙ₋₁⟩]            # Longo alcance

    4. DIMENSIONALIDADE:
       - Input clássico: d dimensões
       - Features quânticas: 3n + (n-1) + 1 = 4n dimensões
       - Mas o espaço de Hilbert tem 2^n dimensões!

       Estamos projetando exponencial → polinomial,
       mas mantendo as features mais informativas.
    """)

    # ============================================================
    # EXEMPLO CONCRETO
    # ============================================================

    print("\n" + "=" * 60)
    print("EXEMPLO CONCRETO: ESTADOS ESPECIAIS")
    print("=" * 60)

    # Vamos criar diferentes estados e ver suas assinaturas
    dev = qml.device("default.qubit", wires=4)

    # Estado 1: Produto (não emaranhado)
    @qml.qnode(dev)
    def product_state():
        # |0000⟩
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    # Estado 2: Bell state (emaranhado de 2 qubits)
    @qml.qnode(dev)
    def bell_state():
        # (|00⟩ + |11⟩)/√2
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        measurements = []
        # Z measurements
        for i in range(4):
            measurements.append(qml.expval(qml.PauliZ(i)))
        # ZZ correlation
        measurements.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
        return measurements

    # Estado 3: GHZ (emaranhado global)
    @qml.qnode(dev)
    def ghz_state():
        # (|0000⟩ + |1111⟩)/√2
        qml.Hadamard(wires=0)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
        measurements = []
        # Z measurements
        for i in range(4):
            measurements.append(qml.expval(qml.PauliZ(i)))
        # All ZZ correlations
        for i in range(3):
            measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
        # Long-range
        measurements.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(3)))
        return measurements

    print("Estado Produto |0000⟩:")
    prod_meas = product_state()
    print(f"  Z measurements: {prod_meas}")
    print("  → Todos os qubits em |0⟩, sem correlações")

    print("\nEstado Bell (|00⟩+|11⟩)/√2:")
    bell_meas = bell_state()
    print(f"  Z₀, Z₁, Z₂, Z₃: {bell_meas[:4]}")
    print(f"  Z₀Z₁ correlation: {bell_meas[4]:.2f}")
    print("  → Qubits 0,1 perfeitamente correlacionados!")

    print("\nEstado GHZ (|0000⟩+|1111⟩)/√2:")
    ghz_meas = ghz_state()
    print(f"  Z measurements: {ghz_meas[:4]}")
    print(f"  Vizinhos próximos Z₀Z₁: {ghz_meas[4]:.2f}")
    print(f"  Longo alcance Z₀Z₃: {ghz_meas[7]:.2f}")
    print("  → TODOS os qubits correlacionados, mesmo distantes!")

    # ============================================================
    # POR QUE ISSO É "MAIS QUÂNTICO"?
    # ============================================================

    print("\n" + "=" * 60)
    print("POR QUE ISSO É GENUINAMENTE QUÂNTICO?")
    print("=" * 60)

    print("""
    1. COMPLEMENTARIDADE:
       - Medições em X, Y, Z são incompatíveis (não-comutativas)
       - Classicamente, poderíamos medir tudo simultaneamente
       - Quanticamente, cada base revela informação diferente

    2. ENTANGLEMENT:
       - Correlações ZZ podem violar desigualdades de Bell
       - Detectam correlações não-locais impossíveis classicamente
       - ⟨Z₀Zₙ⟩ ≠ ⟨Z₀⟩⟨Zₙ⟩ é assinatura de entanglement

    3. CONTEXTUALIDADE:
       - O valor de uma medição depende de QUAIS outras medições fazemos
       - Não existe um "estado oculto" que determine todos os resultados

    4. INFORMAÇÃO HOLOGRÁFICA:
       - Informação sobre o todo está nas correlações, não nas partes
       - Um qubit sozinho pode ter ⟨Z⟩=0 mas estar maximamente emaranhado

    5. EXPONENCIAL → POLINOMIAL:
       - Estado tem 2^n parâmetros
       - Extraímos O(n) features mais informativas
       - Machine learning clássico processa essas features
       - Mas elas carregam assinatura genuinamente quântica!
    """)

    return _quantum_kernel_measurement


# Demonstração
if __name__ == "__main__":
    quantum_kernel_measurement_explained(n_qubits=4)