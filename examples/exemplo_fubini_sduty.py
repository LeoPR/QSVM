import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # salva PNG sem abrir janela
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ativa 3D)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pennylane as qml

# -----------------------------
# 1) Carrega e prepara o IRIS
# -----------------------------
iris = load_iris()
X = iris.data.copy()      # shape (150, 4)  -> [sepal len, sepal wid, petal len, petal wid]
y = iris.target.copy()    # 0,1,2
labels = iris.target_names

# Normalizações:
#   - para RY (ângulo polar): [0, pi]
#   - para RZ (ângulo azimutal): [0, 2pi)
scaler_standard = StandardScaler().fit(X)
Xz = scaler_standard.transform(X)

# mapeia colunas: f0,f1 no qubit0 ; f2,f3 no qubit1
# RY usa faixa [0, pi], RZ usa [0, 2pi]
ry_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
rz_scaler = MinMaxScaler(feature_range=(0.0, 2*np.pi))

f0 = ry_scaler.fit_transform(Xz[:, [0]]).ravel()
f1 = rz_scaler.fit_transform(Xz[:, [1]]).ravel()
f2 = ry_scaler.fit_transform(Xz[:, [2]]).ravel()
f3 = rz_scaler.fit_transform(Xz[:, [3]]).ravel()

features = np.stack([f0, f1, f2, f3], axis=1)  # (150, 4)

# -----------------------------
# 2) Define circuito (2 qubits)
# -----------------------------
dev = qml.device("default.qubit", wires=2)

def prepare(features4):
    """Angle encoding simples:
       - qubit 0: RY(f0) -> RZ(f1)
       - qubit 1: RY(f2) -> RZ(f3)
       - entanglement leve: CZ(0,1)
    """
    qml.RY(features4[0], wires=0)
    qml.RZ(features4[1], wires=0)
    qml.RY(features4[2], wires=1)
    qml.RZ(features4[3], wires=1)
    qml.CZ(wires=[0, 1])

@qml.qnode(dev, interface=None)
def state_qnode(f4):
    prepare(f4)
    return qml.state()  # vetor de estado |psi> (dim 4)

@qml.qnode(dev, interface=None)
def bloch_qnode(f4):
    """Retorna (⟨X⟩,⟨Y⟩,⟨Z⟩) de cada qubit."""
    prepare(f4)
    return (
        qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(1)), qml.expval(qml.PauliZ(1)),
    )

def fs_distance(psi, phi):
    """d_FS = arccos(|<psi|phi>|)"""
    ov = np.vdot(psi, phi)
    return float(np.arccos(np.clip(np.abs(ov), 0.0, 1.0)))

# ------------------------------------------
# 3) Gera estados, Bloch vectors e distâncias
# ------------------------------------------
states = np.array([state_qnode(f) for f in features])               # (150, 4)
bloch = np.array([bloch_qnode(f) for f in features])                # (150, 6)
bloch_q0 = bloch[:, 0:3]  # (rx, ry, rz) qubit 0
bloch_q1 = bloch[:, 3:6]  # (rx, ry, rz) qubit 1

# matriz de distâncias de Fubini–Study (pode levar alguns segundos para 150x150)
n = states.shape[0]
D = np.zeros((n, n), dtype=float)
for i in range(n):
    # preenche só triângulo superior para eficiência
    for j in range(i+1, n):
        D[i, j] = D[j, i] = fs_distance(states[i], states[j])

# ------------------------------------------
# 4) Salva dados e faz os gráficos
# ------------------------------------------
# CSV com Bloch vectors e rótulos
df = pd.DataFrame({
    "label": [labels[t] for t in y],
    "y": y,
    "q0_rx": bloch_q0[:, 0], "q0_ry": bloch_q0[:, 1], "q0_rz": bloch_q0[:, 2],
    "q1_rx": bloch_q1[:, 0], "q1_ry": bloch_q1[:, 1], "q1_rz": bloch_q1[:, 2],
})
df.to_csv("./iris_pennylane_bloch.csv", index=False)

np.savez("./iris_fs_distance_matrix.npz", D=D, y=y)

# --- Heatmap das distâncias FS ---
plt.figure(figsize=(7,6))
plt.title("Iris — Fubini–Study distance matrix (2 qubits, angle encoding)")
plt.imshow(D, interpolation="nearest")
plt.colorbar(label="distance (radians)")
plt.xlabel("sample index")
plt.ylabel("sample index")
plt.tight_layout()
plt.savefig("./iris_fs_heatmap.png", dpi=140)
plt.close()

# --- Scatter 3D Bloch qubit 0 ---
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection="3d")
cmap = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
for cls in np.unique(y):
    sel = (y == cls)
    ax.scatter(bloch_q0[sel,0], bloch_q0[sel,1], bloch_q0[sel,2],
               s=35, alpha=0.8, label=labels[cls], color=cmap[cls])
ax.set_title("Iris — Bloch vectors (qubit 0)")
ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")
ax.legend()
plt.tight_layout()
plt.savefig("./iris_bloch_q0.png", dpi=140)
plt.close()

# --- Scatter 3D Bloch qubit 1 ---
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection="3d")
for cls in np.unique(y):
    sel = (y == cls)
    ax.scatter(bloch_q1[sel,0], bloch_q1[sel,1], bloch_q1[sel,2],
               s=35, alpha=0.8, label=labels[cls], color=cmap[cls])
ax.set_title("Iris — Bloch vectors (qubit 1)")
ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")
ax.legend()
plt.tight_layout()
plt.savefig("./iris_bloch_q1.png", dpi=140)
plt.close()

print("Arquivos salvos na pasta atual:")
print(" - iris_pennylane_bloch.csv")
print(" - iris_fs_distance_matrix.npz")
print(" - iris_fs_heatmap.png")
print(" - iris_bloch_q0.png")
print(" - iris_bloch_q1.png")
