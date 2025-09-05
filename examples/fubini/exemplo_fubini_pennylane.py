# iris_fubini_mds_4q.py
# ------------------------------------------------------------
# Iris -> 4 qubits -> Fubini–Study -> MDS (com etapas salvas)
# + Bloch por qubit e silhouette score (com matriz de distâncias).
# ------------------------------------------------------------
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # salvar PNG sem abrir janela
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

# ---------- álgebra ----------
I2 = np.eye(2, dtype=complex)
PX = np.array([[0,1],[1,0]], dtype=complex)
PY = np.array([[0,-1j],[1j,0]], dtype=complex)
PZ = np.array([[1,0],[0,-1]], dtype=complex)

def kron(*mats):
    out = np.array([[1+0j]])
    for M in mats:
        out = np.kron(out, M)
    return out

def RY(theta):
    c = np.cos(theta/2.0); s = np.sin(theta/2.0)
    return np.array([[c, -s],[s, c]], dtype=complex)

def RZ(phi):
    return np.array([[np.exp(-1j*phi/2), 0],
                     [0, np.exp(1j*phi/2)]], dtype=complex)

def CNOT(ctrl, targ, n):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bits = [(b >> k) & 1 for k in range(n)]
        if bits[ctrl] == 0:
            b2 = b
        else:
            b2 = b ^ (1 << targ)
        U[b2, b] = 1.0
    return U

def CZ(ctrl, targ, n):
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for b in range(dim):
        if ((b >> ctrl) & 1) and ((b >> targ) & 1):
            U[b,b] = -1
    return U

def local_U(qubit, U1, n):
    mats = [I2]*n
    mats[qubit] = U1
    return kron(*mats)

# ---------- codificação 4 qubits ----------
def encode_state_4q(f4, gamma):
    """
    4 qubits; reuse cruzado + entrelaçamento leve.
      q0: RY(f0*γ), RZ(f1*γ)
      q1: RY(f2*γ), RZ(f3*γ)
      q2: RY(f0*γ), RZ(f2*γ)
      q3: RY(f1*γ), RZ(f3*γ)
    Entanglement: anel de CZ (0-1-2-3-0) + CRZ via CNOT-RZ-CNOT.
    """
    n = 4
    a0,a1,a2,a3 = f4
    U = np.eye(2**n, dtype=complex)
    U = local_U(0, RY(gamma*a0) @ RZ(gamma*a1), n) @ U
    U = local_U(1, RY(gamma*a2) @ RZ(gamma*a3), n) @ U
    U = local_U(2, RY(gamma*a0) @ RZ(gamma*a2), n) @ U
    U = local_U(3, RY(gamma*a1) @ RZ(gamma*a3), n) @ U
    for (c,t) in [(0,1),(1,2),(2,3),(3,0)]:
        U = CZ(c,t,n) @ U
    for (c,t) in [(0,2),(1,3)]:
        U = CNOT(c,t,n) @ U
        U = local_U(t, RZ(gamma*0.3), n) @ U
        U = CNOT(c,t,n) @ U
    psi0 = np.zeros(2**n, dtype=complex); psi0[0] = 1.0
    psi  = U @ psi0
    return psi / np.linalg.norm(psi)

# ---------- Fubini–Study ----------
def fs_distance(psi, phi):
    # d_FS = arccos(|<psi|phi>|)
    ov = np.vdot(psi, phi)
    return float(np.arccos(np.clip(np.abs(ov), 0.0, 1.0)))

def distance_matrix_fs(states):
    n = len(states)
    D = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = fs_distance(states[i], states[j])
            D[i,j] = D[j,i] = d
    return D

# ---------- MDS (com visibilidade das etapas) ----------
def mds_verbose(D, save_prefix):
    """
    Salva D, D^2, B=-1/2 J D^2 J, linhas de D e scree plot.
    Retorna X3 (embedding 3D) e autovalores ordenados.
    """
    n = D.shape[0]
    D2 = D**2
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ D2 @ J

    plt.figure(figsize=(6,5))
    plt.title("D (Fubini–Study distances)")
    plt.imshow(D, interpolation="nearest"); plt.colorbar(label="rad")
    plt.tight_layout(); plt.savefig(f"{save_prefix}_D.png", dpi=130); plt.close()

    plt.figure(figsize=(6,5))
    plt.title("D²")
    plt.imshow(D2, interpolation="nearest"); plt.colorbar()
    plt.tight_layout(); plt.savefig(f"{save_prefix}_D2.png", dpi=130); plt.close()

    plt.figure(figsize=(6,5))
    plt.title("B = -1/2 · J D² J")
    plt.imshow(B, interpolation="nearest"); plt.colorbar()
    plt.tight_layout(); plt.savefig(f"{save_prefix}_B.png", dpi=130); plt.close()

    kshow = min(3, n)
    plt.figure(figsize=(7,4))
    for i in range(kshow):
        plt.plot(D[i], label=f"row {i}")
    plt.title("Algumas linhas de D (vetores de distância)")
    plt.xlabel("índice j"); plt.ylabel("d(i,j)"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{save_prefix}_rows.png", dpi=130); plt.close()

    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]

    plt.figure(figsize=(6,4))
    plt.title("Scree plot — autovalores de B")
    plt.plot(w, marker="o"); plt.xlabel("índice (desc)"); plt.ylabel("autovalor")
    plt.tight_layout(); plt.savefig(f"{save_prefix}_scree.png", dpi=130); plt.close()

    pos = np.clip(w[:3], 0, None)
    L = np.diag(np.sqrt(pos))
    X3 = V[:, :3] @ L

    with open(f"{save_prefix}_J_stats.txt","w", encoding="utf-8") as f:
        f.write("J stats (min,max,mean,std)\n")
        f.write(f"{float(J.min())},{float(J.max())},{float(np.mean(J))},{float(np.std(J))}\n")

    return X3, w

def plot_mds_clusters_3d(X3, y, label_names, title, outpath):
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    classes = np.unique(y)
    for cls in classes:
        sel = (y == cls)
        ax.scatter(X3[sel,0], X3[sel,1], X3[sel,2], s=35, alpha=0.9, label=label_names[cls])
    ax.set_title(title)
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2"); ax.set_zlabel("MDS-3")
    ax.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()

# ---------- Bloch ----------
def reduced_rho_for_qubit(rho_full, k, n):
    resh = rho_full.reshape([2]*n + [2]*n)
    keep = [k]; trace_out = [i for i in range(n) if i != k]
    perm = keep + trace_out + [q + n for q in keep] + [q + n for q in trace_out]
    rho_perm = np.transpose(resh, axes=perm)
    d_keep = 2**len(keep); d_tr = 2**len(trace_out)
    rho_perm = rho_perm.reshape(d_keep, d_tr, d_keep, d_tr)
    rho_red = np.einsum('abcb->ac', rho_perm)
    return rho_red  # 2x2

def bloch_vector(rho1):
    rx = float(np.real(np.trace(rho1 @ PX)))
    ry = float(np.real(np.trace(rho1 @ PY)))
    rz = float(np.real(np.trace(rho1 @ PZ)))
    return np.array([rx, ry, rz])

def plot_bloch(points, y, label_names, title, outpath):
    # esfera wireframe
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(xs, ys, zs, rstride=4, cstride=4, linewidth=0.3, alpha=0.3)

    classes = np.unique(y)
    for cls in classes:
        sel = (y == cls)
        P = points[sel]
        # projeção radial para a casca (se necessário)
        norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        P_proj = P / np.clip(norms, 1.0, None)
        ax.scatter(P_proj[:,0], P_proj[:,1], P_proj[:,2], s=35, alpha=0.9, label=label_names[cls])

    ax.set_title(title)
    ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")
    ax.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()

# ---------- pipeline ----------
def build_features():
    iris = load_iris()
    Xdata = iris.data.copy()        # (150,4)
    y = iris.target.copy()          # 0,1,2
    names = iris.target_names       # ['setosa','versicolor','virginica']

    Xz = StandardScaler().fit_transform(Xdata)
    ry = MinMaxScaler((0.0, np.pi))
    rz = MinMaxScaler((0.0, 2*np.pi))
    f0 = ry.fit_transform(Xz[:, [0]]).ravel()
    f1 = rz.fit_transform(Xz[:, [1]]).ravel()
    f2 = ry.fit_transform(Xz[:, [2]]).ravel()
    f3 = rz.fit_transform(Xz[:, [3]]).ravel()
    feats = np.stack([f0,f1,f2,f3], axis=1)    # (150,4)
    return feats, y, names

def run_case(features, y, names, gamma, save_prefix, verbose=False, title_tag=""):
    states = np.array([encode_state_4q(f, gamma) for f in features])
    D = distance_matrix_fs(states)

    # MDS (verbose opcional)
    if verbose:
        X3, w = mds_verbose(D, f"{save_prefix}_verbose")
    else:
        n = D.shape[0]
        D2 = D**2
        J  = np.eye(n) - np.ones((n,n))/n
        B  = -0.5 * J @ D2 @ J
        w, V = np.linalg.eigh(B)
        idx  = np.argsort(w)[::-1]
        w    = w[idx]; V = V[:, idx]
        L    = np.diag(np.sqrt(np.clip(w[:3],0,None)))
        X3   = V[:, :3] @ L

    # silhouette com matriz de distâncias
    try:
        sil = silhouette_score(D, y, metric="precomputed")
        sil_text = f" | Silhouette={sil:.3f}"
    except Exception:
        sil_text = ""

    plot_mds_clusters_3d(
        X3, y, names,
        f"Iris — MDS 3D via Fubini–Study (4 qubits, γ={gamma}){sil_text}\n{title_tag}",
        f"{save_prefix}_mds_gamma_{gamma:.1f}.png"
    )
    np.savez(f"{save_prefix}_fs_dist_gamma_{gamma:.1f}.npz", D=D, y=y)

def main(args):
    feats, y, names = build_features()

    # 3 classes (com MDS verbose na gamma indicada)
    for g in args.gammas:
        run_case(
            features=feats,
            y=y,
            names=names,
            gamma=g,
            save_prefix=f"{args.save_prefix}_full",
            verbose=(args.verbose_gamma is not None and abs(g-args.verbose_gamma)<1e-12),
            title_tag="(3 classes)"
        )

    # 2 classes (opcional): ex. "setosa,versicolor"
    if args.two_classes:
        name_a, name_b = [s.strip().lower() for s in args.two_classes.split(",")]
        # mapear nomes para índices
        name2idx = {n.lower(): i for i, n in enumerate(names)}
        if name_a not in name2idx or name_b not in name2idx:
            raise SystemExit(f"--two-classes precisa nomes entre {list(name2idx.keys())}")
        keep = {name2idx[name_a], name2idx[name_b]}
        mask = np.array([cls in keep for cls in y], dtype=bool)
        feats2 = feats[mask]; y2 = y[mask]
        # reindexa rótulos para 0/1
        uniq = sorted(np.unique(y2))
        remap = {old:i for i, old in enumerate(uniq)}
        y2m = np.array([remap[v] for v in y2], dtype=int)
        names2 = np.array([names[uniq[0]], names[uniq[1]]])

        for g in args.gammas:
            run_case(
                features=feats2,
                y=y2m,
                names=names2,
                gamma=g,
                save_prefix=f"{args.save_prefix}_twocls",
                verbose=(args.verbose_gamma is not None and abs(g-args.verbose_gamma)<1e-12),
                title_tag=f"(2 classes: {names2[0]} vs {names2[1]})"
            )

    # Bloch (por qubit) para última gamma
    gshow = args.gammas[-1]
    states = np.array([encode_state_4q(f, gshow) for f in feats])
    rho_full_list = [np.outer(psi, np.conjugate(psi)) for psi in states]
    for q in range(4):
        pts = np.zeros((len(states), 3))
        for i, rho in enumerate(rho_full_list):
            rho_q = reduced_rho_for_qubit(rho, q, 4)
            pts[i] = bloch_vector(rho_q)
        plot_bloch(
            pts, y, names,
            f"Iris — Esfera de Bloch (qubit {q}), γ={gshow}",
            f"{args.save_prefix}_bloch_q{q}_gamma_{gshow:.1f}.png"
        )

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Iris + Fubini–Study + MDS (4 qubits, local)")
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 3.0], help="valores de γ")
    p.add_argument("--save-prefix", type=str, default="iris4q", help="prefixo para arquivos de saída")
    p.add_argument("--verbose-gamma", type=float, default=0.8, help="γ em que salvar as etapas do MDS")
    p.add_argument("--two-classes", type=str, default="setosa,versicolor", help='ex.: "setosa,versicolor" (gera também o caso 2 classes)')
    args = p.parse_args()
    main(args)
