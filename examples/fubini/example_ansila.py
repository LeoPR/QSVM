# mds_classico_vs_quantico_min.py
import numpy as np, matplotlib.pyplot as plt

def classical_mds(D, k=2):
    n=D.shape[0]; J=np.eye(n)-np.ones((n,n))/n
    B=-0.5*J@(D**2)@J
    w,V=np.linalg.eigh(B); idx=w.argsort()[::-1]; w=w[idx]; V=V[:,idx]
    L=np.diag(np.sqrt(np.clip(w[:k],0,None)))
    return V[:,:k]@L

# 1) Quatro pontos "clássicos" em 2D (parece um losango)
Xc = np.array([[0.0,0.0],[1.0,0.2],[0.8,1.2],[-0.2,0.9]])
Dc = np.linalg.norm(Xc[:,None,:]-Xc[None,:,:], axis=-1)

# 2) Mesmos pontos → ângulos de 1 qubit (θ∈[0,π], φ∈[0,2π))
def norm01(v): # normaliza para [0,1]
    a=(v-v.min())/(v.max()-v.min()+1e-12); return a
θ = norm01(Xc[:,0])*np.pi
φ = norm01(Xc[:,1])*2*np.pi

# Estado 1-qubit: |ψ>=cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>
def psi(th, ph):
    return np.array([np.cos(th/2), np.exp(1j*ph)*np.sin(th/2)], dtype=complex)
Psis = np.array([psi(th,ph) for th,ph in zip(θ,φ)])

# Fubini–Study: d_FS = arccos(|<ψ|φ>|)
def fs_dist(psi,phi):
    ov=np.vdot(psi,phi); return float(np.arccos(np.clip(np.abs(ov),0,1)))
n=len(Psis); Dfs=np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        d=fs_dist(Psis[i],Psis[j]); Dfs[i,j]=Dfs[j,i]=d

# 3) Faz MDS em 2D nas duas matrizes de distância
Yc = classical_mds(Dc, k=2)
Yq = classical_mds(Dfs, k=2)

# 4) Plota lado a lado
fig,axs=plt.subplots(1,2,figsize=(9,4))
for ax,Y,title in [(axs[0],Yc,"MDS clássico (euclidiano)"),
                   (axs[1],Yq,"MDS 'quântico' (Fubini–Study)")]:
    ax.scatter(Y[:,0],Y[:,1],s=80)
    for i in range(n): ax.text(Y[i,0]+0.02,Y[i,1]+0.02,f"P{i}")
    # desenha arestas com transparência proporcional à distância
    D = Dc if "clássico" in title else Dfs
    Dn = D/D.max()
    for i in range(n):
        for j in range(i+1,n):
            ax.plot([Y[i,0],Y[j,0]],[Y[i,1],Y[j,1]],alpha=0.25*(1-Dn[i,j]),lw=2)
    ax.set_title(title); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.show()
