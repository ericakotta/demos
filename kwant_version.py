import kwant
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from types import SimpleNamespace

'''
Version of calculating Majorana modes
copied from Topocondmat course source code
(using Kwant)
'''

pauli = SimpleNamespace(
    s0=np.array([[1., 0.], [0., 1.]]),
    sx=np.array([[0., 1.], [1., 0.]]),
    sy=np.array([[0., -1j], [1j, 0.]]),
    sz=np.array([[1., 0.], [0., -1.]])
)

def kitaev_chain(L=None, periodic=False):
    lat = kwant.lattice.chain()

    if L is None:
        sys = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
        L = 1
    else:
        sys = kwant.Builder()

    # transformation to antisymmetric basis
    U = np.array([[1.0, 1.0], [1.j, -1.j]]) / np.sqrt(2)

    def onsite(onsite, p): 
        return - p.mu * U.dot(pauli.sz.dot(U.T.conj()))
    
    for x in range(L):
        sys[lat(x)] = onsite

    def hop(site1, site2, p):
        return U.dot((-p.t * pauli.sz - 1j * p.delta * pauli.sy).dot(U.T.conj()))
    
    sys[kwant.HoppingKind((1,), lat)] = hop

    if periodic:
        def last_hop(site1, site2, p):
            return hop(site1, site2, p) * (1 - 2 * p.lambda_)
        
        sys[lat(0), lat(L - 1)] = last_hop
    return sys

sys = kitaev_chain(L=3, periodic=False).finalized()
p = SimpleNamespace(t=1, delta=4, lambda_=0, mu=0.5)
ham = sys.hamiltonian_submatrix(args=[p])
ev, evec = np.linalg.eigh(ham)

print(np.round(ham,2))

_, axs = plt.subplots(1,2)
sns.heatmap(ax=axs[0], data=np.real(ham), square=True)
sns.heatmap(ax=axs[1], data=np.imag(ham), square=True)
axs[0].set_title('Real')
axs[1].set_title('Imag')
plt.suptitle('Hamiltonian')
plt.show()