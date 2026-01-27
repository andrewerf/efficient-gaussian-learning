import cvxpy as cp
import numpy as np

from scipy.stats import unitary_group, ortho_group



def make_phi(M):
    def get_n(n):
        return np.eye(1, M, n).T
    phi = 0
    for i in range(M):
        phi += np.kron(get_n(i), get_n(i))

    return phi @ phi.conj().T

# Lemma 5 of https://arxiv.org/pdf/1712.10267
# Alternatively, eq 55 of https://arxiv.org/pdf/1810.12335
def build_diamond_sdp(U1, U2, H, E, M):
    Phi = make_phi(M)
    Id = np.eye(M)
    J = np.kron(U1, Id) @ Phi @ np.kron(U1.conj().T, Id) - \
        np.kron(U2, Id) @ Phi @ np.kron(U2.conj().T, Id)

    W = cp.Variable((M*M, M*M), hermitian=True)
    rho = cp.Variable((M, M), hermitian=True)
    constraints = [
        0 << W,
             W << cp.kron(Id, rho),
        cp.trace(rho) == 1,
        rho >> 0,
        cp.real(cp.trace(rho @ H)) <= E
    ]
    prog = cp.Problem(cp.Maximize(cp.real(cp.trace(J @ W))), constraints)
    prog.solve(solver='SCS')

    return 2 * prog.value


def example():
    M = 2
    U1 = unitary_group.rvs(M, 1)

    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    U2 = U1 @ R

    H = np.random.randn(M, M)
    H = H + H.T
    res = build_diamond_sdp(U1, U2, H,10, M)
    print("The optimal value is", res)

example()