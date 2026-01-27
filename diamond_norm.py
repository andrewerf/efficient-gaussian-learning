import cvxpy as cp
import numpy as np
from gaussian import ec_diamond_norm, random_unitary

from scipy.stats import unitary_group, ortho_group



def make_phi(M):
    def get_n(n):
        return np.eye(1, M, n).T
    phi = 0
    for i in range(M):
        phi += np.kron(get_n(i), get_n(i))

    return phi @ phi.conj().T

def make_N(num_modes, M):
    N = np.zeros((pow(M, num_modes), pow(M, num_modes)))
    for i in range(num_modes):
        n = 1
        for j in range(i):
            n = np.kron(n, np.eye(M))
        n = np.kron(n, np.diag([k for k in range(M)]))
        for j in range(num_modes - i - 1):
            n = np.kron(n, np.eye(M))
        N += n
    return N


# Lemma 5 of https://arxiv.org/pdf/1712.10267
# Alternatively, eq 55 of https://arxiv.org/pdf/1810.12335
def diamond_norm_sdp(U1, U2, H, E, M):
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
    res = diamond_norm_sdp(U1, U2, H, 10, M)
    print("The optimal value is", res)

def example_gaussian():
    num_modes = 1
    np.random.seed(10)
    U1_G = random_unitary(num_modes, 1, 5)
    U2_G = random_unitary(num_modes, 1, 5)
    print(ec_diamond_norm(U1_G, U2_G, 10))

    for cutoff in [4, 6, 8, 10, 12, 14, 16]:
        U1 = U1_G.fock(cutoff)
        U2 = U2_G.fock(cutoff)
        N = make_N(num_modes, cutoff)

        n_sdp = diamond_norm_sdp(U1, U2, N, 10, pow(cutoff, num_modes))
        print(f'Cutoff: {cutoff}, value: {n_sdp}')


example_gaussian()
