import numpy as np
import scipy as sc
from scipy.stats import unitary_group
from scipy.linalg import block_diag

from baby_gauss import GaussianSimulator

def random_symplectic(n):
    U = unitary_group.rvs(n)
    X = U.real
    Y = U.imag
    S = np.block([[X, -Y], [Y, X]])
    assert_symplectic(S)
    return S


def assert_symplectic(S):
    num_modes = S.shape[0] // 2
    Omega = np.block([[np.zeros((num_modes, num_modes)), np.eye(num_modes)],
                      [-np.eye(num_modes), np.zeros((num_modes, num_modes))]])
    symplectic_check = S.T @ Omega @ S
    d = np.linalg.matrix_norm(symplectic_check - Omega, ord='fro')
    assert np.allclose(symplectic_check, Omega), f'Omega L2-distance: {d}'


# Un grand merci à Émilie
def symplectic_correction(S):
    num_modes = S.shape[0] // 2
    Omega = np.block([[np.zeros((num_modes, num_modes)), np.eye(num_modes)],
                      [-np.eye(num_modes), np.zeros((num_modes, num_modes))]])
    T = -Omega @ S.T @ Omega @ S
    Q = sc.linalg.sqrtm(T)
    R =  S @ np.linalg.inv(Q)
    assert_symplectic(R)
    return R


def coherent_probe(S, d, eta, mode, num_samples):
    num_modes = S.shape[0] // 2
    g = GaussianSimulator(num_modes)
    if eta is not None and mode is not None:
        g.coherent_source(mode % num_modes, eta * (1 if mode < num_modes else 1j))
    g.gaussian_unitary((S, d))
    return g.sample_heterodyne(num_samples=num_samples)


def run_symplectic(S, d, num_samples, eta, kind):
    num_modes = S.shape[0] // 2
    est_S = np.zeros((num_modes * 2, num_modes * 2))
    scale = 1 if kind == "shared" else 0.5
    if kind == "shared":
        y_vacuum = np.mean(coherent_probe(S, d, None, None, num_samples), axis=0)

    for i in range(num_modes * 2):
        y = coherent_probe(S, d, eta, i, num_samples)
        if kind == "symmetric":
            y -= coherent_probe(S, d, -eta, i, num_samples)
        else:
            y -= y_vacuum
        est_S[:, i] = np.mean(y, axis=0) / eta * scale
    return symplectic_correction(est_S)


def squeezed_probe(est_S, S, d, z, quadrature, num_samples):
    num_modes = S.shape[0] // 2
    g = GaussianSimulator(num_modes)
    for i in range(num_modes):
        g.squeezed_source(i, z)
    g.gaussian_transform(np.linalg.inv(est_S))
    g.gaussian_unitary((S, d))
    return g.sample_homodyne(quadrature, num_samples=num_samples)


def two_mode_squeezed_probe(est_S, S, d, nu, num_samples):
    num_modes = S.shape[0] // 2
    g = GaussianSimulator(num_modes * 2)

    def xpxp_to_xxpp(s):
        n = s.shape[0]
        indices = np.arange(n)
        indices[:n // 2] = indices[::2]
        indices[n // 2:] = indices[:n // 2] + 1
        s = s[indices, :]
        s = s[:, indices]
        return s

    Z = block_diag(*[[[1, 0], [0, -1]] for i in range(num_modes)])
    Snu = np.block([
        [np.sqrt(nu)*np.eye(num_modes*2), np.sqrt(nu - 1)*Z],
        [np.sqrt(nu - 1)*Z, np.sqrt(nu)*np.eye(num_modes*2)]
    ])
    Snu = xpxp_to_xxpp(Snu)
    g.gaussian_transform(Snu)
    g.gaussian_transform(np.linalg.inv(est_S), modes=range(num_modes))
    g.gaussian_unitary((S, d), modes=range(num_modes))
    g.gaussian_transform(np.linalg.inv(Snu))
    return g.sample_heterodyne(num_samples=num_samples, modes=range(num_modes))


def run_displacement(S, d, num_samples, sqz_param, est_S=None, kind="two_mode"):
    if est_S is None:
        est_S = S
    if kind == "two_mode":
        Y = two_mode_squeezed_probe(est_S, S, d, sqz_param, num_samples).mean(axis=0)
        return Y / np.sqrt(sqz_param)
    else:
        Yx = squeezed_probe(est_S, S, d, 1 / sqz_param, 'x', num_samples).mean(axis=0)
        Yp = squeezed_probe(est_S, S, d, sqz_param, 'p', num_samples).mean(axis=0)
        return np.concat((Yx, Yp))


def main():
    num_modes = 2
    np.random.seed(123)
    S = random_symplectic(num_modes)
    d = np.random.randn(2 * num_modes) * 1000
    sq = 1000
    num_samples = 1000

    pred_S = run_symplectic(S, d, num_samples, sq, kind="symmetric")
    print(f'Symplectic Symmetric Estimator: {np.linalg.matrix_norm(S - pred_S, ord="fro")}')

    pred_S = run_symplectic(S, d, num_samples, sq, kind="shared")
    print(f'Symplectic Shared Estimator: {np.linalg.matrix_norm(S - pred_S, ord="fro")}')

    pred_d = run_displacement(S, d, num_samples, sq, kind="two_mode")
    print(f'Displacement Aux Estimator: {np.linalg.norm(pred_d - d)}')

    pred_d = run_displacement(S, d, num_samples, sq, kind="single_mode")
    print(f'Displacement Singlemode-Squeezed Estimator: {np.linalg.norm(pred_d - d)}')

if __name__ == "__main__":
    main()
