import numpy as np
import scipy as sc
from scipy.stats import unitary_group
from scipy.linalg import block_diag

from baby_gauss import GaussianSimulator, xpxp_to_xxpp

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


def symplectic_correction(S):
    num_modes = S.shape[0] // 2
    Omega = np.block([[np.zeros((num_modes, num_modes)), np.eye(num_modes)],
                      [-np.eye(num_modes), np.zeros((num_modes, num_modes))]])
    T = -Omega @ S.T @ Omega @ S
    Q = sc.linalg.sqrtm(T)
    R = S @ np.linalg.inv(Q)
    assert_symplectic(R)
    return R.real


def coherent_probe(S, d, eta, mode, num_samples):
    num_modes = S.shape[0] // 2
    g = GaussianSimulator(num_modes)
    if eta is not None and mode is not None:
        g.set_coherent(mode % num_modes, eta * (1 if mode < num_modes else 1j))
    g.transform(S, d=d)
    return g.sample_heterodyne(num_samples=num_samples)


def estimate_symplectic(S, d, num_samples, eta, kind="symmetric"):
    if kind not in ["symmetric", "shared"]:
        raise ValueError("kind must be 'symmetric' or 'shared'")
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


def single_mode_squeezed_probe(est_S, S, d, z, quadrature, num_samples):
    num_modes = S.shape[0] // 2
    g = GaussianSimulator(num_modes)
    for i in range(num_modes):
        g.set_squeezed_vacuum(i, z)
    g.transform(np.linalg.inv(est_S))
    g.transform(S, d=d)
    return g.sample_homodyne(quadrature, num_samples=num_samples)


def two_mode_squeezed_probe(est_S, S, d, nu, num_samples):
    num_modes = S.shape[0] // 2
    g = GaussianSimulator(num_modes * 2)

    Z = block_diag(*[[[1, 0], [0, -1]] for i in range(num_modes)])
    Snu = np.block([
        [np.sqrt(nu)*np.eye(num_modes*2), np.sqrt(nu - 1)*Z],
        [np.sqrt(nu - 1)*Z, np.sqrt(nu)*np.eye(num_modes*2)]
    ])
    Snu = xpxp_to_xxpp(Snu)
    g.transform(Snu)
    g.transform(np.linalg.inv(est_S), modes=range(num_modes))
    g.transform(S, d=d, modes=range(num_modes))
    g.transform(np.linalg.inv(Snu))
    return g.sample_heterodyne(num_samples=num_samples, modes=range(num_modes))


def estimate_displacement(S, d, num_samples, sqz_param, est_S=None, kind="two_mode"):
    if kind not in ["single_mode", "two_mode"]:
        raise ValueError("kind must be 'single_mode' or 'two_mode'")

    if est_S is None:
        est_S = S
    if kind == "two_mode":
        Y = two_mode_squeezed_probe(est_S, S, d, sqz_param, num_samples).mean(axis=0)
        return Y / np.sqrt(sqz_param)
    else:
        Yx = single_mode_squeezed_probe(est_S, S, d, 1 / sqz_param, 'x', num_samples).mean(axis=0)
        Yp = single_mode_squeezed_probe(est_S, S, d, sqz_param, 'p', num_samples).mean(axis=0)
        return np.concat((Yx, Yp))


def main():
    num_modes = 2
    np.random.seed(42)

    S = random_symplectic(num_modes)
    d = np.random.randn(2 * num_modes) * 100
    num_samples = 1000

    eta = 100
    sqz = 10

    pred_S_sym = estimate_symplectic(S, d, num_samples, eta, kind="symmetric")
    print(f'Symplectic Symmetric Estimator: {np.linalg.matrix_norm(S - pred_S_sym, ord="fro")}')

    pred_S_shared = estimate_symplectic(S, d, num_samples, eta, kind="shared")
    print(f'Symplectic Shared Estimator: {np.linalg.matrix_norm(S - pred_S_shared, ord="fro")}')

    pred_d_sqz = estimate_displacement(S, d, num_samples, sqz, kind="single_mode", est_S=pred_S_sym)
    print(f'Displacement Singlemode-Squeezed Estimator: {np.linalg.norm(pred_d_sqz - d)}')

    pred_d_tms = estimate_displacement(S, d, num_samples, sqz, kind="two_mode", est_S=pred_S_sym)
    print(f'Displacement Aux Estimator: {np.linalg.norm(pred_d_tms - d)}')


if __name__ == "__main__":
    main()
