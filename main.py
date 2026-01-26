import numpy as np
import scipy as sc
from scipy.linalg import block_diag

from gaussian import (
    GaussianState,
    GaussianUnitary,
    check_symplectic,
    ec_diamond_norm,
    random_coherent_state,
    random_unitary,
    xpxp_to_xxpp,
)


def symplectic_correction(S):
    num_modes = S.shape[0] // 2
    Omega = np.block(
        [
            [np.zeros((num_modes, num_modes)), np.eye(num_modes)],
            [-np.eye(num_modes), np.zeros((num_modes, num_modes))],
        ]
    )
    T = -Omega @ S.T @ Omega @ S
    Q = sc.linalg.sqrtm(T)
    R = S @ np.linalg.inv(Q)
    if not check_symplectic(R):
        raise ValueError("Symplectic correction failed.")
    return R.real


def coherent_probe(U, eta, mode, num_samples):
    num_modes = U.num_modes
    g = GaussianState(num_modes)
    if eta is not None and mode is not None:
        g.set_coherent(mode % num_modes, eta * (1 if mode < num_modes else 1j))
    g.transform(U)
    return g.sample_heterodyne(num_samples=num_samples)


def estimate_symplectic(U, num_samples, eta, kind="symmetric"):
    if kind not in ["symmetric", "shared"]:
        raise ValueError("kind must be 'symmetric' or 'shared'")
    num_modes = U.S.shape[0] // 2
    est_S = np.zeros((num_modes * 2, num_modes * 2))
    scale = 1 if kind == "shared" else 0.5
    if kind == "shared":
        y_vacuum = np.mean(coherent_probe(U, None, None, num_samples), axis=0)

    for i in range(num_modes * 2):
        y = coherent_probe(U, eta, i, num_samples)
        if kind == "symmetric":
            y -= coherent_probe(U, -eta, i, num_samples)
        else:
            y -= y_vacuum
        est_S[:, i] = np.mean(y, axis=0) / eta * scale
    return symplectic_correction(est_S)


def single_mode_squeezed_probe(est_S, U, z, quadrature, num_samples):
    num_modes = U.num_modes
    g = GaussianState(num_modes)
    for i in range(num_modes):
        g.set_squeezed_vacuum(i, z)
    g.transform(np.linalg.inv(est_S))
    g.transform(U)
    return g.sample_homodyne(quadrature, num_samples=num_samples)


def two_mode_squeezed_probe(est_S, U, nu, num_samples):
    num_modes = U.num_modes
    g = GaussianState(num_modes * 2)

    Z = block_diag(*[[[1, 0], [0, -1]] for i in range(num_modes)])
    Snu = np.block(
        [
            [np.sqrt(nu) * np.eye(num_modes * 2), np.sqrt(nu - 1) * Z],
            [np.sqrt(nu - 1) * Z, np.sqrt(nu) * np.eye(num_modes * 2)],
        ]
    )
    Snu = xpxp_to_xxpp(Snu)
    g.transform(Snu)
    g.transform(np.linalg.inv(est_S), modes=range(num_modes))
    g.transform(U, modes=range(num_modes))
    g.transform(np.linalg.inv(Snu))
    return g.sample_heterodyne(num_samples=num_samples, modes=range(num_modes))


def estimate_displacement(U, num_samples, sqz_param, est_S=None, kind="two_mode"):
    if kind not in ["single_mode", "two_mode"]:
        raise ValueError("kind must be 'single_mode' or 'two_mode'")

    if est_S is None:
        est_S = S
    if kind == "two_mode":
        Y = two_mode_squeezed_probe(est_S, U, sqz_param, num_samples)
        return Y.mean(axis=0) / np.sqrt(sqz_param)
    else:
        Yx = single_mode_squeezed_probe(est_S, U, 1 / sqz_param, "x", num_samples)
        Yp = single_mode_squeezed_probe(est_S, U, sqz_param, "p", num_samples)
        return np.concat((Yx.mean(axis=0), Yp.mean(axis=0)))


def main():
    C = random_coherent_state(5, 100)
    print(C)

    num_modes = 2
    np.random.seed(42)

    U = random_unitary(num_modes, 100)
    num_samples = 1000
    eta = 100
    sqz = 10

    def print_cmp(name, A, B):
        if len(A.shape) == 2:
            d = np.linalg.matrix_norm(A - B, ord="fro")
        else:
            d = np.linalg.norm(A - B)
        print(f"{name}: {d}")

    est_S_sym = estimate_symplectic(U, num_samples, eta, kind="symmetric")
    print_cmp("Symplectic symmetric Estimator", est_S_sym, U.S)

    est_S_shared = estimate_symplectic(U, num_samples, eta, kind="shared")
    print_cmp("Symplectic shared Estimator", est_S_shared, U.S)

    est_d_sqz = estimate_displacement(
        U, num_samples, sqz, kind="single_mode", est_S=est_S_sym
    )
    print_cmp("Displacement Singlemode-Squeezed Estimator", est_d_sqz, U.d)

    # TODO(emilie): still fails to generate a symmetric p.s.d matrix sometimes
    est_d_tms = estimate_displacement(
        U, num_samples, sqz, kind="two_mode", est_S=est_S_sym
    )
    print_cmp("Displacement Aux Estimator", est_d_tms, U.d)

    V = GaussianUnitary(est_S_sym, est_d_tms)
    print("ECD:", ec_diamond_norm(U, V, max_n=1e4, method="sampling"))


if __name__ == "__main__":
    main()
