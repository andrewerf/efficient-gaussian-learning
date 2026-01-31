from dataclasses import dataclass

import numpy as np
import scipy as sc
from scipy.linalg import block_diag
import pickle

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
    num_modes = U.num_modes
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
        return np.concatenate((Yx.mean(axis=0), Yp.mean(axis=0)))


def main():
    num_modes = 2
    np.random.seed(42)

    num_samples = 100000
    eta = 10000
    sqz = 10

    U = random_unitary(num_modes, 100, sqz)

    def print_cmp(name, A, B):
        if len(A.shape) == 2:
            d = np.linalg.norm(A - B, ord="fro")
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
    print_cmp("Displacement Singlemode-Squeezed Estimator", est_d_sqz, U.r)

    # TODO(emilie): still fails to generate a symmetric p.s.d matrix sometimes
    est_d_tms = estimate_displacement(
        U, num_samples, sqz, kind="two_mode", est_S=est_S_sym
    )
    print_cmp("Displacement Aux Estimator", est_d_tms, U.r)

    V = GaussianUnitary(est_S_sym, est_d_tms)
    print("ECD:", ec_diamond_norm(U, V, max_n=1e4, method="sampling", sqz_scale=sqz))



# Returns the lower bound for the number of samples Ns
def get_samples_symplectic(kind: str, m: int, z: float, delta: float, tau: float, eta: float):
    if kind == 'shared':
        # Based on Proposition 4.4
        return 324 * m * (z**6) * ( np.sqrt(2 * m) + np.sqrt(2 * np.log(2*m / delta)) )**2 / ( eta**2 * tau**2 )
    elif kind == 'symmetric':
        # Based on Proposition 4.5
        return 81 * (z**6) * ( 2*np.sqrt(2*m) + np.sqrt(2*np.log(1 / delta)) )**2 / ( 2 * eta**2 * tau**2 )
    else:
        raise ValueError("kind must be 'shared' or 'symmetric'")


def get_symplectic_estimation_errors(kind: str,  num_modes: int, max_squeezing: float, num_samples: int, delta: float, tau: float, eta: float):
    errs = []
    N = get_samples_symplectic(kind, num_modes, max_squeezing, delta, tau, eta)
    N = int(np.ceil(N))
    for i in range(num_samples):
        U = random_unitary(num_modes, 10, max_squeezing)
        est_S = estimate_symplectic(U, N, eta, kind=kind)
        errs.append(np.linalg.norm(U.S - est_S, ord=2))
    return N, np.asarray(errs)


@dataclass
class SymplecticEstimationPlotData:
    kind: str
    eta: float
    num_modes: int
    delta: float
    tau: float
    squeezing: float

    N: int
    errs: np.ndarray
    
def get_err_rate(d: SymplecticEstimationPlotData):
    return np.mean(d.errs > d.tau)


def make_symplectic_data() -> list[SymplecticEstimationPlotData]:
    eta = 100
    num_modes = 2
    num_samples = 5000
    delta = 0.1
    tau = 0.1
    max_squeezing = 3
    squeezing_steps = 3

    ret = []
    for kind in ['symmetric', 'shared']:
        for sq in np.arange(1, max_squeezing, max_squeezing / squeezing_steps):
            N, errs = get_symplectic_estimation_errors(kind, num_modes, sq, num_samples, delta, tau, eta)
            d = SymplecticEstimationPlotData(
                kind=kind,
                eta=eta,
                num_modes=num_modes,
                delta=delta,
                tau=tau,
                squeezing=sq,
                N = N,
                errs = errs,
            )
            err_rate = get_err_rate(d)
            print(f'Kind: {kind}\t N: {N}\t err rate: {err_rate}')
            ret.append(d)

    return ret


if __name__ == "__main__":
    symplectic_data = make_symplectic_data()
    pickle.dump(symplectic_data, open("symplectic_data.pickle", "wb"))