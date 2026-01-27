from dataclasses import dataclass

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal, unitary_group


def check_symplectic(S):
    num_modes = S.shape[0] // 2
    Omega = np.block(
        [
            [np.zeros((num_modes, num_modes)), np.eye(num_modes)],
            [-np.eye(num_modes), np.zeros((num_modes, num_modes))],
        ]
    )
    symplectic_check = S.T @ Omega @ S
    return np.allclose(symplectic_check, Omega, atol=1e-7)


@dataclass
class GaussianUnitary:
    S: np.ndarray
    r: np.ndarray

    def __post_init__(self):
        num_modes = self.S.shape[0] // 2

        if self.S.shape != (num_modes * 2, num_modes * 2) or self.r.shape != (
            2 * num_modes,
        ):
            raise ValueError("Invalid dimensions.")

        if not check_symplectic(self.S):
            raise ValueError("S is not symplectic.")
        self.num_modes = num_modes

    def apply(self, state: "GaussianState") -> "GaussianState":
        new_r = self.S @ state.r + self.r
        new_sigma = self.S @ state.sigma @ self.S.T
        return GaussianState(sigma=new_sigma, r=new_r)


def random_unitary(num_modes, r_scale=0, sqz_scale=1):
    def U_to_S(U):
        X = U.real
        Y = U.imag
        return np.block([[X, -Y], [Y, X]])

    o1 = U_to_S(unitary_group.rvs(num_modes))

    if sqz_scale == 1:
        # Just a passive unitary
        S = o1
    else:
        # Euler decomposition
        o2 = U_to_S(unitary_group.rvs(num_modes))
        sqz = np.exp(np.random.randn(num_modes) * np.log(sqz_scale))
        d = np.diag(np.concatenate([sqz, 1 / sqz]))
        S = o1 @ d @ o2

    r = np.random.randn(2 * num_modes) * r_scale
    return GaussianUnitary(S, r)


class GaussianState:
    def __init__(
        self,
        num_modes=1,
        sigma=None,
        r=None,
    ):
        if sigma is not None:
            num_modes = sigma.shape[0] // 2
            self.r = r
            self.sigma = sigma
        else:
            self.r = np.zeros(2 * num_modes)
            self.sigma = np.eye(2 * num_modes)
        self.num_modes = num_modes

    def set_coherent(self, i, alpha):
        alpha = complex(alpha)
        self.r[i] = alpha.real
        self.r[i + self.num_modes] = alpha.imag

    def set_squeezed_vacuum(self, i, z):
        self.sigma[i, i] = z
        self.sigma[i + self.num_modes, i + self.num_modes] = 1 / z

    def _get_indices(self, modes):
        if modes is None:
            return np.arange(2 * self.num_modes)
        modes = np.atleast_1d(modes)
        return np.concatenate([modes, modes + self.num_modes])

    def transform(self, u, modes=None):
        if type(u) is np.ndarray:
            S, r = u, np.zeros((u.shape[0],))
        else:
            S, r = u.S, u.r
        idx = self._get_indices(modes)
        if S.shape != (len(idx), len(idx)) or r.shape != (len(idx),):
            raise ValueError("Dimensions of S or d do not match the number of modes.")
        self.sigma[np.ix_(idx, idx)] = S @ self.sigma[np.ix_(idx, idx)] @ S.T
        self.r[idx] = S @ self.r[idx] + r

    def sample_heterodyne(self, modes=None, num_samples=1):
        idx = self._get_indices(modes)
        n = len(idx)
        mean = self.r[idx]
        cov = (self.sigma[np.ix_(idx, idx)] + np.eye(n)) / 2
        return multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)

    def sample_homodyne(self, quadrature="x", modes=None, num_samples=1):
        modes = np.arange(self.num_modes) if modes is None else np.atleast_1d(modes)
        if quadrature == "x":
            idx = modes
        elif quadrature == "p":
            idx = modes + self.num_modes
        else:
            raise ValueError("Quadrature must be 'x' or 'p'")

        mean = self.r[idx]
        cov = self.sigma[np.ix_(idx, idx)] / 2
        return multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)

    @property
    def mean_photon_number(self):
        return (
            0.25 * (np.trace(self.sigma)) + 0.5 * self.r @ self.r - self.num_modes * 0.5
        )


def xpxp_to_xxpp(s):
    n = s.shape[0]
    indices = np.arange(n)
    indices[: n // 2] = indices[::2]
    indices[n // 2 :] = indices[: n // 2] + 1
    s = s[indices, :]
    s = s[:, indices]
    return s


def overlap(psi_1: GaussianState, psi_2: GaussianState):
    if not psi_1.num_modes == psi_2.num_modes:
        raise ValueError("Incompatible sizes.")
    Sigma = psi_1.sigma + psi_2.sigma
    num_modes = psi_1.num_modes
    try:
        c, lower = sp.linalg.cho_factor(Sigma, check_finite=True)
    except np.linalg.LinAlgError as e:
        raise ValueError("Sigma matrix must be positive definite") from e

    d = psi_1.r - psi_2.r
    inv_Sigma_d = sp.linalg.cho_solve((c, lower), d, check_finite=True)
    exponent = float(d.T @ inv_Sigma_d)
    logdet_Sigma = 2.0 * np.sum(np.log(np.diag(c)))
    overlap_log = -0.5 * (logdet_Sigma + exponent) + num_modes * np.log(2)
    return np.exp(overlap_log)


def random_coherent_state(num_modes, target_mean_photon_number):
    state = GaussianState(
        sigma=np.eye(num_modes * 2, num_modes * 2), r=np.random.randn(num_modes * 2)
    )
    state.r *= (target_mean_photon_number / state.mean_photon_number) ** 0.5
    return state


def random_state(num_modes, target_mean_photon_number, r_scale=10, sqz_scale=1):
    # Start by applying a random unitary to a vacuum state
    state = random_unitary(num_modes, r_scale, sqz_scale).apply(
        GaussianState(num_modes)
    )

    # Kind of hack-ish NLA to reach the target photon number
    g = np.sqrt(target_mean_photon_number / state.mean_photon_number)
    r = g * state.r
    if sqz_scale == 1:
        sigma = (g**2 * state.sigma) - ((g**2 - 1) * np.eye(2 * state.num_modes))
    return GaussianState(sigma=state.sigma, r=r)


def ec_diamond_norm(
    U: GaussianUnitary,
    V: GaussianUnitary,
    max_n=10_000,
    num_samples=1_000,
    sqz_scale=10,
    method="sampling",
):
    def _ecd_overlap(U: GaussianUnitary, V: GaussianUnitary, psi: GaussianState):
        return 2 * (1 - overlap(U.apply(psi), V.apply(psi))) ** 0.5

    num_modes = U.S.shape[0] // 2
    if method == "sampling":
        # Evaluate the overlap on randomly generated coherent and random gaussian states
        c_states = [random_coherent_state(num_modes, max_n) for _ in range(num_samples)]
        r_states = [
            random_state(num_modes, max_n, 10, sqz_scale) for _ in range(num_samples)
        ]
        r_states = [r for r in r_states if r.mean_photon_number <= max_n]
        return np.max([_ecd_overlap(U, V, state) for state in c_states + r_states])
    else:
        raise NotImplementedError("Not yet!")
