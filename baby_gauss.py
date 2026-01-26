import numpy as np
from scipy.stats import multivariate_normal

class GaussianSimulator:
    def __init__(self, num_modes):
        self._num_modes = num_modes
        self._r = np.zeros(2 * self._num_modes)
        self._sigma = np.eye(2 * self._num_modes)
        self._target_mode = None

    def coherent_source(self, i, alpha):
        alpha = complex(alpha)
        self._r[i] = alpha.real
        self._r[i + self._num_modes] = alpha.imag

    def squeezed_source(self, i, z):
        self._sigma[i, i] = z
        self._sigma[i + self._num_modes, i + self._num_modes] = 1 / z

    def _get_indices(self, modes):
        if modes is None:
            return np.arange(2 * self._num_modes)
        modes = np.atleast_1d(modes)
        return np.concatenate([modes, modes + self._num_modes])

    def gaussian_transform(self, S, modes=None):
        idx = self._get_indices(modes)
        if S.shape != (len(idx), len(idx)):
            raise ValueError(f"S must be of shape ({len(idx)}, {len(idx)})")
        self._sigma[np.ix_(idx, idx)] = S @ self._sigma[np.ix_(idx, idx)] @ S.T
        self._r[idx] = S @ self._r[idx]

    def gaussian_unitary(self, decomposed_unitary, modes=None):
        S, d = decomposed_unitary
        idx = self._get_indices(modes)
        if S.shape != (len(idx), len(idx)) or d.shape != (len(idx),):
            raise ValueError("Dimensions of S or d do not match the number of modes.")
        self._sigma[np.ix_(idx, idx)] = S @ self._sigma[np.ix_(idx, idx)] @ S.T
        self._r[idx] = S @ self._r[idx] + d

    def sample_heterodyne(self, modes=None, num_samples=1):
        idx = self._get_indices(modes)
        n = len(idx)
        mean = self._r[idx]
        cov = (self._sigma[np.ix_(idx, idx)] + np.eye(n)) / 2
        return multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)

    def sample_homodyne(self, quadrature='x', modes=None, num_samples=1):
        modes = np.arange(self._num_modes) if modes is None else np.atleast_1d(modes)
        if quadrature == 'x':
            idx = modes
        elif quadrature == 'p':
            idx = modes + self._num_modes
        else:
            raise ValueError("Quadrature must be 'x' or 'p'")

        mean = self._r[idx]
        cov = self._sigma[np.ix_(idx, idx)] / 2
        return multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)
