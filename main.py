import scipy.linalg
import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import scipy as sc
import cmath
from scipy.stats import unitary_group
from scipy.linalg import block_diag


def random_symplectic(n):
    U = unitary_group.rvs(n)

    X = U.real
    Y = U.imag

    S = np.block([[X, -Y], [Y, X]])
    assert_symplectic(S)

    return S

def symplectic_xp2xx(n):
    R = np.zeros((2*n, 2*n))
    for i in range(n):
        R[i, 2*i] = 1
        R[n + i, 2*i + 1] = 1
    return R


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


def run_symplectic_shared(S, d, num_samples, eta):
    num_modes = S.shape[0] // 2
    eng = sf.Engine("gaussian")

    def run(eta, mode):
        prog = sf.Program(num_modes)
        with prog.context as q:
            for i in range(num_modes):
                ops.Vacuum() | q[i]

            if eta is not None:
                ops.Dgate(eta / 2, 0 if mode < num_modes else np.pi / 2) | q[mode % num_modes]

            ops.GaussianTransform(S) | q
            for i in range(num_modes):
                x_disp = d[i]
                p_disp = d[i + num_modes]
                alpha = x_disp + 1j * p_disp
                ops.Dgate(abs(alpha) / 2, cmath.phase(alpha)) | q[i]

            for i in range(num_modes):
                ops.MeasureHeterodyne() | q[i]
        R = eng.run(prog).samples[0]
        R = 2 * np.concat((np.real(R), np.imag(R)))
        return R

    # The operation MeasureHD has not been implemented in GaussianBackend for the arguments {'shots': 100}.
    # Therefore, using hand-crafted samples loop
    est_S = np.zeros_like(S)
    for k in range(num_samples):
        Y0 = run(None, None)
        SS = []
        for j in range(num_modes*2):
            Yj = run(eta, j)
            SS.append((Yj - Y0) / eta)
        est_S = est_S + np.asarray(SS).T

    est_S = est_S / num_samples

    est_S = symplectic_correction(est_S)

    return est_S


def run_symplectic_symmetric(S, d, num_samples, eta):
    num_modes = S.shape[0] // 2
    eng = sf.Engine("gaussian")

    def run(eta, mode):
        prog = sf.Program(num_modes)
        with prog.context as q:
            for i in range(num_modes):
                ops.Vacuum() | q[i]

            ops.Dgate(eta / 2, 0 if mode < num_modes else np.pi / 2) | q[mode % num_modes]

            ops.GaussianTransform(S) | q
            for i in range(num_modes):
                x_disp = d[i]
                p_disp = d[i + num_modes]
                alpha = x_disp + 1j * p_disp
                ops.Dgate(abs(alpha) / 2, cmath.phase(alpha)) | q[i]

            for i in range(num_modes):
                ops.MeasureHeterodyne() | q[i]
        R = eng.run(prog).samples[0]
        R = 2 * np.concat((np.real(R), np.imag(R)))
        return R

    est_S = []
    for j in range(num_modes*2):
        Y_plus = np.zeros_like(d)
        Y_minus = np.zeros_like(d)
        for k in range(num_samples):
            Y_plus += run(eta, j)
            Y_minus += run(-eta, j)

        est_S.append((Y_plus - Y_minus) / (num_samples * 2 * eta))

    est_S = np.asarray(est_S).T

    est_S = symplectic_correction(est_S)

    return est_S


def run_displacement_aux(S, d, num_samples, nu, est_S=None):
    if est_S is None:
        est_S = S
    num_modes = S.shape[0] // 2
    eng = sf.Engine("gaussian")

    Z = block_diag(*[[[1, 0], [0, -1]] for i in range(num_modes)])
    Snu = np.block([
        [np.sqrt(nu)*np.eye(num_modes*2), np.sqrt(nu - 1)*Z],
        [np.sqrt(nu - 1)*Z, np.sqrt(nu)*np.eye(num_modes*2)]
    ])
    R = symplectic_xp2xx(num_modes*2)

    # change ordering from x1 p1 x2 p2 ... to x1 x2 ... p1 p2 ...
    Snu = R @ Snu @ R.T
    assert_symplectic(Snu)

    def run():
        prog = sf.Program(num_modes*2)
        with prog.context as q:
            ops.GaussianTransform(Snu) | q
            ops.GaussianTransform(np.linalg.inv(est_S)) | q[:num_modes]
            ops.GaussianTransform(S) | q[:num_modes]
            for i in range(num_modes):
                x_disp = d[i]
                p_disp = d[i + num_modes]
                alpha = x_disp + 1j * p_disp
                ops.Dgate(abs(alpha) / 2, cmath.phase(alpha)) | q[i]
            ops.GaussianTransform(np.linalg.inv(Snu)) | q
            for i in range(num_modes):
                ops.MeasureHeterodyne() | q[i]
        R = eng.run(prog).samples[0]
        R = 2 * np.concat((np.real(R), np.imag(R)))
        return R

    est_d = np.zeros_like(d)
    for i in range(num_samples):
        est_d += run()

    est_d = est_d / (num_samples * np.sqrt(nu))
    return est_d

def run_displacement_sq(S, d, num_samples, z, est_S=None):
    if est_S is None:
        est_S = S
    num_modes = S.shape[0] // 2
    eng = sf.Engine("gaussian")

    # momentum- and position-squeezed matrices
    Vp = block_diag(z * np.eye(num_modes), (1/z) * np.eye(num_modes))
    Vx = block_diag((1/z) * np.eye(num_modes), z * np.eye(num_modes))

    def run(V, H):
        prog = sf.Program(num_modes)
        with prog.context as q:
            ops.Gaussian(V, decomp=False) | q
            ops.GaussianTransform(np.linalg.inv(est_S)) | q

            ops.GaussianTransform(S) | q
            for i in range(num_modes):
                x_disp = d[i]
                p_disp = d[i + num_modes]
                alpha = x_disp + 1j * p_disp
                ops.Dgate(abs(alpha) / 2, cmath.phase(alpha)) | q[i]

            for i in range(num_modes):
                H | q[i]
        Y = eng.run(prog).samples[0]
        return Y

    est_d = np.zeros_like(d)
    for i in range(num_samples):
        Yx = run(Vx, ops.MeasureX)
        Yp = run(Vp, ops.MeasureP)
        est_d += np.concat((Yx, Yp), axis=0)

    est_d = est_d / num_samples
    return est_d



def main():
    num_modes = 2
    np.random.seed(123)
    S = random_symplectic(num_modes)
    d = np.random.randn(2 * num_modes) * 1000
    sq = 1000

    pred_S = run_symplectic_symmetric(S, d, 100, sq)
    print(f'Symplectic Symmetric Estimator: {np.linalg.matrix_norm(S - pred_S, ord="fro")}')

    pred_S = run_symplectic_shared(S, d, 100, sq)
    print(f'Symplectic Shared Estimator: {np.linalg.matrix_norm(S - pred_S, ord="fro")}')

    pred_d = run_displacement_aux(S, d, 100, sq)
    print(f'Displacement Aux Estimator: {np.linalg.norm(pred_d - d)}')

    pred_d = run_displacement_sq(S, d, 100, sq)
    print(f'Displacement Singlemode-Squeezed Estimator: {np.linalg.norm(pred_d - d)}')

if __name__ == "__main__":
    main()


