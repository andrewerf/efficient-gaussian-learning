from dataclasses import dataclass

import numpy as np
import scipy as sc
from scipy.linalg import block_diag
import pickle
import plotly.graph_objects as go
import tikzplotly
import pandas as pd
from collections import defaultdict

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

def get_err_symplectic(kind: str, m: int, z: float, delta: float, N: int, eta: float):
    if kind == 'shared':
        return np.sqrt(324 * m * (z**6) * ( np.sqrt(2 * m) + np.sqrt(2 * np.log(2*m / delta)) )**2 / ( eta**2 * N ))
    elif kind == 'symmetric':
        return np.sqrt(81 * (z**6) * ( 2*np.sqrt(2*m) + np.sqrt(2*np.log(1 / delta)) )**2 / ( 2 * eta**2 * N ))
    else:
        raise ValueError("kind must be 'shared' or 'symmetric'")

@dataclass
class UnitedSamples:
    Ns: int
    Nr: int

@dataclass
class UnitedErrors:
    eps_S: float
    eps_r: float

def get_samples_united(m: int, z: float, nu: float, eta: float, eps_S: float, eps_r: float, delta: float) -> UnitedSamples:
    Ns = get_samples_symplectic('shared', m, z, delta, eps_S, eta)
    Nr = (1 + 2*nu*z*eps_S + 6*(nu*z*eps_S)**2)*(np.sqrt(2*m) + np.sqrt(np.log(2 / delta)))**2 / (nu * eps_r**2)
    return UnitedSamples(int(np.ceil(Ns)), int(np.ceil(Nr)))

def get_errs_united(m: int, z: float, nu: float, eta: float, Ns: int, Nr: int, delta: float) -> UnitedErrors:
    eps_S = get_err_symplectic('shared', m, z, delta, Ns, eta)
    eps_r = np.sqrt((1 + 2*nu*z*eps_S + 6*(nu*z*eps_S)**2)*(np.sqrt(2*m) + np.sqrt(np.log(2 / delta)))**2 / (nu * Nr))
    return UnitedErrors(eps_S, eps_r)


def get_symplectic_estimation_errors(kind: str, num_modes: int, max_squeezing: float, num_samples: int,
                                     delta: float, tau: float | None, N: int | None, eta: float):
    assert( ( tau is None ) != ( N is None ) )

    if tau is not None:
        N = get_samples_symplectic(kind, num_modes, max_squeezing, delta, tau, eta)
        N = int(np.ceil(N))
    if N is not None:
        tau = get_err_symplectic(kind, num_modes, max_squeezing, delta, N, eta)

    errs = []
    for i in range(num_samples):
        U = random_unitary(num_modes, 10, max_squeezing)
        est_S = estimate_symplectic(U, N, eta, kind=kind)
        errs.append(np.linalg.norm(U.S - est_S, ord=2))
    return N, tau, np.asarray(errs)


def get_united_estimation_errors(num_modes: int, max_S_squeezing: float, num_samples: int,
                                 united_samples: UnitedSamples | None, united_errors: UnitedErrors | None,
                                 delta: float, nu: float, eta: float):
    assert( ( united_samples is None ) != ( united_errors is None ) )

    if united_errors is None:
        united_errors = get_errs_united(num_modes, max_S_squeezing, nu, eta, united_samples.Ns, united_samples.Nr, delta)
    else:
        united_samples = get_samples_united(num_modes, max_S_squeezing, nu, eta, united_errors.eps_S, united_errors.eps_r, delta)

    errs_S, errs_r = [], []
    for i in range(num_samples):
        # > In what follows, we focus on the algorithm that combines the vacuum-shared input protocol from
        # > Section 4.1 with the two-mode squeezed vacuum protocol from Section 5.1
        U = random_unitary(num_modes, 10, max_S_squeezing)
        est_S = estimate_symplectic(U, united_samples.Ns, eta, kind='shared')
        est_r = estimate_displacement(U, united_samples.Nr, nu, est_S, 'two_mode')
        errs_S.append(np.linalg.norm(U.S - est_S, ord=2))
        errs_r.append(np.linalg.norm(U.r - est_r, ord=2))
    return united_samples, united_errors, np.stack((errs_S, errs_r), axis=1)


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


def make_symplectic_data(range_modes, range_queries) -> list[SymplecticEstimationPlotData]:
    eta = 1000
    num_samples = 1000
    delta = 0.1
    sq = 3

    ret = []
    for kind in ['symmetric', 'shared']:
        for m in range_modes:
            for N in range_queries:
                N, tau, errs = get_symplectic_estimation_errors(kind, m, sq, num_samples, delta, None, N, eta)
                d = SymplecticEstimationPlotData(
                    kind=kind,
                    eta=eta,
                    num_modes=m,
                    delta=delta,
                    tau=tau,
                    squeezing=sq,
                    N = N,
                    errs = errs,
                )
                mean_err = np.mean(errs)
                print(f'Kind: {kind}\t N: {N}\t tau: {tau:.5f}\t mean_err: {mean_err:.6f}')
                ret.append(d)

    return ret

@dataclass
class UnitedEstimationPlotData:
    num_modes: int
    max_S_squeezing: float
    nu: float
    eta: float
    eps_S: float
    eps_r: float
    delta: float

    N: UnitedSamples
    errs: np.ndarray

def get_united_err_rate(d: UnitedEstimationPlotData):
    corr_samples = d.errs < [d.eps_S, d.eps_r]
    corr_samples = np.logical_and(corr_samples[:, 0], corr_samples[:, 1])
    return 1 - np.mean(corr_samples)



def make_united_data(range_modes, range_queries) -> list[UnitedEstimationPlotData]:
    eta = 1000
    num_samples = 1000
    delta = 0.1
    sq = 3
    nu = 10

    ret = []
    for num_modes in range_modes:
        for N in range_queries:
            _, error_bounds, errs = get_united_estimation_errors(num_modes, sq, num_samples, UnitedSamples(N, N), None, delta, nu, eta)
            d = UnitedEstimationPlotData(
                num_modes=num_modes,
                max_S_squeezing=sq,
                nu=nu,
                eta=eta,
                eps_S=error_bounds.eps_S,
                eps_r=error_bounds.eps_r,
                delta=delta,
                N = N,
                errs = errs
            )
            mean_err_s = np.mean(errs[:, 0])
            mean_err_r = np.mean(errs[:, 1])
            print(f'N: {N}\t eps_S: {error_bounds.eps_S:.5f}\t eps_r: {error_bounds.eps_r:.5f}\t mean_err_S: {mean_err_s:.6f}\t mean_err_r: {mean_err_r:.6f}')
            ret.append(d)
    return ret


# Claude wrote this in 30 secs, and it is a bit terrifying
def create_histograms_by_parameter(data_list: list[SymplecticEstimationPlotData],
                                   target_params):
    """
    Create histograms grouped by unique combinations of parameters (except target_params),
    with stacked semitransparent histograms for each unique combination of target_params values.

    Parameters:
    -----------
    data_list : List[SymplecticEstimationPlotData]
        List of data objects to visualize
    target_params : str or List[str]
        The parameter(s) to vary in each histogram (e.g., 'kind', ['kind', 'squeezing'], etc.)
        If a single string is provided, it will be converted to a list.

    Returns:
    --------
    List of plotly Figure objects
    """
    # All possible parameters
    all_params = ['kind', 'eta', 'num_modes', 'delta', 'tau', 'squeezing', 'N']

    # Convert single string to list
    if isinstance(target_params, str):
        target_params = [target_params]

    # Validate target parameters
    for param in target_params:
        if param not in all_params:
            raise ValueError(f"Each target parameter must be one of {all_params}, got '{param}'")

    # Get grouping parameters (all except targets)
    grouping_params = [p for p in all_params if p not in target_params]

    # Group data by all parameters except target_params
    groups = defaultdict(list)

    for data in data_list:
        # Create key from all parameters except targets
        key = tuple(getattr(data, param) for param in grouping_params)
        groups[key].append(data)

    # Create figures for each group
    figures = []

    # Color palette
    colors = ['rgba(255, 99, 71, 0.5)', 'rgba(30, 144, 255, 0.5)',
              'rgba(50, 205, 50, 0.5)', 'rgba(255, 215, 0, 0.5)',
              'rgba(138, 43, 226, 0.5)', 'rgba(255, 140, 0, 0.5)',
              'rgba(220, 20, 60, 0.5)', 'rgba(65, 105, 225, 0.5)',
              'rgba(255, 182, 193, 0.5)', 'rgba(144, 238, 144, 0.5)',
              'rgba(173, 216, 230, 0.5)', 'rgba(255, 218, 185, 0.5)']

    for key, group_data in groups.items():
        # Create figure
        fig = go.Figure()

        # Add histogram for each unique combination of target parameter values
        for idx, data in enumerate(group_data):
            # Get all target values
            target_values = [getattr(data, param) for param in target_params]

            # Build label from target parameters
            label_parts = []
            for param, value in zip(target_params, target_values):
                if isinstance(value, float):
                    label_parts.append(f'{param}={value:.3f}')
                else:
                    label_parts.append(f'{param}={value}')
            label = ', '.join(label_parts)

            fig.add_trace(go.Histogram(
                x=data.errs,
                name=label,
                opacity=0.85,
                marker_color=colors[idx % len(colors)],
                nbinsx=50
            ))

        # Build subtitle with grouping parameter values
        param_dict = dict(zip(grouping_params, key))
        subtitle_parts = []
        for param in grouping_params:
            value = param_dict[param]
            if isinstance(value, float):
                subtitle_parts.append(f'{param}={value:.3f}')
            else:
                subtitle_parts.append(f'{param}={value}')
        subtitle = ', '.join(subtitle_parts) if subtitle_parts else 'All data'

        # Build title
        if len(target_params) == 1:
            title_text = f'Error Distribution by {target_params[0].capitalize()}'
        else:
            title_text = f'Error Distribution by {", ".join(p.capitalize() for p in target_params)}'

        # Update layout
        fig.update_layout(
            title=f'{title_text}<br><sub>{subtitle}</sub>',
            xaxis_title='Error',
            yaxis_title='Count',
            barmode='overlay',
            template='plotly_white',
            # width=900,
            # height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        figures.append(fig)

    return figures

def plot_with_error_bars(df, target):
    param_cols = [col for col in df.columns if col not in ['mean', 'std', target]]

    fig = go.Figure()

    if param_cols:
        grouped = df.groupby(param_cols)

        for params, group_df in grouped:
            if len(param_cols) == 1:
                label = f"{param_cols[0]}={params}"
            else:
                label = ", ".join([f"{col}={val}" for col, val in zip(param_cols, params)])

            group_df = group_df.sort_values(target)

            fig.add_trace(go.Scatter(
                x=group_df[target],
                y=group_df['mean'],
                error_y=dict(
                    type='data',
                    array=group_df['std'],
                    visible=True
                ),
                mode='lines+markers',
                name=label
            ))
    else:
        df_sorted = df.sort_values(target)
        fig.add_trace(go.Scatter(
            x=df_sorted[target],
            y=df_sorted['mean'],
            error_y=dict(
                type='data',
                array=df_sorted['std'],
                visible=True
            ),
            mode='lines+markers',
            name='Data'
        ))

    fig.update_layout(
        title='Line Plot with Error Bars',
        xaxis_title=target,
        yaxis_title='Mean Error',
        hovermode='closest',
        template='plotly_white'
    )

    return fig

def title2name(title: str) -> str:
    return str.replace(title, '/', '')

if __name__ == "__main__":
    # range_modes = range(2, 10, 2)
    # range_queries = range(100, 1000, 100)
    #
    # symplectic_data = make_symplectic_data(range_modes, range_queries)
    # pickle.dump(symplectic_data, open('symplectic_data.p', 'wb'))
    #
    # united_data = make_united_data(range_modes, range_queries)
    # pickle.dump(united_data, open('united_data.p', 'wb'))

    symplectic_data: list[SymplecticEstimationPlotData] = pickle.load(open('symplectic_data.p', 'rb'))
    df = pd.DataFrame(symplectic_data)

    mean = df['errs'].map(lambda x: np.mean(x))
    std = df['errs'].map(lambda x: np.std(x))

    df = df.drop('errs', axis=1)
    df.insert(len(df.columns), 'mean', mean)
    df.insert(len(df.columns), 'std', std)
    print(df.columns)
    print(df)

    df = df.drop('tau', axis=1)
    fig = plot_with_error_bars(df.loc[(df['kind'] == 'symmetric')], 'N')
    fig.show()