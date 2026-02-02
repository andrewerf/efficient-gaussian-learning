import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)


ALIASES = {
    "m": "Number of qumodes",
    "diamond": "Diamond norm error",
    "N_total": "Number of queries",
    "eta": "Probe amplitude $\\eta$",
    "sqz": "Probe squeezing parameter $\\nu$",
    "N_tot": "Number of queries",
    "err_sqz": "$||\\tilde{\\mathbf{r}}_{\\mathrm{SMSV}} - \\mathbf{r}||_2$",
    "err_tms": "$||\\tilde{\\mathbf{r}}_{\\mathrm{TMSV}} - \\mathbf{r}||_2$",
    "err_shared": "$||\\tilde{S}_{\\mathrm{shared}} - S||_\\infty$",
    "err_sym": "$||\\tilde{S}_{\\mathrm{symm.}} - S||_\\infty$",
}


def make_label(k):
    return ALIASES.get(k, k)


def plot_metric(
    df,
    x_var,
    y_var,
    z_axis,
    z_axis_values,
    theory_power,
    theory_label,
    theory_line_position=None,
):
    g = (
        df[df[z_axis].isin(z_axis_values)]
        .groupby([z_axis, x_var])[y_var]
        .quantile([0.25, 0.5, 0.75])
        .unstack(level=-1)
        .reset_index()
    )
    x_values = np.sort(df[x_var].unique()).astype(float)
    if theory_line_position is None:
        theory_line_position = df[df[x_var] == x_values[0]][y_var].quantile(0.95) * 2

    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(z_axis_values)))
    colors = {elem: colors[i] for i, elem in enumerate(z_axis_values)}

    fig, ax = plt.subplots(figsize=(5, 4))
    for m, d in g.groupby(z_axis):
        y = d[0.5]
        yerr = [y - d[0.25], d[0.75] - y]
        x = d[x_var]

        ax.fill_between(x, d[0.25], d[0.75], color=colors[m], alpha=0.1)
        ax.plot(
            x,
            np.array(y),
            color=colors[m],
            marker="+",
            label=f"{z_axis}={int(m)}",
        )
        # ax.errorbar(d["N"], y, yerr=yerr, label=f"m={m}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([x_values[0] * 0.75, x_values[-1] * 1.25])

    theory_line = x_values**theory_power
    theory_line *= theory_line_position / theory_line[0]
    ax.plot(
        x_values,
        theory_line,
        label=theory_label,
        color="k",
        linestyle="--",
    )
    ax.set_xlabel(f"{make_label(x_var)}")
    ax.set_ylabel(f"{make_label(y_var)}")
    ax.legend()
    fig.tight_layout()
    return fig


def fix_dataset(df):
    # In the dataset N records the number of samples per homodyne/heterodyne
    # This is not the total number of queries
    df["N_orig"] = df["N"]
    df["N_total"] = df["N_orig"] * (2 * df["m"] + 2)
    df["N"] = (10 ** np.round(np.log10(df["N_total"]))).astype(int)
    df["diamond"] /= 2  # Scale so that it represents probability
    return df


if __name__ == "__main__":
    df = fix_dataset(pd.read_csv("grid_num_modes_num_samples.csv"))

    fig = plot_metric(
        df,
        x_var="N_total",
        y_var="diamond",
        z_axis="m",
        z_axis_values=[2, 8, 32],
        theory_power=-0.25,
        theory_label="$\\varepsilon = 1/\\sqrt[4]{N}$",
    )
    fig.savefig("simulation_N_scaling_diamond.pdf")
    plt.close()

    fig = plot_metric(
        df,
        x_var="N_total",
        y_var="err_shared",
        z_axis="m",
        z_axis_values=[2, 8, 32],
        theory_power=-0.5,
        theory_label="$\\varepsilon = 1/\\sqrt{N}$",
        theory_line_position=0.1,
    )
    fig.savefig("simulation_N_scaling_err_shared.pdf")
    plt.close()

    fig = plot_metric(
        df,
        x_var="N_total",
        y_var="err_tms",
        z_axis="m",
        z_axis_values=[2, 8, 32],
        theory_power=-0.5,
        theory_label="$\\varepsilon = 1/\\sqrt{N}$",
        theory_line_position=0.5,
    )
    fig.savefig("simulation_N_scaling_err_tms.pdf")
    plt.close()

    # TODO: change err_shared to diamond
    fig = plot_metric(
        df,
        x_var="m",
        y_var="err_shared",
        z_axis="N",
        z_axis_values=[10_000, 100_000, 1000_000],
        theory_power=1.5,
        theory_label="$\\varepsilon = m\\sqrt{m}$",
    )
    fig.savefig("simulation_mode_scaling.pdf")
    plt.close()

    df = fix_dataset(pd.read_csv("grid_sqz_eta.csv"))
    fig = plot_metric(
        df,
        x_var="eta",
        y_var="err_shared",
        z_axis="m",
        z_axis_values=[2, 8, 32],
        theory_power=-1,
        theory_label="$\\varepsilon = 1/\\eta$",
        theory_line_position=0.02,
    )
    fig.savefig("simulation_eta_scaling.pdf")
    plt.close()

    fig = plot_metric(
        df,
        x_var="sqz",
        y_var="err_tms",
        z_axis="m",
        z_axis_values=[2, 8, 32],
        theory_power=-0.5,
        theory_label="$\\varepsilon = 1/\\sqrt{\\nu}$",
    )
    fig.savefig("simulation_tms_scaling.pdf")
    plt.close()

    # plot_mode_number_scaling(df)
    # plot_reverse_confidence(df)
