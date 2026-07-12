"""Plot diagnostics saved by ``04_gamma_effect_experiment.py``.

This script deliberately performs no time integration.  It reads the compact
diagnostic arrays from ``results.npz`` so that figure styling can be changed
without rerunning the expensive PDE experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "Time New Roman",
    "font.size": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=float, default=10.5)
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--line-width", type=float, default=1.2)
    parser.add_argument("--font-size", type=float, default=9.0)
    parser.add_argument("--legend-columns", type=int, default=2)
    args = parser.parse_args()

    data = np.load(args.data)
    times = data["times"]
    gammas = data["gammas"]
    abs_r = data["abs_r"]
    omega_error = data["relative_omega_error"]
    enstrophy = data["enstrophy"]
    reference_enstrophy = data["reference_enstrophy"]

    plt.rcParams.update({"font.size": args.font_size})
    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(gammas)))
    for color, gamma, r_values, error_values, ens_values in zip(
        colors, gammas, abs_r, omega_error, enstrophy
    ):
        label = rf"$\gamma={gamma:g}$"
        axes[0].semilogy(times, np.maximum(r_values, 1e-16), color=color,
                         linewidth=args.line_width, label=label)
        axes[1].semilogy(times, np.maximum(error_values, 1e-16), color=color,
                         linewidth=args.line_width)
        axes[2].plot(times, ens_values, color=color, linewidth=args.line_width)

    axes[2].plot(
        times,
        reference_enstrophy,
        color="black",
        linestyle="--",
        linewidth=args.line_width,
        label="ETDRK4 reference",
    )
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel(r"$|r^n|$")
    axes[0].set_ylim(bottom=1e-8)
    axes[0].set_xlim(left=0.0, right=times[-1])
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel(r"relative $L^2$ error")
    axes[1].set_ylim(bottom=1e-9)
    axes[1].set_xlim(left=0.0, right=times[-1])
    axes[2].set_xlabel("$t$")
    axes[2].set_ylabel(r"Enstrophy $\mathcal{E}(t)$")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)

    # Use one shared legend below all three panels.
    gamma_handles, gamma_labels = axes[0].get_legend_handles_labels()
    reference_handles, reference_labels = axes[2].get_legend_handles_labels()
    fig.legend(
        gamma_handles + reference_handles,
        gamma_labels + reference_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=max(6.0, args.font_size - 2.0),
        ncol=3,
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
