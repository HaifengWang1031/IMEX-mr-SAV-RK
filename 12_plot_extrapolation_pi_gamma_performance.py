"""Plot diagnostics for extrapolation PI-gamma performance data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def configure_style() -> None:
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.family": "Times New Roman",
            "font.size": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


ADAPTIVE_STYLE = {"color": "C0", "linestyle": "-", "marker": "o"}
FIXED_GAMMA_STYLE = {"color": "C3", "linestyle": "--", "marker": "s"}
FIXED_STYLES = (
    {"color": "C1", "linestyle": "-.", "marker": "*"},
    {"color": "C8", "linestyle": "--", "marker": "d"},
    {"color": "C9", "linestyle": "--", "marker": "p"},
    {"color": "C2", "linestyle": ":", "marker": "^"},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    data = np.load(args.data)
    output_times = data["output_times"]
    adaptive_t = data["adaptive_t"]
    adaptive_dt = data["adaptive_dt"]
    adaptive_gamma = data["adaptive_gamma"]
    adaptive_q_error = np.abs(1.0 - data["adaptive_q"])
    fixed_gamma_t = data["fixed_gamma_t"]
    fixed_gamma_dt = data["fixed_gamma_dt"]
    fixed_gamma_q_error = np.abs(1.0 - data["fixed_gamma_q"])

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    mark_adaptive = max(1, len(adaptive_dt) // 35)
    mark_fixed = max(1, len(fixed_gamma_dt) // 35)

    axes[0, 0].plot(
        adaptive_t[1:], adaptive_dt, label=r"adaptive $\gamma$ + PI",
        linewidth=1.1, markersize=3, markevery=mark_adaptive, **ADAPTIVE_STYLE,
    )
    axes[0, 0].plot(
        fixed_gamma_t[1:], fixed_gamma_dt, label=r"fixed $\gamma$ + extrapolation I",
        linewidth=1.1, markersize=3, markevery=mark_fixed, **FIXED_GAMMA_STYLE,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"Step size $\tau_n$")
    axes[0, 0].legend(frameon=False, fontsize=9)

    axes[0, 1].plot(
        adaptive_t[1:], adaptive_gamma, label=r"adaptive $\gamma_n$",
        linewidth=1.1, markersize=3, markevery=mark_adaptive, **ADAPTIVE_STYLE,
    )
    axes[0, 1].axhline(
        float(data["gamma_fixed"]), color="C3", linestyle="--", linewidth=1.1,
        label=r"fixed $\gamma$",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"Mean-reversion parameter $\gamma_n$")
    axes[0, 1].legend(frameon=False, fontsize=9)

    axes[0, 2].plot(
        adaptive_t[1:], np.maximum(adaptive_q_error[1:], 1.0e-16),
        label=r"adaptive $\gamma$ + PI", linewidth=1.1, markersize=3,
        markevery=mark_adaptive, **ADAPTIVE_STYLE,
    )
    axes[0, 2].plot(
        fixed_gamma_t[1:], np.maximum(fixed_gamma_q_error[1:], 1.0e-16),
        label=r"fixed $\gamma$ + extrapolation I", linewidth=1.1, markersize=3,
        markevery=mark_fixed, **FIXED_GAMMA_STYLE,
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylabel(r"Auxiliary drift $|1-q|$ (diagnostic)")
    axes[0, 2].legend(frameon=False, fontsize=9)

    adaptive_eomega = data["adaptive_error_omega"]
    adaptive_er = data["adaptive_error_r"]
    valid_omega = np.isfinite(adaptive_eomega)
    valid_r = np.isfinite(adaptive_er)
    axes[1, 0].plot(
        adaptive_t[1:][valid_omega], np.maximum(adaptive_eomega[valid_omega], 1.0e-16),
        label=r"$e_\omega$", linewidth=1.1, markersize=3,
        markevery=mark_adaptive, color="C0", marker="o",
    )
    axes[1, 0].plot(
        adaptive_t[1:][valid_r], np.maximum(adaptive_er[valid_r], 1.0e-16),
        label=r"$e_r$ (diagnostic)", linewidth=1.1, markersize=3,
        markevery=mark_adaptive, color="C3", linestyle="--", marker="s",
    )
    axes[1, 0].axhline(float(data["tol_omega"]), color="C0", alpha=0.5, linewidth=0.8)
    axes[1, 0].axhline(float(data["tol_r"]), color="C3", alpha=0.5, linewidth=0.8, linestyle="--")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Accepted-step indicators")
    axes[1, 0].legend(frameon=False, fontsize=9)

    axes[1, 1].plot(
        output_times, data["adaptive_reference_error"],
        label=r"adaptive $\gamma$ + PI", linewidth=1.2, color="C0",
    )
    axes[1, 1].plot(
        output_times, data["fixed_gamma_reference_error"],
        label=r"fixed $\gamma$ + extrapolation I", linewidth=1.2, color="C3", linestyle="--",
    )
    for i, style in enumerate(FIXED_STYLES):
        key = f"fixed_{i}_reference_error"
        axes[1, 1].plot(
            output_times, data[key], label=rf"fixed $\tau={float(data[f'fixed_{i}_dt_nominal']):g}$",
            linewidth=1.0, markersize=3, markevery=max(1, len(output_times) // 20), **style,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel(r"Reference relative $L^2$ error")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)

    axes[1, 2].plot(
        adaptive_t, np.arange(len(adaptive_t)),
        label=r"adaptive $\gamma$ + PI", linewidth=1.2, color="C0",
    )
    axes[1, 2].plot(
        fixed_gamma_t, np.arange(len(fixed_gamma_t)),
        label=r"fixed $\gamma$ + extrapolation I", linewidth=1.2, color="C3", linestyle="--",
    )
    axes[1, 2].set_ylabel("Cumulative accepted steps")
    axes[1, 2].legend(frameon=False, fontsize=9)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"), axes.flat):
        ax.set_xlim(output_times[0], output_times[-1])
        ax.set_xlabel(r"$t$")
        ax.grid(alpha=0.45, linestyle="dashed", linewidth=0.5)
        ax.text(0.5, -0.22, label, transform=ax.transAxes, fontsize=16,
                fontweight="bold", ha="center", va="center")

    fig.tight_layout(h_pad=3.0, w_pad=1.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
