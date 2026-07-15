"""Plot the six-panel adaptive-performance diagnostics."""

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


FIXED_STYLES = [
    {"color": "C1", "marker": "*", "linestyle": "-.", "markevery": 180},
    {"color": "C8", "marker": "d", "linestyle": "--", "markevery": 280},
    {"color": "C9", "marker": "p", "linestyle": "--", "markevery": 420},
    {"color": "C3", "marker": "s", "linestyle": ":", "markevery": 700},
]


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
    adaptive_cpu = data["adaptive_cpu"]
    adaptive_q = data["adaptive_q"]
    adaptive_eomega = data["adaptive_error_omega"]
    adaptive_er = data["adaptive_error_r"]
    adaptive_ref_error = data["adaptive_reference_error"]
    controller = str(data["controller"]) if "controller" in data else "extrapolation"

    fixed = []
    for i in range(20):
        key = f"fixed_{i}_dt_nominal"
        if key not in data:
            break
        fixed.append(
            {
                "dt_nominal": float(data[key]),
                "t": data[f"fixed_{i}_t"],
                "cpu": data[f"fixed_{i}_cpu"],
                "q": data[f"fixed_{i}_q"],
                "ref_error": data[f"fixed_{i}_reference_error"],
            }
        )
    if len(fixed) > len(FIXED_STYLES):
        raise ValueError("not enough fixed-step styles")

    fig = plt.figure(figsize=(17, 15))
    ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=2, rowspan=2)
    ax6 = plt.subplot2grid((6, 6), (0, 2), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((6, 6), (0, 4), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((6, 6), (2, 0), colspan=2, rowspan=2)
    ax4 = plt.subplot2grid((6, 6), (2, 2), colspan=2, rowspan=2)
    ax5 = plt.subplot2grid((6, 6), (2, 4), colspan=2, rowspan=2)

    adaptive_style = {"color": "C0", "marker": "o", "linestyle": "-"}
    if controller == "pi":
        adaptive_label = rf"PI adaptive $(\mathrm{{tol}}={data['tol_omega']:.1e})$"
        error_label = r"Embedded local $L^2$ error estimate"
    else:
        adaptive_label = (
            rf"adaptive $(\mathrm{{tol}}_\omega={data['tol_omega']:.1e}, "
            rf"\mathrm{{tol}}_r={data['tol_r']:.1e})$"
        )
        error_label = r"Relative extrapolation defect $e_\omega$"
    markevery = max(1, len(adaptive_t) // 30)

    ax1.plot(
        adaptive_t[1:], adaptive_dt, color=adaptive_style["color"],
        marker=adaptive_style["marker"], linestyle="-", linewidth=1,
        markersize=3, markevery=markevery, label=adaptive_label, zorder=5,
    )
    for result, style in zip(fixed, FIXED_STYLES):
        ax1.plot(
            result["t"][1:], np.full(len(result["t"]) - 1, result["dt_nominal"]),
            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            linewidth=1, markersize=4, markevery=style["markevery"],
            label=rf"$\tau={result['dt_nominal']:.3g}$",
        )
    ax1.axhline(float(data["dt_min"]), color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axhline(float(data["dt_max"]), color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_yscale("log")
    ax1.set_xlim(output_times[0], output_times[-1])
    ax1.set_xlabel(r"$t$", fontsize=15)
    ax1.set_ylabel(r"Step size $\Delta t$", fontsize=12)

    for result, style in zip(fixed, FIXED_STYLES):
        ax6.plot(
            result["t"], np.arange(len(result["t"])), color=style["color"],
            marker=style["marker"], linestyle=style["linestyle"], linewidth=1,
            markersize=4, markevery=style["markevery"],
        )
    ax6.plot(adaptive_t, np.arange(len(adaptive_t)), color="C0", linewidth=1)
    ax6.set_xlim(output_times[0], output_times[-1])
    ax6.set_ylim(bottom=0)
    ax6.set_xlabel(r"$t$", fontsize=15)
    ax6.set_ylabel("Cumulative steps", fontsize=12)

    for result, style in zip(fixed, FIXED_STYLES):
        ax2.plot(
            result["t"], result["cpu"], color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], linewidth=1, markersize=4,
            markevery=style["markevery"],
        )
    ax2.plot(adaptive_t, adaptive_cpu, color="C0", linewidth=1)
    ax2.set_xlim(output_times[0], output_times[-1])
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel(r"$t$", fontsize=15)
    ax2.set_ylabel("CPU time (s)", fontsize=12)

    ax3.plot(
        adaptive_t[1:], np.maximum(np.abs(1.0 - adaptive_q[1:]), 1.0e-16),
        color="C0", linewidth=1, marker="o", markersize=3, markevery=markevery,
    )
    for result, style in zip(fixed, FIXED_STYLES):
        ax3.plot(
            result["t"][1:], np.maximum(np.abs(1.0 - result["q"][1:]), 1.0e-16),
            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            linewidth=1, markersize=4, markevery=style["markevery"],
        )
    ax3.set_yscale("log")
    ax3.set_xlim(output_times[0], output_times[-1])
    ax3.set_xlabel(r"$t$", fontsize=15)
    ax3.set_ylabel(r"$|r|=|1-q|$", fontsize=12)

    valid = np.isfinite(adaptive_eomega)
    ax4.plot(
        adaptive_t[1:][valid], np.maximum(adaptive_eomega[valid], 1.0e-16),
        color="C0", linewidth=1, marker="o", markersize=3, markevery=markevery,
    )
    ax4.axhline(float(data["tol_omega"]), color="k", linestyle="--", alpha=0.6, linewidth=1)
    ax4.set_yscale("log")
    ax4.set_xlim(output_times[0], output_times[-1])
    ax4.set_xlabel(r"$t$", fontsize=15)
    ax4.set_ylabel(error_label, fontsize=12)

    ax5.plot(output_times, np.maximum(adaptive_ref_error, 1.0e-16), color="C0", linewidth=1)
    for result, style in zip(fixed, FIXED_STYLES):
        ax5.plot(
            output_times, np.maximum(result["ref_error"], 1.0e-16),
            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            linewidth=1, markersize=4, markevery=max(1, len(output_times) // 20),
        )
    ax5.set_yscale("log")
    ax5.set_xlim(output_times[0], output_times[-1])
    ax5.set_xlabel(r"$t$", fontsize=15)
    ax5.set_ylabel(r"Reference relative $L^2$ error", fontsize=12)

    for ax in (ax1, ax6, ax2, ax3, ax4, ax5):
        ax.grid(alpha=0.5, linestyle="dashed", linewidth=0.5)
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"), (ax1, ax6, ax2, ax3, ax4, ax5)):
        ax.text(0.5, -0.21, label, transform=ax.transAxes, fontsize=18,
                fontweight="bold", ha="center", va="center")

    fig.legend(loc="lower center", ncol=5, fontsize=12, bbox_to_anchor=(0.5, 0.39))
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
