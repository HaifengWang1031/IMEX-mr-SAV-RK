"""Compare cube-root extrapolation defects with square-root ETDRK4 errors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_DATA = (
    (4.0e-3, "0p004"),
    (2.0e-3, "0p002"),
    (1.0e-3, "0p001"),
    (5.0e-4, "0p0005"),
)

STYLES = (
    {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    {"color": "#CC79A7", "linestyle": ":", "marker": "d"},
)


def configure_style() -> None:
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 300,
        }
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.05, 0.25, 0.50, 0.75, 0.95)
    result = np.quantile(values, levels)
    return {f"q{int(100 * level):02d}": float(value) for level, value in zip(levels, result)}


def pointwise_orders(values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    log_steps = np.log(steps)
    return np.array(
        [np.polyfit(log_steps, np.log(values[:, index]), 1)[0] for index in range(values.shape[1])]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    data = np.load(args.data)
    all_times = np.asarray(data["times"], dtype=float)
    valid_time = all_times > 0.0
    times = all_times[valid_time]
    steps = np.array([step for step, _ in STEP_DATA])

    extrapolation = np.vstack(
        [np.asarray(data[f"eext_{tag}"])[valid_time] for _, tag in STEP_DATA]
    )
    reference = np.vstack(
        [np.asarray(data[f"actual_{tag}"])[valid_time] for _, tag in STEP_DATA]
    )
    if not np.all(np.isfinite(extrapolation)) or not np.all(extrapolation > 0.0):
        raise ValueError("extrapolation defects must be finite and positive for t > 0")
    if not np.all(np.isfinite(reference)) or not np.all(reference > 0.0):
        raise ValueError("ETDRK4 reference errors must be finite and positive for t > 0")

    extrapolation_root = np.cbrt(extrapolation)
    reference_root = np.sqrt(reference)
    ratio = reference_root / extrapolation_root
    relative_gap = np.abs(reference_root - extrapolation_root) / np.maximum(
        reference_root, extrapolation_root
    )

    ratio_relative_range = (np.max(ratio, axis=0) - np.min(ratio, axis=0)) / np.mean(
        ratio, axis=0
    )
    extrapolation_root_order = pointwise_orders(extrapolation_root, steps)
    reference_root_order = pointwise_orders(reference_root, steps)

    summary: dict[str, object] = {
        "input": str(args.data.resolve()),
        "comparison": "cuberoot(e_ext) versus sqrt(e_ref)",
        "extrapolation_root_order": quantiles(extrapolation_root_order),
        "reference_root_order": quantiles(reference_root_order),
        "cross_step_ratio_relative_range": quantiles(ratio_relative_range),
        "runs": {},
    }
    for index, (step, tag) in enumerate(STEP_DATA):
        x = extrapolation_root[index]
        y = reference_root[index]
        best_constant = float(np.dot(x, y) / np.dot(x, x))
        prediction = best_constant * x
        centered_r_squared = float(
            1.0
            - np.sum((y - prediction) ** 2)
            / np.sum((y - np.mean(y)) ** 2)
        )
        summary["runs"][tag] = {
            "tau": step,
            "log_correlation": float(np.corrcoef(np.log(x), np.log(y))[0, 1]),
            "sqrt_ref_over_cuberoot_ext": quantiles(ratio[index]),
            "ratio_geometric_mean": float(np.exp(np.mean(np.log(ratio[index])))),
            "direct_relative_gap": quantiles(relative_gap[index]),
            "best_constant_through_origin": best_constant,
            "best_constant_centered_r_squared": centered_r_squared,
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    markevery = max(1, len(times) // 20)
    for index, ((step, _), style) in enumerate(zip(STEP_DATA, STYLES)):
        common = {
            **style,
            "linewidth": 1.3,
            "markersize": 3.5,
            "markevery": markevery,
            "label": rf"$\tau={step:g}$",
        }
        axes[0, 0].plot(times, extrapolation_root[index], **common)
        axes[0, 1].plot(times, reference_root[index], **common)
        axes[1, 0].plot(times, ratio[index], **common)
        axes[1, 1].plot(times, relative_gap[index], **common)

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$\sqrt[3]{e_{\omega,\mathrm{ext}}}$")
    axes[0, 0].legend(frameon=False, ncol=2)

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"$\sqrt{e_{\omega,\mathrm{ref}}}$")
    axes[0, 1].legend(frameon=False, ncol=2)

    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(
        r"$R_\tau(t)=\sqrt{e_{\omega,\mathrm{ref}}}/\sqrt[3]{e_{\omega,\mathrm{ext}}}$"
    )
    axes[1, 0].legend(frameon=False, ncol=2)

    axes[1, 1].set_ylabel(
        r"$|\sqrt{e_{\omega,\mathrm{ref}}}-\sqrt[3]{e_{\omega,\mathrm{ext}}}|/\max$"
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend(frameon=False, ncol=2)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel(r"$t$")
        ax.grid(alpha=0.45, linestyle="dashed", linewidth=0.5)
        ax.text(
            0.5,
            -0.22,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
        )

    fig.tight_layout(h_pad=3.1, w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved figure to {args.output}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
