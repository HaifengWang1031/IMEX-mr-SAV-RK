"""Test the scaling between the extrapolation defect and ETDRK4 error."""

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
    levels = (0.05, 0.50, 0.95)
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
    times = np.asarray(data["times"], dtype=float)
    steps = np.array([item[0] for item in STEP_DATA])
    tags = [item[1] for item in STEP_DATA]

    # t=0 has zero reference error and no three-level extrapolation defect.
    valid_time = times > 0.0
    times = times[valid_time]
    extrapolation = np.vstack([np.asarray(data[f"eext_{tag}"])[valid_time] for tag in tags])
    reference = np.vstack([np.asarray(data[f"actual_{tag}"])[valid_time] for tag in tags])
    if not np.all(np.isfinite(extrapolation)) or not np.all(extrapolation > 0.0):
        raise ValueError("all extrapolation defects at positive output times must be finite and positive")
    if not np.all(np.isfinite(reference)) or not np.all(reference > 0.0):
        raise ValueError("all ETDRK4 reference errors at positive output times must be finite and positive")

    raw_ratio = reference / extrapolation
    scaled_coefficient = steps[:, None] * raw_ratio
    extrapolation_order = pointwise_orders(extrapolation, steps)
    reference_order = pointwise_orders(reference, steps)

    # Calibrate C(t) with the coarsest run and predict the other fixed-step errors.
    calibration = scaled_coefficient[0]
    predicted_reference = calibration[None, :] * extrapolation / steps[:, None]
    prediction_relative_error = np.abs(predicted_reference / reference - 1.0)

    cross_step_relative_range = (
        np.max(scaled_coefficient, axis=0) - np.min(scaled_coefficient, axis=0)
    ) / np.mean(scaled_coefficient, axis=0)

    summary: dict[str, object] = {
        "input": str(args.data.resolve()),
        "definition": "C_tau(t) = tau * e_ref(t) / e_ext(t)",
        "extrapolation_order": quantiles(extrapolation_order),
        "reference_order": quantiles(reference_order),
        "cross_step_relative_range": quantiles(cross_step_relative_range),
        "runs": {},
        "cross_prediction_from_tau_0p004": {},
    }
    for index, (step, tag) in enumerate(STEP_DATA):
        log_correlation = float(np.corrcoef(np.log(extrapolation[index]), np.log(reference[index]))[0, 1])
        same_order_indicator = extrapolation[index] / step
        constant_coefficient = float(
            np.dot(same_order_indicator, reference[index])
            / np.dot(same_order_indicator, same_order_indicator)
        )
        constant_prediction = constant_coefficient * same_order_indicator
        centered_r_squared = float(
            1.0
            - np.sum((reference[index] - constant_prediction) ** 2)
            / np.sum((reference[index] - np.mean(reference[index])) ** 2)
        )
        summary["runs"][tag] = {
            "tau": step,
            "log_correlation": log_correlation,
            "scaled_coefficient": quantiles(scaled_coefficient[index]),
            "scaled_coefficient_geometric_mean": float(
                np.exp(np.mean(np.log(scaled_coefficient[index])))
            ),
            "best_constant_coefficient_through_origin": constant_coefficient,
            "best_constant_centered_r_squared": centered_r_squared,
            "best_constant_relative_prediction_error": quantiles(
                np.abs(constant_prediction / reference[index] - 1.0)
            ),
        }
        if index > 0:
            summary["cross_prediction_from_tau_0p004"][tag] = {
                "tau": step,
                "relative_error": quantiles(prediction_relative_error[index]),
                "maximum_relative_error": float(np.max(prediction_relative_error[index])),
            }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    markevery = max(1, len(times) // 20)

    for index, ((step, _), style) in enumerate(zip(STEP_DATA, STYLES)):
        label = rf"$\tau={step:g}$"
        plot_style = {
            **style,
            "linewidth": 1.3,
            "markersize": 3.5,
            "markevery": markevery,
        }
        axes[0, 0].plot(times, raw_ratio[index], label=label, **plot_style)
        axes[0, 1].plot(times, scaled_coefficient[index], label=label, **plot_style)

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$e_{\omega,\mathrm{ref}}/e_{\omega,\mathrm{ext}}$")
    axes[0, 0].legend(frameon=False, ncol=2)

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"$C_\tau(t)=\tau e_{\omega,\mathrm{ref}}/e_{\omega,\mathrm{ext}}$")
    axes[0, 1].legend(frameon=False, ncol=2)

    axes[1, 0].plot(
        times,
        extrapolation_order,
        color="#0072B2",
        linewidth=1.4,
        label=r"extrapolation defect",
    )
    axes[1, 0].plot(
        times,
        reference_order,
        color="#D55E00",
        linewidth=1.4,
        linestyle="--",
        label=r"ETDRK4 reference error",
    )
    axes[1, 0].axhline(3.0, color="#0072B2", linewidth=0.8, alpha=0.45)
    axes[1, 0].axhline(2.0, color="#D55E00", linewidth=0.8, linestyle="--", alpha=0.45)
    axes[1, 0].set_ylabel("Observed temporal order")
    axes[1, 0].set_ylim(1.8, 3.15)
    axes[1, 0].legend(frameon=False)

    for index, ((step, _), style) in enumerate(zip(STEP_DATA[1:], STYLES[1:]), start=1):
        axes[1, 1].plot(
            times,
            np.maximum(prediction_relative_error[index], 1.0e-8),
            label=rf"predicted at $\tau={step:g}$",
            linewidth=1.3,
            markersize=3.5,
            markevery=markevery,
            **style,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Relative cross-prediction error")
    axes[1, 1].legend(frameon=False)

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
