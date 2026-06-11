"""Plot and tabulate gamma-sweep diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("data/imex_gamma_diagnostics_pilot.h5")
DEFAULT_FIG_DIR = Path("fig")
DEFAULT_TABLE = Path("data/imex_gamma_diagnostics_table.csv")


def gamma_label(gamma: float) -> str:
    return f"gamma_{gamma:g}".replace("-", "m").replace(".", "p")


def choose_sample_times(t: np.ndarray) -> np.ndarray:
    tf = float(t[-1])
    if tf >= 10.0:
        return np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)
    count = min(5, max(1, len(t) - 1))
    return np.linspace(float(t[1]), tf, count)


def nearest_time_index(t: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(t - target)))


def load_data(path: Path):
    with h5py.File(path, "r") as f:
        completed_step = int(f.attrs.get("completed_step", len(f["time/t"]) - 1))
        sl = slice(0, completed_step + 1)
        t = f["time/t"][sl]
        gamma_values = tuple(float(x) for x in f.attrs["gamma_values"])
        gamma_data = {
            gamma: {
                "q": f[gamma_label(gamma)]["q"][sl],
                "Enstrophy": f[gamma_label(gamma)]["Enstrophy"][sl],
                "rel_l2_error": f[f"errors/{gamma_label(gamma)}"]["rel_l2_error"][sl],
                "rel_enstrophy_error": f[f"errors/{gamma_label(gamma)}"]["rel_enstrophy_error"][sl],
            }
            for gamma in gamma_values
        }
        ref_enstrophy = f["etdrk4_ref"]["Enstrophy"][sl]
        attrs = dict(f.attrs)
    return t, gamma_values, gamma_data, ref_enstrophy, attrs


def plot_diagnostics(path: Path, fig_dir: Path) -> tuple[Path, Path]:
    t, gamma_values, gamma_data, ref_enstrophy, attrs = load_data(path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(gamma_values)))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=180)
    ax_err, ax_q, ax_ens = axes

    for color, gamma in zip(colors, gamma_values):
        data = gamma_data[gamma]
        label = rf"$\gamma={gamma:g}$"
        rel_l2 = np.asarray(data["rel_l2_error"], dtype=np.float64)
        q_dev = np.abs(np.asarray(data["q"], dtype=np.float64) - 1.0)
        ax_err.semilogy(t, np.maximum(rel_l2, 1.0e-16), color=color, lw=1.4, label=label)
        ax_q.semilogy(t, np.maximum(q_dev, 1.0e-16), color=color, lw=1.4, label=label)
        ax_ens.plot(t, data["Enstrophy"], color=color, lw=1.2, label=label)

    ax_ens.plot(t, ref_enstrophy, color="black", lw=1.2, ls="--", label="ETDRK4 ref")
    ax_err.set_xlabel("t")
    ax_err.set_ylabel("relative L2 error")
    ax_q.set_xlabel("t")
    ax_q.set_ylabel("|q - 1|")
    ax_ens.set_xlabel("t")
    ax_ens.set_ylabel("Enstrophy")

    title = f"IMEX {attrs.get('method', 'rk?')}, grid={tuple(attrs.get('grid', []))}, dt={attrs.get('dt', np.nan):g}"
    fig.suptitle(title)
    for ax in axes:
        ax.grid(True, which="major", color="0.85", ls="--", lw=0.5)
        ax.tick_params(direction="in", top=True, right=True)

    handles, labels = ax_err.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=True)
    fig.subplots_adjust(bottom=0.28, top=0.82, wspace=0.32)

    pdf_path = fig_dir / "imex_gamma_diagnostics.pdf"
    png_path = fig_dir / "imex_gamma_diagnostics.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def write_error_table(path: Path, table_path: Path) -> Path:
    t, gamma_values, gamma_data, _, _ = load_data(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    sample_times = choose_sample_times(t)

    fieldnames = ["time"]
    for gamma in gamma_values:
        fieldnames.extend([f"gamma_{gamma:g}_rel_l2", f"gamma_{gamma:g}_rel_enstrophy"])

    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for target in sample_times:
            idx = nearest_time_index(t, target)
            row = {"time": f"{t[idx]:.12g}"}
            for gamma in gamma_values:
                data = gamma_data[gamma]
                row[f"gamma_{gamma:g}_rel_l2"] = f"{data['rel_l2_error'][idx]:.8e}"
                row[f"gamma_{gamma:g}_rel_enstrophy"] = f"{data['rel_enstrophy_error'][idx]:.8e}"
            writer.writerow(row)

    return table_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path, png_path = plot_diagnostics(args.input, args.fig_dir)
    table_path = write_error_table(args.input, args.table)
    print(f"Saved figures to {pdf_path} and {png_path}")
    print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
