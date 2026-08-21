"""Compare ETDRK4 solutions at tau=1e-4 and tau=5e-5 up to T=10.

The spatial and forcing setup is copied from the first fixed-step section of
01_Convergence_Analysis.ipynb.  The fine solution is used as the reference at
the common coarse-grid times, and only the scalar L2 error history is stored.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solver import mrSAV_Vorticity_Stream_Periodic_Solve as msSolver


def smooth_trig_initial_vorticity(X, Y, K=10, amplitude=1.0, seed=0):
    """Initial vorticity from 01_Convergence_Analysis.ipynb."""
    # The seed is retained to match the notebook API; this construction is
    # deterministic and does not use the RNG.
    del seed
    omega = np.zeros_like(X)
    for k in range(1, K + 1):
        for m in range(1, K + 1):
            coeff = (k * k + m * m) ** (-3 / 2)
            omega += coeff * np.cos(k * X) * np.cos(m * Y)
    omega -= np.mean(omega[:-1, :-1])
    omega *= amplitude / np.sqrt(np.mean(omega[:-1, :-1] ** 2))
    return omega


def force_term(X, Y, t):
    """Time-independent forcing from 01_Convergence_Analysis.ipynb."""
    del Y, t
    return np.cos(X)


def compare(tau_coarse=1e-4, tau_fine=5e-5, T=10.0, N=128):
    if not np.isclose(tau_coarse / tau_fine, 2.0):
        raise ValueError("This comparison requires tau_coarse/tau_fine = 2.")
    n_steps = int(round(T / tau_coarse))
    if not np.isclose(n_steps * tau_coarse, T):
        raise ValueError("T must be an integer multiple of tau_coarse.")

    domain = (0.0, 0.0, 2 * np.pi, 2 * np.pi)
    nu = 1 / 1000
    gam = 1000
    x = np.linspace(domain[0], domain[2], N + 1)
    y = np.linspace(domain[1], domain[3], N + 1)
    X, Y = np.meshgrid(x, y)
    initial = smooth_trig_initial_vorticity(X, Y, K=10, amplitude=1.0, seed=0)

    coarse_solver = msSolver(nu, gam, domain, [N, N], initial, force_term, "ETDRK4")
    fine_solver = msSolver(nu, gam, domain, [N, N], initial, force_term, "ETDRK4")
    # ETDRK4 only uses q to preserve the common solver interface; its returned
    # scalar is identically one for this method.
    coarse_omega = coarse_solver.Omega0.copy()
    fine_omega = fine_solver.Omega0.copy()
    coarse_solver._prepare_ETDRK4_coefficients(tau_coarse)
    fine_solver._prepare_ETDRK4_coefficients(tau_fine)

    times = tau_coarse * np.arange(n_steps + 1, dtype=np.float64)
    errors = np.empty(n_steps + 1, dtype=np.float64)
    errors[0] = 0.0
    for n in range(n_steps):
        t = times[n]
        coarse_omega, _ = coarse_solver.ETDRK4(
            coarse_omega[None, ...], np.array([1.0]), t, np.array([tau_coarse])
        )
        fine_omega, _ = fine_solver.ETDRK4(
            fine_omega[None, ...], np.array([1.0]), t, np.array([tau_fine])
        )
        fine_omega, _ = fine_solver.ETDRK4(
            fine_omega[None, ...],
            np.array([1.0]),
            t + tau_fine,
            np.array([tau_fine]),
        )
        difference = coarse_omega - fine_omega
        errors[n + 1] = np.sqrt(coarse_solver.h * np.sum(difference * difference))
        if (n + 1) % max(1, n_steps // 20) == 0 or n + 1 == n_steps:
            print(f"progress: {n + 1}/{n_steps} ({100 * (n + 1) / n_steps:.0f}%)", flush=True)

    max_index = int(np.argmax(errors))
    min_index = int(np.argmin(errors))
    positive_min_index = 1 + int(np.argmin(errors[1:]))
    return {
        "time": times,
        "l2_error": errors,
        "tau_coarse": float(tau_coarse),
        "tau_fine": float(tau_fine),
        "T": float(T),
        "N": int(N),
        "max_error": float(errors[max_index]),
        "max_error_time": float(times[max_index]),
        "min_error": float(errors[min_index]),
        "min_error_time": float(times[min_index]),
        "positive_min_error": float(errors[positive_min_index]),
        "positive_min_error_time": float(times[positive_min_index]),
    }


def save_outputs(result, output_dir=Path("fig"), data_dir=Path("data")):
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tag = f"T{result['T']:g}".replace(".", "p")
    npz_path = data_dir / f"etdrk4_reference_tau_comparison_{tag}.npz"
    np.savez_compressed(
        npz_path,
        time=result["time"],
        l2_error=result["l2_error"],
        tau_coarse=result["tau_coarse"],
        tau_fine=result["tau_fine"],
        T=result["T"],
        N=result["N"],
    )

    time = result["time"]
    error = result["l2_error"]
    positive = error > 0
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.semilogy(
        time[positive],
        error[positive],
        color="tab:blue",
        linewidth=1.2,
        label=r"$\|\omega_{\tau=10^{-4}}-\omega_{\tau=5\times10^{-5}}\|_{L^2}$",
    )
    max_index = int(np.argmax(error))
    positive_min_index = 1 + int(np.argmin(error[1:]))
    ax.scatter(
        time[max_index], error[max_index], color="tab:red", s=28, zorder=4,
        label=rf"max = {error[max_index]:.3e} at $t={time[max_index]:.4g}$",
    )
    ax.scatter(
        time[positive_min_index], error[positive_min_index], color="tab:green", s=28,
        zorder=4,
        label=rf"min ($t>0$) = {error[positive_min_index]:.3e} at $t={time[positive_min_index]:.4g}$",
    )
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$L^2$ error in vorticity")
    ax.set_title(
        rf"ETDRK4 reference-step comparison ($T={result['T']:g}$, ${result['N']}^2$ grid)"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.35, alpha=0.7)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    png_path = output_dir / f"etdrk4_reference_tau_comparison_{tag}.png"
    pdf_path = output_dir / f"etdrk4_reference_tau_comparison_{tag}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return npz_path, png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--N", type=int, default=128)
    args = parser.parse_args()
    result = compare(T=args.T, N=args.N)
    paths = save_outputs(result)
    print(f"N={result['N']}, T={result['T']:g}")
    print(f"tau_coarse={result['tau_coarse']:.9g}, tau_fine={result['tau_fine']:.9g}")
    print(f"max L2 error = {result['max_error']:.12e} at t={result['max_error_time']:.9g}")
    print(f"min L2 error = {result['min_error']:.12e} at t={result['min_error_time']:.9g}")
    print(
        "min L2 error for t>0 = "
        f"{result['positive_min_error']:.12e} at t={result['positive_min_error_time']:.9g}"
    )
    print("saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
