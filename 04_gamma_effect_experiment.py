"""Gamma-effect experiment for the SDIRK2-mr-SAV scheme.

This experiment uses a periodically forced Kolmogorov flow to test the
mean-reverting mechanism.  The scalar variable in the solver is ``q=1-r``;
the diagnostics below therefore report ``r=1-q`` as in the manuscript.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import imex_mrsav_rk_solver as solver
from imex_mrsav_rk_solver import (
    advection_term,
    inner_product,
    make_periodic_ops,
    make_tableau,
    make_taylor_v,
    prepare_initial_vorticity,
    step_imex_mrsav_rk,
    vorticity_energy,
)


DOMAIN = (0.0, 0.0, 2.0 * np.pi, 2.0 * np.pi)
NU = 1.0 / 40.0
FORCING_AMPLITUDE = 4.0
FORCING_WAVENUMBER = 4
GAMMAS = (0.0, 100.0, 1000.0, 1500.0, 2000.0)
TAU = 1.0e-3
FINAL_TIME = 20.0
REFERENCE_DT = 2.5e-4
SAMPLE_INTERVAL = 0.1
INITIAL_MODE = 10


def force(X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
    del X, t
    return FORCING_AMPLITUDE * np.cos(FORCING_WAVENUMBER * Y)


def initial_vorticity(ops) -> np.ndarray:
    omega = np.zeros_like(ops.X, dtype=np.float64)
    for k in range(1, INITIAL_MODE + 1):
        for m in range(1, INITIAL_MODE + 1):
            omega += np.cos(k * ops.X) * np.cos(m * ops.Y) / (k * k + m * m) ** 1.5
    return prepare_initial_vorticity(omega, ops)


def l2_norm(ops, value: np.ndarray) -> float:
    return float(np.sqrt(inner_product(ops, value, value)))


def etdrk4_coefficients(ops, dt: float):
    L = NU * ops.Lap
    roots = np.exp(1j * np.pi * (np.arange(1, 17) - 0.5) / 16.0)
    LR = dt * L[..., None] + roots
    E = np.exp(dt * L)
    E2 = np.exp(0.5 * dt * L)
    Q = dt * np.mean((np.exp(0.5 * LR) - 1.0) / LR, axis=-1).real
    f1 = dt * np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3,
        axis=-1,
    ).real
    f2 = dt * np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3,
        axis=-1,
    ).real
    f3 = dt * np.mean(
        (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3,
        axis=-1,
    ).real
    return E, E2, Q, f1, f2, f3


def etdrk4_trajectory(ops, omega0, dt: float, final_time: float, save_every: int):
    nsteps = int(round(final_time / dt))
    if not np.isclose(nsteps * dt, final_time):
        raise ValueError("reference dt must divide the final time")
    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(ops, dt)
    nsaved = nsteps // save_every + 1
    times = np.empty(nsaved, dtype=np.float64)
    snapshots = np.empty((nsaved, ops.ny, ops.nx), dtype=np.float64)
    enstrophy = np.empty(nsaved, dtype=np.float64)

    def nonlinear_hat(omega_hat, t):
        omega = ops.ifft2(omega_hat).real
        return ops.fft2(force(ops.X, ops.Y, t) - advection_term(ops, omega, omega_hat=omega_hat))

    omega_hat = ops.fft2(omega0)
    omega_hat[0, 0] = 0.0
    save_index = 0
    times[0] = 0.0
    snapshots[0] = omega0
    enstrophy[0] = vorticity_energy(ops, omega0)[1]
    for n in range(nsteps):
        t = n * dt
        Nv = nonlinear_hat(omega_hat, t)
        a_hat = E2 * omega_hat + Q * Nv
        Na = nonlinear_hat(a_hat, t + 0.5 * dt)
        b_hat = E2 * omega_hat + Q * Na
        Nb = nonlinear_hat(b_hat, t + 0.5 * dt)
        c_hat = E2 * a_hat + Q * (2.0 * Nb - Nv)
        Nc = nonlinear_hat(c_hat, t + dt)
        omega_hat = E * omega_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        omega_hat[0, 0] = 0.0
        step = n + 1
        if step % save_every == 0:
            save_index += 1
            omega = ops.ifft2(omega_hat).real
            times[save_index] = step * dt
            snapshots[save_index] = omega
            enstrophy[save_index] = vorticity_energy(ops, omega)[1]
    return times, snapshots, enstrophy


def run_gamma(ops, omega0, gamma: float, tau: float, final_time: float, save_every: int):
    nsteps = int(round(final_time / tau))
    if not np.isclose(nsteps * tau, final_time):
        raise ValueError("time step must divide the final time")
    tableau = make_tableau("ars222")
    V, dV = make_taylor_v(tableau.order)
    omega = np.array(omega0, copy=True)
    q = 1.0
    denom_cache = {}
    times = np.arange(0, nsteps + 1, save_every, dtype=np.float64) * tau
    snapshots = np.empty((len(times), ops.ny, ops.nx), dtype=np.float64)
    q_saved = np.empty(len(times), dtype=np.float64)
    enstrophy = np.empty(len(times), dtype=np.float64)
    snapshots[0] = omega
    q_saved[0] = q
    enstrophy[0] = vorticity_energy(ops, omega)[1]
    save_index = 0
    max_newton_iterations = 0
    failed = False
    failure_time = None
    start = time.perf_counter()
    for n in range(nsteps):
        t = n * tau
        try:
            omega, q, info = step_imex_mrsav_rk(
                omega,
                q,
                t,
                tau,
                ops,
                NU,
                gamma,
                tableau,
                force=force,
                V=V,
                dV=dV,
                denom_cache=denom_cache,
            )
            max_newton_iterations = max(max_newton_iterations, info.max_newton_iterations)
            if not np.isfinite(q) or not np.all(np.isfinite(omega)):
                raise FloatingPointError("non-finite state")
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            failed = True
            failure_time = t + tau
            print(f"gamma={gamma:g} failed at t={failure_time:.6g}: {exc}", flush=True)
            break
        step = n + 1
        if step % save_every == 0:
            save_index += 1
            snapshots[save_index] = omega
            q_saved[save_index] = q
            enstrophy[save_index] = vorticity_energy(ops, omega)[1]
    if failed:
        times = times[: save_index + 1]
        snapshots = snapshots[: save_index + 1]
        q_saved = q_saved[: save_index + 1]
        enstrophy = enstrophy[: save_index + 1]
    r_saved = 1.0 - q_saved
    return {
        "gamma": gamma,
        "times": times,
        "snapshots": snapshots,
        "q": q_saved,
        "r": r_saved,
        "enstrophy": enstrophy,
        "max_abs_r": float(np.max(np.abs(r_saved))),
        "max_newton_iterations": int(max_newton_iterations),
        "failed": failed,
        "failure_time": failure_time,
        "cpu_seconds": time.perf_counter() - start,
    }


def main() -> None:
    global NU, TAU, FINAL_TIME, REFERENCE_DT, SAMPLE_INTERVAL, GAMMAS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--nu", type=float, default=NU)
    parser.add_argument("--dt", type=float, default=TAU)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--reference-dt", type=float, default=REFERENCE_DT)
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL)
    parser.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--selected-times", type=float, nargs="+", default=(5.0, 10.0, 15.0, 20.0))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gamma_effect"))
    args = parser.parse_args()
    NU = args.nu
    TAU = args.dt
    FINAL_TIME = args.final_time
    REFERENCE_DT = args.reference_dt
    SAMPLE_INTERVAL = args.sample_interval
    GAMMAS = tuple(args.gammas)
    selected_times = tuple(
        selected_time
        for selected_time in args.selected_times
        if 0.0 <= selected_time <= FINAL_TIME + 1.0e-12
    )
    solver.HAS_FFTW = True
    ops = make_periodic_ops(DOMAIN, (args.resolution, args.resolution), fftw_threads=args.threads)
    omega0 = initial_vorticity(ops)
    save_every = int(round(SAMPLE_INTERVAL / TAU))
    reference_save_every = int(round(SAMPLE_INTERVAL / REFERENCE_DT))
    if not np.isclose(save_every * TAU, SAMPLE_INTERVAL) or not np.isclose(
        reference_save_every * REFERENCE_DT, SAMPLE_INTERVAL
    ):
        raise ValueError("sample interval must be divisible by both time steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing ETDRK4 reference...", flush=True)
    reference_times, reference_omega, reference_enstrophy = etdrk4_trajectory(
        ops, omega0, REFERENCE_DT, FINAL_TIME, reference_save_every
    )
    reference_half_check = None
    print("Reference complete.", flush=True)

    runs = []
    for gamma in GAMMAS:
        print(f"Computing gamma={gamma:g}...", flush=True)
        run = run_gamma(ops, omega0, gamma, TAU, FINAL_TIME, save_every)
        n = min(len(run["times"]), len(reference_times))
        ref_omega = reference_omega[:n]
        ref_enstrophy = reference_enstrophy[:n]
        errors = np.array(
            [l2_norm(ops, run["snapshots"][j] - ref_omega[j]) for j in range(n)]
        )
        relative_errors = errors / np.maximum(
            np.array([l2_norm(ops, ref_omega[j]) for j in range(n)]), 1e-30
        )
        relative_enstrophy = np.abs(run["enstrophy"][:n] - ref_enstrophy) / np.maximum(
            np.abs(ref_enstrophy), 1e-30
        )
        run["times"] = run["times"][:n]
        run["snapshots"] = run["snapshots"][:n]
        run["relative_omega_error"] = relative_errors
        run["relative_enstrophy_error"] = relative_enstrophy
        run["final_relative_omega_error"] = float(relative_errors[-1])
        run["max_relative_omega_error"] = float(np.max(relative_errors))
        run["max_relative_enstrophy_error"] = float(np.max(relative_enstrophy))
        runs.append(run)
        print(
            f"  max|r|={run['max_abs_r']:.4e}, "
            f"final relative omega error={run['final_relative_omega_error']:.4e}",
            flush=True,
        )

    gammas = np.asarray([run["gamma"] for run in runs], dtype=np.float64)
    times = runs[0]["times"]
    abs_r = np.stack([np.abs(run["r"]) for run in runs])
    relative_omega_error = np.stack([run["relative_omega_error"] for run in runs])
    enstrophy = np.stack([run["enstrophy"][: len(times)] for run in runs])
    relative_enstrophy_error = np.stack(
        [run["relative_enstrophy_error"] for run in runs]
    )
    np.savez_compressed(
        args.output_dir / "results.npz",
        times=times,
        gammas=gammas,
        abs_r=abs_r,
        relative_omega_error=relative_omega_error,
        enstrophy=enstrophy,
        relative_enstrophy_error=relative_enstrophy_error,
        reference_enstrophy=reference_enstrophy,
    )
    summary_runs = []
    for run in runs:
        summary_runs.append(
            {
                key: run[key]
                for key in (
                    "gamma",
                    "max_abs_r",
                    "max_newton_iterations",
                    "failed",
                    "failure_time",
                    "cpu_seconds",
                    "final_relative_omega_error",
                    "max_relative_omega_error",
                    "max_relative_enstrophy_error",
                )
            }
        )
    summary = {
        "resolution": args.resolution,
        "nu": NU,
        "forcing": "4 cos(4y)",
        "gamma": list(GAMMAS),
        "dt": TAU,
        "final_time": FINAL_TIME,
        "reference_dt": REFERENCE_DT,
        "sample_interval": SAMPLE_INTERVAL,
        "eta": 1.0 - np.sqrt(2.0) / 2.0,
        "q_less_than_one_threshold_gamma_dt": float(np.sqrt(2.0)),
        "reference_step_halving_difference": reference_half_check,
        "selected_times": list(selected_times),
        "selected_diagnostics": [],
        "runs": summary_runs,
    }
    for selected_time in selected_times:
        index = int(np.argmin(np.abs(times - selected_time)))
        if not np.isclose(times[index], selected_time, rtol=0.0, atol=1e-12):
            raise ValueError(f"selected time {selected_time} is not on the output grid")
        for gamma_index, gamma in enumerate(gammas):
            summary["selected_diagnostics"].append(
                {
                    "time": float(times[index]),
                    "gamma": float(gamma),
                    "abs_r": float(abs_r[gamma_index, index]),
                    "relative_omega_error": float(relative_omega_error[gamma_index, index]),
                    "relative_enstrophy_error": float(
                        relative_enstrophy_error[gamma_index, index]
                    ),
                }
            )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
