"""Accuracy comparison for SDIRK2 and SDIRK2-mr-SAV.

The script compares the classical IMEX-SDIRK2 scheme (r=0) with the
mean-reverting SAV version on the periodic vorticity equation.  A fourth-order
ETDRK4 calculation supplies the reference solution.  Results are written as
NPZ/JSON data and ready-to-include LaTeX tables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import imex_mrsav_rk_solver as solver

from imex_mrsav_rk_solver import (
    Array,
    advection_term,
    inner_product,
    make_periodic_ops,
    make_tableau,
    make_taylor_v,
    prepare_initial_vorticity,
    step_imex_mrsav_rk,
)


DOMAIN = (0.0, 0.0, 2.0 * np.pi, 2.0 * np.pi)
NU = 1.0 / 50.0
GAMMA = 1000.0
FORCE_WAVENUMBER = 1
FIXED_FINAL_TIMES = (0.1, 1.0)
VARIABLE_T_FINAL = 1.0
TAU_BASE = 1.0e-2
FIXED_EXPONENTS = tuple(range(9))
VARIABLE_N = (400, 800, 1600, 3200, 6400)
PERTURBATION = 0.15
RNG_SEED = 2026


@dataclass
class FixedRow:
    final_time: float
    exponent: int
    tau: float
    sdirk2_error: float
    sdirk2_rate: float
    sdirk2_cpu: float
    mrsav_error: float
    mrsav_rate: float
    mrsav_cpu: float


@dataclass
class VariableRow:
    nsteps: int
    tau_max: float
    sdirk2_error: float
    sdirk2_rate: float
    mrsav_error: float
    mrsav_rate: float


def force(X: Array, Y: Array, t: float) -> Array:
    del Y, t
    return np.cos(FORCE_WAVENUMBER * X)


def initial_vorticity(X: Array, Y: Array) -> Array:
    omega = np.zeros_like(X, dtype=np.float64)
    for k in range(1, 11):
        for m in range(1, 11):
            omega += np.cos(k * X) * np.cos(m * Y) / (k * k + m * m) ** 1.5
    return omega


def l2_norm(ops, value: Array) -> float:
    return float(np.sqrt(inner_product(ops, value, value)))


def etdrk4_coefficients(ops, dt: float):
    """Kassam--Trefethen ETDRK4 coefficients for L=nu*Delta."""
    L = NU * ops.Lap
    contour_points = 16
    roots = np.exp(1j * np.pi * (np.arange(1, contour_points + 1) - 0.5) / contour_points)
    LR = dt * L[..., None] + roots
    E = np.exp(dt * L)
    E2 = np.exp(0.5 * dt * L)
    Q = dt * np.mean((np.exp(0.5 * LR) - 1.0) / LR, axis=-1).real
    f1 = dt * np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=-1).real
    f2 = dt * np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=-1).real
    f3 = dt * np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=-1).real
    return E, E2, Q, f1, f2, f3


def etdrk4_reference(ops, omega0: Array, dt: float, final_time: float) -> Array:
    """Integrate the original vorticity equation with fixed-step ETDRK4."""
    nsteps = int(round(final_time / dt))
    if not np.isclose(nsteps * dt, final_time):
        raise ValueError("reference time step must divide the final time")
    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(ops, dt)

    def nonlinear_hat(omega: Array, t: float):
        return ops.fft2(force(ops.X, ops.Y, t) - advection_term(ops, omega))

    omega_hat = ops.fft2(omega0)
    omega_hat[0, 0] = 0.0
    for n in range(nsteps):
        t = n * dt
        omega = ops.ifft2(omega_hat).real
        Nv = nonlinear_hat(omega, t)
        a_hat = E2 * omega_hat + Q * Nv
        a = ops.ifft2(a_hat).real
        Na = nonlinear_hat(a, t + 0.5 * dt)
        b_hat = E2 * omega_hat + Q * Na
        b = ops.ifft2(b_hat).real
        Nb = nonlinear_hat(b, t + 0.5 * dt)
        c_hat = E2 * a_hat + Q * (2.0 * Nb - Nv)
        c = ops.ifft2(c_hat).real
        Nc = nonlinear_hat(c, t + dt)
        omega_hat = E * omega_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        omega_hat[0, 0] = 0.0
    return ops.ifft2(omega_hat).real


def run_sdirk_path(
    ops,
    omega0: Array,
    time_steps: np.ndarray,
    final_time: float,
    *,
    freeze_auxiliary: bool,
) -> tuple[Array, float]:
    """Run one prescribed time-step sequence and return final state and CPU time."""
    tableau = make_tableau("ars222")
    V, dV = make_taylor_v(tableau.order)
    omega = np.array(omega0, copy=True)
    q = 1.0
    t = 0.0
    denom_cache = {}
    start = perf_counter()
    for dt in time_steps:
        omega, q, _ = step_imex_mrsav_rk(
            omega,
            q,
            t,
            float(dt),
            ops,
            NU,
            GAMMA,
            tableau,
            force=force,
            V=V,
            dV=dV,
            denom_cache=denom_cache,
            freeze_auxiliary=freeze_auxiliary,
        )
        t += float(dt)
    # Accumulating thousands of binary64 steps can leave a small endpoint
    # roundoff (notably for long-time tests); this is not a time-stepping error.
    endpoint_tol = max(5.0e-12, 1.0e-12 * abs(final_time))
    if not np.isclose(t, final_time, rtol=0.0, atol=endpoint_tol):
        raise RuntimeError(f"time integration ended at {t}, expected {final_time}")
    return omega, perf_counter() - start


def observed_rates(scales: np.ndarray, errors: np.ndarray) -> np.ndarray:
    rates = np.full_like(errors, np.nan, dtype=float)
    rates[1:] = np.log(errors[:-1] / errors[1:]) / np.log(scales[:-1] / scales[1:])
    return rates


def perturbed_steps(nsteps: int, final_time: float) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + nsteps)
    raw = 1.0 + PERTURBATION * rng.uniform(-1.0, 1.0, nsteps)
    return final_time * raw / np.sum(raw)


def fmt_rate(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:.2f}"


def fmt_sci(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0.0 else 0
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def fixed_table(rows: list[FixedRow]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Vorticity $L^2$ errors, observed convergence rates, and CPU times at",
        r"$T=0.1$ and $T=1$. The reference solution is computed by ETDRK4 with",
        r"$\tau_{\rm ref}=0.1\times2^{-8}$.}",
        r"\label{tab:sdirk2-fixed-convergence}",
        r"\scriptsize",
        r"\begin{tabular}{c c ccc ccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{SDIRK2} & \multicolumn{3}{c}{SDIRK2-mr-SAV}\\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"$k$ & $\tau$ & Error & Rate & CPU (s) & Error & Rate & CPU (s)\\",
        r"\midrule",
    ]
    previous_time = None
    for row in rows:
        if row.final_time != previous_time:
            if previous_time is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{8}}{{c}}{{$T={row.final_time:g}$}}\\")
            previous_time = row.final_time
        lines.append(
            f"{row.exponent:d} & {fmt_sci(row.tau, 5)} & "
            f"{fmt_sci(row.sdirk2_error)} & {fmt_rate(row.sdirk2_rate)} & "
            f"{row.sdirk2_cpu:.2f} & {fmt_sci(row.mrsav_error)} & "
            f"{fmt_rate(row.mrsav_rate)} & {row.mrsav_cpu:.2f}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def variable_table(rows: list[VariableRow]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Vorticity $L^2$ errors and observed convergence rates at $T=1$ under a $15\%$ perturbed variable-step sequence.",
        r"Here $\tau_*$ is the maximum time step.}",
        r"\label{tab:sdirk2-variable-convergence}",
        r"\scriptsize",
        r"\begin{tabular}{c c cc cc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{SDIRK2} & \multicolumn{2}{c}{SDIRK2-mr-SAV}\\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"$N$ & $\tau_*$ & Error & Rate & Error & Rate\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.nsteps:d} & {row.tau_max:.5e} & {row.sdirk2_error:.3e} & {fmt_rate(row.sdirk2_rate)} & "
            f"{row.mrsav_error:.3e} & {fmt_rate(row.mrsav_rate)}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    global FORCE_WAVENUMBER, NU, FIXED_FINAL_TIMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--reference-dt", type=float, default=0.1 * 2.0 ** -8)
    parser.add_argument("--tau-base", type=float, default=TAU_BASE)
    parser.add_argument("--fixed-k-min", type=int, default=FIXED_EXPONENTS[0])
    parser.add_argument("--fixed-k-max", type=int, default=FIXED_EXPONENTS[-1])
    parser.add_argument("--force-wavenumber", type=int, default=FORCE_WAVENUMBER)
    parser.add_argument("--nu", type=float, default=NU)
    parser.add_argument(
        "--final-times", type=float, nargs="+", default=FIXED_FINAL_TIMES,
    )
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fftw-threads", type=int, default=1)
    parser.add_argument("--use-pyfftw", action="store_true", help="use pyFFTW instead of the SciPy FFT backend")
    parser.add_argument("--skip-variable", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sdirk2_accuracy"))
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.tau_base <= 0.0:
        raise ValueError("tau-base must be positive")
    if args.fixed_k_min > args.fixed_k_max:
        raise ValueError("fixed-k-min must not exceed fixed-k-max")
    if args.force_wavenumber < 1:
        raise ValueError("force-wavenumber must be positive")
    if args.nu <= 0.0:
        raise ValueError("nu must be positive")
    if any(final_time <= 0.0 for final_time in args.final_times):
        raise ValueError("final-times must be positive")
    FORCE_WAVENUMBER = args.force_wavenumber
    NU = args.nu
    FIXED_FINAL_TIMES = tuple(args.final_times)

    if not args.use_pyfftw:
        solver.HAS_FFTW = False
    ops = make_periodic_ops(
        DOMAIN, (args.resolution, args.resolution), fftw_threads=args.fftw_threads,
    )
    omega0 = prepare_initial_vorticity(initial_vorticity(ops.X, ops.Y), ops)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed_exponents = tuple(range(args.fixed_k_min, args.fixed_k_max + 1))
    fixed_tau = args.tau_base * 2.0 ** (-np.asarray(fixed_exponents, dtype=int))
    fixed_rows = []
    references = {}
    reference_checks = {}
    for final_time in FIXED_FINAL_TIMES:
        print(
            f"Computing ETDRK4 reference at T={final_time:g} on "
            f"{args.resolution}x{args.resolution} grid with dt={args.reference_dt:.5e}"
        )
        omega_ref = etdrk4_reference(ops, omega0, args.reference_dt, final_time)
        references[final_time] = omega_ref
        reference_checks[final_time] = np.nan
        if args.verify_reference:
            omega_ref_half = etdrk4_reference(
                ops, omega0, 0.5 * args.reference_dt, final_time,
            )
            reference_checks[final_time] = l2_norm(ops, omega_ref - omega_ref_half)
            print(
                f"ETDRK4 step-halving difference at T={final_time:g}: "
                f"{reference_checks[final_time]:.3e}"
            )

        # Short, identical full-path warm-ups initialize the FFT backend.
        warmup_dt = min(args.tau_base, final_time)
        warmup_steps = np.full(int(round(final_time / warmup_dt)), warmup_dt)
        run_sdirk_path(
            ops, omega0, warmup_steps, final_time, freeze_auxiliary=True,
        )
        run_sdirk_path(
            ops, omega0, warmup_steps, final_time, freeze_auxiliary=False,
        )

        fixed_sdirk = np.empty_like(fixed_tau)
        fixed_mrsav = np.empty_like(fixed_tau)
        fixed_cpu_sdirk = np.empty_like(fixed_tau)
        fixed_cpu_mrsav = np.empty_like(fixed_tau)
        for i, dt in enumerate(fixed_tau):
            steps = np.full(int(round(final_time / dt)), dt, dtype=float)
            cpu_sdirk, cpu_mrsav = [], []
            for _ in range(args.repeats):
                omega, elapsed = run_sdirk_path(
                    ops, omega0, steps, final_time, freeze_auxiliary=True,
                )
                fixed_sdirk[i] = l2_norm(ops, omega - omega_ref)
                cpu_sdirk.append(elapsed)
                omega, elapsed = run_sdirk_path(
                    ops, omega0, steps, final_time, freeze_auxiliary=False,
                )
                fixed_mrsav[i] = l2_norm(ops, omega - omega_ref)
                cpu_mrsav.append(elapsed)
            fixed_cpu_sdirk[i] = float(np.median(cpu_sdirk))
            fixed_cpu_mrsav[i] = float(np.median(cpu_mrsav))
            print(
                f"fixed T={final_time:g}, k={fixed_exponents[i]}, dt={dt:.8f}: "
                f"SDIRK2={fixed_sdirk[i]:.3e}, mr-SAV={fixed_mrsav[i]:.3e}"
            )

        fixed_rate_sdirk = observed_rates(fixed_tau, fixed_sdirk)
        fixed_rate_mrsav = observed_rates(fixed_tau, fixed_mrsav)
        fixed_rows.extend(
            FixedRow(
                float(final_time), int(k), float(dt), float(e0), float(p0),
                float(c0), float(e1), float(p1), float(c1),
            )
            for k, dt, e0, p0, c0, e1, p1, c1 in zip(
                fixed_exponents, fixed_tau, fixed_sdirk, fixed_rate_sdirk,
                fixed_cpu_sdirk, fixed_mrsav, fixed_rate_mrsav,
                fixed_cpu_mrsav,
            )
        )

    variable_tau_max = np.empty(len(VARIABLE_N))
    variable_sdirk = np.empty(len(VARIABLE_N))
    variable_mrsav = np.empty(len(VARIABLE_N))
    for i, nsteps in enumerate(VARIABLE_N):
        if args.skip_variable:
            break
        steps = perturbed_steps(nsteps, VARIABLE_T_FINAL)
        variable_tau_max[i] = np.max(steps)
        omega, _ = run_sdirk_path(
            ops, omega0, steps, VARIABLE_T_FINAL, freeze_auxiliary=True,
        )
        variable_sdirk[i] = l2_norm(ops, omega - references[VARIABLE_T_FINAL])
        omega, _ = run_sdirk_path(
            ops, omega0, steps, VARIABLE_T_FINAL, freeze_auxiliary=False,
        )
        variable_mrsav[i] = l2_norm(ops, omega - references[VARIABLE_T_FINAL])
        print(f"variable N={nsteps}: SDIRK2={variable_sdirk[i]:.3e}, mr-SAV={variable_mrsav[i]:.3e}")

    variable_rows = []
    if not args.skip_variable:
        variable_rate_sdirk = observed_rates(variable_tau_max, variable_sdirk)
        variable_rate_mrsav = observed_rates(variable_tau_max, variable_mrsav)
        variable_rows = [
            VariableRow(
                int(n), float(dt), float(e0), float(p0), float(e1), float(p1),
            )
            for n, dt, e0, p0, e1, p1 in zip(
                VARIABLE_N, variable_tau_max, variable_sdirk,
                variable_rate_sdirk, variable_mrsav, variable_rate_mrsav,
            )
        ]

    if args.skip_variable:
        variable_tau_max = np.empty(0)
        variable_sdirk = np.empty(0)
        variable_mrsav = np.empty(0)

    np.savez(
        args.output_dir / "results.npz",
        fixed_final_time=np.asarray([row.final_time for row in fixed_rows]),
        fixed_exponent=np.asarray([row.exponent for row in fixed_rows]),
        fixed_tau=np.asarray([row.tau for row in fixed_rows]),
        fixed_sdirk_error=np.asarray([row.sdirk2_error for row in fixed_rows]),
        fixed_mrsav_error=np.asarray([row.mrsav_error for row in fixed_rows]),
        fixed_sdirk_cpu=np.asarray([row.sdirk2_cpu for row in fixed_rows]),
        fixed_mrsav_cpu=np.asarray([row.mrsav_cpu for row in fixed_rows]),
        variable_n=(
            np.empty(0, dtype=int)
            if args.skip_variable else np.asarray(VARIABLE_N)
        ),
        variable_tau_max=variable_tau_max,
        variable_sdirk_error=variable_sdirk,
        variable_mrsav_error=variable_mrsav,
        reference_final_time=np.asarray(FIXED_FINAL_TIMES),
        reference_check=np.asarray([reference_checks[t] for t in FIXED_FINAL_TIMES]),
    )
    (args.output_dir / "fixed_table.tex").write_text(fixed_table(fixed_rows), encoding="utf-8")
    if variable_rows:
        (args.output_dir / "variable_table.tex").write_text(
            variable_table(variable_rows), encoding="utf-8",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "resolution": args.resolution,
                "nu": NU,
                "gamma": GAMMA,
                "force_wavenumber": FORCE_WAVENUMBER,
                "reference_dt": args.reference_dt,
                "reference_step_halving_difference": {
                    f"T={t:g}": reference_checks[t] for t in FIXED_FINAL_TIMES
                },
                "fixed": [asdict(row) for row in fixed_rows],
                "variable": [asdict(row) for row in variable_rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
