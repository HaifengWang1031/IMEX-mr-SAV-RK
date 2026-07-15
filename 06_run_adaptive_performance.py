"""Run the Kolmogorov-flow adaptive-performance experiment.

The adaptive run uses the quadratic-extrapolation controller implemented in
``solve_adaptive_extrapolation``.  The script writes one compact ``npz`` file
containing the adaptive history, fixed-step histories, and ETDRK4 snapshots.
Plotting is deliberately kept separate from this expensive integration step.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import imex_mrsav_rk_solver as solver
from imex_mrsav_rk_solver import (
    ExtrapolationAdaptiveOptions,
    advection_term,
    inner_product,
    make_periodic_ops,
    make_tableau,
    make_taylor_v,
    prepare_initial_vorticity,
    solve_adaptive_extrapolation,
    step_imex_mrsav_rk,
    vorticity_energy,
)


DOMAIN = (0.0, 0.0, 2.0 * np.pi, 2.0 * np.pi)
NU = 1.0 / 40.0
GAMMA = 1000.0
FORCING_AMPLITUDE = 4.0
FORCING_WAVENUMBER = 4
FINAL_TIME = 20.0
OUTPUT_INTERVAL = 0.1
REFERENCE_DT = 2.5e-4
ADAPTIVE_DT0 = 5.0e-4
TOL_OMEGA = 5.5e-4
TOL_R = 5.5e-4
DT_MIN = 1.0e-5
DT_MAX = 1.0e-2
FIXED_STEPS = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)


def force(X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
    del X, t
    return FORCING_AMPLITUDE * np.cos(FORCING_WAVENUMBER * Y)


def initial_vorticity(ops, K: int, scale: float = 1.0) -> np.ndarray:
    """Build omega_0^(K) = scale*sum_{j,l<=K}(j^2+l^2)^(-3/2) cos(jx)cos(ly)."""
    if K < 1:
        raise ValueError("K must be positive")
    omega = np.zeros_like(ops.X, dtype=np.float64)
    for j in range(1, K + 1):
        for ell in range(1, K + 1):
            omega += (
                np.cos(j * ops.X) * np.cos(ell * ops.Y)
                / (j * j + ell * ell) ** 1.5
            )
    return prepare_initial_vorticity(scale * omega, ops)


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


def etdrk4_reference(ops, omega0: np.ndarray, dt: float, output_times: np.ndarray):
    """Compute ETDRK4 snapshots at the requested times."""
    tf = float(output_times[-1])
    nsteps = int(np.rint(tf / dt))
    if nsteps <= 0 or not np.isclose(nsteps * dt, tf, rtol=0.0, atol=1.0e-11):
        raise ValueError("reference dt must divide the final time")
    output_indices = np.rint(output_times / dt).astype(int)
    if not np.allclose(output_indices * dt, output_times, rtol=0.0, atol=1.0e-10):
        raise ValueError("reference dt must divide every requested output time")

    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(ops, dt)
    snapshots = np.empty((len(output_times), ops.ny, ops.nx), dtype=np.float64)
    omega_hat = ops.fft2(omega0)
    omega_hat[0, 0] = 0.0
    snapshots[0] = omega0
    next_output = 1

    def nonlinear_hat(value_hat, t):
        value = ops.ifft2(value_hat).real
        nonlinear = force(ops.X, ops.Y, t) - advection_term(
            ops, value, omega_hat=value_hat,
        )
        return ops.fft2(nonlinear)

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
        if next_output < len(output_indices) and step == output_indices[next_output]:
            snapshots[next_output] = ops.ifft2(omega_hat).real
            next_output += 1

    if next_output != len(output_indices):
        raise RuntimeError("ETDRK4 reference did not fill all output snapshots")
    return snapshots


def run_fixed(ops, omega0, dt: float, output_times: np.ndarray, gamma: float):
    """Run fixed ARS222 steps, landing on the same output times."""
    tableau = make_tableau("ars222")
    V, dV = make_taylor_v(tableau.order)
    omega = np.array(omega0, copy=True)
    q = 1.0
    t = float(output_times[0])
    tf = float(output_times[-1])
    output_index = 1
    omega_saved = np.empty((len(output_times), ops.ny, ops.nx), dtype=np.float64)
    omega_saved[0] = omega
    t_all = [t]
    q_all = [q]
    dt_all: list[float] = []
    cpu_all = [0.0]
    energy, enstrophy, palinstrophy = vorticity_energy(ops, omega)
    energy_all = [energy]
    enstrophy_all = [enstrophy]
    palinstrophy_all = [palinstrophy]
    max_vorticity_all = [float(np.max(np.abs(omega)))]
    denom_cache = {}
    wall0 = time.perf_counter()
    tolerance = 1.0e-12 * max(1.0, abs(tf))

    while t < tf - tolerance:
        dt_trial = min(dt, tf - t)
        if output_index < len(output_times):
            dt_trial = min(dt_trial, float(output_times[output_index] - t))
        omega, q, _ = step_imex_mrsav_rk(
            omega, q, t, dt_trial, ops, NU, gamma, tableau,
            force=force, V=V, dV=dV, denom_cache=denom_cache,
        )
        t += dt_trial
        if output_index < len(output_times) and abs(t - output_times[output_index]) <= tolerance:
            t = float(output_times[output_index])
            omega_saved[output_index] = omega
            output_index += 1
        if abs(t - tf) <= tolerance:
            t = tf
        dt_all.append(float(dt_trial))
        t_all.append(t)
        q_all.append(float(q))
        energy, enstrophy, palinstrophy = vorticity_energy(ops, omega)
        energy_all.append(energy)
        enstrophy_all.append(enstrophy)
        palinstrophy_all.append(palinstrophy)
        max_vorticity_all.append(float(np.max(np.abs(omega))))
        cpu_all.append(time.perf_counter() - wall0)

    if output_index != len(output_times):
        raise RuntimeError("fixed run did not fill all output snapshots")
    return {
        "t": np.asarray(t_all),
        "q": np.asarray(q_all),
        "dt": np.asarray(dt_all),
        "cpu": np.asarray(cpu_all),
        "omega_snapshot": omega_saved,
        "energy": np.asarray(energy_all),
        "enstrophy": np.asarray(enstrophy_all),
        "palinstrophy": np.asarray(palinstrophy_all),
        "max_vorticity": np.asarray(max_vorticity_all),
    }


def relative_errors(ops, snapshots: np.ndarray, reference: np.ndarray) -> np.ndarray:
    errors = np.empty(len(snapshots), dtype=np.float64)
    for i, (value, ref) in enumerate(zip(snapshots, reference)):
        errors[i] = l2_norm(ops, value - ref) / max(l2_norm(ops, ref), 1.0e-30)
    return errors


def run_experiment(args: argparse.Namespace) -> Path:
    if args.K < 1:
        raise ValueError("K must be positive")
    if args.final_time <= 0.0 or args.output_interval <= 0.0:
        raise ValueError("final time and output interval must be positive")
    if args.reference_dt <= 0.0:
        raise ValueError("reference dt must be positive")

    solver.HAS_FFTW = bool(args.use_pyfftw)
    resolution = (args.resolution, args.resolution)
    ops = make_periodic_ops(DOMAIN, resolution, fftw_threads=args.fftw_threads)
    omega0 = initial_vorticity(ops, args.K, args.initial_scale)
    output_count = int(np.rint(args.final_time / args.output_interval))
    if output_count <= 0 or not np.isclose(output_count * args.output_interval, args.final_time):
        raise ValueError("output interval must divide the final time")
    output_times = np.linspace(0.0, args.final_time, output_count + 1)

    initial_energy = vorticity_energy(ops, omega0)[0]
    initial_velocity_norm = np.sqrt(2.0 * initial_energy)
    initial_reynolds = initial_velocity_norm / NU
    print(
        f"K={args.K}, scale={args.initial_scale:g}, resolution={resolution[0]}^2, T={args.final_time:g}, "
        f"initial ||u||={initial_velocity_norm:.8e}, Re_in={initial_reynolds:.8e}"
    )

    print(f"Computing ETDRK4 reference with dt={args.reference_dt:.3e} ...", flush=True)
    reference_start = time.perf_counter()
    reference = etdrk4_reference(ops, omega0, args.reference_dt, output_times)
    print(f"Reference complete in {time.perf_counter() - reference_start:.2f} s", flush=True)

    adaptive_options = ExtrapolationAdaptiveOptions(
        tol_omega=args.tol_omega,
        tol_r=args.tol_r,
        safety=args.safety,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
    )
    print("Computing extrapolation-adaptive run ...", flush=True)
    adaptive_start = time.perf_counter()
    adaptive = solve_adaptive_extrapolation(
        omega0,
        nu=NU,
        gamma=args.gamma,
        domain=DOMAIN,
        discrete_num=resolution,
        dt0=args.adaptive_dt0,
        t_span=(0.0, args.final_time),
        method="ars222",
        force=force,
        adaptive_options=adaptive_options,
        output_times=output_times,
        fftw_threads=args.fftw_threads,
    )
    print(
        f"Adaptive complete in {time.perf_counter() - adaptive_start:.2f} s: "
        f"accepted={adaptive.accepted_steps}, rejected={adaptive.rejected_steps}, "
        f"floor_accepted={adaptive.floor_accepted_steps}",
        flush=True,
    )

    fixed_results = []
    for dt in FIXED_STEPS:
        print(f"Computing fixed run dt={dt:.4g} ...", flush=True)
        start = time.perf_counter()
        result = run_fixed(ops, omega0, dt, output_times, args.gamma)
        elapsed = time.perf_counter() - start
        print(f"  complete in {elapsed:.2f} s with {len(result['dt'])} steps", flush=True)
        fixed_results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "K": np.array(args.K),
        "initial_scale": np.array(args.initial_scale),
        "resolution": np.array(resolution),
        "nu": np.array(NU),
        "gamma": np.array(args.gamma),
        "forcing_amplitude": np.array(FORCING_AMPLITUDE),
        "forcing_wavenumber": np.array(FORCING_WAVENUMBER),
        "final_time": np.array(args.final_time),
        "output_interval": np.array(args.output_interval),
        "reference_dt": np.array(args.reference_dt),
        "tol_omega": np.array(args.tol_omega),
        "tol_r": np.array(args.tol_r),
        "safety": np.array(args.safety),
        "dt_min": np.array(args.dt_min),
        "dt_max": np.array(args.dt_max),
        "adaptive_dt0": np.array(args.adaptive_dt0),
        "initial_velocity_norm": np.array(initial_velocity_norm),
        "initial_reynolds": np.array(initial_reynolds),
        "output_times": output_times,
        "omega0": omega0,
        "reference_omega": reference,
        "adaptive_t": adaptive.t,
        "adaptive_q": adaptive.q,
        "adaptive_dt": adaptive.dt_history,
        "adaptive_floor_accepted_steps": np.array(adaptive.floor_accepted_steps),
        "adaptive_cpu": adaptive.cpu_time,
        "adaptive_error_omega": adaptive.error_omega_history,
        "adaptive_error_r": adaptive.error_r_history,
        "adaptive_step_cpu": adaptive.step_cpu_times,
        "adaptive_step_accepted": adaptive.step_accepted_mask,
        "adaptive_omega_snapshot": adaptive.omega,
    }
    adaptive_ref_error = relative_errors(ops, adaptive.omega, reference)
    data["adaptive_reference_error"] = adaptive_ref_error

    for i, (dt, result) in enumerate(zip(FIXED_STEPS, fixed_results)):
        prefix = f"fixed_{i}"
        data[f"{prefix}_dt_nominal"] = np.array(dt)
        data[f"{prefix}_t"] = result["t"]
        data[f"{prefix}_q"] = result["q"]
        data[f"{prefix}_dt"] = result["dt"]
        data[f"{prefix}_cpu"] = result["cpu"]
        data[f"{prefix}_omega_snapshot"] = result["omega_snapshot"]
        data[f"{prefix}_reference_error"] = relative_errors(
            ops, result["omega_snapshot"], reference,
        )

    np.savez_compressed(output_path, **data)
    print(f"Saved results to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--initial-scale", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--output-interval", type=float, default=0.1)
    parser.add_argument("--reference-dt", type=float, default=2.5e-4)
    parser.add_argument("--adaptive-dt0", type=float, default=ADAPTIVE_DT0)
    parser.add_argument("--tol-omega", type=float, default=TOL_OMEGA)
    parser.add_argument("--tol-r", type=float, default=TOL_R)
    parser.add_argument("--safety", type=float, default=0.9)
    parser.add_argument("--dt-min", type=float, default=DT_MIN)
    parser.add_argument("--dt-max", type=float, default=DT_MAX)
    parser.add_argument("--fftw-threads", type=int, default=1)
    parser.add_argument("--use-pyfftw", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/adaptive_performance_K10_pilot.npz"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
