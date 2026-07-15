"""Run a short performance comparison for extrapolation PI-gamma control."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

import imex_mrsav_rk_solver as solver
from imex_mrsav_rk_solver import (
    ExtrapolationAdaptiveOptions,
    ExtrapolationPIGammaOptions,
    make_periodic_ops,
    solve_adaptive_extrapolation,
    solve_adaptive_extrapolation_pi_gamma,
)


def load_fixed_experiment_helpers():
    path = Path(__file__).with_name("06_run_adaptive_performance.py")
    spec = importlib.util.spec_from_file_location("fixed_performance_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--initial-scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--output-interval", type=float, default=0.1)
    parser.add_argument("--reference-dt", type=float, default=2.5e-4)
    parser.add_argument(
        "--reference-npz",
        type=Path,
        default=None,
        help="Reuse reference_omega and output_times from an existing NPZ file.",
    )
    parser.add_argument("--adaptive-dt0", type=float, default=1.0e-3)
    parser.add_argument("--tol-omega", type=float, default=1.0e-5)
    parser.add_argument("--tol-r", type=float, default=5.0e-4)
    parser.add_argument("--qbar", type=float, default=0.80)
    parser.add_argument("--gamma-fixed", type=float, default=1000.0)
    parser.add_argument("--dt-min", type=float, default=1.0e-5)
    parser.add_argument("--dt-max", type=float, default=1.0e-2)
    parser.add_argument("--fftw-threads", type=int, default=1)
    parser.add_argument("--use-pyfftw", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/extrapolation_pi_gamma_pilot_K10_T1_64.npz"),
    )
    args = parser.parse_args()

    helpers = load_fixed_experiment_helpers()
    solver.HAS_FFTW = bool(args.use_pyfftw)
    resolution = (args.resolution, args.resolution)
    ops = make_periodic_ops(helpers.DOMAIN, resolution, fftw_threads=args.fftw_threads)
    omega0 = helpers.initial_vorticity(ops, args.K, scale=args.initial_scale)
    output_count = int(np.rint(args.final_time / args.output_interval))
    if output_count <= 0 or not np.isclose(output_count * args.output_interval, args.final_time):
        raise ValueError("output interval must divide final time")
    output_times = np.linspace(0.0, args.final_time, output_count + 1)

    print(
        f"K={args.K}, resolution={args.resolution}^2, T={args.final_time:g}, "
        f"tol=({args.tol_omega:.1e},{args.tol_r:.1e}), qbar={args.qbar:g}"
    )
    if args.reference_npz is None:
        print(f"Computing ETDRK4 reference with dt={args.reference_dt:.3e} ...", flush=True)
        start = time.perf_counter()
        reference = helpers.etdrk4_reference(ops, omega0, args.reference_dt, output_times)
        print(f"  reference CPU={time.perf_counter() - start:.3f}s", flush=True)
    else:
        reference_data = np.load(args.reference_npz)
        reference_times = np.asarray(reference_data["output_times"])
        if not np.allclose(reference_times, output_times, rtol=0.0, atol=1.0e-12):
            raise ValueError("reference NPZ output_times do not match this experiment")
        reference = np.asarray(reference_data["reference_omega"])
        print(f"Reusing ETDRK4 reference from {args.reference_npz}", flush=True)

    common_options = dict(
        tol_omega=args.tol_omega,
        tol_r=args.tol_r,
        safety=0.9,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_rejections=100,
        max_steps=200000,
    )

    print("Computing fixed-gamma extrapolation-I run ...", flush=True)
    start = time.perf_counter()
    fixed_gamma = solve_adaptive_extrapolation(
        omega0,
        nu=helpers.NU,
        gamma=args.gamma_fixed,
        domain=helpers.DOMAIN,
        discrete_num=resolution,
        dt0=args.adaptive_dt0,
        t_span=(0.0, args.final_time),
        method="ars222",
        force=helpers.force,
        adaptive_options=ExtrapolationAdaptiveOptions(**common_options),
        output_times=output_times,
        fftw_threads=args.fftw_threads,
    )
    fixed_gamma_cpu = time.perf_counter() - start
    print(
        f"  accepted={fixed_gamma.accepted_steps}, rejected={fixed_gamma.rejected_steps}, "
        f"CPU={fixed_gamma_cpu:.3f}s",
        flush=True,
    )

    print("Computing adaptive-gamma extrapolation-PI run ...", flush=True)
    start = time.perf_counter()
    adaptive_gamma = solve_adaptive_extrapolation_pi_gamma(
        omega0,
        nu=helpers.NU,
        domain=helpers.DOMAIN,
        discrete_num=resolution,
        dt0=args.adaptive_dt0,
        t_span=(0.0, args.final_time),
        method="ars222",
        force=helpers.force,
        adaptive_options=ExtrapolationPIGammaOptions(
            **common_options,
            qbar=args.qbar,
        ),
        output_times=output_times,
        fftw_threads=args.fftw_threads,
    )
    adaptive_gamma_cpu = time.perf_counter() - start
    print(
        f"  accepted={adaptive_gamma.accepted_steps}, rejected={adaptive_gamma.rejected_steps}, "
        f"floor={adaptive_gamma.floor_accepted_steps}, CPU={adaptive_gamma_cpu:.3f}s",
        flush=True,
    )

    print("Computing fixed-step baselines ...", flush=True)
    fixed_steps = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)
    fixed_results = []
    for dt in fixed_steps:
        start = time.perf_counter()
        result = helpers.run_fixed(ops, omega0, dt, output_times, args.gamma_fixed)
        elapsed = time.perf_counter() - start
        print(f"  dt={dt:.4g}, steps={len(result['dt'])}, CPU={elapsed:.3f}s", flush=True)
        fixed_results.append((dt, result, elapsed))

    data = {
        "K": np.array(args.K),
        "initial_scale": np.array(args.initial_scale),
        "resolution": np.array(resolution),
        "nu": np.array(helpers.NU),
        "final_time": np.array(args.final_time),
        "output_interval": np.array(args.output_interval),
        "reference_dt": np.array(args.reference_dt),
        "tol_omega": np.array(args.tol_omega),
        "tol_r": np.array(args.tol_r),
        "gamma_fixed": np.array(args.gamma_fixed),
        "qbar": np.array(args.qbar),
        "output_times": output_times,
        "omega0": omega0,
        "reference_omega": reference,
        "fixed_gamma_t": fixed_gamma.t,
        "fixed_gamma_q": fixed_gamma.q,
        "fixed_gamma_dt": fixed_gamma.dt_history,
        "fixed_gamma_cpu": fixed_gamma.cpu_time,
        "fixed_gamma_error_omega": fixed_gamma.error_omega_history,
        "fixed_gamma_error_r": fixed_gamma.error_r_history,
        "fixed_gamma_reference_error": helpers.relative_errors(ops, fixed_gamma.omega, reference),
        "fixed_gamma_accepted_steps": np.array(fixed_gamma.accepted_steps),
        "fixed_gamma_rejected_steps": np.array(fixed_gamma.rejected_steps),
        "adaptive_t": adaptive_gamma.t,
        "adaptive_q": adaptive_gamma.q,
        "adaptive_dt": adaptive_gamma.dt_history,
        "adaptive_gamma": adaptive_gamma.gamma_history,
        "adaptive_gamma_dt": adaptive_gamma.gamma_dt_history,
        "adaptive_damping_factor": adaptive_gamma.damping_factor_history,
        "adaptive_cpu": adaptive_gamma.cpu_time,
        "adaptive_error_omega": adaptive_gamma.error_omega_history,
        "adaptive_error_r": adaptive_gamma.error_r_history,
        "adaptive_effective_error": adaptive_gamma.effective_error_history,
        "adaptive_reference_error": helpers.relative_errors(ops, adaptive_gamma.omega, reference),
        "adaptive_accepted_steps": np.array(adaptive_gamma.accepted_steps),
        "adaptive_rejected_steps": np.array(adaptive_gamma.rejected_steps),
        "adaptive_floor_accepted_steps": np.array(adaptive_gamma.floor_accepted_steps),
        "adaptive_chi_qbar": np.array(adaptive_gamma.chi_qbar),
        "fixed_gamma_wall_cpu": np.array(fixed_gamma_cpu),
        "adaptive_wall_cpu": np.array(adaptive_gamma_cpu),
    }
    for i, (dt, result, elapsed) in enumerate(fixed_results):
        prefix = f"fixed_{i}"
        data[f"{prefix}_dt_nominal"] = np.array(dt)
        data[f"{prefix}_t"] = result["t"]
        data[f"{prefix}_q"] = result["q"]
        data[f"{prefix}_dt"] = result["dt"]
        data[f"{prefix}_cpu"] = result["cpu"]
        data[f"{prefix}_reference_error"] = helpers.relative_errors(
            ops, result["omega_snapshot"], reference
        )
        data[f"{prefix}_wall_cpu"] = np.array(elapsed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
