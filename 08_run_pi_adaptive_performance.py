"""Run the same performance experiment with the repository PI controller.

This is a diagnostic companion to ``06_run_adaptive_performance.py``.  It
uses the same physical problem and fixed-step baselines, but calls the
embedded-error PI controller already exposed by the solver.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from imex_mrsav_rk_solver import (
    PIAdaptiveOptions,
    make_periodic_ops,
    solve_adaptive,
    vorticity_energy,
)


_SPEC = importlib.util.spec_from_file_location(
    "adaptive_performance_helpers",
    Path(__file__).with_name("06_run_adaptive_performance.py"),
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load adaptive-performance helpers")
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--initial-scale", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--output-interval", type=float, default=0.1)
    parser.add_argument("--reference-dt", type=float, default=2.5e-4)
    parser.add_argument("--adaptive-dt0", type=float, default=5.0e-4)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--safety", type=float, default=0.9)
    parser.add_argument("--dt-min", type=float, default=1.0e-5)
    parser.add_argument("--dt-max", type=float, default=1.0e-2)
    parser.add_argument("--fftw-threads", type=int, default=1)
    parser.add_argument("--use-pyfftw", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pi_adaptive_performance_scale20_T20_64.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.K < 1 or args.initial_scale <= 0.0:
        raise ValueError("K and initial scale must be positive")
    if args.final_time <= 0.0 or args.output_interval <= 0.0:
        raise ValueError("final time and output interval must be positive")

    _HELPERS.solver.HAS_FFTW = bool(args.use_pyfftw)
    if args.use_pyfftw and _HELPERS.solver.pyfftw is not None:
        # The high-level pyFFTW interface can retain large temporary arrays
        # during very long ETDRK4 runs.  Plan caching is unnecessary here
        # because all transforms use fixed shapes, and disabling it keeps the
        # production reference memory bounded.
        _HELPERS.solver.pyfftw.interfaces.cache.disable()
    resolution = (args.resolution, args.resolution)
    ops = make_periodic_ops(_HELPERS.DOMAIN, resolution, fftw_threads=args.fftw_threads)
    omega0 = _HELPERS.initial_vorticity(ops, args.K, args.initial_scale)
    output_count = int(np.rint(args.final_time / args.output_interval))
    if not np.isclose(output_count * args.output_interval, args.final_time):
        raise ValueError("output interval must divide final time")
    output_times = np.linspace(0.0, args.final_time, output_count + 1)

    initial_energy = vorticity_energy(ops, omega0)[0]
    initial_velocity_norm = np.sqrt(2.0 * initial_energy)
    initial_reynolds = initial_velocity_norm / _HELPERS.NU
    print(
        f"PI controller, K={args.K}, scale={args.initial_scale:g}, "
        f"Re_in={initial_reynolds:.8e}"
    )

    print("Computing ETDRK4 reference ...", flush=True)
    start = time.perf_counter()
    reference = _HELPERS.etdrk4_reference(
        ops, omega0, args.reference_dt, output_times,
    )
    print(f"Reference complete in {time.perf_counter() - start:.2f} s", flush=True)

    print("Computing PI-adaptive run ...", flush=True)
    start = time.perf_counter()
    adaptive = solve_adaptive(
        omega0,
        nu=_HELPERS.NU,
        gamma=args.gamma,
        domain=_HELPERS.DOMAIN,
        discrete_num=resolution,
        dt0=args.adaptive_dt0,
        t_span=(0.0, args.final_time),
        method="ars222",
        force=_HELPERS.force,
        adaptive_options=PIAdaptiveOptions(
            tol=args.tol,
            safety=args.safety,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
        ),
        output_times=output_times,
        keep_omega=True,
        fftw_threads=args.fftw_threads,
    )
    print(
        f"PI-adaptive complete in {time.perf_counter() - start:.2f} s: "
        f"accepted={adaptive.accepted_steps}, rejected={adaptive.rejected_steps}",
        flush=True,
    )

    fixed_results = []
    for dt in _HELPERS.FIXED_STEPS:
        print(f"Computing fixed run dt={dt:.4g} ...", flush=True)
        start = time.perf_counter()
        fixed_results.append(
            _HELPERS.run_fixed(ops, omega0, dt, output_times, args.gamma)
        )
        print(f"  complete in {time.perf_counter() - start:.2f} s", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "controller": np.array("pi"),
        "K": np.array(args.K),
        "initial_scale": np.array(args.initial_scale),
        "resolution": np.array(resolution),
        "nu": np.array(_HELPERS.NU),
        "gamma": np.array(args.gamma),
        "forcing_amplitude": np.array(_HELPERS.FORCING_AMPLITUDE),
        "forcing_wavenumber": np.array(_HELPERS.FORCING_WAVENUMBER),
        "final_time": np.array(args.final_time),
        "output_interval": np.array(args.output_interval),
        "reference_dt": np.array(args.reference_dt),
        "tol_omega": np.array(args.tol),
        "tol_r": np.array(np.nan),
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
        "adaptive_cpu": adaptive.cpu_time,
        "adaptive_error_omega": adaptive.error_history,
        "adaptive_error_r": np.abs(1.0 - adaptive.q[1:]),
        "adaptive_step_cpu": adaptive.step_cpu_times,
        "adaptive_step_accepted": adaptive.step_accepted_mask,
        "adaptive_omega_snapshot": adaptive.omega,
    }
    data["adaptive_reference_error"] = _HELPERS.relative_errors(
        ops, adaptive.omega, reference,
    )

    for i, (dt, result) in enumerate(zip(_HELPERS.FIXED_STEPS, fixed_results)):
        prefix = f"fixed_{i}"
        data[f"{prefix}_dt_nominal"] = np.array(dt)
        data[f"{prefix}_t"] = result["t"]
        data[f"{prefix}_q"] = result["q"]
        data[f"{prefix}_dt"] = result["dt"]
        data[f"{prefix}_cpu"] = result["cpu"]
        data[f"{prefix}_omega_snapshot"] = result["omega_snapshot"]
        data[f"{prefix}_reference_error"] = _HELPERS.relative_errors(
            ops, result["omega_snapshot"], reference,
        )

    np.savez_compressed(output_path, **data)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
