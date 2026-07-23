"""Compare adaptive and fixed-step SDIRK2-mr-SAV integrations."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from solver import mrSAV_Vorticity_Stream_Periodic_Solve as Solver


METHOD = "SDIRK2_mr_SAV"


def make_force_term(m: float):
    def force_term(X, Y, t):
        return m * np.cos(m * Y)
    return force_term

def initial_streamfunction(x: np.ndarray, y: np.ndarray, nu: float, m: float, eps: float) -> np.ndarray:
    base_flow = -(1.0 / (nu * m**3)) * np.cos(m * y)

    k_max = 10
    k1_vals = np.arange(-k_max, k_max + 1)
    k2_vals = np.arange(-k_max, k_max + 1)
    k1_grid, k2_grid = np.meshgrid(k1_vals, k2_vals, indexing="ij")
    k_mod = np.sqrt(k1_grid**2 + k2_grid**2)

    mask = k_mod <= 10
    k1_valid = k1_grid[mask]
    k2_valid = k2_grid[mask]
    k_mod_valid = k_mod[mask]

    perturbation = np.zeros_like(x, dtype=np.float64)
    for k1, k2, k_abs in zip(k1_valid, k2_valid, k_mod_valid):
        if k_abs < 1e-10:
            continue
        term = (1 / (k_abs**3)) * (
            np.cos(k1 * x) * np.cos(k2 * y)
            + np.sin(k1 * x) * np.cos(k2 * y)
            + np.cos(k1 * x) * np.sin(k2 * y)
            + np.sin(k1 * x) * np.sin(k2 * y)
        )
        perturbation += term

    return 0 * base_flow + eps * perturbation


def tol_key(rtol, rtol_q):
    return (float(f"{rtol:.12g}"), float(f"{rtol_q:.12g}"))


def default_tolerance_lists():
    tol_grid_list = [
        (5e-5, 1e-2)
        # (5e-4, 1e-4),
        # (5e-4, 1e-3),
        # (1e-4, 1e-4),
        # (1e-4, 5e-4),
        # (1e-4, 1e-3),
        # (1e-3, 1e-4),
        # (1e-3, 5e-4),
        # (1e-3, 1e-3),
    ]

    tol_perturb_list = [
        (5e-5*1.1, 1e-2*1.1),
        (5e-5*1.1, 1e-2    ),
        (5e-5*1.1, 1e-2*0.9),
        (5e-5,     1e-2*1.1),
        (5e-5,     1e-2    ),
        (5e-5,     1e-2*0.9),
        (5e-5*0.9, 1e-2*1.1),
        (5e-5*0.9, 1e-2    ),
        (5e-5*0.9, 1e-2*0.9),
    ]

    # Compute every requested pair once, including single-entry lists.
    tol_list = list(dict.fromkeys(tol_perturb_list + tol_grid_list))
    return tol_grid_list, tol_perturb_list, tol_list


def default_fixed_configs():
    return [
        {
            "key": "tau_0p004",
            "tau": 0.004,
        },
        {
            "key": "tau_0p002",
            "tau": 0.002,
        },
        {
            "key": "tau_0p0015",
            "tau": 0.0015,
        },
        {
            "key": "tau_0p001",
            "tau": 0.001,
        },
        {
            "key": "tau_0p0005",
            "tau": 0.0005,
        },
    ]


def fixed_end_time(t_period, tau):
    n_steps = int(np.ceil((t_period[1] - t_period[0]) / tau))
    return t_period[0] + n_steps * tau


def compute_reference_error(metric_solver, omega, omega_t, ref_omega, ref_t):
    ref_dt = ref_t[1] - ref_t[0]
    ref_idx = np.rint((omega_t - ref_t[0]) / ref_dt).astype(int)

    if np.any(ref_idx < 0) or np.any(ref_idx >= len(ref_t)):
        raise ValueError("Fixed-step time grid is outside the reference grid.")

    if not np.allclose(ref_t[ref_idx], omega_t, atol=1e-10, rtol=1e-10):
        raise ValueError("Fixed-step time grid is not aligned with reference grid.")

    ref_error = np.zeros(len(omega))
    for i, j in enumerate(ref_idx):
        diff = omega[i] - ref_omega[j]
        ref_norm = metric_solver.inner_product(ref_omega[j], ref_omega[j])
        ref_error[i] = np.sqrt(metric_solver.inner_product(diff, diff) / ref_norm)
    return ref_error


def format_param_for_filename(value: float) -> str:
    return f"{value:g}"


def add_parameter_suffix(path: Path, eps: float, t_ini: float) -> Path:
    suffix = f"_{format_param_for_filename(eps)}_{format_param_for_filename(t_ini)}"
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def make_initial_vorticity(nu, gam, m, eps, t_ini, s_domain, discrete_num, force_term):
    xn = np.linspace(s_domain[0], s_domain[2], discrete_num[0] + 1)
    yn = np.linspace(s_domain[1], s_domain[3], discrete_num[1] + 1)
    X, Y = np.meshgrid(xn, yn)

    initial_phi = initial_streamfunction(X[:-1, :-1], Y[:-1, :-1], nu, m, eps)
    solver_init = Solver(
        nu,
        gam,
        s_domain,
        discrete_num,
        np.zeros_like(X),
        force_term,
        "ETDRK4",
        force_time_dependent=False,
    )
    u, v = solver_init.stream2velocity(initial_phi)
    solver_init.Omega0 = solver_init.velocity2vorticity(u, v)
    initial_vorticity = np.pad(solver_init.Omega0, ((0, 1), (0, 1)))
    if t_ini <= 0:
        return initial_vorticity
    solver_init.solve_fix_step((0, t_ini), 0.0025)
    return np.pad(solver_init.Omega[-1], ((0, 1), (0, 1)))


def load_fixed_cache(path, metadata, fixed_configs):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            for key, expected in metadata.items():
                if key not in data:
                    return None
                actual = np.asarray(data[key])
                expected = np.asarray(expected)
                if actual.shape != expected.shape:
                    return None
                if expected.dtype.kind in "US":
                    if not np.array_equal(actual, expected):
                        return None
                elif not np.allclose(actual, expected, rtol=0.0, atol=1e-14):
                    return None

            if int(data["fixed_count"]) != len(fixed_configs):
                return None
            fixed_results = []
            for i, cfg in enumerate(fixed_configs):
                if str(data[f"fixed_{i}_key"].item()) != cfg["key"]:
                    return None
                if not np.isclose(float(data[f"fixed_{i}_tau"]), cfg["tau"]):
                    return None
                result = {
                    **cfg,
                    "tn": data[f"fixed_{i}_tn"].copy(),
                    "tau_values": data[f"fixed_{i}_tau_values"].copy(),
                    "q": data[f"fixed_{i}_q"].copy(),
                    "cpu_time": data[f"fixed_{i}_cpu_time"].copy(),
                    "ref_error": data[f"fixed_{i}_ref_error"].copy(),
                }
                if (
                    len(result["q"]) != len(result["tn"])
                    or len(result["cpu_time"]) != len(result["tn"])
                    or len(result["ref_error"]) != len(result["tn"])
                    or len(result["tau_values"]) != len(result["tn"]) - 1
                    or not all(np.all(np.isfinite(result[key])) for key in (
                        "tn", "tau_values", "q", "cpu_time", "ref_error"
                    ))
                ):
                    return None
                fixed_results.append(result)
            ref_tn = data["ref_tn"].copy()
            if ref_tn.ndim != 1 or len(ref_tn) < 2 or not np.all(np.isfinite(ref_tn)):
                return None
            return fixed_results, ref_tn
    except (KeyError, OSError, ValueError):
        return None


def run_experiment(
    output_path: Path,
    force: bool = False,
    recompute_fixed: bool = False,
    adaptive_tolerances=None,
    compute_ref_err: bool = True,
    *,
    nu: float = 1 / 40,
    m: float = 4,
    eps: float = 3,
    t_ini: float = 0,
    gam: float = 1000,
):
    output_path = add_parameter_suffix(output_path, eps, t_ini)

    if output_path.exists() and not (force or recompute_fixed):
        logging.info("Output already exists: %s", output_path)
        logging.info("Use --force to recompute and overwrite it.")
        return

    np.random.seed(1)

    s_domain = (0.0, 0.0, 2 * np.pi, 2 * np.pi)
    discrete_num = [256, 256]
    t_period = (0.0, 20.0)
    adaptive_cache_version = 5
    fixed_cache_version = 4
    ref_tau = 0.00025
    force_term = make_force_term(m)

    if adaptive_tolerances is None:
        tol_grid_list, tol_perturb_list, tol_list = default_tolerance_lists()
    else:
        tol_grid_list = list(dict.fromkeys(adaptive_tolerances))
        tol_perturb_list = []
        tol_list = tol_grid_list
    fixed_configs = default_fixed_configs()
    fixed_metadata = {
        "method": METHOD,
        "fixed_cache_version": fixed_cache_version,
        "nu": nu,
        "m": m,
        "eps": eps,
        "t_ini": t_ini,
        "gam": gam,
        "s_domain": s_domain,
        "discrete_num": discrete_num,
        "t_period": t_period,
        "ref_tau": ref_tau,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Parameters: nu=%g, m=%g, eps=%g, t_ini=%g, gam=%g",
        nu,
        m,
        eps,
        t_ini,
        gam,
    )
    logging.info("Building initial vorticity")
    t0 = time.perf_counter()
    initial_vorticity = make_initial_vorticity(
        nu,
        gam,
        m,
        eps,
        t_ini,
        s_domain,
        discrete_num,
        force_term,
    )
    logging.info("Initial vorticity ready in %.2f s", time.perf_counter() - t0)

    all_adaptive_results = []
    for idx, (rtol, rtol_q) in enumerate(tol_list, start=1):
        logging.info(
            "Adaptive run %d/%d: rtol=%.3e, rtol_q=%.3e",
            idx,
            len(tol_list),
            rtol,
            rtol_q,
        )
        t0 = time.perf_counter()
        solver = Solver(
            nu,
            gam,
            s_domain,
            discrete_num,
            initial_vorticity,
            force_term,
            METHOD,
            force_time_dependent=False,
        )
        solver.solve_adaptive_step(
            t_period,
            1e-5,
            1e-2,
            snapshot=np.linspace(0, t_period[1], 201),
            compute_ref_err=compute_ref_err,
            rho   = 0.9,
            rtol  = rtol,
            rtol_q= rtol_q,
            r     = 1/2,
            ref_substeps = 4
        )
        wall_time = time.perf_counter() - t0
        all_adaptive_results.append(
            {
                "rtol": rtol,
                "rtol_q": rtol_q,
                "cpu_time": solver.cpu_time.copy(),
                "step_count": len(solver.tn),
                "tn": solver.tn.copy(),
                "tau": solver.tau.copy(),
                "q": solver.q.copy(),
                "ref_err": solver.ref_err.copy(),
                "rel_err": solver.rel_err.copy(),
                "controller_err": solver.controller_err.copy(),
                "ref_err_p": solver.ref_err_p.copy(),
                "accepted_steps": solver.accepted_steps,
                "rejected_steps": solver.rejected_steps,
                "forced_accept_steps": solver.forced_accept_steps,
                "wall_time": wall_time,
            }
        )
        logging.info(
            "Adaptive run %d finished in %.2f s with %d steps",
            idx,
            wall_time,
            len(solver.tn),
        )
        logging.info(
            "  accepted=%d, rejected=%d, forced=%d",
            solver.accepted_steps,
            solver.rejected_steps,
            solver.forced_accept_steps,
        )

    def save_output(fixed_results, ref_tn):
        save_data = {
            "method": np.array(METHOD),
            "adaptive_cache_version": np.array(adaptive_cache_version),
            "fixed_cache_version": np.array(fixed_cache_version),
            "nu": np.array(nu),
            "m": np.array(m),
            "eps": np.array(eps),
            "t_ini": np.array(t_ini),
            "gam": np.array(gam),
            "s_domain": np.array(s_domain),
            "discrete_num": np.array(discrete_num),
            "t_period": np.array(t_period),
            "ref_tau": np.array(ref_tau),
            "adaptive_ref_error_computed": np.array(compute_ref_err),
            "tol_grid_rtol": np.array([tol[0] for tol in tol_grid_list]),
            "tol_grid_rtol_q": np.array([tol[1] for tol in tol_grid_list]),
            "tol_perturb_rtol": np.array([tol[0] for tol in tol_perturb_list]),
            "tol_perturb_rtol_q": np.array([tol[1] for tol in tol_perturb_list]),
            "adaptive_count": np.array(len(all_adaptive_results)),
            "fixed_count": np.array(len(fixed_results)),
            "ref_tn": ref_tn,
        }
        for i, res in enumerate(all_adaptive_results):
            save_data.update({
                f"adaptive_{i}_rtol": np.array(res["rtol"]),
                f"adaptive_{i}_rtol_q": np.array(res["rtol_q"]),
                f"adaptive_{i}_step_count": np.array(res["step_count"]),
                f"adaptive_{i}_cpu_time": res["cpu_time"],
                f"adaptive_{i}_tn": res["tn"],
                f"adaptive_{i}_tau": res["tau"],
                f"adaptive_{i}_q": res["q"],
                f"adaptive_{i}_ref_err": res["ref_err"],
                f"adaptive_{i}_rel_err": res["rel_err"],
                f"adaptive_{i}_controller_err": res["controller_err"],
                f"adaptive_{i}_ref_err_p": res["ref_err_p"],
                f"adaptive_{i}_accepted_steps": np.array(res["accepted_steps"]),
                f"adaptive_{i}_rejected_steps": np.array(res["rejected_steps"]),
                f"adaptive_{i}_forced_accept_steps": np.array(res["forced_accept_steps"]),
                f"adaptive_{i}_wall_time": np.array(res["wall_time"]),
            })
        for i, res in enumerate(fixed_results):
            save_data.update({
                f"fixed_{i}_key": np.array(res["key"]),
                f"fixed_{i}_tau": np.array(res["tau"]),
                f"fixed_{i}_tn": res["tn"],
                f"fixed_{i}_tau_values": res["tau_values"],
                f"fixed_{i}_q": res["q"],
                f"fixed_{i}_cpu_time": res["cpu_time"],
                f"fixed_{i}_ref_error": res["ref_error"],
            })
        np.savez(output_path, **save_data)
        logging.info("Saved data to %s", output_path)

    cached_fixed = None if recompute_fixed else load_fixed_cache(
        output_path, fixed_metadata, fixed_configs
    )
    if cached_fixed is not None:
        fixed_results, ref_tn = cached_fixed
        logging.info("Reusing %d cached fixed-step results from %s", len(fixed_results), output_path)
        save_output(fixed_results, ref_tn)
        return

    ref_t_period = (
        t_period[0],
        max(fixed_end_time(t_period, cfg["tau"]) for cfg in fixed_configs),
    )
    for cfg in fixed_configs:
        ratio = cfg["tau"] / ref_tau
        if not np.isclose(ratio, np.rint(ratio), rtol=0.0, atol=1e-10):
            raise ValueError(f"tau={cfg['tau']} is not aligned with ref_tau={ref_tau}")

    ref_step_count = (ref_t_period[1] - ref_t_period[0]) / ref_tau
    if not np.isclose(ref_step_count, np.rint(ref_step_count), rtol=0.0, atol=1e-10):
        raise ValueError("Reference end time is not aligned with ref_tau.")

    logging.info("Fixed-step reference run: tau=%.4g, t_period=%s", ref_tau, ref_t_period)
    t0 = time.perf_counter()
    ref_solver = Solver(
        nu,
        gam,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        "ETDRK4",
        force_time_dependent=False,
    )
    ref_snapshot_times = (
        ref_t_period[0]
        + ref_tau * np.arange(int(np.rint(ref_step_count)) + 1, dtype=np.float64)
    )
    ref_solver.solve_fix_step(ref_t_period, ref_tau, snapshot=ref_snapshot_times)
    logging.info("Reference run finished in %.2f s", time.perf_counter() - t0)

    metric_solver = Solver(
        nu,
        gam,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        METHOD,
        force_time_dependent=False,
    )

    fixed_results = []
    for idx, cfg in enumerate(fixed_configs, start=1):
        logging.info("Fixed run %d/%d: tau=%.4g", idx, len(fixed_configs), cfg["tau"])
        t0 = time.perf_counter()
        solver = Solver(
            nu,
            gam,
            s_domain,
            discrete_num,
            initial_vorticity,
            force_term,
            METHOD,
            force_time_dependent=False,
        )
        fixed_step_count = int(np.ceil((t_period[1] - t_period[0]) / cfg["tau"]))
        fixed_snapshot_times = (
            t_period[0]
            + cfg["tau"] * np.arange(fixed_step_count + 1, dtype=np.float64)
        )
        solver.solve_fix_step(t_period, cfg["tau"], snapshot=fixed_snapshot_times)
        ref_error = compute_reference_error(
            metric_solver,
            solver.Omega,
            solver.tn_s,
            ref_solver.Omega,
            ref_solver.tn_s,
        )
        fixed_results.append(
            {
                **cfg,
                "tn": solver.tn_s.copy(),
                "tau_values": solver.tau.copy(),
                "q": solver.q.copy(),
                "cpu_time": solver.cpu_time.copy(),
                "ref_error": ref_error,
            }
        )
        logging.info(
            "Fixed run %d finished in %.2f s with %d steps",
            idx,
            time.perf_counter() - t0,
            len(solver.tn),
        )

    save_output(fixed_results, ref_solver.tn.copy())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 03_mrSAV-VarStep-Exp simulations and save plotting data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/03_mrSAV_varstep_exp_data.npz"),
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute adaptive runs; reuse compatible fixed-step results.",
    )
    parser.add_argument(
        "--recompute-fixed",
        action="store_true",
        help="Recompute the ETDRK4 reference and all fixed-step results.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/03_mrSAV_varstep_exp.log"),
        help="Log file path.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=1/40,
        help="Viscosity parameter.",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=4,
        help="Mode parameter used in the initial condition and forcing.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=3,
        help="Perturbation amplitude in the initial condition.",
    )
    parser.add_argument(
        "--t-ini",
        "--t_ini",
        dest="t_ini",
        type=float,
        default=0,
        help="Initial spin-up time used to build the starting vorticity.",
    )
    parser.add_argument(
        "--gam",
        "--gamma",
        dest="gam",
        type=float,
        default=1000,
        help="Mean-reverting gamma parameter.",
    )
    parser.add_argument("--rtol", type=float, help="Run one adaptive vorticity tolerance.")
    parser.add_argument("--rtol-q", type=float, help="Run one adaptive auxiliary tolerance.")
    parser.add_argument(
        "--skip-ref-error",
        action="store_true",
        help="Skip the costly stepwise ETDRK4 reference integration for fast screening.",
    )
    args = parser.parse_args()
    if (args.rtol is None) != (args.rtol_q is None):
        parser.error("--rtol and --rtol-q must be specified together")
    if args.rtol is not None and (args.rtol <= 0 or args.rtol_q <= 0):
        parser.error("--rtol and --rtol-q must be positive")
    return args


def main():
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log, mode="a"),
        ],
    )
    run_experiment(
        args.output,
        force=args.force,
        recompute_fixed=args.recompute_fixed,
        adaptive_tolerances=(None if args.rtol is None else [(args.rtol, args.rtol_q)]),
        compute_ref_err=not args.skip_ref_error,
        nu=args.nu,
        m=args.m,
        eps=args.eps,
        t_ini=args.t_ini,
        gam=args.gam,
    )


if __name__ == "__main__":
    main()
