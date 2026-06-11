"""Run gamma-sweep diagnostics for the IMEX-mr-SAV-RK solver.

The setup follows the ETD mean-reverting test in ../ETD-mr-SAV-MS2:
the initial condition is generated from the same streamfunction and uses
stream2velocity(initial_phi)[0] as the vorticity-like initial data.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import h5py
import numpy as np

from imex_mrsav_rk_solver import (
    NewtonOptions,
    advection_term,
    inner_product,
    laplacian_hat,
    make_periodic_ops,
    make_tableau,
    make_taylor_v,
    prepare_initial_vorticity,
    step_imex_mrsav_rk,
    vorticity_energy,
)


DEFAULT_ETD_DIR = Path("/Users/wanghaifeng/Desktop/ETD-mr-SAV-MS2")
DEFAULT_OUTPUT = Path("data/imex_gamma_diagnostics_pilot.h5")
DEFAULT_GAMMAS = (0.0, 1.0, 10.0, 100.0, 1000.0)
DIAGNOSTIC_KEYS = (
    "q",
    "Energy",
    "Enstrophy",
    "Enstrophy_rate",
    "Palinstrophy",
    "Mx",
    "CPU_time",
)
ERROR_KEYS = ("l2_error", "rel_l2_error", "rel_enstrophy_error")


Array = np.ndarray
ForceFn = Callable[[Array, Array, float], Array]


@dataclass
class ImexState:
    gamma: float
    omega: Array
    q: float
    cpu_time: float
    denom_cache: dict


@dataclass
class EtdState:
    solver: object
    tau: float
    t: float
    omega_hist: list[Array]
    q_hist: list[float]
    cpu_time: float


def gamma_label(gamma: float) -> str:
    return f"gamma_{gamma:g}".replace("-", "m").replace(".", "p")


def initial_streamfunction(
    x: Array,
    y: Array,
    nu: float,
    m: float,
    eps: float,
) -> Array:
    base_flow = (1.0 / (nu * m**3)) * np.cos(m * y)

    k_max = 10
    k1_vals = np.arange(-k_max, k_max + 1)
    k2_vals = np.arange(-k_max, k_max + 1)
    k1_grid, k2_grid = np.meshgrid(k1_vals, k2_vals, indexing="ij")
    k_mod = np.sqrt(k1_grid**2 + k2_grid**2)

    mask = k_mod <= 10
    perturbation = np.zeros_like(x, dtype=np.float64)
    for k1, k2, k_abs in zip(k1_grid[mask], k2_grid[mask], k_mod[mask]):
        if k_abs < 1.0e-10:
            continue
        perturbation += (1.0 / k_abs**3) * (
            np.cos(k1 * x) * np.cos(k2 * y)
            + np.sin(k1 * x) * np.cos(k2 * y)
            + np.cos(k1 * x) * np.sin(k2 * y)
            + np.sin(k1 * x) * np.sin(k2 * y)
        )

    return 0.0 * base_flow + eps * perturbation


def make_force(m: float) -> ForceFn:
    def force(X: Array, Y: Array, t: float) -> Array:
        return m * np.cos(m * Y)

    return force


def load_etd_solver_class(etd_dir: Path):
    etd_dir = etd_dir.expanduser().resolve()
    if not (etd_dir / "vs_ns_periodic_mrSAV_solver.py").exists():
        raise FileNotFoundError(f"Cannot find ETD solver in {etd_dir}")
    if str(etd_dir) not in sys.path:
        sys.path.insert(0, str(etd_dir))
    from vs_ns_periodic_mrSAV_solver import (  # type: ignore
        vs_mrSAV_Vorticity_Stream_Periodic_Solver,
    )

    return vs_mrSAV_Vorticity_Stream_Periodic_Solver


def make_initial_vorticity(
    solver_class,
    nu: float,
    gamma_for_init: float,
    m: float,
    eps: float,
    domain: tuple[float, float, float, float],
    discrete_num: tuple[int, int],
    force: ForceFn,
) -> Array:
    xn = np.linspace(domain[0], domain[2], discrete_num[0] + 1)
    yn = np.linspace(domain[1], domain[3], discrete_num[1] + 1)
    x_grid, y_grid = np.meshgrid(xn, yn)
    initial_phi = initial_streamfunction(x_grid[:-1, :-1], y_grid[:-1, :-1], nu, m, eps)
    solver_init = solver_class(
        nu,
        gamma_for_init,
        domain,
        discrete_num,
        initial_phi,
        force,
        "ETDRK4",
    )
    return np.pad(solver_init.stream2velocity(initial_phi)[0], ((0, 1), (0, 1)))


def enstrophy_rate(ops, omega: Array, nu: float, force: ForceFn, t: float) -> float:
    omega_hat = ops.fft2(omega)
    diffusion = ops.ifft2(nu * laplacian_hat(ops, omega_hat)).real
    nonlinear = -advection_term(ops, omega, omega_hat=omega_hat)
    rhs = diffusion + nonlinear + force(ops.X, ops.Y, t)
    return inner_product(ops, rhs, omega)


def record_diagnostics(group, ops, omega: Array, q: float, t: float, cpu_time: float, nu: float, force: ForceFn, idx: int) -> None:
    energy, enstrophy, palinstrophy = vorticity_energy(ops, omega)
    group["q"][idx] = q
    group["Energy"][idx] = energy
    group["Enstrophy"][idx] = enstrophy
    group["Enstrophy_rate"][idx] = enstrophy_rate(ops, omega, nu, force, t)
    group["Palinstrophy"][idx] = palinstrophy
    group["Mx"][idx] = float(np.max(omega))
    group["CPU_time"][idx] = cpu_time


def init_etd_reference(solver_class, nu: float, gamma: float, domain, discrete_num, omega0: Array, force: ForceFn, tau_ref: float) -> EtdState:
    solver = solver_class(nu, gamma, domain, discrete_num, omega0, force, "ETDRK4")
    solver._prepare_ETDRK4_coefficients(tau_ref)
    return EtdState(
        solver=solver,
        tau=float(tau_ref),
        t=0.0,
        omega_hist=[solver.Omega0.copy()],
        q_hist=[solver.q0],
        cpu_time=0.0,
    )


def advance_etd_one_step(state: EtdState) -> tuple[Array, float]:
    solver = state.solver
    start = perf_counter()
    omega_new, q_new = solver.step(
        np.asarray(state.omega_hist[-solver.setup_step :]),
        np.asarray(state.q_hist[-solver.setup_step :]),
        state.t,
        np.full(solver.setup_step, state.tau, dtype=np.float64),
    )
    state.cpu_time += perf_counter() - start
    state.t += state.tau
    state.omega_hist.append(omega_new)
    state.q_hist.append(q_new)
    del state.omega_hist[:-solver.setup_step]
    del state.q_hist[:-solver.setup_step]
    return omega_new, float(q_new)


def create_output_file(path: Path, n_steps: int, t: Array, gamma_values: tuple[float, ...], attrs: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.create_group("time").create_dataset("t", data=t)
    chunk = (min(10_000, n_steps + 1),)

    for gamma in gamma_values:
        group = f.create_group(gamma_label(gamma))
        for key in DIAGNOSTIC_KEYS:
            group.create_dataset(key, shape=(n_steps + 1,), dtype=np.float64, chunks=chunk, fillvalue=np.nan)
        err_group = f.create_group(f"errors/{gamma_label(gamma)}")
        for key in ERROR_KEYS:
            err_group.create_dataset(key, shape=(n_steps + 1,), dtype=np.float64, chunks=chunk, fillvalue=np.nan)

    ref_group = f.create_group("etdrk4_ref")
    for key in DIAGNOSTIC_KEYS:
        ref_group.create_dataset(key, shape=(n_steps + 1,), dtype=np.float64, chunks=chunk, fillvalue=np.nan)

    for key, value in attrs.items():
        f.attrs[key] = value
    f.attrs["complete"] = False
    f.attrs["completed_step"] = 0
    f.attrs["completed_time"] = float(t[0])
    return f


def run_gamma_diagnostics(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    etd_dir: Path = DEFAULT_ETD_DIR,
    T: float = 10.0,
    dt: float = 1.0e-3,
    dt_ref: float = 5.0e-4,
    Re: float = 40.0,
    m: float = 4.0,
    eps: float = 0.25,
    gamma_values: tuple[float, ...] = DEFAULT_GAMMAS,
    reference_gamma: float = 1000.0,
    method: str = "rk3",
    discrete_num: tuple[int, int] = (64, 64),
    domain: tuple[float, float, float, float] = (0.0, 0.0, 2.0 * np.pi, 2.0 * np.pi),
    checkpoint_every: int = 100,
) -> Path:
    n_steps = int(round(T / dt))
    if n_steps <= 0 or not np.isclose(n_steps * dt, T, rtol=0.0, atol=max(1.0e-12, dt * 1.0e-8)):
        raise ValueError("T must be a positive integer multiple of dt")
    ref_steps_per_step = int(round(dt / dt_ref))
    if ref_steps_per_step <= 0 or not np.isclose(ref_steps_per_step * dt_ref, dt, rtol=0.0, atol=max(1.0e-12, dt_ref * 1.0e-8)):
        raise ValueError("dt must be a positive integer multiple of dt_ref")

    solver_class = load_etd_solver_class(etd_dir)
    nu = 1.0 / Re
    force = make_force(m)
    ops = make_periodic_ops(domain, discrete_num)
    tableau = make_tableau(method)
    V, dV = make_taylor_v(tableau.order)
    newton_options = NewtonOptions()
    omega0 = make_initial_vorticity(solver_class, nu, reference_gamma, m, eps, domain, discrete_num, force)
    omega0_imex = prepare_initial_vorticity(omega0, ops, project_mean=True)

    states = [
        ImexState(
            gamma=float(gamma),
            omega=omega0_imex.copy(),
            q=1.0,
            cpu_time=0.0,
            denom_cache={},
        )
        for gamma in gamma_values
    ]
    ref_state = init_etd_reference(solver_class, nu, reference_gamma, domain, discrete_num, omega0, force, dt_ref)
    ref_omega = prepare_initial_vorticity(ref_state.omega_hist[-1], ops, project_mean=True)
    ref_q = float(ref_state.q_hist[-1])

    t_values = dt * np.arange(n_steps + 1, dtype=np.float64)
    attrs = {
        "gamma_values": np.asarray(gamma_values, dtype=np.float64),
        "reference_gamma": float(reference_gamma),
        "method": method,
        "grid": np.asarray(discrete_num, dtype=np.int64),
        "T": float(T),
        "dt": float(dt),
        "dt_ref": float(dt_ref),
        "Re": float(Re),
        "m": float(m),
        "eps": float(eps),
        "reference": "ETDRK4",
        "etd_dir": str(etd_dir),
    }

    checkpoint_every = max(1, int(checkpoint_every))
    with create_output_file(output_path, n_steps, t_values, gamma_values, attrs) as f:
        record_diagnostics(f["etdrk4_ref"], ops, ref_omega, ref_q, 0.0, 0.0, nu, force, 0)
        ref_energy, ref_enstrophy, _ = vorticity_energy(ops, ref_omega)
        del ref_energy
        for state in states:
            group = f[gamma_label(state.gamma)]
            record_diagnostics(group, ops, state.omega, state.q, 0.0, 0.0, nu, force, 0)
            err_group = f[f"errors/{gamma_label(state.gamma)}"]
            err_group["l2_error"][0] = 0.0
            err_group["rel_l2_error"][0] = 0.0
            err_group["rel_enstrophy_error"][0] = 0.0

        f.flush()
        wall_start = perf_counter()
        for step in range(1, n_steps + 1):
            t_n = t_values[step - 1]
            for state in states:
                start = perf_counter()
                omega_new, q_new, _ = step_imex_mrsav_rk(
                    state.omega,
                    state.q,
                    t_n,
                    dt,
                    ops,
                    nu,
                    state.gamma,
                    tableau,
                    force=force,
                    V=V,
                    dV=dV,
                    newton_options=newton_options,
                    denom_cache=state.denom_cache,
                    project_mean=True,
                )
                state.cpu_time += perf_counter() - start
                state.omega = omega_new
                state.q = q_new

            for _ in range(ref_steps_per_step):
                ref_omega_raw, ref_q = advance_etd_one_step(ref_state)
            ref_omega = prepare_initial_vorticity(ref_omega_raw, ops, project_mean=True)

            t_now = t_values[step]
            record_diagnostics(f["etdrk4_ref"], ops, ref_omega, ref_q, t_now, ref_state.cpu_time, nu, force, step)
            _, ref_enstrophy, _ = vorticity_energy(ops, ref_omega)
            ref_norm = np.sqrt(inner_product(ops, ref_omega, ref_omega))

            for state in states:
                group_name = gamma_label(state.gamma)
                record_diagnostics(f[group_name], ops, state.omega, state.q, t_now, state.cpu_time, nu, force, step)
                _, ens, _ = vorticity_energy(ops, state.omega)
                diff = state.omega - ref_omega
                l2_error = np.sqrt(inner_product(ops, diff, diff))
                err_group = f[f"errors/{group_name}"]
                err_group["l2_error"][step] = l2_error
                err_group["rel_l2_error"][step] = l2_error / ref_norm if ref_norm > 0.0 else np.nan
                err_group["rel_enstrophy_error"][step] = abs(ens - ref_enstrophy) / abs(ref_enstrophy) if ref_enstrophy != 0.0 else np.nan

            if step % checkpoint_every == 0 or step == n_steps:
                f.attrs["completed_step"] = step
                f.attrs["completed_time"] = float(t_now)
                f.flush()
                elapsed = perf_counter() - wall_start
                print(f"step {step}/{n_steps}, t={t_now:.6g}, elapsed={elapsed:.2f}s", flush=True)

        f.attrs["complete"] = True
        f.attrs["completed_step"] = n_steps
        f.attrs["completed_time"] = float(t_values[-1])
        f.flush()

    return output_path


def parse_gamma_values(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--etd-dir", type=Path, default=DEFAULT_ETD_DIR)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--dt-ref", type=float, default=5.0e-4)
    parser.add_argument("--Re", type=float, default=40.0)
    parser.add_argument("--m", type=float, default=4.0)
    parser.add_argument("--eps", type=float, default=0.25)
    parser.add_argument("--gammas", type=parse_gamma_values, default=DEFAULT_GAMMAS)
    parser.add_argument("--reference-gamma", type=float, default=1000.0)
    parser.add_argument("--method", choices=("rk1", "rk2", "rk3", "rk4"), default="rk3")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run_gamma_diagnostics(
        output_path=args.output,
        etd_dir=args.etd_dir,
        T=args.T,
        dt=args.dt,
        dt_ref=args.dt_ref,
        Re=args.Re,
        m=args.m,
        eps=args.eps,
        gamma_values=args.gammas,
        reference_gamma=args.reference_gamma,
        method=args.method,
        discrete_num=(args.nx, args.ny),
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Saved diagnostics to {path}")


if __name__ == "__main__":
    main()
