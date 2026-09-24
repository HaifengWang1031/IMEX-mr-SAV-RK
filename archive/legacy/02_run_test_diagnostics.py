from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from solver import mrSAV_Vorticity_Stream_Periodic_Solve as Solver


GAMMAS = (0.0, 50.0, 500, 1000.0, 2000.0)
DIAGNOSTICS = (
    "q", "Energy", "Enstrophy", "Enstrophy_rate",
    "Palinstrophy", "Mx", "CPU_time",
)
DEFAULT_OUTPUT = Path("data/test_bursting_diagnostics.h5")


def gamma_tag(gamma):
    return f"{gamma:g}".replace("-", "m").replace(".", "p")


def initial_data(domain, shape, m=4):
    x = np.linspace(domain[0], domain[2], shape[0] + 1)
    y = np.linspace(domain[1], domain[3], shape[1] + 1)
    X, Y = np.meshgrid(x, y)
    omega = sum(
        (k*k + j*j)**(-1.5) * np.cos(k*X) * np.cos(j*Y)
        for k in range(1, 11) for j in range(1, 11)
    )
    omega -= np.mean(omega[:-1, :-1])
    omega /= np.sqrt(np.mean(omega[:-1, :-1]**2))
    return omega, lambda x, y, t: m * np.cos(m*y)


def make_state(nu, gamma, domain, shape, omega0, force, method, tau):
    solver = Solver(
        nu, gamma, domain, shape, omega0, force, method,
        force_time_dependent=False,
    )
    solver._fixed_step_cache = {}
    solver._enable_fixed_step_cache = True
    return {
        "solver": solver,
        "tau": float(tau),
        "t": 0.0,
        "omega": [solver.Omega0.copy()],
        "q": [solver.q0],
        "cpu": 0.0,
    }


def advance(state):
    solver = state["solver"]
    tau = state["tau"]
    omega_hist, q_hist = state["omega"], state["q"]
    start = perf_counter()
    omega, q = solver.step(
        np.asarray(omega_hist[-solver.setup_step:]),
        np.asarray(q_hist[-solver.setup_step:]),
        state["t"],
        np.full(solver.setup_step, tau),
    )
    state["cpu"] += perf_counter() - start
    state["t"] += tau
    omega_hist.append(omega)
    q_hist.append(q)
    del omega_hist[:-solver.setup_step]
    del q_hist[:-solver.setup_step]
    return omega, q


def empty_diagnostics(n_steps):
    return {key: np.empty(n_steps + 1) for key in DIAGNOSTICS}


def record(data, solver, omega, q, t, cpu, index):
    energy, enstrophy, palinstrophy = solver.vorticity_energy(omega)
    data["q"][index] = q
    data["Energy"][index] = energy
    data["Enstrophy"][index] = enstrophy
    data["Enstrophy_rate"][index] = solver.enstrophy_rate(omega, t)
    data["Palinstrophy"][index] = palinstrophy
    data["Mx"][index] = np.max(omega)
    data["CPU_time"][index] = cpu


def save_results(path, times, gamma_data, reference_data, errors, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_group("time").create_dataset("t", data=times)
        for gamma, data in gamma_data.items():
            group = f.create_group(f"gamma_{gamma_tag(gamma)}")
            for key, values in data.items():
                group.create_dataset(key, data=values, compression="gzip")
        group = f.create_group("etdrk4_ref")
        for key, values in reference_data.items():
            group.create_dataset(key, data=values, compression="gzip")
        group = f.create_group("errors")
        for gamma, data in errors.items():
            tag = gamma_tag(gamma)
            group.create_dataset(f"l2_error_gamma_{tag}", data=data["l2"], compression="gzip")
            group.create_dataset(f"rel_l2_gamma_{tag}", data=data["relative"], compression="gzip")
        for key, value in attrs.items():
            f.attrs[key] = value


def run_diagnostics(
    output=DEFAULT_OUTPUT,
    *,
    gammas=GAMMAS,
    T=20.0,
    tau=1e-3,
    tau_ref=2.5e-4,
    Re=40.0,
    m=4,
    shape=(128, 128),
    domain=(0.0, 0.0, 2*np.pi, 2*np.pi),
    progress_every=100,
):
    gammas = tuple(float(gamma) for gamma in gammas)
    n_steps = int(round(T/tau))
    ref_substeps = int(round(tau/tau_ref))
    if not np.isclose(n_steps*tau, T) or not np.isclose(ref_substeps*tau_ref, tau):
        raise ValueError("T/tau and tau/tau_ref must be integers")

    omega0, force = initial_data(domain, shape, m)
    states = {
        gamma: make_state(1/Re, gamma, domain, shape, omega0, force, "SDIRK2_mr_SAV", tau)
        for gamma in gammas
    }
    reference = make_state(1/Re, 0.0, domain, shape, omega0, force, "ETDRK4", tau_ref)

    times = tau*np.arange(n_steps + 1)
    data = {gamma: empty_diagnostics(n_steps) for gamma in gammas}
    ref_data = empty_diagnostics(n_steps)
    errors = {
        gamma: {"l2": np.empty(n_steps), "relative": np.empty(n_steps)}
        for gamma in gammas
    }

    for gamma, state in states.items():
        record(data[gamma], state["solver"], state["omega"][-1], 1.0, 0.0, 0.0, 0)
    record(ref_data, reference["solver"], reference["omega"][-1], 1.0, 0.0, 0.0, 0)

    for step in range(1, n_steps + 1):
        for _ in range(ref_substeps):
            omega_ref, q_ref = advance(reference)
        record(ref_data, reference["solver"], omega_ref, q_ref, times[step], reference["cpu"], step)
        ref_norm = np.sqrt(reference["solver"].h*np.sum(omega_ref**2))

        for gamma, state in states.items():
            omega, q = advance(state)
            record(data[gamma], state["solver"], omega, q, times[step], state["cpu"], step)
            error = np.sqrt(state["solver"].h*np.sum((omega - omega_ref)**2))
            errors[gamma]["l2"][step - 1] = error
            errors[gamma]["relative"][step - 1] = error/ref_norm

        if step % max(1, progress_every) == 0 or step == n_steps:
            print(f"step {step:>7}/{n_steps}, t={times[step]:.6f}", flush=True)

    output = Path(output)
    save_results(output, times, data, ref_data, errors, {
        "gammas": gammas, "T": T, "tau": tau, "tau_ref": tau_ref,
        "Re": Re, "m": m, "shape": shape, "domain": domain,
    })

    for gamma in gammas:
        print(
            f"gamma={gamma:g}: final L2={errors[gamma]['l2'][-1]:.6e}, "
            f"relative={errors[gamma]['relative'][-1]:.6e}, "
            f"max|q-1|={np.max(np.abs(data[gamma]['q'] - 1)):.6e}, "
            f"CPU={data[gamma]['CPU_time'][-1]:.3f}s"
        )
    print(f"saved: {output}")
    return output


if __name__ == "__main__":
    run_diagnostics()
