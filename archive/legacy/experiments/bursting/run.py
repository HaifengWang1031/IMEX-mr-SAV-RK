"""Forced periodic NS bursting: explicit computation with per-run evidence.

Purpose: compare fixed/adaptive integration of the original forced-flow case.
Controls: the existing initial field, forcing, warmup and numerical formulas.
Observables: vorticity snapshots, q, energy/enstrophy, step times and CPU time.
Completion means finite saved output, not accuracy or physical convergence.
Compatible completed runs are reused; --rerun always preserves a new attempt.
Interrupted attempts restart from the initial condition, not a checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic
import uuid
import warnings

import h5py
import numpy as np
import pyfftw
import scipy

from solver import mrSAV_Vorticity_Stream_Periodic_Solve as Solver

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
SCHEMA = 1
FIXED_METHODS = (
    "IMEX", "IMEX_RK2", "ETD", "ETDMS2", "ETDRK4", "SDIRK2_mr_SAV",
    "ETD_mrSAV_MS2_b", "mr_SAV_BDF2", "ETD_mrSAV_MS2_L",
)
ADAPTIVE_METHODS = ("SDIRK2_mr_SAV", "ETD_mrSAV_MS2_b", "ETD_mrSAV_MS2_L")
ADAPTIVE_KEYS = (
    "tau_min", "tau_max", "rtol", "rtol_q", "atol", "rho", "r",
    "tau_initial", "max_step_ratio",
)
DATASETS = (
    "Omega", "tn_s", "q", "tn", "Mx", "Energy", "Energy_rate",
    "Enstrophy", "Enstrophy_rate", "Palinstrophy", "CPU_time", "tau",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def resolve_config(argv=None):
    """One documented default file, optional JSON overrides, then CLI overrides."""
    document = json.loads(DEFAULT_CONFIG.read_text())
    defaults = {key: value for key, value in document.items() if key != "_help"}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON configuration; omitted keys use the documented defaults")
    parser.add_argument("--rerun", action="store_true", help="new attempt, even if compatible results exist")
    parser.add_argument("--show-config", action="store_true", help="print effective configuration without computing")
    for key, value in defaults.items():
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=type(value),
                            default=argparse.SUPPRESS, help=document["_help"].get(key))
    args = parser.parse_args(argv)
    try:
        custom = json.loads(args.config.read_text())
        if not isinstance(custom, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = custom.keys() - defaults.keys() - {"_help"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        values = defaults | {k: v for k, v in custom.items() if k != "_help"}
        values.update({k: v for k, v in vars(args).items() if k in defaults})
        for key, default in defaults.items():
            value = values[key]
            if isinstance(default, str):
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a string")
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"{key} must be a finite number")
                if isinstance(default, int) and value != int(value):
                    raise ValueError(f"{key} must be an integer")
                values[key] = type(default)(value)
        validate_config(values)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    output_root = Path(values.pop("output_root")).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    execution = {"output_root": str(output_root.resolve()),
                 "log_interval": values.pop("log_interval"),
                 "config_file": str(args.config.resolve())}
    if values["mode"] == "fix":
        for key in ADAPTIVE_KEYS:
            values.pop(key)
    else:
        values.pop("tau")
    # Derived scientific settings are persisted rather than inferred from filenames.
    values.update(nu=1 / values["Re"], s_domain=[0, 0, 2*np.pi, 2*np.pi],
                  discrete_num=[values["N"], values["N"]], t_period=[0, values["T"]],
                  force_time_dependent=False, initial_modes=10, seed=1,
                  warmup_method="ETDRK4",
                  warmup_actual_end=float(np.ceil(values["warmup_time"] / values["warmup_tau"]) * values["warmup_tau"]))
    if values["mode"] == "adaptive":
        values.update(compute_ref_err=False, ref_substeps=2)
    return {"parameters": values, "execution": execution}, args


def validate_config(c):
    if c["mode"] not in ("fix", "adaptive") or c["M"] not in FIXED_METHODS:
        raise ValueError("Unsupported mode or method")
    if c["mode"] == "adaptive" and c["M"] not in ADAPTIVE_METHODS:
        raise ValueError(f"Adaptive methods: {ADAPTIVE_METHODS}")
    positive = ("Re", "T", "snapshot_dt", "warmup_tau", "log_interval")
    if any(c[key] <= 0 for key in positive):
        raise ValueError(f"These settings must be positive: {positive}")
    if c["N"] < 4 or c["N"] % 2 or c["m"] <= 0 or c["gamma"] < 0 or c["warmup_time"] < 0:
        raise ValueError("Require even N>=4, m>0, gamma>=0 and warmup_time>=0")
    if c["mode"] == "fix":
        if c["tau"] <= 0:
            raise ValueError("tau must be positive")
        steps = round(c["T"] / c["tau"])
        if steps < 1 or not np.isclose(steps*c["tau"], c["T"], rtol=1e-12, atol=1e-14):
            raise ValueError("Fixed mode requires T/tau to be an integer")
    else:
        if any(c[key] <= 0 for key in ADAPTIVE_KEYS) or c["tau_min"] >= c["tau_max"]:
            raise ValueError("Adaptive parameters must be positive, with tau_min < tau_max")
        if c["rho"] >= 1 or c["max_step_ratio"] < 1:
            raise ValueError("Require rho<1 and max_step_ratio>=1")


def identity_for(parameters):
    files = [Path(__file__), ROOT / "solver/__init__.py", ROOT / "solver/ns_periodic_mrSAV_solver.py"]
    return {"schema": SCHEMA, "parameters": parameters,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "pyfftw": pyfftw.__version__,
                            "h5py": h5py.__version__, "platform": platform.platform(),
                            "cpu_count": os.cpu_count()}}


def find_completed(output_root, identity, signature):
    for path in sorted(output_root.glob(f"{signature[:16]}-*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text())
            if manifest.get("status") != "completed" or manifest.get("identity") != identity:
                continue
            config = json.loads(path.with_name("config.json").read_text())
            if config["parameters"] != identity["parameters"]:
                raise ValueError("saved configuration disagrees with manifest")
            with h5py.File(path.with_name("results.h5"), "r") as result:
                if result.attrs["identity"] != canonical(identity) or result.attrs["status"] != "completed":
                    raise ValueError("saved HDF5 identity/status mismatch")
                if result.attrs["run_id"] != path.parent.name:
                    raise ValueError("saved HDF5 run_id mismatch")
                n, ns = len(result["tn"]), len(result["tn_s"])
                grid = identity["parameters"]["N"]
                if result["Omega"].shape != (ns, grid, grid) or len(result["tau"]) != n-1:
                    raise ValueError("saved time/field shapes disagree")
                for name in DATASETS[2:-1]:
                    if result[name].shape != (n,):
                        raise ValueError(f"saved {name} shape disagrees with tn")
                if n < 2 or ns < 2 or not np.isclose(result["tn"][-1], manifest["final_time"], rtol=0, atol=1e-12):
                    raise ValueError("saved endpoint mismatch")
            return path.parent
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Never silently compute over a supposedly completed but damaged record.
            raise RuntimeError(f"Cannot reuse {path.parent}: {exc}. Inspect it or use --rerun.") from exc
    return None


class ProgressLog(io.TextIOBase):
    """Adapt the existing solver's carriage-return output without changing it."""
    def __init__(self, logger, interval, phase):
        self.logger, self.interval, self.phase = logger, interval, phase
        self.started = monotonic()
        self.last_emit = -float("inf")
        self.pending = ""
        self.latest = ""
        self.last_logged = ""

    def write(self, text):
        # print() writes a complete progress string followed by an empty end string.
        for index, part in enumerate(text.split("\r")):
            if index:
                self._emit(force=False)
                self.pending = ""
            lines = part.split("\n")
            for line_index, line in enumerate(lines):
                if line_index:
                    self._emit(force=True)
                self.pending += line
        self._emit(force=False)
        return len(text)

    def _emit(self, force):
        if self.pending.strip():
            self.latest = self.pending.strip()
        now = monotonic()
        if self.latest and self.latest != self.last_logged and (force or now-self.last_emit >= self.interval):
            self.logger.info("%s elapsed_wall=%.1fs | %s", self.phase, now-self.started, self.latest)
            self.last_emit, self.last_logged = now, self.latest
        if force:
            self.pending = ""

    def flush(self):
        self._emit(force=True)


def initial_streamfunction(x, y, nu, m, eps):
    # Preserve the existing initial condition exactly (base flow is multiplied by 0).
    base_flow = (1.0 / (nu * m**3)) * np.cos(m * y)
    k1_grid, k2_grid = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11), indexing="ij")
    k_mod = np.sqrt(k1_grid**2 + k2_grid**2)
    mask = k_mod <= 10
    perturbation = np.zeros_like(x, dtype=np.float64)
    for k1, k2, k_abs in zip(k1_grid[mask], k2_grid[mask], k_mod[mask]):
        if k_abs < 1e-10:
            continue
        term = (1 / (k_abs**3)) * (
            np.cos(k1*x)*np.cos(k2*y) + np.sin(k1*x)*np.cos(k2*y)
            + np.cos(k1*x)*np.sin(k2*y) + np.sin(k1*x)*np.sin(k2*y)
        )
        perturbation += term
    return 0 * base_flow + eps * perturbation


def integrate(c, logger, interval):
    np.random.seed(c["seed"])
    x = np.linspace(0, 2*np.pi, c["N"]+1)
    X, Y = np.meshgrid(x, x)
    def force_term(X, Y, t):
        return c["m"] * np.cos(c["m"] * Y)
    initial_phi = initial_streamfunction(X[:-1, :-1], Y[:-1, :-1], c["nu"], c["m"], c["eps"])
    warmup = Solver(c["nu"], c["gamma"], c["s_domain"], c["discrete_num"],
                    np.zeros_like(X), force_term, "ETDRK4", force_time_dependent=False)
    u, v = warmup.stream2velocity(initial_phi)
    warmup.Omega0 = warmup.velocity2vorticity(u, v)
    if c["warmup_time"] > 0:
        logger.info("Warmup begins: requested_T=%g actual_T=%g", c["warmup_time"], c["warmup_actual_end"])
        with ProgressLog(logger, interval, "warmup") as progress, contextlib.redirect_stdout(progress):
            warmup.solve_fix_step((0, c["warmup_time"]), c["warmup_tau"])
        omega0 = warmup.Omega[-1]
    else:
        omega0 = warmup.Omega0
    solver = Solver(c["nu"], c["gamma"], c["s_domain"], c["discrete_num"],
                    np.pad(omega0, ((0, 1), (0, 1))), force_term, c["M"], force_time_dependent=False)
    if c["mode"] == "fix":
        steps = int(round(c["T"] / c["tau"]))
        stride = max(1, int(round(c["snapshot_dt"] / c["tau"])))
        indices = np.arange(0, steps+1, stride, dtype=np.int64)
        if indices[-1] != steps:
            indices = np.append(indices, steps)
        snapshots = c["tau"] * indices
    else:
        snapshots = np.append(np.arange(0, c["T"], c["snapshot_dt"]), c["T"])
    logger.info("Main integration begins: mode=%s method=%s T=%g snapshots=%d FFTW_threads=%d",
                c["mode"], c["M"], c["T"], len(snapshots), pyfftw.config.NUM_THREADS)
    logger.info("Snapshot fields alone require approximately %.3f GiB; scalar trajectories and FFT workspaces are additional",
                len(snapshots)*c["N"]**2*8/1024**3)
    with ProgressLog(logger, interval, "main") as progress, contextlib.redirect_stdout(progress):
        if c["mode"] == "fix":
            solver.solve_fix_step((0, c["T"]), c["tau"], snapshot=snapshots)
        else:
            solver.solve_adaptive_step((0, c["T"]), snapshot=snapshots,
                                       **{key: c[key] for key in ADAPTIVE_KEYS},
                                       compute_ref_err=c["compute_ref_err"], ref_substeps=c["ref_substeps"])
    return solver


def save_results(path, solver, config, identity, run_id):
    """Save old HDF5 field names plus explicit run/time semantics; preserve nonfinite data."""
    c = config["parameters"]
    arrays = {name: getattr(solver, "cpu_time" if name == "CPU_time" else name) for name in DATASETS}
    if c["mode"] == "adaptive":
        arrays.update({name: getattr(solver, name) for name in ("rel_err", "controller_err", "ref_err", "ref_err_p", "ref_err_b")})
    # Bound validation scratch space even when snapshot arrays are large.
    finite = all(np.isfinite(flat[start:start+1_000_000]).all()
                 for array in arrays.values() for flat in (np.asarray(array).reshape(-1),)
                 for start in range(0, flat.size, 1_000_000))
    status = "completed" if finite else "nonfinite"
    n, ns = len(arrays["tn"]), len(arrays["tn_s"])
    if arrays["Omega"].shape != (ns, c["N"], c["N"]) or len(arrays["tau"]) != n-1:
        raise ValueError("Solver output shapes disagree with physical time grids")
    for name, array in arrays.items():
        if name not in ("Omega", "tn_s", "tau") and array.shape != (n,):
            raise ValueError(f"{name} shape disagrees with tn")
    if not np.isclose(arrays["tn"][-1], c["T"], rtol=1e-12, atol=1e-12):
        raise ValueError("Solver did not reach the requested endpoint")
    temporary = path.with_suffix(".h5.tmp")
    with h5py.File(temporary, "w") as f:
        for key in ("mode", "Re", "m", "eps", "gamma", "nu", "s_domain", "discrete_num", "t_period", "snapshot_dt", "warmup_time", "warmup_tau"):
            f.attrs[key] = c[key]
        f.attrs["method"] = c["M"]
        for key in (("tau",) if c["mode"] == "fix" else ADAPTIVE_KEYS):
            f.attrs[key] = c[key]
        f.attrs.update(schema_version=SCHEMA, run_id=run_id, identity=canonical(identity),
                       effective_config=canonical(config), status=status,
                       warmup_actual_end=c["warmup_actual_end"], fftw_threads=pyfftw.config.NUM_THREADS,
                       snapshot_policy="fixed_grid" if c["mode"] == "fix" else "linear_interpolation",
                       cpu_time_scope="solver accumulated step timings; distinct from run wall time")
        for name, array in arrays.items():
            f[name] = array
        if c["mode"] == "adaptive":
            for name in ("accepted_steps", "rejected_steps", "forced_accept_steps"):
                f.attrs[name] = getattr(solver, name)
            f.attrs["compute_ref_err"] = c["compute_ref_err"]
        f.flush()
    os.replace(temporary, path)
    return status, {"final_time": float(arrays["tn"][-1]), "steps": n-1,
                    "snapshots": ns, "fftw_threads": int(pyfftw.config.NUM_THREADS)}


def run_experiment(config, *, rerun=False):
    identity = identity_for(config["parameters"])
    signature = hashlib.sha256(canonical(identity).encode()).hexdigest()
    output_root = Path(config["execution"]["output_root"])
    if not rerun:
        previous = find_completed(output_root, identity, signature)
        if previous is not None:
            print(f"REUSED completed run: {previous}\nResults: {previous / 'results.h5'}\nLog: {previous / 'run.log'}", flush=True)
            return previous
    run_id = f"{signature[:16]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"
    directory = output_root / run_id
    directory.mkdir(parents=True)
    for name in ("figures", "tables"):
        (directory / name).mkdir()
    atomic_json(directory / "config.json", config)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest = {"schema": SCHEMA, "run_id": run_id, "signature": signature, "identity": identity,
                "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_head": git.stdout.strip() if git.returncode == 0 else None,
                "command": sys.argv, "restart_policy": "restart_from_initial_condition",
                "artifacts": {"config": "config.json", "results": "results.h5", "log": "run.log",
                              "figures": "figures/", "tables": "tables/"}}
    atomic_json(directory / "manifest.json", manifest)
    logger = logging.getLogger(f"bursting.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(directory / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    started = monotonic()
    try:
        logger.info("STARTED run_id=%s directory=%s", run_id, directory)
        logger.info("Effective configuration: %s", canonical(config))
        logger.info("Code/environment identity: %s", canonical(identity))
        with warnings.catch_warnings():
            def log_warning(message, category, filename, lineno, file=None, line=None):
                logger.warning("%s: %s (%s:%s)", category.__name__, message, filename, lineno)
            warnings.showwarning = log_warning
            solver = integrate(config["parameters"], logger, config["execution"]["log_interval"])
        logger.info("Saving results to %s", directory / "results.h5")
        status, summary = save_results(directory / "results.h5", solver, config, identity, run_id)
        if status != "completed":
            raise FloatingPointError("Nonfinite output preserved in results.h5; this run is not reusable")
        manifest.update(status="completed", **summary, elapsed_wall_seconds=monotonic()-started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(directory / "manifest.json", manifest)
        logger.info("COMPLETED elapsed_wall=%.3fs results=%s", manifest["elapsed_wall_seconds"], directory / "results.h5")
        return directory
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        elapsed_wall_seconds=monotonic()-started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(directory / "manifest.json", manifest)
        logger.exception("FAILED; attempt preserved at %s; a retry starts from the initial condition", directory)
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main(argv=None):
    config, args = resolve_config(argv)
    if args.show_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    try:
        run_experiment(config, rerun=args.rerun)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
