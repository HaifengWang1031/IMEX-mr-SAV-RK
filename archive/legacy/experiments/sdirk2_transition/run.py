"""Forced periodic NS: fixed-step SDIRK2 versus SDIRK2-mr-SAV.

Purpose: search for a resolved transition interval with smaller physical errors
for mr-SAV, between small-step agreement and original-method numerical blowup.
Controls: identical initial field, forcing, viscosity, grid, physical times and
steps for each pair. Only mr-SAV uses gamma. Nonzero forcing is mandatory.
Observables: vorticity/velocity errors, energy/enstrophy, q, forcing work, status.
Interpretation: finite completion is not accuracy; a coarse-grid candidate needs
reference-step halving and spatial refinement before a scientific conclusion.
Run computation explicitly; analyze only reads saved, compatible run files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings

import numpy as np
import scipy
import pyfftw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from solver import mrSAV_Vorticity_Stream_Periodic_Solve as Solver

SCHEMA = 1
DEFAULT_OUTPUT = ROOT / 'data/sdirk2_transition'
METHODS = ('IMEX_RK2', 'SDIRK2_mr_SAV')
LABELS = {'IMEX_RK2': 'SDIRK2', 'SDIRK2_mr_SAV': 'SDIRK2-mr-SAV'}


@dataclass(frozen=True)
class Case:
    # omega_t + u.grad(omega) = nu*Delta(omega) + force*cos(force_k*x)
    grid: int = 64
    nu: float = 0.001
    gamma: float = 2000.0
    amplitude: float = 1.0  # initial vorticity RMS, independent of grid
    force: float = 1.0  # nonzero vorticity forcing amplitude
    force_k: int = 1
    initial: str = 'trig'
    modes: int = 10  # same physical initial modes on every refinement grid
    seed: int = 0  # used only by random-phase initial fields
    final_time: float = 2.0
    root_selection: str = 'legacy'
    threads: int = 1  # fixed FFTW thread count; also recorded for timings


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def provenance():
    files = [Path(__file__), ROOT / 'solver/ns_periodic_mrSAV_solver.py']
    return {'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files},
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'pyfftw': pyfftw.__version__}


def make_solver(c, method):
    if c.grid < 16 or c.grid % 2 or c.nu <= 0 or c.gamma < 0 or c.force == 0:
        raise ValueError('Require an even grid >=16, nu>0, gamma>=0 and nonzero forcing.')
    if c.amplitude <= 0 or c.final_time <= 0 or c.threads < 1:
        raise ValueError('Require positive amplitude, final time and thread count.')
    if not 1 <= c.force_k < c.grid // 3 or not 1 <= c.modes < c.grid // 3:
        raise ValueError('Initial and forcing modes must be inside the dealiased band.')
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, c.grid+1), np.linspace(0, 2*np.pi, c.grid+1))
    if c.initial in ('trig', 'random'):
        w = np.zeros_like(x)
        rng = np.random.default_rng(c.seed)
        for k in range(1, c.modes+1):
            for m in range(1, c.modes+1):
                phases = rng.uniform(0, 2*np.pi, 2) if c.initial == 'random' else (0, 0)
                w += (k*k+m*m)**(-1.5)*np.cos(k*x+phases[0])*np.cos(m*y+phases[1])
    elif c.initial == 'shear':
        # Smooth periodic shear with a nonparallel perturbation.
        w = np.cos(2*y) + 0.2*np.cos(x)*np.cos(y) + 0.1*np.sin(3*x+2*y)
    else:
        raise ValueError(f'Unknown initial field: {c.initial}')
    w -= np.mean(w[:-1, :-1])
    w *= c.amplitude / np.sqrt(np.mean(w[:-1, :-1]**2))
    def forcing(x, y, t):
        return c.force*np.cos(c.force_k*x)
    solver = Solver(c.nu, c.gamma, (0, 0, 2*np.pi, 2*np.pi), (c.grid, c.grid),
                    w, forcing, method, force_time_dependent=False,
                    root_selection=c.root_selection)
    pyfftw.config.NUM_THREADS = c.threads
    solver._enable_fixed_step_cache = True
    return solver


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def run(c, method, steps, output=DEFAULT_OUTPUT):
    """Rolling one-step driver; does not alter the project's numerical methods.

    Save the endpoint and quarter-time snapshots only when they lie on the grid.
    Any positive integer step count >=4 is allowed, with no interpolation.
    A run stops at nonfinite state or RMS exceeding 100 times the continuous
    forced-enstrophy upper bound A + |F| t/sqrt(2). This empirical stop is labeled
    solution_blowup and is distinct from merely exceeding the physical bound.
    Interrupted entries restart from t=0; they are not continuation checkpoints.
    """
    if steps < 4 or int(steps) != steps:
        raise ValueError('steps must be an integer >=4.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    config = {'schema': SCHEMA, 'case': asdict(c), 'method': method, 'steps': steps,
              'tau': c.final_time/steps, 'source_sha256': prov['source_sha256'],
              'blowup_bound_factor': 100.0}
    key = digest(config)
    path = output / f'{key}.npz'
    manifest = output / f'{key}.json'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            old = json.loads(str(saved['metadata']))
            if old['config'] != config:
                raise ValueError(f'Configuration mismatch: {path}')
            if old['status'] != 'incomplete':
                return {name: saved[name].copy() for name in saved.files}
        raise RuntimeError(f'Unexpected incomplete NPZ: {path}')
    meta = {'config': config, 'provenance': prov, 'status': 'incomplete', 'run_id': key}
    atomic_json(manifest, meta)
    s = make_solver(c, method)
    w, q = s.Omega0.copy(), 1.0
    last_state_time = 0.0
    tau = c.final_time/steps
    snapshots, snapshot_times = [w.copy()], [0.0]
    diagnostics = []
    status, message = 'completed', ''
    start = perf_counter()
    caught = []
    def record(i, field, scalar):
        t = i*tau
        rms = float(np.sqrt(np.mean(field**2)))
        bound = c.amplitude + abs(c.force)*t/np.sqrt(2)
        diagnostics.append((t, scalar, rms, float(np.max(np.abs(field))), float(np.mean(field)), bound))
    record(0, w, q)
    for i in range(1, steps+1):
        try:
            with warnings.catch_warnings(record=True) as ws, np.errstate(over='raise', invalid='raise', divide='raise'):
                warnings.simplefilter('always', RuntimeWarning)
                wn, qn = s.step(w[None], np.array([q]), (i-1)*tau, np.array([tau]))
            caught.extend(str(item.message) for item in ws)
            w, q = wn, float(qn)
            last_state_time = i*tau
            if not np.isfinite(w).all() or not np.isfinite(q):
                status = 'nonfinite'
                # Preserve the actual nonfinite state and time.
                break
            record(i, w, q)
            if (4*i) % steps == 0:
                snapshots.append(w.copy())
                snapshot_times.append(i*tau)
            if diagnostics[-1][2] > 100*diagnostics[-1][-1]:
                status = 'solution_blowup'
                message = 'RMS exceeds 100 times forced-enstrophy bound.'
                break
        except FloatingPointError as exc:
            status, message = 'floating_point_error', str(exc)
            break
        except (ValueError, RuntimeError, OverflowError) as exc:
            status, message = 'solver_failure', f'{type(exc).__name__}: {exc}'
            break
    elapsed = perf_counter()-start
    meta.update(status=status, message=message, elapsed_seconds=elapsed,
                warnings=sorted(set(caught)), attempted_time=i*tau,
                last_diagnostic_time=diagnostics[-1][0],
                last_state_time=last_state_time,
                snapshot_times=snapshot_times)
    arrays = {'metadata': np.array(canonical(meta)), 'omega': np.array(snapshots),
              'times': np.array(snapshot_times), 'diagnostics': np.array(diagnostics),
              'last_state': w, 'last_q': np.array(q)}
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)
    atomic_json(manifest, meta)
    print(f'{key} N={c.grid} A={c.amplitude:g} F={c.force:g} nu={c.nu:g} ga={c.gamma:g}'
          f' {method} tau={tau:.6g}: {status} ({elapsed:.2f}s)', flush=True)
    return arrays


def relative_errors(c, result, reference):
    s = make_solver(c, 'ETDRK4')
    rows = []
    for t, w in zip(result['times'][1:], result['omega'][1:]):
        indices = np.flatnonzero(np.isclose(reference['times'], t, rtol=0, atol=1e-12))
        if len(indices) != 1:
            raise ValueError(f'Reference has no unique sample at {t}.')
        wr = reference['omega'][indices[0]]
        u, v = s.stream2velocity(s.vorticity2stream(w))
        ur, vr = s.stream2velocity(s.vorticity2stream(wr))
        ew_abs = np.sqrt(s.h*np.sum((w-wr)**2))
        ew = ew_abs/np.sqrt(s.h*np.sum(wr**2))
        eu = np.sqrt(np.sum((u-ur)**2+(v-vr)**2)/np.sum(ur**2+vr**2))
        energy, enstrophy, _ = s.vorticity_energy(w)
        work = s.inner_product(s.vorticity2stream(w), s.f(s.X[:-1,:-1], s.Y[:-1,:-1], t))
        rows.append({'time': float(t), 'omega_relative': float(ew), 'omega_absolute': float(ew_abs),
                     'velocity_relative': float(eu), 'energy': float(energy),
                     'enstrophy': float(enstrophy), 'forcing_energy_work': float(work)})
    return rows


def compare(c, counts, reference_steps, output):
    ref = run(c, 'ETDRK4', reference_steps, output)
    ref_meta = json.loads(str(ref['metadata']))
    if ref_meta['status'] != 'completed':
        raise RuntimeError('Reference failed; reduce reference step before comparison.')
    records = []
    for n in counts:
        pair = {}
        for method in METHODS:
            value = run(c, method, n, output)
            meta = json.loads(str(value['metadata']))
            pair[method] = {'run_id': meta['run_id'], 'status': meta['status'],
                            'elapsed_seconds': meta['elapsed_seconds'],
                            'errors': relative_errors(c, value, ref)}
        records.append({'tau': c.final_time/n, 'steps': n, 'methods': pair})
    summary = {'case': asdict(c), 'reference_id': ref_meta['run_id'],
               'reference_steps': reference_steps, 'records': records,
               'validation': 'same-grid reference, not yet time/spatial validated'}
    key = digest({'case': asdict(c), 'reference_id': ref_meta['run_id'], 'counts': counts})
    atomic_json(Path(output)/f'comparison_{key}.json', summary)
    return summary


def report(output):
    """Read-only ranking; only complete pairs contribute to transition ratios."""
    rankings = []
    for path in Path(output).glob('comparison_*.json'):
        summary = json.loads(path.read_text())
        good, finite = [], []
        for row in sorted(summary['records'], key=lambda x: x['tau']):
            a, b = [row['methods'][m] for m in METHODS]
            if a['status'] != 'completed' or b['status'] != 'completed':
                finite.append(None)
                continue
            ratios = [eb['omega_relative']/ea['omega_relative']
                      for ea, eb in zip(a['errors'], b['errors']) if ea['omega_relative'] > 0]
            item = {'tau': row['tau'], 'ratios': ratios,
                    'original_error': a['errors'][-1]['omega_relative'],
                    'mrsav_error': b['errors'][-1]['omega_relative']}
            finite.append(item)
            if ratios[-1] <= .8:
                good.append(item)
        streak = best = 0
        for item in finite:
            streak = streak+1 if item is not None and item['ratios'][-1] <= .8 else 0
            best = max(best, streak)
        rankings.append({'file': path.name, 'case': summary['case'], 'good': good,
                         'longest_terminal_streak': best})
    rankings.sort(key=lambda r: (r['longest_terminal_streak'], len(r['good'])), reverse=True)
    atomic_json(Path(output)/'ranking.json', rankings)
    for rank in rankings[:10]:
        print(json.dumps(rank, ensure_ascii=False))
    return rankings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['compare', 'screen', 'report'])
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--grid', type=int, default=64)
    p.add_argument('--nu', type=float, default=.001)
    p.add_argument('--gamma', type=float, default=2000)
    p.add_argument('--amplitude', type=float, default=1)
    p.add_argument('--force', type=float, default=1)
    p.add_argument('--force-k', type=int, default=1)
    p.add_argument('--initial', choices=['trig', 'random', 'shear'], default='trig')
    p.add_argument('--modes', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--final-time', type=float, default=2)
    p.add_argument('--root-selection', choices=['legacy', 'nearest', 'farthest'], default='legacy')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--counts', type=int, nargs='+', default=[16, 24, 32, 40, 48, 64, 80, 96, 128, 192, 256, 512])
    p.add_argument('--reference-steps', type=int, default=2048)
    return p.parse_args()


def main():
    a = parse_args()
    c = Case(**{name: getattr(a, name) for name in Case.__dataclass_fields__})
    if a.command == 'report':
        report(a.output)
    elif a.command == 'compare':
        compare(c, a.counts, a.reference_steps, a.output)
    else:
        # Staged screening, all cases forced. This list is the documented search
        # membership; individual results have independent immutable identities.
        cases = [c]
        cases += [replace(c, amplitude=A, gamma=g) for A in (2., 4.) for g in (20., 200., 2000.)]
        cases += [replace(c, force=F, gamma=g) for F in (4., 8.) for g in (20., 200., 2000.)]
        cases += [replace(c, initial=ic, amplitude=2., gamma=g)
                  for ic in ('random', 'shear') for g in (20., 200., 2000.)]
        cases += [replace(c, nu=nu, amplitude=2., gamma=200.) for nu in (.0005, .005)]
        for candidate in cases:
            compare(candidate, a.counts, a.reference_steps, a.output)
        report(a.output)


if __name__ == '__main__':
    main()
