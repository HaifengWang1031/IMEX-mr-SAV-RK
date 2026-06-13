# IMEX-mr-SAV-RK solver

2D periodic Navier-Stokes solver in vorticity-streamfunction form using
IMEX multirate scalar auxiliary variable (mr-SAV) Runge-Kutta timestepping.

## Quick start

```python
import numpy as np
from imex_mrsav_rk_solver import solve_fixed_step, solve_adaptive, PIAdaptiveOptions

domain = (0.0, 0.0, 2*np.pi, 2*np.pi)
nx, ny = 128, 128
x = np.linspace(domain[0], domain[2], nx, endpoint=False)
y = np.linspace(domain[1], domain[3], ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")
omega0 = np.sin(X) * np.sin(Y)

def force(X, Y, t):
    return np.cos(X)

# ── Fixed-step solve ──────────────────────────────────────────────────
result = solve_fixed_step(
    omega0,
    q0=1.0,
    nu=1/100,
    gamma=1000.0,
    domain=domain,
    discrete_num=(nx, ny),
    dt=1e-3,
    t_span=(0.0, 1.0),
    method="rk3",
    force=force,
    save_every=100,
)
print(f"Final energy: {result.energy[-1]:.6e}")
# Embedded solution available at result.omega_embed (when method supports it)

# ── Adaptive solve ────────────────────────────────────────────────────
opts = PIAdaptiveOptions(tol=1e-4)
result = solve_adaptive(
    omega0, q0=1.0,
    nu=1/100, gamma=1000.0,
    domain=domain, discrete_num=(nx, ny),
    dt0=1e-2, t_span=(0.0, 1.0),
    method="rk3", force=force,
    adaptive_options=opts,
    print_progress=True,
)
```

## Built-in methods

| Key | Order | Embedding | Embed order | Adaptive |
|-----|-------|-----------|-------------|----------|
| `rk1` | 1 | none | — | no |
| `rk2` | 2 | embedded RK weights | 1 | yes |
| `ars222` | 2 | b_tilde/bhat_tilde | 1 | yes |
| `rk3` | 3 | embedded RK weights | 2 | yes |
| `rk4` | 4 | embedded RK weights | 3 | yes |

All methods accept case-insensitive keys: `"rk3"`, `"imex-rk3"`, `"imex-mrsav-rk3"`.

## Embedded RK weights

`rk2`, `rk3`, and `rk4` use coefficient-based embedded IMEX-RK outputs of
orders 1, 2, and 3 respectively.  The old stage interpolation estimator has
been removed, so adaptive stepping always uses the embedded RK weights carried
by the tableau.  The embedded output costs one extra scalar Newton solve per
accepted step.

## Adaptive step-size control

`PIAdaptiveOptions` configures the PI controller:

```python
PIAdaptiveOptions(tol=1e-4,        # target L2 error tolerance
                  safety=0.9,      # safety factor
                  k_I=0.133,       # integral gain (0.4/(p_embed+1) for p=2)
                  k_P=0.1,         # proportional gain (0.3/(p_embed+1) for p=2)
                  dt_min=0.0,      # minimum step size
                  dt_max=float("inf"))  # maximum step size
```

Default gains are tuned for p_embed=2 (RK3/RK4).  For ARS222 (p_embed=1),
use `k_I=0.15, k_P=0.20`.

## Public API

| Function / Class | Description |
|---|---|
| `solve_fixed_step` | Fixed-step solve |
| `solve_adaptive` | Adaptive solve with PI controller |
| `step_imex_mrsav_rk` | Single time step |
| `make_tableau` | Build Butcher tableau for a named method |
| `make_periodic_ops` | Create Fourier operators for a periodic box |
| `make_taylor_v` | Build Taylor polynomial for 1/q at q=1 |
| `advection_term` | Skew-symmetric dealiased advection |
| `velocity_from_vorticity` | Recover (u, v) from ω |
| `vorticity_energy` | Energy, enstrophy, palinstrophy |
| `PeriodicOps` | Fourier operators and grid data |
| `IMEXTableau` | Butcher tableau coefficients |
| `NewtonOptions` | Newton solver parameters |
| `PIAdaptiveOptions` | PI controller parameters |
| `SolverResult` | Fixed-step result |
| `AdaptiveSolverResult` | Adaptive result |
| `StepDiagnostics` | Per-step diagnostics |
| `NewtonInfo` | Per-stage Newton convergence info |

## Files

- `imex_mrsav_rk_solver.py` — solver module
- `test_ars222_convergence.py` — convergence test for ARS(2,2,2)
- `verify_embedded_rk_convergence.py` — manufactured-solution convergence check for embedded RK pairs

## Notes

The solver uses `pyfftw` if installed; otherwise falls back to `scipy.fft`.
Adding a new adaptive method requires registering `A`, `Ahat`, `b_tilde`, and
`bhat_tilde` in `make_tableau`.
