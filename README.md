# IMEX-mr-SAV-RK fixed-step solver

This folder contains a functional fixed-step solver for the 2D periodic
Navier-Stokes equation in vorticity-streamfunction form.

## Files

- `imex_mrsav_rk_solver.py`: solver implementation.

## Usage

```python
import numpy as np
from imex_mrsav_rk_solver import solve_fixed_step

domain = (0.0, 0.0, 2*np.pi, 2*np.pi)
nx, ny = 128, 128
x = np.linspace(domain[0], domain[2], nx, endpoint=False)
y = np.linspace(domain[1], domain[3], ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")
omega0 = np.sin(X) * np.sin(Y)

def force(X, Y, t):
    return np.zeros_like(X)

result = solve_fixed_step(
    omega0,
    q0=1.0,
    nu=1/100,
    gamma=1000,
    domain=domain,
    discrete_num=(nx, ny),
    dt=1e-3,
    t_span=(0.0, 1.0),
    method="rk3",
    force=force,
    save_every=100,
)
```

## Notes

The built-in methods are `rk1`, `rk2`, `rk3`, and `rk4`.  The RK coefficients are
centralized in `make_tableau`, so adding a new fixed-step or embedded
adaptive method only requires registering a new pair of matrices `A` and
`Ahat`.

The solver uses `pyfftw` if it is installed.  Numba kernels are optional:
when `numba` is installed, selected scalar reductions and stage recovery
operations are JIT-compiled; otherwise the code falls back to NumPy.

## Gamma diagnostics

Run the pilot gamma sweep against the ETDRK4 reference from the companion ETD
folder:

```bash
python 02_run_gamma_diagnostics.py
python 02_analyze_gamma_diagnostics.py
```

The runner writes `data/imex_gamma_diagnostics_pilot.h5` with gamma groups for
`0, 1, 10, 100, 1000`.  The analyzer writes a PDF/PNG summary figure under
`fig/` and a CSV error table under `data/`.  For a quick smoke-sized run, reduce
the grid and final time, for example:

```bash
python 02_run_gamma_diagnostics.py --output data/imex_gamma_diagnostics_pilot_quick.h5 --T 0.05 --nx 32 --ny 32
python 02_analyze_gamma_diagnostics.py --input data/imex_gamma_diagnostics_pilot_quick.h5 --fig-dir fig/gamma_pilot_quick --table data/imex_gamma_diagnostics_pilot_quick_table.csv
```
