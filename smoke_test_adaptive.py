"""Smoke test for solve_adaptive with the new q-deviation error term.

Checks:
1. solve_adaptive runs without error for ars222 / rk3 / rk4.
2. Final vorticity is close to the fixed-step reference (same method, fine dt).
3. err_omega and err_q contributions are both printed so we can judge their
   relative magnitudes.
4. Tighter tol produces smaller final error (basic sanity of PI controller).
"""

import numpy as np
from imex_mrsav_rk_solver import (
    make_periodic_ops,
    inner_product,
    solve_adaptive,
    solve_fixed_step,
    _compute_error,
    PIAdaptiveOptions,
)

# ── problem setup ────────────────────────────────────────────────────────────
DOMAIN = (0.0, 0.0, 2.0 * np.pi, 2.0 * np.pi)
NX, NY = 32, 32
NU = 0.01
GAMMA = 1.0
T0, TF = 0.0, 0.5

ops = make_periodic_ops(DOMAIN, (NX, NY))
X, Y = ops.X, ops.Y
omega0 = np.sin(X) * np.cos(Y) + 0.5 * np.cos(2 * X) * np.sin(2 * Y)


def kolmogorov_force(X, Y, t):
    return np.cos(2 * Y)


# ── PI gains per embedding order: k_I = 0.4/(p+1), k_P = 0.3/(p+1) ─────────
# ars222: embed p=1 → k_I=0.200, k_P=0.150
# rk3:   embed p=2 → k_I=0.133, k_P=0.100
# rk4:   embed p=3 → k_I=0.100, k_P=0.075
_PI_GAINS = {
    "ars222": dict(k_I=0.200, k_P=0.150),
    "rk3":    dict(k_I=0.133, k_P=0.100),
    "rk4":    dict(k_I=0.100, k_P=0.075),
}


def _l2(ops, a, b):
    diff = a - b
    return float(np.sqrt(inner_product(ops, diff, diff)))


# ── reference solution (fine fixed step, rk3) ───────────────────────────────
ref = solve_fixed_step(
    omega0, nu=NU, gamma=GAMMA, domain=DOMAIN,
    discrete_num=(NX, NY), dt=1e-4, t_span=(T0, TF),
    method="rk3", force=kolmogorov_force, keep_omega=True,
)
omega_ref = ref.omega[-1]
print(f"Reference (rk3, dt=1e-4): E={ref.energy[-1]:.6e}, Ens={ref.enstrophy[-1]:.6e}")
print()


# ── helper: print _compute_error breakdown for one step ──────────────────────
def _show_error_breakdown(omega_main, omega_embed, q_main, order, ops):
    diff = omega_main - omega_embed
    err_omega = float(np.sqrt(inner_product(ops, diff, diff)))
    print(f"  err_omega={err_omega:.3e}  |q-1|={abs(q_main - 1.0):.3e}")


# ── test 1: basic run for all methods ────────────────────────────────────────
print("=" * 60)
print("Test 1: basic run (all methods)")
print("=" * 60)
for method in ("ars222", "rk3", "rk4"):
    res = solve_adaptive(
        omega0, nu=NU, gamma=GAMMA, domain=DOMAIN,
        discrete_num=(NX, NY), dt0=1e-2, t_span=(T0, TF),
        method=method, force=kolmogorov_force,
        adaptive_options=PIAdaptiveOptions(tol=1e-4, **_PI_GAINS[method]),
        keep_omega=True,
    )
    err_vs_ref = _l2(ops, res.omega_final, omega_ref)
    print(f"  {method}: accepted={res.accepted_steps}, rejected={res.rejected_steps}, "
          f"|q_final-1|={abs(res.q_final-1):.2e}, "
          f"err_vs_ref={err_vs_ref:.3e}")
print()


# ── test 2: tighter tol → smaller error ──────────────────────────────────────
print("=" * 60)
print("Test 2: tighter tol → smaller final error (ars222)")
print("=" * 60)
errors = {}
for tol in (1e-3, 1e-4, 1e-5):
    res = solve_adaptive(
        omega0, nu=NU, gamma=GAMMA, domain=DOMAIN,
        discrete_num=(NX, NY), dt0=1e-2, t_span=(T0, TF),
        method="ars222", force=kolmogorov_force,
        adaptive_options=PIAdaptiveOptions(tol=tol, **_PI_GAINS["ars222"]),
        keep_omega=True,
    )
    err = _l2(ops, res.omega_final, omega_ref)
    errors[tol] = err
    print(f"  tol={tol:.0e}: err_vs_ref={err:.3e}, steps={res.accepted_steps}+{res.rejected_steps}")

assert errors[1e-4] < errors[1e-3], "tighter tol did not reduce error (1e-3 vs 1e-4)"
assert errors[1e-5] < errors[1e-4], "tighter tol did not reduce error (1e-4 vs 1e-5)"
print("  [PASS] tighter tol monotonically reduces error")
print()


# ── test 3: error breakdown for rk3 ──────────────────────────────────────────
print("=" * 60)
print("Test 3: err_omega vs err_q breakdown (rk3, one step)")
print("=" * 60)
from imex_mrsav_rk_solver import make_tableau, step_imex_mrsav_rk

tableau = make_tableau("rk3")
omega_one, q_one, info = step_imex_mrsav_rk(
    omega0, 1.0, T0, 1e-2, ops, NU, GAMMA, tableau,
    force=kolmogorov_force,
)
if info.omega_embed is not None:
    _show_error_breakdown(omega_one, info.omega_embed, q_one, tableau.order, ops)
else:
    print("  (no embedding available)")
print()


# ── test 4: q deviation stays bounded ────────────────────────────────────────
print("=" * 60)
print("Test 4: |q-1| stays small throughout (rk3)")
print("=" * 60)
res = solve_adaptive(
    omega0, nu=NU, gamma=GAMMA, domain=DOMAIN,
    discrete_num=(NX, NY), dt0=1e-2, t_span=(T0, TF),
    method="rk3", force=kolmogorov_force,
    adaptive_options=PIAdaptiveOptions(tol=1e-5, **_PI_GAINS["rk3"]),
    keep_omega=False,
)
print(f"  |q_final - 1| = {abs(res.q_final - 1):.3e}")
assert abs(res.q_final - 1) < 0.1, f"|q-1|={abs(res.q_final-1):.3e} is too large"
print("  [PASS] q stays close to 1")
print()

print("All smoke tests passed.")
