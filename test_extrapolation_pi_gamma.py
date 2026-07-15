"""Regression tests for extrapolation-PI control with adaptive gamma."""

from __future__ import annotations

import numpy as np

from imex_mrsav_rk_solver import (
    ExtrapolationPIGammaOptions,
    make_periodic_ops,
    solve_adaptive_extrapolation_pi_gamma,
)


def main() -> None:
    domain = (0.0, 0.0, 2.0 * np.pi, 2.0 * np.pi)
    resolution = (24, 24)
    ops = make_periodic_ops(domain, resolution, fftw_threads=1)
    omega0 = np.sin(3.0 * ops.X) * np.cos(2.0 * ops.Y)

    def force(X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        del X, t
        return 2.0 * np.cos(3.0 * Y)

    output_times = np.linspace(0.0, 0.04, 5)
    options = ExtrapolationPIGammaOptions(
        tol_omega=1.0e-7,
        tol_r=1.0e-6,
        qbar=0.95,
        dt_min=1.0e-5,
        dt_max=1.0e-2,
        max_rejections=100,
    )
    result = solve_adaptive_extrapolation_pi_gamma(
        omega0,
        nu=0.02,
        domain=domain,
        discrete_num=resolution,
        dt0=5.0e-3,
        t_span=(0.0, 0.04),
        method="ars222",
        force=force,
        adaptive_options=options,
        output_times=output_times,
        fftw_threads=1,
    )

    assert result.rejected_steps >= 0
    assert result.accepted_steps + result.rejected_steps == len(result.step_cpu_times)
    assert len(result.step_cpu_times) == len(result.step_accepted_mask)
    assert np.allclose(result.t_snapshot, output_times)
    assert np.isclose(result.t[-1], 0.04)
    assert np.all(result.dt_history > 0.0)
    assert np.allclose(result.gamma_dt_history, result.chi_qbar, rtol=0.0, atol=2.0e-13)
    assert np.allclose(result.damping_factor_history, result.qbar, rtol=0.0, atol=2.0e-13)
    assert len(result.effective_error_history) == result.accepted_steps

    valid = np.isfinite(result.error_omega_history[2:])
    if np.any(valid):
        assert np.nanmax(result.error_omega_history[2:][valid]) <= options.tol_omega + 1.0e-12
        assert np.nanmax(result.effective_error_history[2:][valid]) <= 1.0 + 1.0e-12

    floor_result = solve_adaptive_extrapolation_pi_gamma(
        omega0,
        nu=0.02,
        domain=domain,
        discrete_num=resolution,
        dt0=5.0e-4,
        t_span=(0.0, 0.003),
        method="ars222",
        force=force,
        adaptive_options=ExtrapolationPIGammaOptions(
            tol_omega=1.0e-15,
            tol_r=1.0e-15,
            qbar=0.95,
            dt_min=1.0e-4,
            dt_max=1.0e-2,
            max_rejections=100,
        ),
        output_times=np.array([0.0, 0.003]),
        fftw_threads=1,
    )
    assert floor_result.floor_accepted_steps > 0
    assert np.isclose(floor_result.t[-1], 0.003)
    assert np.allclose(
        floor_result.damping_factor_history,
        floor_result.qbar,
        rtol=0.0,
        atol=2.0e-13,
    )

    print(
        "PASS: extrapolation PI-gamma controller; "
        f"accepted={result.accepted_steps}, rejected={result.rejected_steps}, "
        f"dt=[{result.dt_history.min():.3e}, {result.dt_history.max():.3e}], "
        f"gamma=[{result.gamma_history.min():.3e}, {result.gamma_history.max():.3e}]"
    )


if __name__ == "__main__":
    main()
