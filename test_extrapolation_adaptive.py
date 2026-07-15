"""Small deterministic checks for the manuscript's extrapolation controller."""

from __future__ import annotations

import numpy as np

from imex_mrsav_rk_solver import (
    ExtrapolationAdaptiveOptions,
    make_periodic_ops,
    solve_adaptive_extrapolation,
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
    result = solve_adaptive_extrapolation(
        omega0,
        nu=0.02,
        gamma=10.0,
        domain=domain,
        discrete_num=resolution,
        dt0=0.01,
        t_span=(0.0, 0.04),
        method="ars222",
        force=force,
        adaptive_options=ExtrapolationAdaptiveOptions(
            tol_omega=1.0e-8,
            tol_r=1.0e-7,
            dt_min=1.0e-5,
            dt_max=1.0e-2,
            max_rejections=100,
        ),
        output_times=output_times,
        fftw_threads=1,
    )

    assert result.rejected_steps > 0
    assert result.accepted_steps + result.rejected_steps == len(result.step_cpu_times)
    assert len(result.step_cpu_times) == len(result.step_accepted_mask)
    assert np.allclose(result.t_snapshot, output_times)
    assert np.isclose(result.t[-1], 0.04)
    assert np.all(result.dt_history > 0.0)
    non_floor = result.dt_history[2:] > 1.0e-5 * (1.0 + 1.0e-12)
    if np.any(non_floor):
        assert np.nanmax(result.error_omega_history[2:][non_floor]) <= 1.0e-8 + 1.0e-13
        assert np.nanmax(result.error_r_history[2:][non_floor]) <= 1.0e-7 + 1.0e-13

    floor_result = solve_adaptive_extrapolation(
        omega0,
        nu=0.02,
        gamma=10.0,
        domain=domain,
        discrete_num=resolution,
        dt0=5.0e-4,
        t_span=(0.0, 0.003),
        method="ars222",
        force=force,
        adaptive_options=ExtrapolationAdaptiveOptions(
            tol_omega=1.0e-15,
            tol_r=1.0e-15,
            dt_min=1.0e-4,
            dt_max=1.0e-2,
            max_rejections=100,
        ),
        output_times=np.array([0.0, 0.003]),
        fftw_threads=1,
    )
    assert floor_result.floor_accepted_steps > 0
    assert np.isclose(floor_result.t[-1], 0.003)

    print(
        "PASS: extrapolation adaptive controller; "
        f"accepted={result.accepted_steps}, rejected={result.rejected_steps}, "
        f"dt=[{result.dt_history.min():.3e}, {result.dt_history.max():.3e}]"
    )


if __name__ == "__main__":
    main()
