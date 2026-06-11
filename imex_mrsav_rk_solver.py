"""Fixed-step IMEX-mr-SAV-RK solver for 2D periodic Navier-Stokes.

The solver uses the vorticity-streamfunction formulation on a periodic box,
Fourier pseudo-spectral differentiation, 2/3 dealiasing, and FFTW when available.

The time discretization follows the stage equations

    omega_i = omega_n + nu*dt*sum_j a_ij Delta omega_j + dt*sum_j ahat_ij (f_j - q_i*V(q_i)*N_j),

    q_i = q_n - gamma*dt*sum_j a_ij q_j + dt*sum_j ahat_ij (gamma + V(q_i)*(N_j, omega_i)),

where N_j = u_j dot grad(omega_j).  Each stage is reduced to two Helmholtz solves and one scalar Newton solve for q_i.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fftw_fft

    pyfftw.interfaces.cache.enable()
    HAS_FFTW = True
except ImportError:  # pragma: no cover - SciPy fallback is for portability.
    import scipy.fft as scipy_fft

    pyfftw = None
    fftw_fft = None
    HAS_FFTW = False


Array = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
ForceFn = Callable[[Array, Array, float], Array]
ScalarFn = Callable[[float], float]


def optimal_fftw_threads(n: int) -> int:
    """Thread heuristic for pyFFTW, tuned by benchmarking on a 10-core CPU.

    For small transforms the thread spin-up cost dwarfs the FFT itself
    (multithreading measured ~6x slower than a single thread at 64x64), so use
    one thread there.  Larger transforms scale well, so use up to the available
    core count.  Thresholds adapt to the machine via os.cpu_count().
    """
    cores = os.cpu_count() or 1
    if n <= 64:
        return 1
    if n <= 128:
        return min(8, cores)
    return cores


@dataclass(frozen=True)
class PeriodicOps:
    """Fourier operators and grid data for a 2D periodic box."""

    xa: float
    ya: float
    xb: float
    yb: float
    nx: int
    ny: int
    x: Array
    y: Array
    X: Array
    Y: Array
    hx: float
    hy: float
    cell_area: float
    Dx: ComplexArray
    Dy: ComplexArray
    Lap: Array
    inv_Lap: Array
    dealias_mask: npt.NDArray[np.bool_]
    fftw_threads: int

    def fft2(self, u: Array | ComplexArray) -> ComplexArray:
        """Real forward transform; returns the half spectrum (ny, nx//2+1)."""
        if HAS_FFTW:
            return fftw_fft.rfft2(np.asarray(u, dtype=np.float64), threads=self.fftw_threads)
        return scipy_fft.rfft2(np.asarray(u, dtype=np.float64))

    def ifft2(self, u_hat: ComplexArray) -> Array:
        """Inverse real transform; returns a real (ny, nx) array."""
        shape = (self.ny, self.nx)
        if HAS_FFTW:
            return fftw_fft.irfft2(u_hat, s=shape, threads=self.fftw_threads)
        return scipy_fft.irfft2(u_hat, s=shape)

    def dealias(self, u_hat: ComplexArray) -> ComplexArray:
        return u_hat * self.dealias_mask


@dataclass(frozen=True)
class IMEXTableau:
    """Stage coefficients in the notation of the article.

    The stage index 0 denotes the known value at t_n.  The arrays A and Ahat
    have shape (s, s).  At stage i=1,...,s, only A[i-1, :i] and
    Ahat[i-1, :i] are used.
    """

    name: str
    order: int
    A: Array
    Ahat: Array
    c: Array
    chat: Array

    @property
    def stages(self) -> int:
        return int(self.A.shape[0])

    def time_of_stage(self, stage_index: int) -> float:
        """Return c_j for known stage j, where j=0 is the old solution."""
        if stage_index == 0:
            return 0.0
        return float(self.c[stage_index - 1])


@dataclass(frozen=True)
class NewtonOptions:
    tol: float = 1.0e-12
    max_iter: int = 30
    min_abs_q: float = 1.0e-12
    max_step: float = 5.0
    damping_steps: int = 12


@dataclass(frozen=True)
class NewtonInfo:
    converged: bool
    iterations: int
    residual: float

@dataclass
class StepDiagnostics:
    q_newton: Tuple[NewtonInfo, ...]
    max_newton_iterations: int
    mean_vorticity: float

@dataclass
class SolverResult:
    t: Array
    q: Array
    omega: Optional[Array]
    energy: Array
    enstrophy: Array
    palinstrophy: Array
    max_vorticity: Array
    cpu_time: Array
    method: str
    dt: float


def make_periodic_ops(
    domain: Tuple[float, float, float, float],
    discrete_num: Tuple[int, int],
    fftw_threads: Optional[int] = None,
) -> PeriodicOps:
    """Create periodic Fourier operators.

    Parameters
    ----------
    domain:
        (xa, ya, xb, yb).
    discrete_num:
        (Nx, Ny).  Physical arrays use shape (Ny, Nx), matching
        np.meshgrid(x, y, indexing="xy").
    """
    xa, ya, xb, yb = map(float, domain)
    nx, ny = map(int, discrete_num)
    if nx <= 0 or ny <= 0:
        raise ValueError("discrete_num must contain positive integers")

    if fftw_threads is None:
        fftw_threads = optimal_fftw_threads(max(nx, ny))
    if HAS_FFTW:
        pyfftw.config.NUM_THREADS = int(fftw_threads)

    lx = xb - xa
    ly = yb - ya
    hx = lx / nx
    hy = ly / ny
    x = np.linspace(xa, xb, nx, endpoint=False, dtype=np.float64)
    y = np.linspace(ya, yb, ny, endpoint=False, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Real-FFT layout: the last axis (x) is reduced to nx//2+1 modes; the
    # first axis (y) keeps the full ny modes.  All operators live on this half
    # spectrum, matching ops.fft2 == rfft2.
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=hx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=hy)
    Dx = (1j * kx)[np.newaxis, :].astype(np.complex128)
    Dy = (1j * ky)[:, np.newaxis].astype(np.complex128)
    Lap = ((1j * kx)[np.newaxis, :] ** 2 + (1j * ky)[:, np.newaxis] ** 2).real

    # First derivatives: zero the Nyquist mode for even N.  The full-complex
    # code dropped it implicitly via .real truncation; irfft2 would otherwise
    # keep a spurious contribution.  Even-order operators (Lap) keep Nyquist.
    if nx % 2 == 0:
        Dx[:, -1] = 0.0
    if ny % 2 == 0:
        Dy[ny // 2, :] = 0.0

    inv_Lap = np.zeros_like(Lap, dtype=np.float64)
    nonzero = Lap != 0.0
    inv_Lap[nonzero] = 1.0 / Lap[nonzero]

    kx_index = np.fft.rfftfreq(nx) * nx
    ky_index = np.fft.fftfreq(ny) * ny
    KX_index = kx_index[np.newaxis, :]
    KY_index = ky_index[:, np.newaxis]
    dealias_mask = (np.abs(KX_index) <= nx // 3) & (np.abs(KY_index) <= ny // 3)

    return PeriodicOps(
        xa=xa,
        ya=ya,
        xb=xb,
        yb=yb,
        nx=nx,
        ny=ny,
        x=x,
        y=y,
        X=X,
        Y=Y,
        hx=hx,
        hy=hy,
        cell_area=hx * hy,
        Dx=Dx,
        Dy=Dy,
        Lap=Lap,
        inv_Lap=inv_Lap,
        dealias_mask=dealias_mask,
        fftw_threads=int(fftw_threads),
    )


def _validate_tableau(tableau: IMEXTableau) -> IMEXTableau:
    A = np.asarray(tableau.A, dtype=np.float64)
    Ahat = np.asarray(tableau.Ahat, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if Ahat.shape != A.shape:
        raise ValueError("Ahat must have the same shape as A")
    s = A.shape[0]
    lower = np.tril(np.ones_like(A, dtype=bool))
    if not np.allclose(A[~lower], 0.0):
        raise ValueError("A must be lower triangular in the stage ordering")
    if not np.allclose(Ahat[~np.tril(np.ones_like(Ahat, dtype=bool), k=0)], 0.0):
        raise ValueError("Ahat must only use known stages j=0,...,i-1")

    c = np.sum(A, axis=1)
    chat = np.sum(Ahat, axis=1)
    if not np.allclose(c, chat, atol=1e-13, rtol=1e-13):
        raise ValueError(f"implicit and explicit stage times differ: {c} vs {chat}")

    return IMEXTableau(
        name=tableau.name,
        order=int(tableau.order),
        A=A,
        Ahat=Ahat,
        c=c.astype(np.float64),
        chat=chat.astype(np.float64),
    )


def make_tableau(method: str) -> IMEXTableau:
    """Return built-in IMEX-mr-SAV-RK coefficients.

    Names
    -----
    rk1, imex-mrsav-rk1:
        One-stage first-order IMEX Euler.
    rk2, imex-mrsav-rk2:
        Second-order IMEX scheme with c2=0.3.  The zero first row of the
        standard Butcher tableau is represented by the known value omega_n,
        so only the two nontrivial output stages are stored here.
    rk3, imex-mrsav-rk3:
        Third-order IMEX scheme with a55=1.  The zero first row of the
        standard Butcher tableau is represented by the known value omega_n,
        so only the four nontrivial output stages are stored here.
    rk4, imex-mrsav-rk4:
        Fourth-order IMEX scheme with ahat43=1.  As above, the zero first row
        of the standard Butcher tableau is represented by omega_n, so only
        the six nontrivial output stages are stored here.
    """
    key = method.lower().replace("_", "-")
    if key in {"rk1", "imex-rk1", "imex-mrsav-rk1"}:
        A = np.array([[1.0]], dtype=np.float64)
        Ahat = np.array([[1.0]], dtype=np.float64)
        return _validate_tableau(IMEXTableau("imex-mrsav-rk1", 1, A, Ahat, A.sum(1), Ahat.sum(1)))

    if key in {"rk2", "imex-rk2", "imex-mrsav-rk2"}:
        c2 = 0.3
        b2 = 1.0 / (2.0 - 2.0 * c2)
        b3 = (1.0 - 2.0 * c2) / (2.0 - 2.0 * c2)
        bhat1 = 1.0 - 1.0 / (2.0 * c2)
        bhat2 = 1.0 / (2.0 * c2)
        A = np.array(
            [
                [c2, 0.0],
                [b2, b3],
            ],
            dtype=np.float64,
        )
        Ahat = np.array(
            [
                [c2, 0.0],
                [bhat1, bhat2],
            ],
            dtype=np.float64,
        )
        return _validate_tableau(IMEXTableau("imex-mrsav-rk2", 2, A, Ahat, A.sum(1), Ahat.sum(1)))

    if key in {"rk3", "imex-rk3", "imex-mrsav-rk3"}:
        a55 = 1.0
        a52 = -2.0 * (36.0 * a55 - 5.0) / 147.0
        a53 = (75.0 * a55 + 598.0) / 588.0
        a54 = -25.0 * (15.0 * a55 + 2.0) / 588.0

        ahat31 = (939.0 * a55 + 282.0) / (2640.0 * a55 + 940.0)
        ahat32 = (381.0 * a55 + 188.0) / (2640.0 * a55 + 940.0)
        ahat41 = 9.0 * (639.0 * a55 + 1222.0) / (250.0 * (132.0 * a55 + 47.0))
        A = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-3.0 / 10.0, 4.0 / 5.0, 0.0, 0.0],
                [-367.0 / 250.0, 196.0 / 125.0, 4.0 / 5.0, 0.0],
                [a52, a53, a54, a55],
            ],
            dtype=np.float64,
        )
        Ahat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [ahat31, ahat32, 0.0, 0.0],
                [ahat41, 9.0 / 10.0, -ahat41, 0.0],
                [47.0 / 270.0, 1.0 / 10.0, 19.0 / 30.0, 5.0 / 54.0],
            ],
            dtype=np.float64,
        )
        return _validate_tableau(IMEXTableau("imex-mrsav-rk3", 3, A, Ahat, A.sum(1), Ahat.sum(1)))

    if key in {"rk4", "imex-rk4", "imex-mrsav-rk4"}:
        ahat43 = 1.0

        a42 = -11099846794473413537.0 / 13545655559296875000.0
        a43 = 5938991227245191.0 / 56762747105625000.0
        a52 = -15012700453574148059759.0 / 355573458431542968750000.0
        a53 = 37751222339857820917.0 / 135456555592968750000.0

        ahat51 = (70997500000.0 * ahat43 + 1042842334347.0) / 2411160939000.0
        ahat52 = (-283990000000.0 * ahat43 - 5126845621293.0) / 2411160939000.0
        ahat53 = 70997500.0 * ahat43 / 803720313.0 + 570851989.0 / 676532250.0
        ahat61 = (1913150328903.0 - 672359500000.0 * ahat43) / 2893393126800.0
        ahat62 = (2689438000000.0 * ahat43 + 3223449241353.0) / 2893393126800.0
        ahat63 = (-168089875000.0 * ahat43 - 138406721733.0) / 241116093900.0

        A = np.array(
            [
                [3.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0 / 2.0, 3.0 / 2.0, 0.0, 0.0, 0.0, 0.0],
                [-169.0 / 800.0, 129.0 / 800.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
                [
                    a42,
                    a43,
                    4021588899578801.0 / 4257206032921875.0,
                    144648284471.0 / 278085937500.0,
                    0.0,
                    0.0,
                ],
                [
                    a52,
                    a53,
                    2547104330002710487.0 / 10159241669472656250.0,
                    -3921377950657453.0 / 7299755859375000.0,
                    4.0 / 5.0,
                    0.0,
                ],
                [
                    94181.0 / 262500.0,
                    -53.0 / 100.0,
                    3.0 / 5.0,
                    -125681.0 / 262500.0,
                    4.0 / 5.0,
                    1.0 / 4.0,
                ],
            ],
            dtype=np.float64,
        )
        Ahat = np.array(
            [
                [3.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [7.0 / 10.0, 3.0 / 10.0, 0.0, 0.0, 0.0, 0.0],
                [
                    ahat43 / 3.0 + 1557.0 / 4000.0,
                    243.0 / 4000.0 - 4.0 * ahat43 / 3.0,
                    ahat43,
                    0.0,
                    0.0,
                    0.0,
                ],
                [ahat51, ahat52, ahat53, 8.0 / 5.0, 0.0, 0.0],
                [
                    ahat61,
                    ahat62,
                    ahat63,
                    -201267778.0 / 267906771.0,
                    3.0 / 10.0,
                    0.0,
                ],
                [
                    25.0 / 162.0,
                    -811.0 / 540.0,
                    3.0 / 22.0,
                    500.0 / 891.0,
                    3.0 / 4.0,
                    9.0 / 10.0,
                ],
            ],
            dtype=np.float64,
        )
        return _validate_tableau(IMEXTableau("imex-mrsav-rk4", 4, A, Ahat, A.sum(1), Ahat.sum(1)))

    raise ValueError("unknown method {!r}; use rk1, rk2, rk3, or rk4".format(method))


def make_taylor_v(order: int) -> Tuple[ScalarFn, ScalarFn]:
    """Return V_k and V_k' where V_k is the Taylor polynomial of 1/q at q=1."""
    if order <= 0:
        raise ValueError("order must be positive")

    def V(q: float) -> float:
        x = q - 1.0
        total = 0.0
        power = 1.0
        sign = 1.0
        for _ in range(order):
            total += sign * power
            power *= x
            sign *= -1.0
        return float(total)

    def dV(q: float) -> float:
        x = q - 1.0
        total = 0.0
        power = 1.0
        sign = -1.0
        for m in range(1, order):
            total += sign * m * power
            power *= x
            sign *= -1.0
        return float(total)

    return V, dV


def prepare_initial_vorticity(omega0: Array, ops: PeriodicOps, project_mean: bool = True) -> Array:
    """Accept either (Ny, Nx) or endpoint-including (Ny+1, Nx+1) data."""
    omega = np.asarray(omega0, dtype=np.float64)
    if omega.shape == (ops.ny + 1, ops.nx + 1):
        omega = omega[:-1, :-1]
    elif omega.shape != (ops.ny, ops.nx):
        raise ValueError(
            "omega0 has shape {}, expected ({}, {}) or ({}, {})".format(
                omega.shape, ops.ny, ops.nx, ops.ny + 1, ops.nx + 1
            )
        )
    omega = np.array(omega, dtype=np.float64, copy=True)
    if project_mean:
        omega -= float(np.mean(omega))
    return omega


def inner_product(ops: PeriodicOps, f: Array, g: Array) -> float:
    return float(ops.cell_area * np.sum(f * g))


def poisson_solve(ops: PeriodicOps, omega: Array) -> Array:
    omega_hat = ops.fft2(omega)
    psi_hat = -omega_hat * ops.inv_Lap
    psi_hat[0, 0] = 0.0
    return ops.ifft2(psi_hat).real


def velocity_from_vorticity(ops: PeriodicOps, omega: Array) -> Tuple[Array, Array]:
    omega_hat = ops.dealias(ops.fft2(omega))
    psi_hat = -omega_hat * ops.inv_Lap
    psi_hat[0, 0] = 0.0
    u = ops.ifft2(ops.Dy * psi_hat).real
    v = ops.ifft2(-ops.Dx * psi_hat).real
    return u, v


def advection_term(
    ops: PeriodicOps, omega: Array, omega_hat: Optional[ComplexArray] = None
) -> Array:
    """Return u dot grad(omega) using a skew-symmetric dealiased form.

    If the (un-dealiased) Fourier transform of omega is already available it can
    be passed as omega_hat to skip a redundant forward FFT.  The [0,0] mode does
    not affect the advection, so a mean-projected transform may be passed.
    """
    if omega_hat is None:
        omega_hat = ops.fft2(omega)
    omega_hat = ops.dealias(omega_hat)
    omega_d = ops.ifft2(omega_hat).real

    psi_hat = -omega_hat * ops.inv_Lap
    psi_hat[0, 0] = 0.0
    u = ops.ifft2(ops.Dy * psi_hat).real
    v = ops.ifft2(-ops.Dx * psi_hat).real

    omega_x = ops.ifft2(ops.Dx * omega_hat).real
    omega_y = ops.ifft2(ops.Dy * omega_hat).real

    flux_x = ops.ifft2(ops.Dx * ops.dealias(ops.fft2(u * omega_d))).real
    flux_y = ops.ifft2(ops.Dy * ops.dealias(ops.fft2(v * omega_d))).real
    return 0.5 * (u * omega_x + v * omega_y + flux_x + flux_y)


def laplacian_hat(ops: PeriodicOps, omega_hat: ComplexArray) -> ComplexArray:
    return ops.Lap * omega_hat


def project_zero_mean(ops: PeriodicOps, omega: Array) -> Array:
    omega_hat = ops.fft2(omega)
    omega_hat[0, 0] = 0.0
    return ops.ifft2(omega_hat).real


def helmholtz_solve_hat(
    ops: PeriodicOps,
    rhs_hat: ComplexArray,
    nu: float,
    dt: float,
    a_diag: float,
    denom_cache: Optional[Dict[Tuple[float, float], Array]] = None,
    project_mean: bool = True,
) -> Tuple[Array, ComplexArray]:
    """Solve (I - nu*dt*a_diag*Delta) x = rhs in Fourier space."""
    key = (float(dt), float(a_diag))
    if denom_cache is not None and key in denom_cache:
        denom = denom_cache[key]
    else:
        denom = 1.0 - nu * dt * a_diag * ops.Lap
        if denom_cache is not None:
            denom_cache[key] = denom

    x_hat = rhs_hat / denom
    if project_mean:
        x_hat[0, 0] = 0.0
    return ops.ifft2(x_hat).real, x_hat


def _force_value(ops: PeriodicOps, force: Optional[ForceFn], t: float) -> Array:
    if force is None:
        return np.zeros((ops.ny, ops.nx), dtype=np.float64)
    value = np.asarray(force(ops.X, ops.Y, float(t)), dtype=np.float64)
    if value.shape == (ops.ny + 1, ops.nx + 1):
        value = value[:-1, :-1]
    if value.shape != (ops.ny, ops.nx):
        raise ValueError("force returned shape {}, expected {}".format(value.shape, (ops.ny, ops.nx)))
    return value


def _linear_combination(arrays: Iterable[Array], coeffs: Array, shape: Tuple[int, int]) -> Array:
    out = np.zeros(shape, dtype=np.float64)
    for coef, arr in zip(coeffs, arrays):
        if coef != 0.0:
            out += float(coef) * arr
    return out


def scalar_q_residual(
    q: float,
    m: float,
    Rq: float,
    dt: float,
    Acoef: float,
    Bcoef: float,
    V: ScalarFn,
) -> float:
    vq = V(q)
    return m * q - Rq - dt * Acoef * vq + dt * dt * Bcoef * q * vq * vq


def scalar_q_jacobian(
    q: float,
    m: float,
    dt: float,
    Acoef: float,
    Bcoef: float,
    V: ScalarFn,
    dV: ScalarFn,
) -> float:
    vq = V(q)
    dvq = dV(q)
    return m - dt * Acoef * dvq + dt * dt * Bcoef * (vq * vq + 2.0 * q * vq * dvq)


def newton_solve_q(
    m: float,
    Rq: float,
    dt: float,
    Acoef: float,
    Bcoef: float,
    V: ScalarFn,
    dV: ScalarFn,
    q0: float,
    options: NewtonOptions = NewtonOptions(),
) -> Tuple[float, NewtonInfo]:
    """Safeguarded scalar Newton solve for one stage q variable."""
    q = float(q0)
    residual = scalar_q_residual(q, m, Rq, dt, Acoef, Bcoef, V)
    scale = 1.0 + abs(Rq) + abs(m * q)
    if abs(residual) <= options.tol * scale:
        return q, NewtonInfo(True, 0, abs(residual))

    for iteration in range(1, options.max_iter + 1):
        jac = scalar_q_jacobian(q, m, dt, Acoef, Bcoef, V, dV)
        if not np.isfinite(jac) or abs(jac) < 1.0e-15:
            break
        step = residual / jac
        if not np.isfinite(step):
            break
        step = float(np.clip(step, -options.max_step, options.max_step))

        accepted = False
        old_abs = abs(residual)
        damping = 1.0
        q_trial = q
        r_trial = residual
        for _ in range(options.damping_steps):
            q_trial = q - damping * step
            if abs(q_trial) <= options.min_abs_q:
                damping *= 0.5
                continue
            try:
                r_trial = scalar_q_residual(q_trial, m, Rq, dt, Acoef, Bcoef, V)
            except FloatingPointError:
                damping *= 0.5
                continue
            if np.isfinite(r_trial) and abs(r_trial) <= (1.0 - 1.0e-4 * damping) * old_abs:
                accepted = True
                break
            damping *= 0.5

        if not accepted:
            q_trial = q - step
            if abs(q_trial) <= options.min_abs_q:
                q_trial = np.copysign(options.min_abs_q, q_trial if q_trial != 0.0 else 1.0)
            r_trial = scalar_q_residual(q_trial, m, Rq, dt, Acoef, Bcoef, V)

        q = float(q_trial)
        residual = float(r_trial)
        scale = 1.0 + abs(Rq) + abs(m * q)
        if abs(residual) <= options.tol * scale:
            return q, NewtonInfo(True, iteration, abs(residual))

    return q, NewtonInfo(False, options.max_iter, abs(residual))


def recover_stage(phi: Array, z: Array, scale: float) -> Array:
    return phi - scale * z


def step_imex_mrsav_rk(
    omega_n: Array,
    q_n: float,
    t_n: float,
    dt: float,
    ops: PeriodicOps,
    nu: float,
    gamma: float,
    tableau: IMEXTableau,
    force: Optional[ForceFn] = None,
    V: Optional[ScalarFn] = None,
    dV: Optional[ScalarFn] = None,
    newton_options: NewtonOptions = NewtonOptions(),
    denom_cache: Optional[Dict[Tuple[float, float], Array]] = None,
    project_mean: bool = True,
) -> Tuple[Array, float, StepDiagnostics]:
    """Advance one fixed-size IMEX-mr-SAV-RK step."""
    if V is None or dV is None:
        V, dV = make_taylor_v(tableau.order)

    s = tableau.stages
    shape = (ops.ny, ops.nx)
    omega_stages = [np.asarray(omega_n, dtype=np.float64)]
    q_stages = [float(q_n)]
    omega_hat_0 = ops.fft2(omega_stages[0])
    omega_hat_0[0, 0] = 0.0
    omega_hat_stages = [omega_hat_0]
    N_stages = [advection_term(ops, omega_stages[0], omega_hat=omega_hat_0)]

    force_stages = []
    for j in range(s):
        force_time = t_n + tableau.time_of_stage(j) * dt
        force_stages.append(_force_value(ops, force, force_time))

    q_infos = []

    for stage in range(s):
        known_count = stage + 1
        ahat = tableau.Ahat[stage, :known_count]
        a_known = tableau.A[stage, :stage]
        a_diag = float(tableau.A[stage, stage])

        Fhat_phys = _linear_combination(force_stages[:known_count], ahat, shape)
        Nhat_phys = _linear_combination(N_stages[:known_count], ahat, shape)
        Fhat_hat = ops.fft2(Fhat_phys)
        Nhat_hat = ops.fft2(Nhat_phys)

        Rw_hat = np.array(omega_hat_stages[0], copy=True)
        for coeff, omega_hat_j in zip(a_known, omega_hat_stages[1:known_count]):
            if coeff != 0.0:
                Rw_hat += nu * dt * float(coeff) * laplacian_hat(ops, omega_hat_j)
        Rw_hat += dt * Fhat_hat
        if project_mean:
            Rw_hat[0, 0] = 0.0

        phi, phi_hat = helmholtz_solve_hat(ops, Rw_hat, nu, dt, a_diag, denom_cache, project_mean)
        z, z_hat = helmholtz_solve_hat(ops, Nhat_hat, nu, dt, a_diag, denom_cache, project_mean=True)

        q_known = 0.0
        for coeff, qj in zip(a_known, q_stages[1:known_count]):
            q_known += float(coeff) * float(qj)
        Rq = float(q_n) - gamma * dt * q_known + gamma * dt * float(tableau.chat[stage])
        m = 1.0 + gamma * dt * a_diag

        Acoef = inner_product(ops, Nhat_phys, phi)
        Bcoef = inner_product(ops, Nhat_phys, z)

        q_guess = q_stages[-1]
        q_i, q_info = newton_solve_q(m, Rq, dt, Acoef, Bcoef, V, dV, q_guess, newton_options)
        q_infos.append(q_info)

        scale = dt * q_i * V(q_i)
        # Reconstruct the stage directly in Fourier space (omega_i = phi -
        # scale*z holds mode by mode), avoiding the forward/inverse FFT round
        # trip of recover_stage + project_zero_mean + fft2.  The [0,0] mode is
        # zeroed after the inverse transform, so omega_i keeps its mean when
        # project_mean is False, matching the previous behaviour exactly.
        omega_hat_i = phi_hat - scale * z_hat
        omega_i = ops.ifft2(omega_hat_i).real
        omega_hat_i[0, 0] = 0.0

        omega_stages.append(omega_i)
        omega_hat_stages.append(omega_hat_i)
        q_stages.append(float(q_i))
        if stage < s - 1:
            N_stages.append(advection_term(ops, omega_i, omega_hat=omega_hat_i))

    omega_np1 = omega_stages[-1]
    q_np1 = q_stages[-1]
    diagnostics = StepDiagnostics(
        q_newton=tuple(q_infos),
        max_newton_iterations=max((info.iterations for info in q_infos), default=0),
        mean_vorticity=float(np.mean(omega_np1)),
    )
    return omega_np1, q_np1, diagnostics


def vorticity_energy(ops: PeriodicOps, omega: Array) -> Tuple[float, float, float]:
    omega_hat = ops.fft2(omega)
    psi_hat = -omega_hat * ops.inv_Lap
    psi_hat[0, 0] = 0.0

    u = ops.ifft2(ops.Dy * psi_hat).real
    v = ops.ifft2(-ops.Dx * psi_hat).real
    omega_x = ops.ifft2(ops.Dx * omega_hat).real
    omega_y = ops.ifft2(ops.Dy * omega_hat).real

    energy = 0.5 * (inner_product(ops, u, u) + inner_product(ops, v, v))
    enstrophy = 0.5 * inner_product(ops, omega, omega)
    palinstrophy = 0.5 * (inner_product(ops, omega_x, omega_x) + inner_product(ops, omega_y, omega_y))
    return float(energy), float(enstrophy), float(palinstrophy)


def _allocate_snapshots(
    n_saved: int,
    ops: PeriodicOps,
    keep_omega: bool,
) -> Optional[Array]:
    if not keep_omega:
        return None
    return np.empty((n_saved, ops.ny, ops.nx), dtype=np.float64)


def _saved_count(nsteps: int, save_every: int) -> int:
    return 1 + sum(1 for n in range(1, nsteps + 1) if n % save_every == 0)


def solve_fixed_step(
    omega0: Array,
    q0: float = 1.0,
    *,
    nu: float,
    gamma: float,
    domain: Tuple[float, float, float, float] = (0.0, 0.0, 2.0 * np.pi, 2.0 * np.pi),
    discrete_num: Optional[Tuple[int, int]] = None,
    dt: float,
    t_span: Tuple[float, float],
    method: str = "rk1",
    force: Optional[ForceFn] = None,
    V: Optional[ScalarFn] = None,
    dV: Optional[ScalarFn] = None,
    save_every: int = 1,
    keep_omega: bool = True,
    project_mean: bool = True,
    fftw_threads: Optional[int] = None,
    newton_options: NewtonOptions = NewtonOptions(),
    print_progress: bool = False,
) -> SolverResult:
    """Solve the periodic 2D NSE vorticity equation with fixed time steps."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if save_every <= 0:
        raise ValueError("save_every must be a positive integer")
    t0, tf = map(float, t_span)
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0")

    raw_omega0 = np.asarray(omega0, dtype=np.float64)
    if discrete_num is None:
        if raw_omega0.ndim != 2:
            raise ValueError("omega0 must be two-dimensional")
        ny0, nx0 = raw_omega0.shape
        if ny0 > 1 and nx0 > 1 and np.allclose(raw_omega0[0, :], raw_omega0[-1, :]) and np.allclose(
            raw_omega0[:, 0], raw_omega0[:, -1]
        ):
            discrete_num = (nx0 - 1, ny0 - 1)
        else:
            discrete_num = (nx0, ny0)

    ops = make_periodic_ops(domain, discrete_num, fftw_threads)
    omega = prepare_initial_vorticity(raw_omega0, ops, project_mean=project_mean)
    q = float(q0)
    tableau = make_tableau(method)
    if V is None or dV is None:
        V, dV = make_taylor_v(tableau.order)

    nsteps = int(np.rint((tf - t0) / dt))
    if nsteps <= 0:
        raise ValueError("dt is larger than the requested time interval")
    actual_tf = t0 + nsteps * dt
    if abs(actual_tf - tf) > 100.0 * np.finfo(float).eps * max(1.0, abs(tf)):
        raise ValueError(
            "fixed-step solve requires (tf-t0)/dt to be an integer; got final time {:.16g}".format(actual_tf)
        )

    n_saved = _saved_count(nsteps, save_every)
    omega_saved = _allocate_snapshots(n_saved, ops, keep_omega)
    t_saved = np.empty(n_saved, dtype=np.float64)

    q_values = np.empty(nsteps + 1, dtype=np.float64)
    energy = np.empty(nsteps + 1, dtype=np.float64)
    enstrophy = np.empty(nsteps + 1, dtype=np.float64)
    palinstrophy = np.empty(nsteps + 1, dtype=np.float64)
    max_vorticity = np.empty(nsteps + 1, dtype=np.float64)
    cpu_time = np.empty(nsteps + 1, dtype=np.float64)

    t = t0
    q_values[0] = q
    energy[0], enstrophy[0], palinstrophy[0] = vorticity_energy(ops, omega)
    max_vorticity[0] = float(np.max(np.abs(omega)))
    cpu_time[0] = 0.0
    t_saved[0] = t
    if keep_omega:
        omega_saved[0] = omega

    denom_cache: Dict[Tuple[float, float], Array] = {}
    save_index = 1
    wall0 = perf_counter()
    for n in range(1, nsteps + 1):
        omega, q, step_info = step_imex_mrsav_rk(
            omega,
            q,
            t,
            dt,
            ops,
            nu,
            gamma,
            tableau,
            force=force,
            V=V,
            dV=dV,
            newton_options=newton_options,
            denom_cache=denom_cache,
            project_mean=project_mean,
        )
        t = t0 + n * dt
        q_values[n] = q
        cpu_time[n] = perf_counter() - wall0
        step_elapsed = cpu_time[n] - cpu_time[n - 1]
        energy[n], enstrophy[n], palinstrophy[n] = vorticity_energy(ops, omega)
        max_vorticity[n] = float(np.max(np.abs(omega)))

        if n % save_every == 0:
            t_saved[save_index] = t
            if keep_omega:
                omega_saved[save_index] = omega
            save_index += 1

        if print_progress:
            print(
                "\rt={:.6f}/{:.6f}, q={:.8e}, E={:.8e}, Ens={:.8e}, Newton={}, "
                "step={:.4f}s, total={:.2f}s".format(
                    t, tf, q, energy[n], enstrophy[n], step_info.max_newton_iterations,
                    step_elapsed, cpu_time[n],
                ),
                end="",
                flush=True,
            )

    if print_progress:
        print()

    return SolverResult(
        t=t_saved,
        q=q_values,
        omega=omega_saved,
        energy=energy,
        enstrophy=enstrophy,
        palinstrophy=palinstrophy,
        max_vorticity=max_vorticity,
        cpu_time=cpu_time,
        method=tableau.name,
        dt=float(dt),
    )


__all__ = [
    "HAS_FFTW",
    "IMEXTableau",
    "NewtonInfo",
    "NewtonOptions",
    "PeriodicOps",
    "SolverResult",
    "StepDiagnostics",
    "advection_term",
    "make_periodic_ops",
    "make_tableau",
    "make_taylor_v",
    "solve_fixed_step",
    "step_imex_mrsav_rk",
    "velocity_from_vorticity",
    "vorticity_energy",
]
