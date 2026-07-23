from typing import Tuple
from time import perf_counter

import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
from scipy.optimize import newton, brentq
import numpy.typing as tnp
import os

def _optimal_fftw_threads(n: int) -> int:
      available = os.cpu_count() or 1
      if n <= 64:
          return 1
      if n <= 128:
          return min(4, available)
      return min(8, available)

class mrSAV_Vorticity_Stream_Periodic_Solve():
    """
    variable step mean reverting Vorticity-Streamfunction formulation Solver in periodic domain.
    """
    def __init__(
            self,
            nu: float,
            ga: float,
            s_domain: Tuple[float,float ,float, float],
            discrete_num: Tuple[int,int],
            initial_condition: tnp.NDArray,
            force_term,
            step_method: str,
            force_time_dependent: bool = True):

        pyfftw.config.NUM_THREADS = _optimal_fftw_threads(max(discrete_num))

        step_methods = {
            "IMEX" : (self.IMEX, 1),
            "IMEX_RK2" : (self.IMEX_RK2, 1),
            "ETD" : (self.ETD, 1),
            "ETDMS2": (self.ETDMS2, 2),
            "ETDRK4" : (self.ETDRK4, 1),
            "SDIRK2_mr_SAV": (self.SDIRK2_mr_SAV, 1),
            "ETD_mrSAV_MS2_b": (self.ETD_mrSAV_MS2_b, 2),
            "mr_SAV_BDF2": (self.mr_SAV_BDF2, 2),
            "ETD_mrSAV_MS2_L" : (self.ETD_mrSAV_MS2_L,3),
        }

        if step_method not in step_methods:
            raise ValueError(f"Not supported step method: {step_method}")

        self.step, self.setup_step = step_methods[step_method]

        # viscous coefficient
        self.nu = nu
        # free parameter gamma
        self.ga = ga
        self.talbot_node_count = 10

        self.xa, self.ya, self.xb, self.yb = s_domain
        self.Nx, self.Ny = discrete_num
        self.hx = (self.xb - self.xa) / self.Nx
        self.hy = (self.yb - self.ya) / self.Ny
        self.h = self.hx*self.hy

        self.xn = np.linspace(self.xa, self.xb, self.Nx + 1)
        self.yn = np.linspace(self.ya, self.yb, self.Ny + 1)
        self.X,self.Y = np.meshgrid(self.xn,self.yn)

        # presudo spectral method
        self.mu_x = 2 * np.pi / (self.xb - self.xa)
        self.mu_y = 2 * np.pi / (self.yb - self.ya)

        k_x = np.zeros(self.Nx); k_x[0:self.Nx//2] = np.arange(0,self.Nx//2); k_x[self.Nx//2+1:] = np.arange(-self.Nx//2+1,0)
        k_y = np.zeros(self.Ny); k_y[0:self.Ny//2] = np.arange(0,self.Ny//2); k_y[self.Ny//2+1:] = np.arange(-self.Ny//2+1,0)
        self.D_x = (1j*self.mu_x*k_x)[np.newaxis,:]
        self.D_y = (1j*self.mu_y*k_y)[:,np.newaxis]

        k_xx = np.zeros(self.Nx); k_xx[0:self.Nx//2] = np.arange(0,self.Nx//2); k_xx[self.Nx//2:] = np.arange(-self.Nx//2,0)
        k_yy = np.zeros(self.Ny); k_yy[0:self.Ny//2] = np.arange(0,self.Ny//2); k_yy[self.Ny//2:] = np.arange(-self.Ny//2,0)
        self.D_xx = ((1j*self.mu_x*k_xx)**2)[np.newaxis,:]
        self.D_yy = ((1j*self.mu_y*k_yy)**2)[:,np.newaxis]

        self.Lap = self.D_xx + self.D_yy
        mask = np.zeros_like(self.Lap); mask[0,0] = 1
        self.inv_Lap = 1/(self.Lap + mask); self.inv_Lap[0,0] = 0

        # 2/3-rule dealiasing mask: zero wavenumbers |k| > N/3
        self.dealias_mask = np.ones((self.Nx, self.Ny), dtype=bool)
        kx_max = self.Nx // 3
        ky_max = self.Ny // 3
        self.dealias_mask[kx_max+1 : self.Nx-kx_max, :] = False
        self.dealias_mask[:, ky_max+1 : self.Ny-ky_max] = False

        # Linear Operator
        self.L = self.linear_operator()

        self.f = force_term
        self.force_time_dependent = force_time_dependent

        self.Omega0 = initial_condition[:-1,:-1] - np.mean(initial_condition[:-1,:-1])
        self.q0 = 1.0
        self._fN_cache = None
        self._enable_fixed_step_cache = False
        self._fixed_step_cache = {}
        self._force_hat_cache = {}
        self._force_hat_cache_limit = 256
        self._force_hat_const = None

    def ft(self,u):
        return fft.fft2(u)

    def ift(self,u):
        return fft.ifft2(u)

    def dealias(self, u_hat):
        return u_hat * self.dealias_mask

    def velocity2vorticity(self,u,v):
        f_u = self.ft(u); f_u[0,0] = 0
        f_v = self.ft(v); f_v[0,0] = 0
        omega = self.ift(f_v*self.D_x - f_u*self.D_y).real
        return omega

    def vorticity2stream(self,omega):
        fomega = self.ft(omega)
        return self.ift(-fomega*self.inv_Lap).real

    def stream2velocity(self,psi):
        u =  self.ift(self.ft(psi)*self.D_y).real
        v = -self.ift(self.ft(psi)*self.D_x).real
        return u,v

    def N_hat(self,omega,omega_hat=None):
        if omega_hat is None:
            omega_hat = self.ft(omega)
        omega_hat = self.dealias(omega_hat)
        omega_d   = self.ift(omega_hat).real

        psi_hat = self.dealias(-omega_hat * self.inv_Lap)
        u = self.ift(psi_hat * self.D_y).real
        v = self.ift(-psi_hat * self.D_x).real

        omega_x = self.ift(self.D_x * omega_hat).real
        omega_y = self.ift(self.D_y * omega_hat).real

        u_omega_x_hat = self.D_x * self.dealias(self.ft(u * omega_d))
        v_omega_y_hat = self.D_y * self.dealias(self.ft(v * omega_d))

        return -(self.ft(u*omega_x + v*omega_y) + u_omega_x_hat + v_omega_y_hat)/2

    def N(self,omega):
        return self.ift(self.N_hat(omega)).real

    def linear_operator(self):
        return  -self.nu*self.Lap

    def vorticity_energy(self,omega):
        omega_x = self.ift(self.D_x*self.ft(omega)).real
        omega_y = self.ift(self.D_y*self.ft(omega)).real
        u,v = self.stream2velocity(self.vorticity2stream(omega))

        Energy = (self.inner_product(u,u) + self.inner_product(v,v))/2
        Enstrophy = self.inner_product(omega,omega)/2
        Palinstrophy = (self.inner_product(omega_x,omega_x) + self.inner_product(omega_y,omega_y))/2
        return Energy, Enstrophy, Palinstrophy

    def vorticity_rhs(self, omega, t):
        omega_hat = self.ft(omega)
        linear = self.ift(-self.L*omega_hat).real
        nonlinear = self.ift(self.N_hat(omega, omega_hat)).real
        force = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], t)
        return linear + nonlinear + force

    def enstrophy_rate(self, omega, t):
        return self.inner_product(self.vorticity_rhs(omega, t), omega)

    def energy_rate(self, omega, t, enstrophy=None):
        if enstrophy is None:
            _, enstrophy, _ = self.vorticity_energy(omega)
        psi = self.vorticity2stream(omega)
        force = self.f(self.X[:-1, :-1], self.Y[:-1, :-1], t)
        injection = self.inner_product(psi, force)
        dissipation = -2 * self.nu * enstrophy
        return dissipation + injection

    def inner_product(self,f,g):
        return self.h*np.sum(f*g)

    def inner_product_ft(self,f_hat,g_hat):
        return (self.h*np.sum(f_hat*np.conj(g_hat))/self.Nx/self.Ny).real

    def _cached(self, key, factory):
        if not self._enable_fixed_step_cache:
            return factory()
        if key not in self._fixed_step_cache:
            self._fixed_step_cache[key] = factory()
        return self._fixed_step_cache[key]

    def _reset_force_cache(self):
        self._force_hat_cache = {}
        self._force_hat_const = None

    def force_hat_at(self, t):
        if not self.force_time_dependent:
            if self._force_hat_const is None:
                f_value = self.f(self.X[:-1, :-1], self.Y[:-1, :-1], t)
                self._force_hat_const = self.ft(f_value)
            return self._force_hat_const

        key = round(float(t), 14)
        if key not in self._force_hat_cache:
            if len(self._force_hat_cache) >= self._force_hat_cache_limit:
                self._force_hat_cache.clear()
            f_value = self.f(self.X[:-1, :-1], self.Y[:-1, :-1], t)
            self._force_hat_cache[key] = self.ft(f_value)
        return self._force_hat_cache[key]

    def IMEX(self,Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]
        self.phi1 = self._imex_phi(tau_n)

        omega_n = Omega_s[-1]
        fomega_n = self.ft(omega_n)

        fN_1 = self.N_hat(omega_n, fomega_n)
        f_n = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn)

        fomega_n1 = self.phi1*fomega_n + tau_n*self.phi1*(fN_1 + self.ft(f_n))
        return self.ift(fomega_n1).real, 1

    def _etd_phi(self, tau):
        key = ("etd_phi", float(tau))

        def factory():
            M = 16
            dim = self.L.ndim
            r = np.expand_dims(
                np.exp(1j*np.pi*(np.arange(1, M+1) - .5)/M),
                axis=list(range(dim))
            )
            Lr = np.expand_dims(self.L, axis=-1) + r
            phi0_L = np.exp(-tau*self.L)
            phi1_L = np.mean((1-np.exp(-tau*Lr))/(tau*Lr), axis=-1).real
            return phi0_L, phi1_L

        return self._cached(key, factory)

    def _etd_ga_phi(self, tau):
        key = ("etd_ga_phi", float(tau))

        def factory():
            M = 16
            dim = self.L.ndim
            r = np.expand_dims(
                np.exp(1j*np.pi*(np.arange(1, M+1) - .5)/M),
                axis=list(range(dim))
            )
            Lr = np.expand_dims(self.L, axis=-1) + r
            gar = np.expand_dims(self.ga, axis=-1) + r
            phi0_L = np.exp(-tau*self.L)
            phi1_L = np.mean((1-np.exp(-tau*Lr))/(tau*Lr), axis=-1).real
            phi0_ga = np.exp(-tau*self.ga)
            phi1_ga = np.mean((1-np.exp(-tau*gar))/(tau*gar), axis=-1).real
            return phi0_L, phi1_L, phi0_ga, phi1_ga

        return self._cached(key, factory)

    def _imex_phi(self, tau):
        return self._cached(("imex_phi", float(tau)), lambda: 1/(1 + tau*self.L))

    def _bdf_phis(self, dt):
        key = ("bdf_phis", float(dt))
        return self._cached(key, lambda: (1/(1 + dt*self.L), 1/(1 + dt*self.ga)))

    def ETD(self, Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]

        self.phi0_L, self.phi1_L = self._etd_phi(tau_n)

        omega_n   = Omega_s[-1]
        fomega_n  = self.ft(omega_n)

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn)
        f_fn = self.ft(fn)

        fomega_n1 = self.phi0_L*fomega_n + tau_n*self.phi1_L*(f_fn); fomega_n1[0,0] = 0.+0j
        return self.ift(fomega_n1).real, p_n + 1

    def ETDMS2(self, Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]
        tau_nm = tau_s[-2]

        self.phi0_L, self.phi1_L = self._etd_phi(tau_n)

        omega_n   = Omega_s[-1]
        fomega_n  = self.ft(omega_n)
        omega_nm  = Omega_s[-2]

        fN_n = self.N_hat(omega_n, fomega_n)
        fN_nm = self._fN_cache if self._fN_cache is not None else self.N_hat(omega_nm)
        self._fN_cache = fN_n
        f_N12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn+tau_n/2)
        f_fn = self.ft(fn)

        fomega_n1 = self.phi0_L*fomega_n + tau_n*self.phi1_L*(f_N12 + f_fn); fomega_n1[0,0] = 0.+0j
        return self.ift(fomega_n1).real, p_n + 1

    def BDF2(self, Omega_s, q_s, tn, tau_s):
        tau_n  = tau_s[-1]
        tau_nm = tau_s[-2]
        rho = tau_n / tau_nm

        # variable-step BDF2 coefficients
        a0 = (1 + 2*rho) / (1 + rho)
        c1 = (1 + rho)**2 / (1 + 2*rho)     # coeff for omega_n
        c2 = rho**2 / (1 + 2*rho)           # coeff for omega_nm (subtracted)
        dt = tau_n / a0                     # effective step size

        phi_L = self._imex_phi(dt)

        omega_n  = Omega_s[-1]; omega_nm = Omega_s[-2]
        fomega_n  = self.ft(omega_n); fomega_nm = self.ft(omega_nm)

        q_n  = q_s[-1]

        f_n = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn + tau_n)

        omega_2 = (1+rho)*omega_n - rho*omega_nm
        fomega_2 = (1+rho)*fomega_n - rho*fomega_nm
        fN_2 = self.N_hat(omega_2, fomega_2)

        fomega_n1 = phi_L*(c1*fomega_n - c2*fomega_nm + dt*(fN_2 + self.ft(f_n)))

        return self.ift(fomega_n1).real, q_n

    def _prepare_ETDRK4_coefficients(self, tau):
        tau = float(tau)
        if getattr(self, "_etdrk4_tau", None) == tau:
            return
        M = 16
        dim = self.L.ndim
        r  = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ),axis = list(range(dim)) )
        Lr = tau*np.expand_dims(-self.L,axis=-1) + r

        self.phi10 = np.exp(-tau*self.L/2)
        self.phi11 = np.mean((1-np.exp(Lr/2))/(-Lr),axis=-1).real

        self.phi30 = np.exp(-tau*self.L)
        self.phi31 = np.mean((- 4 - Lr + np.exp(Lr)*(4 - 3*Lr + Lr**2))/(Lr)**3,axis=-1).real
        self.phi32 = np.mean((  2 + Lr + np.exp(Lr)*(-2+Lr))           /(Lr)**3,axis=-1).real
        self.phi33 = np.mean((- 4 - 3*Lr - Lr**2 + np.exp(Lr)*(4-Lr))  /(Lr)**3,axis=-1).real
        self._etdrk4_tau = tau

    def ETDRK4(self, Omega_s, q_s, tn, tau_s):
        tau = tau_s[-1]
        omega_n = Omega_s[-1]
        self._prepare_ETDRK4_coefficients(tau)

        fomega_n = self.ft(omega_n)
        N0 = self.N_hat(omega_n, fomega_n) + self.ft(self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn))

        fomega_n1 = self.phi10*fomega_n  + tau*self.phi11*N0
        omega_n1 = self.ift(fomega_n1).real
        N1 = self.N_hat(omega_n1, fomega_n1) + self.ft(self.f(self.X[:-1,:-1],self.Y[:-1,:-1], tn+tau/2))

        fomega_n2 = self.phi10*fomega_n  + tau*self.phi11*N1
        omega_n2 = self.ift(fomega_n2).real
        N2 = self.N_hat(omega_n2, fomega_n2) + self.ft(self.f(self.X[:-1,:-1],self.Y[:-1,:-1], tn+tau/2))

        fomega_n3 = self.phi10*fomega_n1 + tau*self.phi11*(2*N2-N0)
        omega_n3 = self.ift(fomega_n3).real
        N3 = self.N_hat(omega_n3, fomega_n3) + self.ft(self.f(self.X[:-1,:-1],self.Y[:-1,:-1], tn+tau))

        fomega_n4 = self.phi30*fomega_n  + tau*(self.phi31*N0 + 2*self.phi32*(N1 + N2) + self.phi33*N3)
        return self.ift(fomega_n4).real, 1

    def IMEX_RK2(self, Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]

        eta = 1 - np.sqrt(2)/2
        delta = 1 - 1/(2*eta)

        omega_n = Omega_s[-1]
        fomega_n = self.ft(omega_n)
        phi_L = self._imex_phi(eta*tau_n)

        # Stage 1
        fN_0 = self.N_hat(omega_n, fomega_n)
        f_0 = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn)
        ff_0 = self.ft(f_0)

        fN_hat_1 = eta*fN_0
        ff_hat_1 = eta*ff_0

        fomega_11 = phi_L*(fomega_n + tau_n*ff_hat_1); fomega_11[0,0] = 0.+0j
        fomega_12 = phi_L*fN_hat_1; fomega_12[0,0] = 0.+0j

        fomega_1 = fomega_11 + tau_n*fomega_12; fomega_1[0,0] = 0.+0j
        omega_1 = self.ift(fomega_1).real

        # Stage 2
        a_21 = 1 - 2*eta
        ahat_20 = delta - eta
        ahat_21 = 1 - delta

        fN_1 = self.N_hat(omega_1, fomega_1)
        f_1 = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn + eta*tau_n)

        fN_hat_2 = ahat_20*fN_0 + ahat_21*fN_1
        ff_hat_2 = ahat_20*ff_0 + ahat_21*self.ft(f_1)

        fomega_21 = phi_L*(fomega_1 - tau_n*a_21*self.L*fomega_1 + tau_n*ff_hat_2); fomega_21[0,0] = 0.+0j
        fomega_22 = phi_L*fN_hat_2; fomega_22[0,0] = 0.+0j

        fomega_2 = fomega_21 + tau_n*fomega_22; fomega_2[0,0] = 0.+0j
        return self.ift(fomega_2).real, 1.0

    def SDIRK2_mr_SAV(self, Omega_s, q_s, tn, tau_s, adaptive=False, fN_0=None):
        tau_n = tau_s[-1]

        eta = 1 - np.sqrt(2)/2
        delta = 1 - 1/(2*eta)

        omega_n = Omega_s[-1]
        fomega_n = self.ft(omega_n)
        r_n = 1 - q_s[-1]

        phi_L = self._imex_phi(eta*tau_n)
        scalar_mass = 1 + eta*tau_n*self.ga

        # Stage 1
        if fN_0 is None:
            fN_0 = self.N_hat(omega_n, fomega_n)
        f_0 = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn)
        ff_0 = self.ft(f_0)

        fN_hat_1 = eta*fN_0
        ff_hat_1 = eta*ff_0

        fomega_11 = phi_L*(fomega_n + tau_n*ff_hat_1); fomega_11[0,0] = 0.+0j
        fomega_12 = phi_L*fN_hat_1; fomega_12[0,0] = 0.+0j

        A_1 = self.inner_product_ft(fN_hat_1, fomega_11)
        B_1 = self.inner_product_ft(fN_hat_1, fomega_12)

        def scalar_1(r):
            return scalar_mass*r - r_n - tau_n*(1 + r)*(A_1 + tau_n*(1 - r**2)*B_1)

        def scalar_1_prime(r):
            return scalar_mass - tau_n*A_1 + tau_n**2*B_1*(3*r**2 + 2*r - 1)

        try:
            r_1 = newton(scalar_1, r_n, fprime=scalar_1_prime)
        except RuntimeError:
            lo, hi = -1., 1.
            while scalar_1(lo)*scalar_1(hi) > 0:
                lo *= 2
                hi *= 2
            r_1 = brentq(scalar_1, lo, hi)

        fomega_1 = fomega_11 + tau_n*(1 - r_1**2)*fomega_12; fomega_1[0,0] = 0.+0j
        omega_1 = self.ift(fomega_1).real

        # Stage 2
        a_21 = 1 - 2*eta
        ahat_20 = delta - eta
        ahat_21 = 1 - delta

        fN_1 = self.N_hat(omega_1, fomega_1)
        f_1 = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn + eta*tau_n)

        fN_hat_2 = ahat_20*fN_0 + ahat_21*fN_1
        ff_hat_2 = ahat_20*ff_0 + ahat_21*self.ft(f_1)

        fomega_21 = phi_L*(fomega_1 - tau_n*a_21*self.L*fomega_1 + tau_n*ff_hat_2); fomega_21[0,0] = 0.+0j
        fomega_22 = phi_L*fN_hat_2; fomega_22[0,0] = 0.+0j

        A_2 = self.inner_product_ft(fN_hat_2, fomega_21)
        B_2 = self.inner_product_ft(fN_hat_2, fomega_22)
        R_2 = (1 - tau_n*a_21*self.ga)*r_1

        def scalar_2(r):
            return scalar_mass*r - R_2 - tau_n*(1 + r)*(A_2 + tau_n*(1 - r**2)*B_2)

        def scalar_2_prime(r):
            return scalar_mass - tau_n*A_2 + tau_n**2*B_2*(3*r**2 + 2*r - 1)

        try:
            r_2 = newton(scalar_2, r_1, fprime=scalar_2_prime)
        except RuntimeError:
            lo, hi = -1., 1.
            while scalar_2(lo)*scalar_2(hi) > 0:
                lo *= 2
                hi *= 2
            r_2 = brentq(scalar_2, lo, hi)

        fomega_2 = fomega_21 + tau_n*(1 - r_2**2)*fomega_22; fomega_2[0,0] = 0.+0j
        Omega_2 = self.ift(fomega_2).real
        q_2 = 1 - r_2
        if not adaptive:
            return Omega_2, q_2

        Omega_1 = omega_n + (omega_1 - omega_n)/eta
        return Omega_2, Omega_1, q_2

    def ETD_mrSAV_MS2_b(self, Omega_s, q_s, tn, tau_s, fN_n=None, fN_nm=None, adaptive=False):
        tau_n = tau_s[-1]
        tau_nm = tau_s[-2]

        self.phi0_L, self.phi1_L, self.phi0_ga, self.phi1_ga = self._etd_ga_phi(tau_n)

        omega_n   = Omega_s[-1]
        fomega_n  = self.ft(omega_n)
        omega_nm  = Omega_s[-2]

        if fN_n is None:
            fN_n = self.N_hat(omega_n, fomega_n)
        if fN_nm is None:
            fN_nm = self._fN_cache if self._fN_cache is not None else self.N_hat(omega_nm)
        self._fN_cache = fN_n
        f_N12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn+tau_n/2)
        f_fn = self.ft(fn)

        A = tau_n*self.inner_product_ft(self.phi1_L * f_N12, self.phi0_L*fomega_n + tau_n*self.phi1_L*f_fn)
        B = tau_n**2*self.inner_product_ft(self.phi1_L * f_N12, self.phi1_L*f_N12)
        C =  self.phi0_ga*p_n

        Tgam = 0.1
        f = lambda p: p + (1-p)*A*Tgam + (p**3 - p**2 - p + 1)*B*Tgam - C
        try:
            p_n1 = newton(f, 0.)
        except RuntimeError:
            lo, hi = -10., 10.
            if f(lo) * f(hi) > 0:
                lo, hi = -100., 100.
            p_n1 = brentq(f, lo, hi)

        fomega_2 = self.phi0_L*fomega_n + tau_n*self.phi1_L*((1 - p_n1**2)*f_N12 + f_fn); fomega_2[0,0] = 0.+0j
        Omega_2 = self.ift(fomega_2).real
        q_2 = p_n1 + 1
        if not adaptive:
            return Omega_2, q_2

        fomega_1 = self.phi0_L*fomega_n + tau_n*self.phi1_L*((1 + p_n1)*f_N12 + f_fn); fomega_1[0,0] = 0.+0j
        return Omega_2, self.ift(fomega_1).real, q_2

    def mr_SAV_BDF2(self, Omega_s, q_s, tn, tau_s):
        tau_n  = tau_s[-1]
        tau_nm = tau_s[-2]
        rho = tau_n / tau_nm

        # variable-step BDF2 coefficients
        a0 = (1 + 2*rho) / (1 + rho)
        c1 = (1 + rho)**2 / (1 + 2*rho)     # coeff for omega_n
        c2 = rho**2 / (1 + 2*rho)           # coeff for omega_nm (subtracted)
        dt = tau_n / a0                     # effective step size

        phi_L, phi_ga = self._bdf_phis(dt)

        omega_n  = Omega_s[-1]
        omega_nm = Omega_s[-2]
        fomega_n  = self.ft(omega_n)
        fomega_nm = self.ft(omega_nm)

        q_n  = q_s[-1]
        q_nm = q_s[-2]

        f_n = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn + tau_n)

        omega_2 = (1+rho)*omega_n - rho*omega_nm
        fomega_2 = (1+rho)*fomega_n - rho*fomega_nm
        fN_2 = self.N_hat(omega_2, fomega_2)
        f_fn = self.ft(f_n)
        fomega_base = phi_L*(c1*fomega_n - c2*fomega_nm + dt*f_fn)

        LHS = 1 + dt**2*phi_ga*self.inner_product_ft(fN_2, phi_L*fN_2)
        RHS = phi_ga*(c1*q_n - c2*q_nm) \
            + dt*phi_ga*self.ga \
            - dt*phi_ga*self.inner_product_ft(fN_2, fomega_base)

        q_n1 = RHS / LHS
        fomega_n1 = fomega_base + dt*phi_L*q_n1*fN_2

        return self.ift(fomega_n1).real, q_n1

    def ETD_mrSAV_MS2_L(self, Omega_s, q_s, tn, tau_s,
                        fN_n=None, fN_nm=None, adaptive=False):
            tau_n =  tau_s[-1]
            tau_nm = tau_s[-2]

            self.phi0_L, self.phi1_L = self._etd_phi(tau_n)

            omega_n  = Omega_s[-1]
            omega_nm = Omega_s[-2]
            fomega_n = self.ft(omega_n)

            if fN_n is None:
                fN_n = self.N_hat(omega_n, fomega_n)
            if fN_nm is None:
                fN_nm = self._fN_cache if self._fN_cache is not None else self.N_hat(omega_nm)
            self._fN_cache = fN_n
            fN_12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

            q_n = q_s[-1]

            q_cache = {}

            def force_hat(tau):
                return self.force_hat_at(tn + tau)

            def talbot_nodes(t: float, N=16):
                key = ("talbot_nodes", float(t), int(N))

                def factory():
                    mu = 0.6443*N/t
                    nu = 0.5653
                    sigma = -0.4814*N/t

                    theta_k = np.pi/(2*N)*(2*np.arange(0,N) + 1)
                    z = sigma + mu*(theta_k/np.tan(theta_k) + nu*1j*theta_k)
                    dz_dtheta = mu*(1 / np.tan(theta_k) - theta_k / np.sin(theta_k)**2 + nu*1j)
                    return z, dz_dtheta

                return self._cached(key, factory)

            def talbot_inverse(f_tau_hat, t: float, N=8):
                """
                Use Talbot Contour Method to calculate the inverse Laplace transform.
                The forcing transform is cached outside the Talbot nodes because it is
                independent of the Laplace parameter s.
                """
                z, dz_dtheta = talbot_nodes(t, N)

                inv_key = ("talbot_inv_zL", float(t), int(N))

                def inv_factory():
                    return 1 / (z[:, np.newaxis, np.newaxis] + self.L[np.newaxis, :, :])

                inv_zL_values = self._cached(inv_key, inv_factory)

                Fz = np.empty(N, dtype=np.complex128)
                inner_scale = self.h / (self.Nx*self.Ny)
                conj_fN_12 = np.conj(fN_12)
                conj_fomega_n = np.conj(fomega_n)
                conj_f_tau_hat = np.conj(f_tau_hat)
                for i, zi in enumerate(z):
                    inv_zL = inv_zL_values[i]
                    weighted_fN = fN_12 * np.conj(inv_zL)
                    forcing_inner = (
                        inner_scale * np.sum(weighted_fN * (conj_fomega_n + conj_f_tau_hat/np.conj(zi)))
                    ).real
                    denom_inner = (inner_scale*np.sum(weighted_fN*conj_fN_12)).real
                    Fz[i] = (
                        self.ga/zi
                        - forcing_inner
                        + q_n) / (zi + self.ga + denom_inner)
                return np.sum(np.exp(z * t)*dz_dtheta*Fz/N).imag

            def q(tau):
                key = float(tau)
                if key not in q_cache:
                    q_cache[key] = talbot_inverse(force_hat(tau), tau, self.talbot_node_count)
                return q_cache[key]

            q_hn = q(tau_n/2)
            q_nn = q(tau_n)

            p = np.array([0, 1/2, 1], dtype=np.float64)
            w = np.array([1/6, 2/3, 1/6], dtype=np.float64)
            exp_key = ("ms2l_simpson_exp", float(tau_n))

            def exp_factory():
                return np.exp(-tau_n*(1-p)[:, np.newaxis, np.newaxis]*self.L[np.newaxis, :, :])

            exp_factors = self._cached(exp_key, exp_factory)
            force_values = np.stack([force_hat(0.0), force_hat(tau_n/2), force_hat(tau_n)], axis=0)
            q_values = np.array([q_n, q_hn, q_nn], dtype=np.float64)[:, np.newaxis, np.newaxis]

            fomega_n1 = self.phi0_L*fomega_n + tau_n*np.sum(
                w[:, np.newaxis, np.newaxis] * exp_factors * (force_values + q_values*fN_12),
                axis=0
            )

            omega_n1 = self.ift(fomega_n1).real
            z = self.ga*tau_n
            exp_ga = np.exp(-z)
            if abs(z) < 1e-12:
                phi1_ga = 1 - z/2 + z*z/6
            else:
                phi1_ga = (1 - exp_ga)/z
            fomega_mid = 0.5*(fomega_n + fomega_n1)
            q_nn = exp_ga*q_n + (1 - exp_ga) + tau_n*phi1_ga*self.inner_product_ft(fN_12, fomega_mid)

            if not adaptive:
                return omega_n1, q_nn

            fomega_n2 = self.phi0_L*fomega_n + tau_n*self.phi1_L*(force_values[-1] + q_nn*fN_12)
            return omega_n1, self.ift(fomega_n2).real, q_nn

    def ETD_mrGSAV_MS12_b(self, Omega_s, q_s, tn, tau_s, fN_n=None, fN_nm=None):
        return self.ETD_mrGSAV_MS2_b(
            Omega_s, q_s, tn, tau_s,
            fN_n=fN_n, fN_nm=fN_nm, adaptive=True,
        )

    def ETD_mrSAV_MS12_L(self, Omega_s, q_s, tn, tau_s, fN_n=None, fN_nm=None):
        return self.ETD_mrSAV_MS2_L(
            Omega_s, q_s, tn, tau_s,
            fN_n=fN_n, fN_nm=fN_nm, adaptive=True,
        )

    def init_record(self, M_max):
        self.Energy = np.empty(M_max + 1, dtype=np.float64)
        self.Energy_rate = np.empty(M_max + 1, dtype=np.float64)
        self.Enstrophy = np.empty(M_max + 1, dtype=np.float64)
        self.Enstrophy_rate = np.empty(M_max + 1, dtype=np.float64)
        self.Palinstrophy =  np.empty(M_max + 1, dtype=np.float64)
        self.Mx = np.empty(M_max + 1, dtype=np.float64)
        self.Energy[0], self.Enstrophy[0], self.Palinstrophy[0] = self.vorticity_energy(self.Omega0)
        self.Energy_rate[0] = self.energy_rate(self.Omega0, getattr(self, "T0", 0.0), self.Enstrophy[0])
        self.Enstrophy_rate[0] = self.enstrophy_rate(self.Omega0, getattr(self, "T0", 0.0))
        self.Mx[0] = np.max(self.Omega0)

    def result_record(self,i,Omega,q):
        self.Energy[i+1], self.Enstrophy[i+1], self.Palinstrophy[i+1] = self.vorticity_energy(Omega)
        self.Energy_rate[i+1] = self.energy_rate(Omega, self.tn[i+1], self.Enstrophy[i+1])
        self.Enstrophy_rate[i+1] = self.enstrophy_rate(Omega, self.tn[i+1])
        self.Mx[i+1] = np.max(Omega)
        msg = f"Vorticity Energy:{self.Energy[i+1]:.4f}, Energy rate:{self.Energy_rate[i+1]:.4e}, Enstrophy:{self.Enstrophy[i+1]:.4f}, Enstrophy rate:{self.Enstrophy_rate[i+1]:.4e}, Palinstrophy:{self.Palinstrophy[i+1]:.4f}, Maximum:{self.Mx[i+1]:.2f}, |q-1|:{np.abs(q - 1):.4e}"
        return msg

    def extend_array(self):
        # 扩展数组容量
        new_M_max = int(len(self.tn) * 1.5)  # 扩展为原来的1.5倍

        # 扩展 tau 数组
        new_tau = np.empty(new_M_max, dtype=np.float64)
        new_tau[:len(self.tau)] = self.tau
        self.tau = new_tau

        # 扩展 tn, q 数组
        new_tn = np.empty(new_M_max + 1, dtype=np.float64)
        new_tn[:len(self.tn)] = self.tn
        self.tn = new_tn

        new_q = np.empty(new_M_max + 1, dtype=np.float64)
        new_q[:len(self.q)] = self.q
        self.q = new_q

        if hasattr(self, 'ref_err'):
            new_ref_err = np.empty(new_M_max + 1, dtype=np.float64)
            new_ref_err[:len(self.ref_err)] = self.ref_err
            self.ref_err = new_ref_err

            new_rel_err = np.empty(new_M_max + 1, dtype=np.float64)
            new_rel_err[:len(self.rel_err)] = self.rel_err
            self.rel_err = new_rel_err

            new_controller_err = np.empty(new_M_max + 1, dtype=np.float64)
            new_controller_err[:len(self.controller_err)] = self.controller_err
            self.controller_err = new_controller_err

            new_ref_err_p = np.empty(new_M_max + 1, dtype=np.float64)
            new_ref_err_p[:len(self.ref_err_p)] = self.ref_err_p
            self.ref_err_p = new_ref_err_p

            new_ref_err_b = np.empty(new_M_max + 1, dtype=np.float64)
            new_ref_err_b[:len(self.ref_err_b)] = self.ref_err_b
            self.ref_err_b = new_ref_err_b


        # 扩展 Omega 数组
        if hasattr(self, 'Omega'):
            new_Omega = np.empty([new_M_max + 1] + list(self.Omega0.shape), dtype=np.float64)
            new_Omega[:len(self.Omega)] = self.Omega
            self.Omega = new_Omega

        # 扩展记录数组
        new_Energy = np.empty(new_M_max + 1, dtype=np.float64)
        new_Energy[:len(self.Energy)] = self.Energy
        self.Energy = new_Energy

        new_Energy_rate = np.empty(new_M_max + 1, dtype=np.float64)
        new_Energy_rate[:len(self.Energy_rate)] = self.Energy_rate
        self.Energy_rate = new_Energy_rate

        new_Enstrophy = np.empty(new_M_max + 1, dtype=np.float64)
        new_Enstrophy[:len(self.Enstrophy)] = self.Enstrophy
        self.Enstrophy = new_Enstrophy

        new_Enstrophy_rate = np.empty(new_M_max + 1, dtype=np.float64)
        new_Enstrophy_rate[:len(self.Enstrophy_rate)] = self.Enstrophy_rate
        self.Enstrophy_rate = new_Enstrophy_rate

        new_Palinstrophy = np.empty(new_M_max + 1, dtype=np.float64)
        new_Palinstrophy[:len(self.Palinstrophy)] = self.Palinstrophy
        self.Palinstrophy = new_Palinstrophy

        new_Mx = np.empty(new_M_max + 1, dtype=np.float64)
        new_Mx[:len(self.Mx)] = self.Mx
        self.Mx = new_Mx

        new_cpu_time = np.empty(new_M_max + 1, dtype=np.float64)
        new_cpu_time[:len(self.cpu_time)] = self.cpu_time
        self.cpu_time = new_cpu_time

    def solve_adaptive_step(
            self, t_span, tau_min, tau_max, snapshot=None, compute_ref_err=False,
            rho=0.9, rtol=1e-3, rtol_q=1e-3, r=1/2, ref_substeps=2, atol=1e-12, tau_initial=None, max_step_ratio=2.0):

        # 初始化自适应时间步长参数
        self.rho = rho
        self.rtol = rtol
        self.rtol_q = rtol_q
        self.r = r
        self.atol = atol
        self.max_step_ratio = max_step_ratio

        if not isinstance(t_span, (list, tuple)) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")
        if t_span[1] <= t_span[0]:
            raise ValueError("自适应步长模式要求 T > T0")
        if tau_min <= 0 or tau_max <= 0 or tau_min >= tau_max:
            raise ValueError("时间步长 tau_min, tau_max 必须大于 0, 且 tau_min < tau_max")
        if not 0 < rho < 1:
            raise ValueError("rho 必须满足 0 < rho < 1")
        if rtol <= 0 or rtol_q <= 0 or atol <= 0 or r <= 0:
            raise ValueError("rtol, rtol_q, atol 和 r 必须大于 0")
        if not isinstance(ref_substeps, int) or ref_substeps < 1:
            raise ValueError("ref_substeps 必须是 >= 1 的整数")
        if max_step_ratio < 1:
            raise ValueError("max_step_ratio 必须大于等于 1")
        if tau_initial is None:
            tau_initial = 5e-4
        if tau_initial <= 0:
            raise ValueError("tau_initial 必须大于 0")

        self.T0, self.T = t_span

        step_name = getattr(self.step, "__name__", "")
        adaptive_methods = {"SDIRK2_mr_SAV", "ETD_mrSAV_MS2_b", "ETD_mrSAV_MS2_L"}
        if step_name not in adaptive_methods:
            raise ValueError(f"自适应步长暂不支持格式: {step_name}")

        snapshot_mode = snapshot is not None
        if snapshot_mode:
            snapshot = np.sort(np.asarray(snapshot, dtype=np.float64))
            if snapshot.ndim != 1:
                raise ValueError("snapshot 必须是一维数组")
            snapshot_atol = 1e-12
            if np.any(snapshot < self.T0 - snapshot_atol) or np.any(snapshot > self.T + snapshot_atol):
                raise ValueError("snapshot 必须位于 t_span 内")
            snapshot_index = 0
            has_snapshots = len(snapshot) > 0

        M_max = int(np.ceil((self.T - self.T0)/tau_max)) + 200
        self.tau     = np.empty(M_max, dtype=np.float64)
        self.tn      = np.empty(M_max + 1, dtype=np.float64)
        self.q       = np.empty(M_max + 1, dtype=np.float64)
        self.ref_err = np.zeros(M_max + 1, dtype=np.float64)
        self.rel_err = np.zeros(M_max + 1, dtype=np.float64)
        self.controller_err = np.zeros(M_max + 1, dtype=np.float64)
        self.ref_err_p = np.zeros(M_max + 1, dtype=np.float64)
        self.ref_err_b = np.zeros(M_max + 1, dtype=np.float64)
        self.cpu_time  = np.empty(M_max + 1, dtype=np.float64)

        # Omega_temp_2 在两种模式下都分配，用于计算参考误差
        self.Omega_temp_2 = np.empty([self.setup_step+1] + list(self.Omega0.shape), dtype=np.float64)

        if hasattr(self, 'Omega'):
            del self.Omega
        self.Omega_temp = np.empty([self.setup_step+1] + list(self.Omega0.shape), dtype=np.float64)
        if snapshot_mode:
            if has_snapshots:
                self.snapshot_Omega = np.empty([len(snapshot)] + list(self.Omega0.shape), dtype=np.float64)
                self.snapshot_tn    = np.empty(len(snapshot), dtype=np.float64)
            else:
                self.snapshot_Omega = np.array([])
                self.snapshot_tn    = np.array([])

        # 初始化记录
        self.init_record(M_max)

        # 初始步设置
        self.tn[0]       = self.T0
        self.q[0]        = self.q0
        self.cpu_time[0] = 0.0
        self.Omega_temp[0]   = self.Omega0
        self.Omega_temp_2[0] = self.Omega0
        final_omega = self.Omega0


        if snapshot_mode:
            def save_initial_snapshots():
                nonlocal snapshot_index
                while has_snapshots and snapshot_index < len(snapshot) and snapshot[snapshot_index] <= self.T0 + snapshot_atol:
                    self.snapshot_Omega[snapshot_index] = self.Omega0
                    self.snapshot_tn[snapshot_index] = snapshot[snapshot_index]
                    snapshot_index += 1

            def save_interpolated_snapshots(t_prev, omega_prev, t_curr, omega_curr):
                nonlocal snapshot_index
                while has_snapshots and snapshot_index < len(snapshot) and snapshot[snapshot_index] <= t_curr + snapshot_atol:
                    target = snapshot[snapshot_index]
                    theta = np.clip((target - t_prev)/(t_curr - t_prev), 0.0, 1.0)
                    self.snapshot_Omega[snapshot_index] = (1 - theta)*omega_prev + theta*omega_curr
                    self.snapshot_tn[snapshot_index] = target
                    snapshot_index += 1

            save_initial_snapshots()

        self._fN_cache = None
        self._reset_force_cache()
        self._fixed_step_cache = {}
        self._enable_fixed_step_cache = False
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.forced_accept_steps = 0

        # 使用 ETDRK4 计算初始步，确保有足够的历史数据用于多步方法。
        initial_tau = max(tau_min, min(tau_initial, tau_max))
        time_atol = max(1e-14, 10*np.finfo(np.float64).eps*max(1.0, abs(self.T)))
        index = 1
        while index < self.setup_step and self.tn[index-1] < self.T - time_atol:
            i = index - 1
            self.tau[i] = min(initial_tau, self.T - self.tn[i])
            self.Omega_temp[index], self.q[index] = self.ETDRK4(
                self.Omega_temp[i:i+1], self.q[i:i+1], self.tn[i], self.tau[i:i+1]
            )
            self.Omega_temp_2[index] = self.Omega_temp[index]
            self.tn[index] = min(self.T, self.tn[i] + self.tau[i])
            self.cpu_time[index] = 0.0
            self.result_record(i, self.Omega_temp[index], self.q[index])
            final_omega = self.Omega_temp[index]
            if snapshot_mode:
                save_interpolated_snapshots(
                    self.tn[i], self.Omega_temp[i], self.tn[index], self.Omega_temp[index]
                )
            self.accepted_steps += 1
            index += 1

        if index == self.setup_step and self.tn[index-1] < self.T - time_atol:
            self.tau[index-1] = min(initial_tau, self.T - self.tn[index-1])

        # 主时间循环
        prev_fN_n = None
        while self.tn[index-1] < self.T - time_atol:
            if index >= len(self.tn) - 10:
                self.extend_array()

            omega_hist = self.Omega_temp[:-1]
            remaining_time = self.T - self.tn[index-1]
            tau_trial = min(self.tau[index-1], remaining_time)
            self.tau[index-1] = tau_trial

            start_time = perf_counter()  # 包含 N 预计算和所有被拒绝步的计算时间，与 solve_fix_step 计时口径一致
            # 预计算非线性项的傅里叶变换，避免步长被拒绝时重复计算
            if step_name == "SDIRK2_mr_SAV":
                fN_nm = None
            else:
                fN_nm = self.N_hat(omega_hist[-2]) if prev_fN_n is None else prev_fN_n
            fN_n = self.N_hat(omega_hist[-1])
            while True:
                if step_name == "SDIRK2_mr_SAV":
                    Omega_2, Omega_1, q_2 = self.SDIRK2_mr_SAV(
                        omega_hist,
                        self.q[index-self.setup_step:index],
                        self.tn[index-1],
                        self.tau[index-self.setup_step:index],
                        adaptive=True,
                        fN_0=fN_n,
                    )
                elif step_name == "ETD_mrSAV_MS2_L":
                    Omega_2, Omega_1, q_2 = self.ETD_mrSAV_MS2_L(
                        omega_hist,
                        self.q[index-self.setup_step:index],
                        self.tn[index-1],
                        self.tau[index-self.setup_step:index],
                        fN_n=fN_n,
                        fN_nm=fN_nm,
                        adaptive=True,
                    )
                else:
                    Omega_2, Omega_1, q_2 = self.ETD_mrSAV_MS2_b(
                        omega_hist,
                        self.q[index-self.setup_step:index],
                        self.tn[index-1],
                        self.tau[index-self.setup_step:index],
                        fN_n=fN_n,
                        fN_nm=fN_nm,
                        adaptive=True,
                    )

                # 计算误差
                diff_Omega = Omega_2 - Omega_1
                error_norm = np.sqrt(max(self.inner_product(diff_Omega, diff_Omega), 0.0))
                solution_norm = np.sqrt(max(self.inner_product(Omega_2, Omega_2), 0.0))
                Error_u = error_norm/(self.atol + self.rtol*solution_norm)
                Error_q = np.abs(q_2 - 1) + 1e-16

                # 计算新的时间步长
                factor_u = self.max_step_ratio if Error_u == 0 else Error_u**(-self.r)
                factor_q = self.rtol_q/Error_q
                factor = self.rho*min(factor_u, factor_q)
                tau_next = tau_trial*factor
                tau_next = min(tau_next, self.max_step_ratio*tau_trial, tau_max)
                tau_next = max(tau_min, tau_next)

                errors_pass = Error_u <= 1.0 and Error_q <= self.rtol_q
                at_min_step = tau_trial <= tau_min*(1 + 10*np.finfo(np.float64).eps)
                if errors_pass or at_min_step:
                    # 接受当前时间步
                    if not errors_pass:
                        self.forced_accept_steps += 1
                    self.Omega_temp[-1] = Omega_2
                    break

                self.rejected_steps += 1
                tau_trial = tau_next
                self.tau[index-1] = tau_trial

            end_time = perf_counter()
            cpu_time = end_time - start_time
            self.cpu_time[index] = self.cpu_time[index-1] + cpu_time

            self.q[index]         = q_2
            self.tn[index]        = min(self.T, self.tn[index-1] + tau_trial)
            self.rel_err[index]   = error_norm/max(solution_norm, self.atol)
            self.controller_err[index] = Error_u
            self.ref_err_p[index] = Error_q
            self.accepted_steps  += 1
            final_omega = Omega_2

            if self.tn[index] < self.T - time_atol:
                remaining_time = self.T - self.tn[index]
                self.tau[index] = min(tau_next, remaining_time)

            if compute_ref_err:
                fN_2 = self.N_hat(Omega_2)
                fN_1 = self.N_hat(Omega_1)
                diff_b = fN_2 - fN_1
                norm_b = np.sqrt(max(self.inner_product_ft(fN_2, fN_2), 0.0))
                Error_b = np.sqrt(max(self.inner_product_ft(diff_b, diff_b), 0.0))/max(norm_b, self.atol)
            else:
                Error_b = 0.0
            self.ref_err_b[index] = Error_b

            if compute_ref_err:
                ref_tau = self.tau[index-1] / ref_substeps
                ref_Omega = self.Omega_temp_2[-2]
                ref_q = self.q[index-1]
                ref_t = self.tn[index-1]
                for _ in range(ref_substeps):
                    ref_Omega, ref_q = self.ETDRK4(
                        np.array([ref_Omega]),
                        np.array([ref_q]),
                        ref_t,
                        np.array([ref_tau])
                    )
                    ref_t += ref_tau
                ref_err = np.sqrt(self.inner_product(Omega_2 - ref_Omega, Omega_2 - ref_Omega))
                ref_norm = np.sqrt(max(self.inner_product(ref_Omega, ref_Omega), 0.0))
                self.ref_err[index] = ref_err/max(ref_norm, self.atol)
                self.Omega_temp_2[-1] = ref_Omega
            else:
                self.ref_err[index] = 0.0

            if snapshot_mode:
                save_interpolated_snapshots(
                    self.tn[index-1], self.Omega_temp[-2], self.tn[index], self.Omega_temp[-1]
                )

            msg = self.result_record(index-1, self.Omega_temp[-1], self.q[index])
            print(f"\r {self.tn[index]:.6f}\\{self.T}, tau = {tau_trial:.6e}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")

            # 滚动更新
            self.Omega_temp[:-1] = self.Omega_temp[1:]
            self.Omega_temp_2[:-1] = self.Omega_temp_2[1:]

            prev_fN_n = fN_n
            index += 1

        print("")
        # 结果整理 - 裁剪数组到实际使用的大小
        if abs(self.tn[index-1] - self.T) <= time_atol:
            self.tn[index-1] = self.T
        self.tau          = self.tau[:max(index-1, 0)]
        self.tn           = self.tn[:index]
        self.q            = self.q[:index]
        self.Energy       = self.Energy[:index]
        self.Energy_rate  = self.Energy_rate[:index]
        self.Enstrophy    = self.Enstrophy[:index]
        self.Enstrophy_rate = self.Enstrophy_rate[:index]
        self.Palinstrophy = self.Palinstrophy[:index]
        self.Mx           = self.Mx[:index]
        self.ref_err      = self.ref_err[:index]
        self.rel_err      = self.rel_err[:index]
        self.controller_err = self.controller_err[:index]
        self.ref_err_p    = self.ref_err_p[:index]
        self.ref_err_b    = self.ref_err_b[:index]
        self.cpu_time     = self.cpu_time[:index]

        if snapshot_mode:
            self.Omega = self.snapshot_Omega
            self.tn_s  = self.snapshot_tn
        else:
            self.Omega = np.asarray(final_omega)[None, ...].copy()
        del self.Omega_temp, self.Omega_temp_2

    def solve_fix_step(self, t_span, tau, snapshot=None):

        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")
        if tau <= 0:
            raise ValueError("时间步长 tau 必须大于 0")

        self.T0, self.T = t_span
        M_float = (self.T - self.T0) / tau
        M = int(np.ceil(M_float))
        if M <= 0:
            raise ValueError("固定步长模式要求 T > T0")
        self.tau = np.full(M, tau, dtype=np.float64)
        self.tn = self.T0 + tau * np.arange(M + 1, dtype=np.float64)

        snapshot_mode = snapshot is not None
        if hasattr(self, 'Omega'):
            del self.Omega
        self.Omega_temp = np.empty([self.setup_step + 1] + list(self.Omega0.shape), dtype=np.float64)
        if snapshot_mode:
            snapshot = np.sort(np.asarray(snapshot, dtype=np.float64))
            snapshot_indices = np.rint((snapshot - self.T0) / tau).astype(np.int64)
            grid_atol = max(1e-12, abs(tau) * 1e-8)
            if (
                np.any(snapshot_indices < 0)
                or np.any(snapshot_indices > M)
                or not np.allclose(self.tn[snapshot_indices], snapshot, rtol=0.0, atol=grid_atol)
            ):
                raise ValueError("snapshot 必须落在固定步长时间网格上")
            snapshot_cursor = 0
            self.snapshot_Omega = np.empty([len(snapshot)] + list(self.Omega0.shape), dtype=np.float64)
            self.snapshot_tn = self.tn[snapshot_indices].copy()
        else:
            snapshot_indices = np.array([], dtype=np.int64)
            snapshot_cursor = 0

        self.q        = np.zeros(M + 1, dtype=np.float64)
        self.cpu_time = np.zeros(M + 1, dtype=np.float64)

        # 初始化记录
        self.init_record(M)

        def save_snapshot(index, omega):
            nonlocal snapshot_cursor
            while snapshot_cursor < len(snapshot_indices) and snapshot_indices[snapshot_cursor] == index:
                self.snapshot_Omega[snapshot_cursor] = omega
                snapshot_cursor += 1

        # 初始步设置
        self.q[0] = self.q0
        self.cpu_time[0] = 0.0
        self.Omega_temp[0] = self.Omega0
        save_snapshot(0, self.Omega0)

        # 使用 ETDRK4 启动多步法。
        for i in range(1, self.setup_step):
            omega_new, self.q[i] = self.ETDRK4(
                self.Omega_temp[i-1:i],
                self.q[i-1:i],
                self.tn[i-1],
                self.tau[i-1:i]
            )

            self.Omega_temp[i] = omega_new
            self.cpu_time[i] = 0.0
            self.result_record(i-1, omega_new, self.q[i])
            save_snapshot(i, omega_new)

        # 主时间循环
        self._fN_cache = None
        self._reset_force_cache()
        self._fixed_step_cache = {}
        self._enable_fixed_step_cache = True
        if getattr(self.step, "__name__", "") == "ETDRK4":
            self._prepare_ETDRK4_coefficients(tau)

        for index in range(self.setup_step, M + 1):
            start_time = perf_counter()
            omega_new, self.q[index] = self.step(
                self.Omega_temp[:-1],
                self.q[index-self.setup_step:index],
                self.tn[index-1],
                self.tau[index-self.setup_step:index]
            )
            self.Omega_temp[-1] = omega_new
            end_time = perf_counter()

            cpu_time = end_time - start_time
            self.cpu_time[index] = self.cpu_time[index-1] + cpu_time

            msg = self.result_record(index-1, omega_new, self.q[index])
            print(f"\r {self.tn[index]:.6f}\\{self.T}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")
            save_snapshot(index, omega_new)

            self.Omega_temp[:-1] = self.Omega_temp[1:]

        self._enable_fixed_step_cache = False
        print("")
        if snapshot_mode:
            self.Omega = self.snapshot_Omega
            self.tn_s  = self.snapshot_tn
        else:
            self.Omega = self.Omega_temp[-1:].copy()
        del self.Omega_temp

    def solve_random_step(self, t_span, M):

        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")
        if M <= 0:
            raise ValueError("时间步数 M 必须大于 0")

        self.T0, self.T = t_span
        eps        = np.random.uniform(0., 1., M)
        self.tau   = (self.T - self.T0)* eps / np.sum(eps)
        self.tn    = np.zeros(M + 1, dtype=np.float64)
        self.tn[0] = self.T0; self.tn[1:] = np.cumsum(self.tau) + self.T0

        self.q     = np.empty(M + 1, dtype=np.float64)
        self.Omega = np.zeros([M + 1] + list(self.Omega0.shape), dtype=np.float64)

        # 初始化记录
        self.init_record(M)

        # 初始步设置
        self.q[0]   = self.q0
        self.Omega[0] = self.Omega0

        # 使用ETDRK4计算初始步，确保有足够的历史数据用于高阶方法
        for i in range(1, self.setup_step):
            # 对于每个初始步，使用固定的tau_min
            # 调用ETDRK4计算下一步
            self.Omega[i], self.q[i] = self.ETDRK4(
                self.Omega[i-1:i],   # 传递最近1个Omega值（ETDRK4需要）
                self.q[i-1:i],       # 传递最近1个q值（ETDRK4需要）
                self.tn[i-1],        # 当前时间点
                self.tau[i-1:i]                  # 时间步长
            )
            # 记录结果
            self.result_record(i-1, self.Omega[i], self.q[i])

        # 主时间循环
        self._fN_cache = None
        self._reset_force_cache()
        self._enable_fixed_step_cache = False
        index = self.setup_step
        for index in range(self.setup_step, M + 1):

            start_time = perf_counter()
            self.Omega[index], self.q[index] = self.step(
                self.Omega[index-self.setup_step:index],
                self.q[index-self.setup_step:index],
                self.tn[index-1],
                self.tau[index-self.setup_step:index]
            )

            end_time = perf_counter()
            cpu_time = end_time - start_time

            # 更新显示信息
            msg = self.result_record(index-1, self.Omega[index], self.q[index])
            print(f"\r {self.tn[index]:.6f}\\{self.T}, tau = {self.tau[index-1]:.6e}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")

        print("")

    def solve_given_tau(self, t_span, tau, snapshot=None):

        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")

        self.tau = np.asarray(tau, dtype=np.float64)
        if self.tau.ndim != 1 or len(self.tau) == 0 or np.any(self.tau <= 0):
            raise ValueError("tau 必须是一维正数数组")

        self.T0, self.T = t_span
        self.T = self.T0 + np.sum(self.tau)
        M = len(self.tau)

        self.tn = np.zeros(M + 1, dtype=np.float64)
        self.tn[0] = self.T0; self.tn[1:] = np.cumsum(self.tau) + self.T0

        snapshot_mode = snapshot is not None
        if hasattr(self, 'Omega'):
            del self.Omega
        self.Omega_temp = np.empty([self.setup_step + 1] + list(self.Omega0.shape), dtype=np.float64)

        if snapshot_mode:
            snapshot = np.sort(np.asarray(snapshot, dtype=np.float64))
            if snapshot.ndim != 1:
                raise ValueError("snapshot 必须是一维数组")
            right = np.clip(np.searchsorted(self.tn, snapshot), 0, M)
            left = np.clip(right - 1, 0, M)
            use_left = np.abs(self.tn[left] - snapshot) <= np.abs(self.tn[right] - snapshot)
            snapshot_indices = np.where(use_left, left, right)
            grid_atol = max(1e-12, np.max(self.tau) * 1e-8)
            if not np.allclose(self.tn[snapshot_indices], snapshot, rtol=0.0, atol=grid_atol):
                raise ValueError("snapshot 必须落在给定 tau 生成的时间网格上")
            snapshot_cursor = 0
            self.snapshot_Omega = np.empty([len(snapshot)] + list(self.Omega0.shape), dtype=np.float64)
            self.snapshot_tn = snapshot.copy()
        else:
            snapshot_indices = np.array([], dtype=np.int64)
            snapshot_cursor = 0

        self.q = np.empty(M + 1, dtype=np.float64)

        # 初始化记录
        self.init_record(M)

        def save_snapshot(index, omega):
            nonlocal snapshot_cursor
            while snapshot_cursor < len(snapshot_indices) and snapshot_indices[snapshot_cursor] == index:
                self.snapshot_Omega[snapshot_cursor] = omega
                snapshot_cursor += 1

        # 初始步设置
        self.q[0] = self.q0
        self.Omega_temp[0] = self.Omega0
        save_snapshot(0, self.Omega0)

        # 使用 ETDRK4 启动多步格式；单步格式直接从第一个时间步开始。
        for i in range(1, min(self.setup_step, M + 1)):
            omega_new, self.q[i] = self.ETDRK4(
                self.Omega_temp[i-1:i],
                self.q[i-1:i],
                self.tn[i-1],
                self.tau[i-1:i]
            )
            self.Omega_temp[i] = omega_new
            self.result_record(i-1, omega_new, self.q[i])
            save_snapshot(i, omega_new)

        # 主时间循环
        self._fN_cache = None
        self._reset_force_cache()
        self._enable_fixed_step_cache = False
        final_omega = self.Omega_temp[min(self.setup_step - 1, M)]
        for index in range(self.setup_step, M + 1):

            start_time = perf_counter()
            omega_new, self.q[index] = self.step(
                self.Omega_temp[:-1],
                self.q[index-self.setup_step:index],
                self.tn[index-1],
                self.tau[index-self.setup_step:index]
            )

            end_time = perf_counter()
            cpu_time = end_time - start_time

            # 更新显示信息
            self.Omega_temp[-1] = omega_new
            final_omega = omega_new
            msg = self.result_record(index-1, omega_new, self.q[index])
            print(f"\r {self.tn[index]:.6f}\\{self.T}, tau = {self.tau[index-1]:.6f}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")
            save_snapshot(index, omega_new)

            self.Omega_temp[:-1] = self.Omega_temp[1:]

        print("")
        if snapshot_mode:
            self.Omega = self.snapshot_Omega
            self.tn_s = self.snapshot_tn
        else:
            self.Omega = np.asarray(final_omega)[None, ...].copy()
        del self.Omega_temp
