"""Independent Fourier pseudospectral NS model; spatial formulas copied from the legacy baseline.

L is positive -nu*Delta, so omega_t = -L*omega + N(omega) + f.
This module owns no time scheme, auxiliary variable or integration history.
"""
from collections import OrderedDict
import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft


class FourierNS:
    def initial_state(self, omega):
        """Build the zero-mean interior vorticity state, matching legacy initialization."""
        from .core import State
        omega = np.asarray(omega)
        if omega.shape != (self.Ny, self.Nx):
            raise ValueError("Supply the interior vorticity array without periodic padding")
        return State({"omega": omega-np.mean(omega)})

    def __init__(self, nu, shape=(32, 32), domain=(0, 0, 2*np.pi, 2*np.pi), forcing=None, threads=1):
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError("nu must be positive and finite")
        if len(shape) != 2 or any(isinstance(n, bool) or int(n) != n or n < 4 or n % 2 for n in shape):
            raise ValueError("Two even grid dimensions >=4 are required")
        if len(domain) != 4 or not np.isfinite(domain).all() or domain[2] <= domain[0] or domain[3] <= domain[1]:
            raise ValueError("Invalid rectangular domain")
        if int(threads) != threads or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.nu = float(nu)
        self.threads = int(threads)
        pyfftw.config.NUM_THREADS = self.threads
        pyfftw.interfaces.cache.enable()
        s_domain, discrete_num = domain, tuple(map(int, shape))
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

        self.f = forcing if forcing is not None else lambda X, Y, t: np.zeros_like(X)
        self._coefficients = OrderedDict()

    def _cached(self, key, factory):
        # Bounded coefficient-only cache: safe across rejected attempts.
        if key not in self._coefficients:
            self._coefficients[key] = factory()
            if len(self._coefficients) > 8:
                self._coefficients.popitem(last=False)
        return self._coefficients[key]

    def diagnostics(self, state, t):
        omega = state.fields["omega"]
        energy, enstrophy, palinstrophy = self.vorticity_energy(omega)
        return {"energy": float(energy), "enstrophy": float(enstrophy),
                "palinstrophy": float(palinstrophy),
                "energy_rate": float(self.energy_rate(omega, t, enstrophy)),
                "enstrophy_rate": float(self.enstrophy_rate(omega, t)),
                "max_vorticity": float(np.max(omega))}

    def norm(self, name, field):
        return float(np.sqrt(max(self.inner_product(field, field), 0.0)))

    def ft(self,u):
        return fft.fft2(u, threads=self.threads)

    def ift(self,u):
        return fft.ifft2(u, threads=self.threads)

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

    def _imex_phi(self, tau):
        return self._cached(("imex_phi", float(tau)), lambda: 1/(1 + tau*self.L))
