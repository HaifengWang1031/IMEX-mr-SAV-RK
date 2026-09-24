"""ETDMrSAVMS2L: independent stage formulas migrated from legacy ETD_mrSAV_MS2_L."""
import numpy as np
from ..core import Scheme, State, Trial
from scipy.optimize import newton, brentq


class ETDMrSAVMS2L(Scheme):
    name = "ETDMrSAVMS2L"
    order = 2
    history_size = 3
    variable_step = True
    embedded = True
    auxiliary_defaults = {"q": 1.0}

    def __init__(self, gamma=1000.0, talbot_nodes=10):
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be nonnegative and finite")
        self.gamma = float(gamma)
        if int(talbot_nodes) != talbot_nodes or talbot_nodes < 1:
            raise ValueError("talbot_nodes must be positive integer")
        self.talbot_nodes = int(talbot_nodes)

    def step(self, model, history, dt, *, estimate=False):
        fields = [s.fields["omega"] for s in history.states]
        auxiliary = [s.aux["q"] for s in history.states]
        value = self._formula(model, fields, auxiliary, history.time, (*history.steps, dt), adaptive=estimate)
        if estimate:
            omega, embedded, q = value
            return Trial(State({"omega": omega}, {"q": float(q)}), {"omega": embedded})
        omega, q = value
        return Trial(State({"omega": omega}, {"q": float(q)}))

    def startup(self, model, history, dt):
        from .etdrk4 import ETDRK4
        if any(value != 1.0 for value in history.state.aux.values()):
            raise ValueError("The default ETDRK4 starter requires initial auxiliary values 1; supply a custom startup")
        trial = ETDRK4().step(model, history, dt)
        return Trial(State(trial.state.fields, history.state.aux))

    def _formula(self, model, Omega_s, q_s, tn, tau_s,
                            fN_n=None, fN_nm=None, adaptive=False):
                tau_n =  tau_s[-1]
                tau_nm = tau_s[-2]

                phi0_L, phi1_L = model._etd_phi(tau_n)

                omega_n  = Omega_s[-1]
                omega_nm = Omega_s[-2]
                fomega_n = model.ft(omega_n)

                if fN_n is None:
                    fN_n = model.N_hat(omega_n, fomega_n)
                if fN_nm is None:
                    fN_nm = model.N_hat(omega_nm)
                pass  # Rejected trials must not mutate nonlinear history.
                fN_12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

                q_n = q_s[-1]

                q_cache = {}

                def force_hat(tau):
                    return model.ft(model.f(model.X[:-1, :-1], model.Y[:-1, :-1], tn + tau))

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

                    return model._cached(key, factory)

                def talbot_inverse(f_tau_hat, t: float, N=8):
                    """
                    Use Talbot Contour Method to calculate the inverse Laplace transform.
                    The forcing transform is cached outside the Talbot nodes because it is
                    independent of the Laplace parameter s.
                    """
                    z, dz_dtheta = talbot_nodes(t, N)

                    inv_key = ("talbot_inv_zL", float(t), int(N))

                    def inv_factory():
                        return 1 / (z[:, np.newaxis, np.newaxis] + model.L[np.newaxis, :, :])

                    inv_zL_values = model._cached(inv_key, inv_factory)

                    Fz = np.empty(N, dtype=np.complex128)
                    inner_scale = model.h / (model.Nx*model.Ny)
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
                            self.gamma/zi
                            - forcing_inner
                            + q_n) / (zi + self.gamma + denom_inner)
                    return np.sum(np.exp(z * t)*dz_dtheta*Fz/N).imag

                def q(tau):
                    key = float(tau)
                    if key not in q_cache:
                        q_cache[key] = talbot_inverse(force_hat(tau), tau, self.talbot_nodes)
                    return q_cache[key]

                q_hn = q(tau_n/2)
                q_nn = q(tau_n)

                p = np.array([0, 1/2, 1], dtype=np.float64)
                w = np.array([1/6, 2/3, 1/6], dtype=np.float64)
                exp_key = ("ms2l_simpson_exp", float(tau_n))

                def exp_factory():
                    return np.exp(-tau_n*(1-p)[:, np.newaxis, np.newaxis]*model.L[np.newaxis, :, :])

                exp_factors = model._cached(exp_key, exp_factory)
                force_values = np.stack([force_hat(0.0), force_hat(tau_n/2), force_hat(tau_n)], axis=0)
                q_values = np.array([q_n, q_hn, q_nn], dtype=np.float64)[:, np.newaxis, np.newaxis]

                fomega_n1 = phi0_L*fomega_n + tau_n*np.sum(
                    w[:, np.newaxis, np.newaxis] * exp_factors * (force_values + q_values*fN_12),
                    axis=0
                )

                omega_n1 = model.ift(fomega_n1).real
                z = self.gamma*tau_n
                exp_ga = np.exp(-z)
                if abs(z) < 1e-12:
                    phi1_ga = 1 - z/2 + z*z/6
                else:
                    phi1_ga = (1 - exp_ga)/z
                fomega_mid = 0.5*(fomega_n + fomega_n1)
                q_nn = exp_ga*q_n + (1 - exp_ga) + tau_n*phi1_ga*model.inner_product_ft(fN_12, fomega_mid)

                if not adaptive:
                    return omega_n1, q_nn

                fomega_n2 = phi0_L*fomega_n + tau_n*phi1_L*(force_values[-1] + q_nn*fN_12)
                return omega_n1, model.ift(fomega_n2).real, q_nn
