"""SDIRK2MrSAV: independent stage formulas migrated from legacy SDIRK2_mr_SAV."""
import numpy as np
from ..core import Scheme, State, Trial
from scipy.optimize import newton, brentq


class SDIRK2MrSAV(Scheme):
    name = "SDIRK2MrSAV"
    order = 2
    history_size = 1
    variable_step = True
    embedded = True
    auxiliary_defaults = {"q": 1.0}

    def __init__(self, gamma=1000.0):
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be nonnegative and finite")
        self.gamma = float(gamma)

    def step(self, model, history, dt, *, estimate=False):
        fields = [s.fields["omega"] for s in history.states]
        auxiliary = [s.aux["q"] for s in history.states]
        value = self._formula(model, fields, auxiliary, history.time, (*history.steps, dt), adaptive=estimate)
        if estimate:
            omega, embedded, q = value
            return Trial(State({"omega": omega}, {"q": float(q)}), {"omega": embedded})
        omega, q = value
        return Trial(State({"omega": omega}, {"q": float(q)}))

    def _formula(self, model, Omega_s, q_s, tn, tau_s, adaptive=False, fN_0=None):
        tau_n = tau_s[-1]

        eta = 1 - np.sqrt(2)/2
        delta = 1 - 1/(2*eta)

        omega_n = Omega_s[-1]
        fomega_n = model.ft(omega_n)
        r_n = 1 - q_s[-1]

        phi_L = model._imex_phi(eta*tau_n)
        scalar_mass = 1 + eta*tau_n*self.gamma

        # Stage 1
        if fN_0 is None:
            fN_0 = model.N_hat(omega_n, fomega_n)
        f_0 = model.f(model.X[:-1,:-1], model.Y[:-1,:-1], tn)
        ff_0 = model.ft(f_0)

        fN_hat_1 = eta*fN_0
        ff_hat_1 = eta*ff_0

        fomega_11 = phi_L*(fomega_n + tau_n*ff_hat_1); fomega_11[0,0] = 0.+0j
        fomega_12 = phi_L*fN_hat_1; fomega_12[0,0] = 0.+0j

        A_1 = model.inner_product_ft(fN_hat_1, fomega_11)
        B_1 = model.inner_product_ft(fN_hat_1, fomega_12)

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
        omega_1 = model.ift(fomega_1).real

        # Stage 2
        a_21 = 1 - 2*eta
        ahat_20 = delta - eta
        ahat_21 = 1 - delta

        fN_1 = model.N_hat(omega_1, fomega_1)
        f_1 = model.f(model.X[:-1,:-1], model.Y[:-1,:-1], tn + eta*tau_n)

        fN_hat_2 = ahat_20*fN_0 + ahat_21*fN_1
        ff_hat_2 = ahat_20*ff_0 + ahat_21*model.ft(f_1)

        fomega_21 = phi_L*(fomega_1 - tau_n*a_21*model.L*fomega_1 + tau_n*ff_hat_2); fomega_21[0,0] = 0.+0j
        fomega_22 = phi_L*fN_hat_2; fomega_22[0,0] = 0.+0j

        A_2 = model.inner_product_ft(fN_hat_2, fomega_21)
        B_2 = model.inner_product_ft(fN_hat_2, fomega_22)
        R_2 = (1 - tau_n*a_21*self.gamma)*r_1

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
        Omega_2 = model.ift(fomega_2).real
        q_2 = 1 - r_2
        if not adaptive:
            return Omega_2, q_2

        Omega_1 = omega_n + (omega_1 - omega_n)/eta
        return Omega_2, Omega_1, q_2
