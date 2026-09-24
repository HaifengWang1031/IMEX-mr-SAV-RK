"""ETDMrSAVMS2B: independent stage formulas migrated from legacy ETD_mrSAV_MS2_b."""
import numpy as np
from ..core import Scheme, State, Trial
from scipy.optimize import newton, brentq


class ETDMrSAVMS2B(Scheme):
    name = "ETDMrSAVMS2B"
    order = 2
    history_size = 2
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

    def startup(self, model, history, dt):
        from .etdrk4 import ETDRK4
        if any(value != 1.0 for value in history.state.aux.values()):
            raise ValueError("The default ETDRK4 starter requires initial auxiliary values 1; supply a custom startup")
        trial = ETDRK4().step(model, history, dt)
        return Trial(State(trial.state.fields, history.state.aux))

    def _formula(self, model, Omega_s, q_s, tn, tau_s, fN_n=None, fN_nm=None, adaptive=False):
        tau_n = tau_s[-1]
        tau_nm = tau_s[-2]

        phi0_L, phi1_L = model._etd_phi(tau_n)
        phi0_ga = np.exp(-tau_n*self.gamma)

        omega_n   = Omega_s[-1]
        fomega_n  = model.ft(omega_n)
        omega_nm  = Omega_s[-2]

        if fN_n is None:
            fN_n = model.N_hat(omega_n, fomega_n)
        if fN_nm is None:
            fN_nm = model.N_hat(omega_nm)
        f_N12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = model.f(model.X[:-1,:-1],model.Y[:-1,:-1],tn+tau_n/2)
        f_fn = model.ft(fn)

        A = tau_n*model.inner_product_ft(phi1_L * f_N12, phi0_L*fomega_n + tau_n*phi1_L*f_fn)
        B = tau_n**2*model.inner_product_ft(phi1_L * f_N12, phi1_L*f_N12)
        C =  phi0_ga*p_n

        Tgam = 0.1
        f = lambda p: p + (1-p)*A*Tgam + (p**3 - p**2 - p + 1)*B*Tgam - C
        try:
            p_n1 = newton(f, 0.)
        except RuntimeError:
            lo, hi = -10., 10.
            if f(lo) * f(hi) > 0:
                lo, hi = -100., 100.
            p_n1 = brentq(f, lo, hi)

        fomega_2 = phi0_L*fomega_n + tau_n*phi1_L*((1 - p_n1**2)*f_N12 + f_fn); fomega_2[0,0] = 0.+0j
        Omega_2 = model.ift(fomega_2).real
        q_2 = p_n1 + 1
        if not adaptive:
            return Omega_2, q_2

        fomega_1 = phi0_L*fomega_n + tau_n*phi1_L*((1 + p_n1)*f_N12 + f_fn); fomega_1[0,0] = 0.+0j
        return Omega_2, model.ift(fomega_1).real, q_2
