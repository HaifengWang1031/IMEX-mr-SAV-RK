"""MrSAVBDF2: independent stage formulas migrated from legacy mr_SAV_BDF2."""
import numpy as np
from ..core import Scheme, State, Trial
from scipy.optimize import newton, brentq


class MrSAVBDF2(Scheme):
    name = "MrSAVBDF2"
    order = 2
    history_size = 2
    variable_step = True
    embedded = False
    auxiliary_defaults = {"q": 1.0}

    def __init__(self, gamma=1000.0):
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be nonnegative and finite")
        self.gamma = float(gamma)

    def step(self, model, history, dt, *, estimate=False):
        fields = [s.fields["omega"] for s in history.states]
        auxiliary = [s.aux["q"] for s in history.states]
        if estimate:
            raise ValueError("No embedded estimate for BDF2")
        omega, q = self._formula(model, fields, auxiliary, history.time, (*history.steps, dt))
        return Trial(State({"omega": omega}, {"q": float(q)}))

    def startup(self, model, history, dt):
        from .etdrk4 import ETDRK4
        if any(value != 1.0 for value in history.state.aux.values()):
            raise ValueError("The default ETDRK4 starter requires initial auxiliary values 1; supply a custom startup")
        trial = ETDRK4().step(model, history, dt)
        return Trial(State(trial.state.fields, history.state.aux))

    def _formula(self, model, Omega_s, q_s, tn, tau_s, fN_n=None, fN_nm=None):
            tau_n  = tau_s[-1]
            tau_nm = tau_s[-2]
            rho = tau_n / tau_nm

            # variable-step BDF2 coefficients
            a0 = (1 + 2*rho) / (1 + rho)
            c1 = (1 + rho)**2 / (1 + 2*rho)     # coeff for omega_n
            c2 = rho**2 / (1 + 2*rho)           # coeff for omega_nm (subtracted)
            dt = tau_n / a0                     # effective step size

            phi_L, phi_ga = (1/(1+dt*model.L), 1/(1+dt*self.gamma))

            omega_n  = Omega_s[-1]
            omega_nm = Omega_s[-2]
            fomega_n  = model.ft(omega_n)
            fomega_nm = model.ft(omega_nm)

            if fN_n is None:
                fN_n = model.N_hat(omega_n, fomega_n)
            if fN_nm is None:
                fN_nm = model.N_hat(omega_nm, fomega_nm)
            pass  # No mutable nonlinear-history cache
            # BDF2 extrapolates N to t_{n+1}, rather than to the ETD midpoint.
            fN_2 = (1 + rho)*fN_n - rho*fN_nm

            q_n  = q_s[-1]
            q_nm = q_s[-2]

            f_n = model.f(model.X[:-1,:-1], model.Y[:-1,:-1], tn + tau_n)

            f_fn = model.ft(f_n)
            fomega_base = phi_L*(c1*fomega_n - c2*fomega_nm + dt*f_fn)

            LHS = 1 + dt**2*phi_ga*model.inner_product_ft(fN_2, phi_L*fN_2)
            RHS = phi_ga*(c1*q_n - c2*q_nm) \
                + dt*phi_ga*self.gamma \
                - dt*phi_ga*model.inner_product_ft(fN_2, fomega_base)

            q_n1 = RHS / LHS
            fomega_n1 = fomega_base + dt*phi_L*q_n1*fN_2

            return model.ift(fomega_n1).real, q_n1
