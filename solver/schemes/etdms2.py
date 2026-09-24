"""ETDMS2: independent stage formulas migrated from legacy ETDMS2."""
import numpy as np
from ..core import Scheme, State, Trial


class ETDMS2(Scheme):
    name = "ETDMS2"
    order = 2
    history_size = 2
    variable_step = True
    embedded = False

    def step(self, model, history, dt, *, estimate=False):
        if estimate:
            raise ValueError("No embedded estimate supplied for this format")
        fields = [s.fields["omega"] for s in history.states]
        omega = self._formula(model, fields, history.time, (*history.steps, dt))
        return Trial(State({"omega": omega}))

    def startup(self, model, history, dt):
        from .etdrk4 import ETDRK4
        if any(value != 1.0 for value in history.state.aux.values()):
            raise ValueError("The default ETDRK4 starter requires initial auxiliary values 1; supply a custom startup")
        trial = ETDRK4().step(model, history, dt)
        return Trial(State(trial.state.fields, history.state.aux))

    def _formula(self, model, Omega_s, tn, tau_s):
        tau_n = tau_s[-1]
        tau_nm = tau_s[-2]

        phi0_L, phi1_L = model._etd_phi(tau_n)

        omega_n   = Omega_s[-1]
        fomega_n  = model.ft(omega_n)
        omega_nm  = Omega_s[-2]

        fN_n = model.N_hat(omega_n, fomega_n)
        fN_nm = model.N_hat(omega_nm)
        f_N12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm


        fn = model.f(model.X[:-1,:-1],model.Y[:-1,:-1],tn+tau_n/2)
        f_fn = model.ft(fn)

        fomega_n1 = phi0_L*fomega_n + tau_n*phi1_L*(f_N12 + f_fn); fomega_n1[0,0] = 0.+0j
        return model.ift(fomega_n1).real
