"""SDIRK2: independent stage formulas migrated from legacy IMEX_RK2."""
import numpy as np
from ..core import Scheme, State, Trial


class SDIRK2(Scheme):
    name = "SDIRK2"
    order = 2
    history_size = 1
    variable_step = True
    embedded = False

    def step(self, model, history, dt, *, estimate=False):
        if estimate:
            raise ValueError("No embedded estimate supplied for this format")
        fields = [s.fields["omega"] for s in history.states]
        omega = self._formula(model, fields, history.time, (*history.steps, dt))
        return Trial(State({"omega": omega}))

    def _formula(self, model, Omega_s, tn, tau_s):
        tau_n = tau_s[-1]

        eta = 1 - np.sqrt(2)/2
        delta = 1 - 1/(2*eta)

        omega_n = Omega_s[-1]
        fomega_n = model.ft(omega_n)
        phi_L = model._imex_phi(eta*tau_n)

        # Stage 1
        fN_0 = model.N_hat(omega_n, fomega_n)
        f_0 = model.f(model.X[:-1,:-1], model.Y[:-1,:-1], tn)
        ff_0 = model.ft(f_0)

        fN_hat_1 = eta*fN_0
        ff_hat_1 = eta*ff_0

        fomega_11 = phi_L*(fomega_n + tau_n*ff_hat_1); fomega_11[0,0] = 0.+0j
        fomega_12 = phi_L*fN_hat_1; fomega_12[0,0] = 0.+0j

        fomega_1 = fomega_11 + tau_n*fomega_12; fomega_1[0,0] = 0.+0j
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

        fomega_2 = fomega_21 + tau_n*fomega_22; fomega_2[0,0] = 0.+0j
        return model.ift(fomega_2).real
