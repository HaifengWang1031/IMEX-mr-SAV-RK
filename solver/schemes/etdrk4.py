"""ETDRK4: independent stage formulas migrated from legacy ETDRK4."""
import numpy as np
from ..core import Scheme, State, Trial


class ETDRK4(Scheme):
    name = "ETDRK4"
    order = 4
    history_size = 1
    variable_step = True
    embedded = False

    def step(self, model, history, dt, *, estimate=False):
        if estimate:
            raise ValueError("No embedded estimate supplied for this format")
        fields = [s.fields["omega"] for s in history.states]
        omega = self._formula(model, fields, history.time, (*history.steps, dt))
        return Trial(State({"omega": omega}))

    def _coefficients(self, model, tau):
        M = 16
        dim = model.L.ndim
        r  = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ),axis = list(range(dim)) )
        Lr = tau*np.expand_dims(-model.L,axis=-1) + r

        phi10 = np.exp(-tau*model.L/2)
        phi11 = np.mean((1-np.exp(Lr/2))/(-Lr),axis=-1).real

        phi30 = np.exp(-tau*model.L)
        phi31 = np.mean((- 4 - Lr + np.exp(Lr)*(4 - 3*Lr + Lr**2))/(Lr)**3,axis=-1).real
        phi32 = np.mean((  2 + Lr + np.exp(Lr)*(-2+Lr))           /(Lr)**3,axis=-1).real
        phi33 = np.mean((- 4 - 3*Lr - Lr**2 + np.exp(Lr)*(4-Lr))  /(Lr)**3,axis=-1).real
        return phi10, phi11, phi30, phi31, phi32, phi33

    def _formula(self, model, Omega_s, tn, tau_s):
        tau = tau_s[-1]
        omega_n = Omega_s[-1]
        phi10, phi11, phi30, phi31, phi32, phi33 = model._cached(("etdrk4", float(tau)), lambda: self._coefficients(model, tau))

        fomega_n = model.ft(omega_n)
        N0 = model.N_hat(omega_n, fomega_n) + model.ft(model.f(model.X[:-1,:-1],model.Y[:-1,:-1],tn))

        fomega_n1 = phi10*fomega_n  + tau*phi11*N0
        omega_n1 = model.ift(fomega_n1).real
        N1 = model.N_hat(omega_n1, fomega_n1) + model.ft(model.f(model.X[:-1,:-1],model.Y[:-1,:-1], tn+tau/2))

        fomega_n2 = phi10*fomega_n  + tau*phi11*N1
        omega_n2 = model.ift(fomega_n2).real
        N2 = model.N_hat(omega_n2, fomega_n2) + model.ft(model.f(model.X[:-1,:-1],model.Y[:-1,:-1], tn+tau/2))

        fomega_n3 = phi10*fomega_n1 + tau*phi11*(2*N2-N0)
        omega_n3 = model.ift(fomega_n3).real
        N3 = model.N_hat(omega_n3, fomega_n3) + model.ft(model.f(model.X[:-1,:-1],model.Y[:-1,:-1], tn+tau))

        fomega_n4 = phi30*fomega_n  + tau*(phi31*N0 + 2*phi32*(N1 + N2) + phi33*N3)
        return model.ift(fomega_n4).real
