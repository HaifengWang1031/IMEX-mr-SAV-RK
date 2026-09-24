"""First-order IMEX Euler, preserving the legacy IMEX formula."""
from ..core import Scheme, State, Trial

class IMEXEuler(Scheme):
    name='IMEXEuler'
    order=1
    history_size=1
    variable_step=True
    embedded=False

    def step(self, model, history, dt, *, estimate=False):
        if estimate:raise ValueError('No embedded estimate')
        w=history.state.fields['omega'];fw=model.ft(w)
        f=model.f(model.X[:-1,:-1],model.Y[:-1,:-1],history.time)
        out=(fw+dt*(model.N_hat(w,fw)+model.ft(f)))/(1+dt*model.L)
        return Trial(State({'omega':model.ift(out).real}))
