"""Legacy ETD: linear diffusion plus forcing ONLY; does not discretize full NS advection."""
from ..core import Scheme, State, Trial

class LegacyLinearETD(Scheme):
    name='LegacyLinearETD'
    order=1
    history_size=1
    variable_step=True
    embedded=False

    def step(self, model, history, dt, *, estimate=False):
        if estimate:raise ValueError('No embedded estimate')
        p0,p1=model._etd_phi(dt)
        f=model.f(model.X[:-1,:-1],model.Y[:-1,:-1],history.time)
        out=p0*model.ft(history.state.fields['omega'])+dt*p1*model.ft(f)
        out[0,0]=0
        return Trial(State({'omega':model.ift(out).real}))
