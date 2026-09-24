"""Legacy SAV policy: embedded physical error plus named scalar deviation limits."""
import numpy as np
from .error_based import EmbeddedErrorControl, ProportionalController
from ..core import Assessment


class SAVControl(EmbeddedErrorControl):
    def __init__(self, tolerances, scalar_limits, exponent=0.5, safety=0.9, max_growth=2.0):
        """scalar_limits maps names to (target, tolerance); q:(1,rtol_q) reproduces the old criterion."""
        super().__init__(tolerances, ProportionalController(exponent, safety, max_growth))
        self.scalar_limits = dict(scalar_limits)
        if not self.scalar_limits:
            raise ValueError("Choose at least one scalar control indicator")
        for target, tolerance in self.scalar_limits.values():
            if not np.isfinite([target, tolerance]).all() or tolerance <= 0:
                raise ValueError("Scalar targets must be finite and tolerances positive")

    def validate(self, scheme, state):
        super().validate(scheme, state)
        if self.scalar_limits.keys() - state.aux.keys():
            raise ValueError("Controlled auxiliary variable is absent")

    def attempt(self, model, scheme, history, dt):
        trial = scheme.step(model, history, dt, estimate=True)
        metrics = self.assess(model, history, trial, dt)
        error = metrics["error"]
        controller = self.controller
        factor = controller.max_growth if error == 0 else error**(-controller.exponent)
        accept = error <= 1
        for name, (target, tolerance) in self.scalar_limits.items():
            deviation = abs(trial.state.aux[name]-target) + 1e-16
            metrics[f"deviation/{name}"] = float(deviation)
            metrics[f"scaled_aux/{name}"] = float(deviation/tolerance)
            factor = min(factor, tolerance/deviation)
            accept = accept and deviation <= tolerance
        next_dt = dt * min(controller.max_growth, controller.safety*factor)
        return Assessment(trial, bool(accept), float(next_dt), metrics)
