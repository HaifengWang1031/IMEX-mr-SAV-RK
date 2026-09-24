"""Embedded-error algorithm; normalization and controller are replaceable callables."""
import numpy as np
from ..core import AdaptiveAlgorithm, Assessment


class ProportionalController:
    def __init__(self, exponent=0.5, safety=0.9, max_growth=2.0):
        if not np.isfinite([exponent, safety, max_growth]).all() or exponent <= 0 or not 0 < safety < 1 or max_growth < 1:
            raise ValueError("Invalid proportional-controller settings")
        self.exponent, self.safety, self.max_growth = exponent, safety, max_growth

    def __call__(self, error, dt):
        factor = self.max_growth if error == 0 else error**(-self.exponent)
        return dt * min(self.max_growth, self.safety*factor)

    def reset(self):
        pass

    def on_accept(self, metrics, forced):
        pass

    def on_reject(self, metrics):
        pass


class EmbeddedErrorControl(AdaptiveAlgorithm):
    def __init__(self, tolerances, controller=None, indicator=None):
        """tolerances maps physical field names to (atol, rtol).

        indicator(model, history, trial, dt) may replace the default normalization;
        it returns numeric metrics including a dimensionless 'error' (pass <=1).
        Auxiliary drift is not an auxiliary local-error estimate.
        """
        self.tolerances = dict(tolerances)
        if not self.tolerances:
            raise ValueError("Choose at least one controlled field")
        for pair in self.tolerances.values():
            if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] <= 0 or pair[1] < 0:
                raise ValueError("Require finite atol>0 and rtol>=0")
        self.controller = controller or ProportionalController()
        self.indicator = indicator

    def validate(self, scheme, state):
        if self.tolerances.keys() - state.fields.keys():
            raise ValueError("Controlled field is absent from the state")
        if not scheme.embedded and self.indicator is None:
            raise ValueError("This scheme has no supplied embedded estimate; choose a compatible algorithm")

    def reset(self):
        if hasattr(self.controller, "reset"):
            self.controller.reset()

    def assess(self, model, history, trial, dt):
        if self.indicator is not None:
            return self.indicator(model, history, trial, dt)
        if trial.embedded_fields is None:
            raise ValueError("No embedded fields supplied")
        metrics = {}
        for name, (atol, rtol) in self.tolerances.items():
            high, low = trial.state.fields[name], trial.embedded_fields[name]
            if high.shape != np.shape(low):
                raise ValueError("Embedded estimate shape mismatch")
            norm = lambda array: model.norm(name, array)
            difference, magnitude = norm(high-low), norm(high)
            metrics[f"relative/{name}"] = difference/max(magnitude, atol)
            metrics[f"scaled/{name}"] = difference/(atol + rtol*magnitude)
        metrics["error"] = max(metrics[f"scaled/{name}"] for name in self.tolerances)
        return metrics

    def attempt(self, model, scheme, history, dt):
        trial = scheme.step(model, history, dt, estimate=self.indicator is None)
        metrics = self.assess(model, history, trial, dt)
        error = float(metrics["error"])
        return Assessment(trial, error <= 1, float(self.controller(error, dt)), metrics)

    def on_accept(self, assessment, forced):
        if hasattr(self.controller, "on_accept"):
            self.controller.on_accept(assessment.metrics, forced)

    def on_reject(self, assessment):
        if hasattr(self.controller, "on_reject"):
            self.controller.on_reject(assessment.metrics)
