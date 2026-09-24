"""Explicit single-step doubling option; not a fallback for arbitrary multistep schemes."""
from .error_based import EmbeddedErrorControl
from ..core import History, Trial, Assessment


class StepDoubling(EmbeddedErrorControl):
    def validate(self, scheme, state):
        if scheme.history_size != 1 or scheme.order is None or scheme.order <= 0:
            raise ValueError("Step doubling requires a single-step scheme with declared positive order")
        if self.tolerances.keys() - state.fields.keys():
            raise ValueError("Controlled field is absent")

    def attempt(self, model, scheme, history, dt):
        coarse = scheme.step(model, history, dt)
        half = scheme.step(model, history, dt/2)
        mid = History((half.state,), (history.time+dt/2,))
        fine = scheme.step(model, mid, dt/2)
        divisor = 2**scheme.order-1
        # assess() measures high-low; accept the two-half-step state without extrapolation.
        embedded = {name: high-(high-coarse.state.fields[name])/divisor
                    for name, high in fine.state.fields.items()}
        trial = Trial(fine.state, embedded)
        metrics = self.assess(model, history, trial, dt)
        error = float(metrics["error"])
        return Assessment(trial, error <= 1, float(self.controller(error, dt)), metrics)
