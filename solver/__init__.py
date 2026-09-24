"""Public API for the modular Fourier research solver."""
from .core import State, History, Trial, Assessment, Scheme, AdaptiveAlgorithm, Result, IntegrationError
from .fourier_ns import FourierNS
from .integrate import integrate

__all__ = ["State", "History", "Trial", "Assessment", "Scheme", "AdaptiveAlgorithm",
           "Result", "IntegrationError", "FourierNS", "integrate"]
