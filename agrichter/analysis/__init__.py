"""Analysis engine for AgRichter calculations."""

from .envelope_v2 import HPEnvelopeCalculatorV2, EnvelopeData
from .envelope_builder import build_envelope
from .event_calculator import EventCalculator
from .convergence_validator import ConvergenceValidator, ValidationResult, MathematicalProperties
from .envelope_diagnostics import EnvelopeDiagnostics

__all__ = [
    "HPEnvelopeCalculatorV2",
    "EnvelopeData",
    "build_envelope",
    "EventCalculator",
    "ConvergenceValidator",
    "ValidationResult",
    "MathematicalProperties",
    "EnvelopeDiagnostics",
]
