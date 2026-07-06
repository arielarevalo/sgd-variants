"""NumPy implementations of stochastic gradient descent and variants.

Companion code for the MA-1006 research project. The optimizer step
functions here are mirrored line for line by the presentation's
playground.js; check.py verifies that parity numerically.
"""

from .optimizers import OPTIMIZERS
from .problems import Ravine, Rosenbrock, SyntheticLS, LogisticRegression
from .runner import run

__all__ = [
    "OPTIMIZERS",
    "Ravine",
    "Rosenbrock",
    "SyntheticLS",
    "LogisticRegression",
    "run",
]
