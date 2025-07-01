"""AlgoQ: Reusable building blocks for quantum algorithms.

This package provides implementations of common quantum algorithms with support for
multiple quantum computing backends, including Qiskit and PennyLane.
"""

__version__ = "0.1.0"

# Import core modules
from algoq.core import *  # noqa: F403

# Import algorithm implementations
from algoq.algorithms import *  # noqa: F403

# Import backends
from algoq.backends import *  # noqa: F403

__all__ = ["__version__"]
# Add other public API exports here
