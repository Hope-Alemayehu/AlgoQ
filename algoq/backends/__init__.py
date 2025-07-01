"""Backend interfaces for quantum computing frameworks.

This module provides a unified interface to different quantum computing backends,
including Qiskit and PennyLane.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class QuantumBackend(ABC):
    """Abstract base class for quantum backends."""
    
    @abstractmethod
    def run(self, circuit: Any, **kwargs) -> Any:
        """Execute a quantum circuit.
        
        Args:
            circuit: The quantum circuit to execute
            **kwargs: Additional backend-specific arguments
            
        Returns:
            The execution result
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend."""
        pass

# Import specific backend implementations
from algoq.backends.qiskit_backend import QiskitBackend  # noqa: F401
from algoq.backends.pennylane_backend import PennyLaneBackend  # noqa: F401

__all__ = [
    "QuantumBackend",
    "QiskitBackend",
    "PennyLaneBackend",
]
