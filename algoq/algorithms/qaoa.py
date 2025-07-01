"""QAOA (Quantum Approximate Optimization Algorithm) implementation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..backends import QuantumBackend


class QAOA(ABC):
    """Base class for QAOA (Quantum Approximate Optimization Algorithm).
    
    This class provides a common interface for running the QAOA algorithm
    using different quantum backends.
    """
    
    def __init__(
        self,
        backend: Optional[QuantumBackend] = None,
        num_layers: int = 1,
        optimizer: str = 'COBYLA',
        **kwargs
    ):
        """Initialize the QAOA solver.
        
        Args:
            backend: The quantum backend to use. If None, a default backend will be used.
            num_layers: Number of QAOA layers (p parameter).
            optimizer: Classical optimizer to use for parameter optimization.
            **kwargs: Additional QAOA parameters.
        """
        self.backend = backend
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.params = kwargs
        
    @abstractmethod
    def minimize(
        self,
        cost_hamiltonian: Any,
        initial_state: Optional[Any] = None,
        initial_params: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """Minimize the given cost Hamiltonian using QAOA.
        
        Args:
            cost_hamiltonian: The Hamiltonian whose ground state we want to find.
            initial_state: Optional initial state for the QAOA circuit.
            initial_params: Optional initial parameters for the QAOA circuit.
            **kwargs: Additional parameters for the QAOA algorithm.
            
        Returns:
            A dictionary containing the optimization result.
        """
        pass
    
    @abstractmethod
    def get_optimal_params(self) -> np.ndarray:
        """Get the optimal parameters found during optimization.
        
        Returns:
            The optimal parameters as a numpy array.
        """
        pass
    
    @abstractmethod
    def get_optimal_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the optimal state and its energy.
        
        Returns:
            A tuple (state, energy) where state is the optimal state vector
            and energy is its corresponding energy.
        """
        pass
    
    def _validate_hamiltonian(self, hamiltonian: Any) -> None:
        """Validate the input Hamiltonian.
        
        Args:
            hamiltonian: The Hamiltonian to validate.
            
        Raises:
            ValueError: If the Hamiltonian is not valid.
        """
        if hamiltonian is None:
            raise ValueError("Hamiltonian cannot be None")
