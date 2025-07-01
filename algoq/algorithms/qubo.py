"""QUBO (Quadratic Unconstrained Binary Optimization) solver implementation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
from ..backends import QuantumBackend


class QUBOSolver(ABC):
    """Base class for QUBO solvers.
    
    This class provides a common interface for solving QUBO problems
    using different quantum backends.
    """
    
    def __init__(self, backend: Optional[QuantumBackend] = None):
        """Initialize the QUBO solver.
        
        Args:
            backend: The quantum backend to use. If None, a default backend will be used.
        """
        self.backend = backend
        
    @abstractmethod
    def solve(
        self,
        Q: Union[np.ndarray, Dict[tuple, float]],
        num_vars: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Solve a QUBO problem.
        
        Args:
            Q: The QUBO matrix as a 2D numpy array or a dictionary of (i,j) -> coefficient.
            num_vars: Number of variables. Required if Q is a dictionary.
            **kwargs: Additional solver-specific parameters.
            
        Returns:
            A dictionary containing the solution and additional information.
        """
        pass
    
    def _validate_qubo_matrix(self, Q: np.ndarray) -> None:
        """Validate the QUBO matrix.
        
        Args:
            Q: The QUBO matrix to validate.
            
        Raises:
            ValueError: If the matrix is not square.
            ValueError: If the matrix is not symmetric.
        """
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("QUBO matrix must be square")
            
        if not np.allclose(Q, Q.T):
            raise ValueError("QUBO matrix must be symmetric")
