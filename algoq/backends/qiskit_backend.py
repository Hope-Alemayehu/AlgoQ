"""Qiskit backend implementation for AlgoQ."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp
from qiskit.providers import Backend

from . import QuantumBackend


class QiskitBackend(QuantumBackend):
    """Qiskit implementation of the QuantumBackend interface."""
    
    def __init__(self, backend: Optional[Backend] = None, shots: int = 1024):
        """Initialize the Qiskit backend.
        
        Args:
            backend: Qiskit backend to use. If None, uses Aer's statevector simulator.
            shots: Number of shots for measurement. Ignored for statevector simulation.
        """
        self._backend = backend or Aer.get_backend('statevector_simulator')
        self._shots = shots
        self._is_statevector = 'statevector' in str(self._backend).lower()
    
    @property
    def name(self) -> str:
        """Return the name of the backend."""
        return f"qiskit_{self._backend.name()}"
    
    def run(self, circuit: QuantumCircuit, **kwargs) -> Dict[str, Any]:
        """Execute a quantum circuit.
        
        Args:
            circuit: The quantum circuit to execute.
            **kwargs: Additional execution parameters.
            
        Returns:
            A dictionary containing the execution results.
        """
        shots = kwargs.get('shots', self._shots)
        
        if self._is_statevector:
            result = execute(circuit, self._backend).result()
            statevector = result.get_statevector()
            return {
                'statevector': statevector,
                'counts': self._statevector_to_counts(statevector, shots)
            }
        else:
            result = execute(circuit, self._backend, shots=shots).result()
            return {
                'counts': result.get_counts(),
                'statevector': None
            }
    
    def _statevector_to_counts(self, statevector: Statevector, shots: int) -> Dict[str, int]:
        """Convert a statevector to measurement counts.
        
        Args:
            statevector: The statevector to sample from.
            shots: Number of shots to sample.
            
        Returns:
            A dictionary of bitstring counts.
        """
        probs = np.abs(statevector.data) ** 2
        possible_outcomes = [format(i, f'0{len(statevector.dims())}b') for i in range(len(probs))]
        counts = np.random.multinomial(shots, probs)
        return {outcome: count for outcome, count in zip(possible_outcomes, counts) if count > 0}
    
    def get_operator(self, pauli_str: str, coeff: float = 1.0) -> PauliSumOp:
        """Convert a Pauli string to a Qiskit operator.
        
        Args:
            pauli_str: String of I, X, Y, Z operators.
            coeff: Coefficient for the operator.
            
        Returns:
            A PauliSumOp representing the operator.
        """
        return PauliSumOp.from_list([(pauli_str, coeff)])
    
    def create_parameter(self, name: str) -> Parameter:
        """Create a parameter for parameterized circuits.
        
        Args:
            name: Name of the parameter.
            
        Returns:
            A Qiskit Parameter object.
        """
        return Parameter(name)
    
    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        parameters: Dict[Parameter, float]
    ) -> QuantumCircuit:
        """Bind parameters to a circuit.
        
        Args:
            circuit: The circuit to bind parameters to.
            parameters: Dictionary of Parameter to value mappings.
            
        Returns:
            A new circuit with parameters bound to values.
        """
        return circuit.assign_parameters(parameters)
