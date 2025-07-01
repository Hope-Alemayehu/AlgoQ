"""PennyLane backend implementation for AlgoQ."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.operation import Tensor

from . import QuantumBackend


class PennyLaneBackend(QuantumBackend):
    """PennyLane implementation of the QuantumBackend interface."""
    
    def __init__(self, device: str = 'default.qubit', wires: int = 1, shots: int = 1024):
        """Initialize the PennyLane backend.
        
        Args:
            device: Name of the PennyLane device to use.
            wires: Number of qubits to use.
            shots: Number of shots for measurement.
        """
        self._wires = wires
        self._shots = shots
        self._device = qml.device(device, wires=wires, shots=shots)
        self._is_statevector = 'statevector' in device
    
    @property
    def name(self) -> str:
        """Return the name of the backend."""
        return f"pennylane_{self._device.short_name}"
    
    def run(self, circuit: callable, **kwargs) -> Dict[str, Any]:
        """Execute a quantum circuit.
        
        Args:
            circuit: A function that defines the quantum circuit.
            **kwargs: Additional execution parameters.
            
        Returns:
            A dictionary containing the execution results.
        """
        shots = kwargs.get('shots', self._shots)
        
        @qml.qnode(self._device, interface="autograd")
        def qnode():
            circuit()
            if self._is_statevector:
                return qml.state()
            return qml.counts()
        
        result = qnode()
        
        if self._is_statevector:
            return {
                'statevector': result,
                'counts': self._statevector_to_counts(result, shots)
            }
        else:
            return {
                'counts': result,
                'statevector': None
            }
    
    def _statevector_to_counts(self, statevector: np.ndarray, shots: int) -> Dict[str, int]:
        """Convert a statevector to measurement counts.
        
        Args:
            statevector: The statevector to sample from.
            shots: Number of shots to sample.
            
        Returns:
            A dictionary of bitstring counts.
        """
        probs = np.abs(statevector) ** 2
        n_qubits = int(np.log2(len(statevector)))
        possible_outcomes = [format(i, f'0{n_qubits}b') for i in range(len(probs))]
        counts = np.random.multinomial(shots, probs)
        return {outcome: int(count) for outcome, count in zip(possible_outcomes, counts) if count > 0}
    
    def get_operator(self, pauli_str: str, coeff: float = 1.0) -> Tensor:
        """Convert a Pauli string to a PennyLane operator.
        
        Args:
            pauli_str: String of I, X, Y, Z operators.
            coeff: Coefficient for the operator.
            
        Returns:
            A PennyLane Tensor representing the operator.
        """
        ops = []
        for i, pauli in enumerate(pauli_str):
            if pauli == 'I':
                continue
            elif pauli == 'X':
                ops.append(qml.PauliX(i))
            elif pauli == 'Y':
                ops.append(qml.PauliY(i))
            elif pauli == 'Z':
                ops.append(qml.PauliZ(i))
            else:
                raise ValueError(f"Invalid Pauli operator: {pauli}")
        
        if not ops:
            return coeff * qml.Identity(0)
        return coeff * qml.operation.Tensor(*ops)
    
    def create_parameter(self, name: str) -> pnp.tensor:
        """Create a parameter for parameterized circuits.
        
        Args:
            name: Name of the parameter (not used in PennyLane, but kept for interface compatibility).
            
        Returns:
            A PennyLane numpy tensor that can be used as a parameter.
        """
        return pnp.array(0.0, requires_grad=True)
    
    def bind_parameters(
        self,
        circuit: callable,
        parameters: Dict[str, float]
    ) -> callable:
        """Bind parameters to a circuit.
        
        Args:
            circuit: The circuit function to bind parameters to.
            parameters: Dictionary of parameter names to values.
            
        Returns:
            A new circuit function with parameters bound to values.
        
        Note:
            In PennyLane, parameters are typically handled through function arguments
            and automatic differentiation, so this is mostly a passthrough.
        """
        return circuit
