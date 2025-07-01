# AlgoQ: Quantum Algorithm Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AlgoQ is a Python library that provides reusable building blocks for quantum algorithms, with support for multiple quantum computing backends including Qiskit and PennyLane.

## Features

- **Multiple Backend Support**: Seamlessly switch between Qiskit and PennyLane
- **Algorithm Implementations**: Pre-built implementations of popular quantum algorithms
- **Easy to Use**: Simple and consistent API across different backends
- **Extensible**: Easy to add new algorithms and backends

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hope-Alemayehu/AlgoQ.git
   cd AlgoQ
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Install optional dependencies for specific backends:
   ```bash
   # For Qiskit support
   pip install qiskit qiskit-optimization
   
   # For PennyLane support
   pip install pennylane
   ```

## Quick Start

Here's a quick example of solving a QUBO problem using AlgoQ:

```python
import numpy as np
from algoq.algorithms.qubo_solvers import QiskitQUBOSolver, PennyLaneQUBOSolver

# Define a simple QUBO problem
Q = np.array([
    [1, -2],
    [0, 1]
])

# Solve using Qiskit backend
qiskit_solver = QiskitQUBOSolver()
result = qiskit_solver.solve(Q)
print("Qiskit solution:", result['solution'])
print("Optimal value:", result['optimal_value'])

# Solve using PennyLane backend
pennylane_solver = PennyLaneQUBOSolver()
result = pennylane_solver.solve(Q)
print("PennyLane solution:", result['solution'])
```

## Documentation

### QUBO Solvers

AlgoQ provides QUBO (Quadratic Unconstrained Binary Optimization) solvers with the following features:

- Support for both matrix and dictionary input formats
- Multiple backend support (Qiskit and PennyLane)
- Customizable optimization parameters

#### Example: Solving a QUBO Problem

```python
from algoq.algorithms.qubo_solvers import QiskitQUBOSolver

# Using dictionary input
Q = {
    (0, 0): 1,
    (0, 1): -2,
    (1, 1): 1
}

solver = QiskitQUBOSolver(reps=2)  # 2 QAOA layers
result = solver.solve(Q, num_vars=2)
print("Solution:", result['solution'])
print("Optimal value:", result['optimal_value'])
```

### Backends

AlgoQ supports multiple quantum computing backends:

- **QiskitBackend**: Uses Qiskit's simulators and real quantum devices
- **PennyLaneBackend**: Uses PennyLane's device interface

#### Example: Using Different Backends

```python
from algoq.backends import QiskitBackend, PennyLaneBackend
from qiskit import Aer

# Using Qiskit with a specific backend
qiskit_backend = QiskitBackend(backend=Aer.get_backend('qasm_simulator'), shots=1000)

# Using PennyLane with default qubit simulator
pennylane_backend = PennyLaneBackend(device='default.qubit', wires=4, shots=1000)
```

## Testing

To run the test suite:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.