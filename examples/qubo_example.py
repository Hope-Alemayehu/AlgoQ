"""
Example: Solving a QUBO problem with AlgoQ

This example demonstrates how to use the AlgoQ library to solve a simple
Quadratic Unconstrained Binary Optimization (QUBO) problem using both
Qiskit and PennyLane backends.
"""

import numpy as np
from algoq.algorithms.qubo_solvers import QiskitQUBOSolver, PennyLaneQUBOSolver

def main():
    # Define a simple QUBO problem: min x1 + x2 - 2x1x2
    # The optimal solutions are x1=1, x2=0 or x1=0, x2=1 with value 0
    Q = np.array([
        [1, -2],
        [0, 1]
    ])
    
    print("QUBO Matrix:")
    print(Q)
    
    # Solve using Qiskit backend
    print("\nSolving with Qiskit backend...")
    qiskit_solver = QiskitQUBOSolver()
    qiskit_result = qiskit_solver.solve(Q)
    print("Qiskit solution:", qiskit_result['solution'])
    print("Optimal value:", qiskit_result['optimal_value'])
    
    # Solve using PennyLane backend
    print("\nSolving with PennyLane backend...")
    pennylane_solver = PennyLaneQUBOSolver()
    pennylane_result = pennylane_solver.solve(Q)
    print("PennyLane solution:", pennylane_result['solution'])
    print("Optimal value:", pennylane_result['optimal_value'])
    
    # Example with dictionary input
    print("\nSolving with dictionary input...")
    Q_dict = {
        (0, 0): 1,
        (0, 1): -2,
        (1, 1): 1
    }
    dict_result = qiskit_solver.solve(Q_dict, num_vars=2)
    print("Solution with dictionary input:", dict_result['solution'])

if __name__ == "__main__":
    main()
