"""Tests for QUBO solvers."""

import pytest
import numpy as np
from algoq.algorithms.qubo_solvers import QiskitQUBOSolver, PennyLaneQUBOSolver


class TestQUBOSolvers:
    """Test cases for QUBO solvers."""
    
    @pytest.fixture(params=[QiskitQUBOSolver, PennyLaneQUBOSolver])
    def solver(self, request):
        """Fixture that provides a solver instance for testing."""
        solver_class = request.param
        return solver_class()
    
    def test_solve_simple_qubo(self, solver):
        """Test solving a simple QUBO problem."""
        # Simple QUBO: min x1 + x2 - 2x1x2
        # Solution: x1=1, x2=0 or x1=0, x2=1
        Q = np.array([
            [1, -2],
            [0, 1]
        ])
        
        result = solver.solve(Q)
        solution = result['solution']
        
        # Check that the solution is valid
        assert len(solution) == 2
        assert sum(solution) == 1  # Exactly one variable should be 1
        assert result['optimal_value'] == 0  # Minimum value is 0
    
    def test_solve_qubo_dict_input(self, solver):
        """Test solving QUBO with dictionary input."""
        # Same as above but with dictionary input
        Q = {
            (0, 0): 1,
            (0, 1): -2,
            (1, 1): 1
        }
        
        result = solver.solve(Q, num_vars=2)
        solution = result['solution']
        
        assert len(solution) == 2
        assert sum(solution) == 1
    
    def test_validate_qubo_matrix(self, solver):
        """Test QUBO matrix validation."""
        # Non-square matrix should raise ValueError
        with pytest.raises(ValueError):
            Q = np.array([[1, 2, 3], [4, 5, 6]])
            solver._validate_qubo_matrix(Q)
        
        # Non-symmetric matrix should raise ValueError
        with pytest.raises(ValueError):
            Q = np.array([[1, 2], [3, 4]])
            solver._validate_qubo_matrix(Q)
        
        # Valid QUBO matrix should not raise
        Q = np.array([[1, -1], [-1, 2]])
        solver._validate_qubo_matrix(Q)  # Should not raise
    
    @pytest.mark.parametrize("solver_class,backend_kwargs", [
        (QiskitQUBOSolver, {'reps': 2}),
        (PennyLaneQUBOSolver, {'reps': 2})
    ])
    def test_custom_parameters(self, solver_class, backend_kwargs):
        """Test passing custom parameters to the solver."""
        solver = solver_class(**backend_kwargs)
        assert solver.reps == 2
