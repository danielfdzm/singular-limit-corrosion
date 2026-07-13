"""The main ingredients needed to solve one finite-element problem."""

from .model import Parameters, Problem, build_benchmarks
from .mesh import build_problem, solve_mixed_reference
from .nonlinear_solver import solve_problem


if __name__ == "__main__":
    from .solve_experiments import main

    main()