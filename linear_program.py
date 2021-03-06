import strategy
from simplex import Simplex


class StandardLinearProgram(object):
    def __init__(self, objective_function, lefthand_side, righthand_side):
        self.objective_function = objective_function
        self.lefthand_side = lefthand_side
        self.righthand_side = righthand_side
        self.constraints_count = len(righthand_side)
        self.variables_count = lefthand_side.shape[1]


class LinearProgramSolver(object):
    DEFAULT_MAX_ITERATIONS_COUNT = 1000

    @staticmethod
    def solve_simplex(linear_program, pivot_strategy=None, max_iterations=None):
        if max_iterations is None:
            max_iterations = LinearProgramSolver.DEFAULT_MAX_ITERATIONS_COUNT
        if pivot_strategy is None:
            pivot_strategy = strategy.MaxCoefficientStrategy()

        solver = Simplex(pivot_strategy, max_iterations)
        return solver.solve(linear_program)

    @staticmethod
    def solve_simplex_steps(linear_program, pivot_strategy=None, max_iterations=None):
        if max_iterations is None:
            max_iterations = LinearProgramSolver.DEFAULT_MAX_ITERATIONS_COUNT
        if pivot_strategy is None:
            pivot_strategy = strategy.MaxCoefficientStrategy()

        solver = Simplex(pivot_strategy, max_iterations)
        return solver.solution_steps(linear_program)
