import strategy
from simplex import Simplex

class StandardLinearProgram(object):
    def __init__(self, objective_function, lefthand_side, righthand_side):
        self.objective_function = objective_function
        self.lefthand_side = lefthand_side
        self.righthand_side = righthand_side
        self.constraints_count = len(righthand_side)
        self.variables_count = lefthand_side.shape[1]

    def solve_simplex(self, pivot_strategy=None, max_iterations=1000):
        if pivot_strategy is None:
            pivot_strategy = strategy.MaxCoefficientStrategy()
        solver = Simplex(self, pivot_strategy, max_iterations)
        return solver.solve()

    def solve_simplex_steps(self, pivot_strategy=None, max_iterations=1000):
        if pivot_strategy is None:
            pivot_strategy = strategy.MaxCoefficientStrategy()
        solver = Simplex(self, pivot_strategy, max_iterations)
        return solver.solution_steps()
