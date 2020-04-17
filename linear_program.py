import strategy
from simplex import Simplex

class StandardLinearProgram(object):
    def __init__(self, objective_function, lefthand_side, righthand_side):
        self.objective_function = objective_function
        self.lefthand_side = lefthand_side
        self.righthand_side = righthand_side
        self.constraints_count = len(righthand_side)
        self.variables_count = lefthand_side.shape[1]

    def solve_simplex(self, startegy=None, max_iterations=1000):
        if startegy is None:
            startegy = strategy.MaxCoefficientStrategy()
        solver = Simplex(self, startegy, max_iterations)
        return solver.solve()

    def solve_simplex_steps(self, startegy=None, max_iterations=1000):
        if startegy is None:
            startegy = strategy.MaxCoefficientStrategy()
        solver = Simplex(self, startegy, max_iterations)
        return solver.solution_steps()


if __name__ == '__main__':
    from utils import zeros, array

    def basic():
        objective_func = array([5, 4, 3])
        constraint_lhs = array([[2, 3, 1], [4, 1, 2], [3, 4, 2]])
        constraint_rhs = array([5, 11 ,8])
        return objective_func, constraint_lhs, constraint_rhs

    def need_init():
        objective_func = array([-2, -1])
        constraint_lhs = array([[-1, 1], [-1, -2], [0, 1]])
        constraint_rhs = array([-1, -2 ,1])
        return objective_func, constraint_lhs, constraint_rhs

    def klee_minty():
        objective_func = array([100, 10, 1])
        constraint_lhs = array([[1, 0, 0], [20, 1, 0], [200, 20, 1]])
        constraint_rhs = array([1, 100, 10000])
        return objective_func, constraint_lhs, constraint_rhs

    def unbounded():
        objective_func = array([1, -1])
        constraint_lhs = array([[-2, 3], [0, 4], [0, -1]])
        constraint_rhs = array([5, 7, 0])
        return objective_func, constraint_lhs, constraint_rhs

    obj_func, cons_lhs, cons_rhs = need_init()
    lp = StandardLinearProgram(obj_func, cons_lhs, cons_rhs)
    print(lp.solve_simplex(max_iterations=10))
    # for sol in lp.solve_simplex_steps():
    #    print(sol)
