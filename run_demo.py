import linear_program
from simplex_plotter import KleeMintyPlotter
import strategy
from utils import zeros, array

############ LP parameters creators ##############
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

def klee_minty2():
    objective_func = array([4, 2, 1])
    constraint_lhs = array([[1, 0, 0], [4, 1, 0], [8, 4, 1]])
    constraint_rhs = array([5, 25, 125])
    return objective_func, constraint_lhs, constraint_rhs

def unbounded():
    objective_func = array([1, -1])
    constraint_lhs = array([[-2, 3], [0, 4], [0, -1]])
    constraint_rhs = array([5, 7, 0])
    return objective_func, constraint_lhs, constraint_rhs
##################################################

def shows_steps():
    obj_func, cons_lhs, cons_rhs = klee_minty2()
    lp = linear_program.StandardLinearProgram(obj_func, cons_lhs, cons_rhs)
    print(lp.solve_simplex(max_iterations=10))
    # for sol in lp.solve_simplex_steps(pivot_strategy=strategy.MaxCoefficientStrategy()):
    #    print(sol)

def main():
    plotter = KleeMintyPlotter.create()
    plotter.demo()
    plotter = KleeMintyPlotter.create()
    plotter.demo(pivot_strategy=strategy.MinCoefficientStrategy())


if __name__ == '__main__':
    main()
