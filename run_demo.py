import exceptions
from linear_program import LinearProgramSolver, StandardLinearProgram
from simplex_plotter import KleeMintyPlotter
import strategy
from utils import array

############ LP parameters creators ##############
def basic():
    objective_func = array([5, 4, 3])
    constraint_lhs = array([[2, 3, 1], [4, 1, 2], [3, 4, 2]])
    constraint_rhs = array([5, 11 ,8])
    return StandardLinearProgram(objective_func, constraint_lhs, constraint_rhs)

def need_init():
    objective_func = array([-2, -1])
    constraint_lhs = array([[-1, 1], [-1, -2], [0, 1]])
    constraint_rhs = array([-1, -2 ,1])
    return StandardLinearProgram(objective_func, constraint_lhs, constraint_rhs)

def klee_minty():
    objective_func = array([100, 10, 1])
    constraint_lhs = array([[1, 0, 0], [20, 1, 0], [200, 20, 1]])
    constraint_rhs = array([1, 100, 10000])
    return StandardLinearProgram(objective_func, constraint_lhs, constraint_rhs)

def klee_minty2():
    objective_func = array([4, 2, 1])
    constraint_lhs = array([[1, 0, 0], [4, 1, 0], [8, 4, 1]])
    constraint_rhs = array([5, 25, 125])
    return StandardLinearProgram(objective_func, constraint_lhs, constraint_rhs)

def unbounded():
    objective_func = array([1, -1])
    constraint_lhs = array([[-2, 3], [0, 4], [0, -1]])
    constraint_rhs = array([5, 7, 0])
    return StandardLinearProgram(objective_func, constraint_lhs, constraint_rhs)

def cycle():
    objective_func = array([10, -57, -9, -24])
    constraint_lhs = array([[1/2, -11/2, -5/2, 9], [1/2, -3/2, -1/2, 1], [1, 0, 0, 0]])
    constraint_rhs = array([0, 0, 1])
    return StandardLinearProgram(objective_func, constraint_lhs, constraint_rhs)
##################################################

def shows_steps():
    lp_creator_methods = [basic, need_init, klee_minty, klee_minty2, unbounded, cycle]
    for lp_create in lp_creator_methods:
        try:
            print(LinearProgramSolver.solve_simplex(lp_create(), max_iterations=20))
            # for sol in LinearProgramSolver.solve_simplex_steps(lp_create(), pivot_strategy=strategy.MaxCoefficientStrategy(), max_iterations=20):
            #     print(sol)
        except exceptions.SimplexError as e:
            print(e)

def main():
    # shows_steps()

    plotter = KleeMintyPlotter()
    plotter.demo()
    plotter = KleeMintyPlotter()
    plotter.demo(pivot_strategy=strategy.MinCoefficientStrategy())


if __name__ == '__main__':
    main()
