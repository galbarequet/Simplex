import exceptions
import itertools
import tableau
from solution import Solution

class Simplex(object):
    def __init__(self, strategy, max_iterations):
        self._strategy = strategy
        self._max_iterations = max_iterations

    def _is_optimal_solution(self, tableau):
        '''
        Solution is optimal if all variable coefficients are non-positive in objective function
        '''
        # Note: first item is free variables coefficient
        return all(x <= 0 for i, x in enumerate(tableau.get_objective_function_coefficients()) if i > 0)

    def _optimize_solution(self, tableau):
        entering_var = self._strategy.find_entering(tableau)
        leaving_var = self._strategy.find_leaving(tableau, entering_var)
        if leaving_var is None:
            raise exceptions.SimplexProblemUnboundedError()

        tableau.change_base(entering_var, leaving_var)

    def _solve_phase_steps(self, tableau):
        while not self._is_optimal_solution(tableau) and tableau.pivots_count < self._max_iterations:
            self._optimize_solution(tableau)
            yield Solution(tableau, self._strategy)

        if not self._is_optimal_solution(tableau) and tableau.pivots_count >= self._max_iterations:
            raise exceptions.SimplexIterationsLimitExceedError()

    def _phase1_steps(self, tableau):
        if tableau.pivots_count >= self._max_iterations:
            raise exceptions.SimplexIterationsLimitExceedError()

        if not tableau.should_initialize:
            yield Solution(tableau, self._strategy)
            return

        leaving_var, free_var_value = tableau.get_most_infeasible_basic_variable_info()
        if free_var_value <= 0:
            # nothing to do
            return
        
        with tableau.use_artificial_argument():
            # perfrom first mandatory pivot
            # CR: (GB) use regular change base!
            tableau._change_base_internal(-1, leaving_var)

            if tableau.pivots_count >= self._max_iterations:
                raise exceptions.SimplexIterationsLimitExceedError()

            yield Solution(tableau, self._strategy)

            # solve single phase normally
            for solution in self._solve_phase_steps(tableau):
                yield solution

            if tableau.get_objective_function_coefficients()[0] > 0:
                raise exceptions.SimplexProblemInfeasibleError()

    def _phase2_steps(self, tableau):
        tableau.use_objective_function()
        got_solution = False
        for solution in self._solve_phase_steps(tableau):
            yield solution
            if not got_solution:
                got_solution = True

        if not got_solution:
            yield Solution(tableau, self._strategy)

    def solution_steps(self, linear_program):
        tableau_obj = tableau.tableau(linear_program)
        return itertools.chain(self._phase1_steps(tableau_obj), self._phase2_steps(tableau_obj))

    def solve(self, linear_program):
        return list(self.solution_steps(linear_program))[-1]
