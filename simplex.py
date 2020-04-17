import numpy as np
from utils import Status, zeros, eye, ones
from solution import Solution
from contextlib import contextmanager
import itertools

class Simplex(object):
    '''
    The tableau has the following form:
    ----------------------------------------------------------------
    | free_num | <obj_coeffs> | <zeros_row>  | <artificial_coeffs> |
    ----------------------------------------------------------------
    |   b_i    | <real_vars>  | <slack_vars> |  <artificial_vars>  |
    ----------------------------------------------------------------

    The basic variables list is a map between each variable and the constraint index (1-based) representing it (if it is non-basic then will have zero)
    Note that the first index (0) is reserved for the free variable, and isn't used and should always be zero.

    The tight variables list is a map between the constraints and the basic variables represented by them. This is the reciprocal of the basic variables list.
    Note that the first index (0) is reserved for the objective function, and isn't used and should always be zero.
    '''
    _OBJECTIVE_ROW_INDEX = 0
    _CONSTRAINT_ROW_START_INDEX = 1
    _VARIABLES_FREE_VARIABLE_COL_INDEX = 0
    _VARIABLES_COL_START_INDEX = 1

    def __init__(self, linear_program, strategy):
        self.startegy = strategy
        self._objective_function = linear_program.objective_function
        self.constraints_count = linear_program.constraints_count
        self.real_variables_count = linear_program.variables_count
        self._artificial_variables_count = 0
        self._variables_count = self.constraints_count + self.real_variables_count

        if min(linear_program.righthand_side) < 0:
            self._artificial_variables_count += 1

        # 1 col for free variable, 1 row for objective_function
        self.tableau = zeros((self.constraints_count + 1, self._variables_count + 1))

        # lefthand-side:
        # real variales
        self.tableau[self._CONSTRAINT_ROW_START_INDEX:, self._VARIABLES_COL_START_INDEX: self.real_variables_count + 1] = linear_program.lefthand_side
        # slack variables - we assume every line is <= (LE) so we just need to add a slack variable per constraint
        slack_start_index = self._VARIABLES_COL_START_INDEX + self.real_variables_count
        self.tableau[self._CONSTRAINT_ROW_START_INDEX:, slack_start_index: slack_start_index + self.constraints_count] = eye(self.constraints_count)

        # righthand-side:
        self.tableau[self._CONSTRAINT_ROW_START_INDEX:, self._VARIABLES_FREE_VARIABLE_COL_INDEX] = linear_program.righthand_side * -1

        # initially all slack variables are basic ones
        self.basic_vars = np.zeros((self._variables_count + 1,), dtype='int')  # including free variable, unused and should always be 0
        self.basic_vars[slack_start_index: slack_start_index + self.constraints_count] = range(1, self.constraints_count + 1)
        self._tight_vars = np.array(range(self.real_variables_count, self._variables_count + 1), dtype='int')
        self._tight_vars[0] = 0

        assert len(self.basic_vars) == self.tableau.shape[1], 'basic variables array must be the same size as tablue row size'


    def _is_optimal_solution(self):
        '''
        Solution is optimal if all variable coefficients are non-positive in objective function
        '''
        return all(x <= 0 for x in self.tableau[self._OBJECTIVE_ROW_INDEX, self._VARIABLES_COL_START_INDEX:])

    def _perform_pivot(self, pivot_row_index, pivot_col_index):
        # canonize according to pivot_col_index
        self.tableau[pivot_row_index] /= self.tableau[pivot_row_index, pivot_col_index]

        # perform Gauss elimination
        num_rows = self.tableau.shape[0]
        for row_index in range(num_rows):
            if row_index == pivot_row_index:
                continue
            self.tableau[row_index] -= self.tableau[row_index, pivot_col_index] * self.tableau[pivot_row_index]

    def _change_base_internal(self, entering_var, leaving_var):
        self._perform_pivot(self.basic_vars[leaving_var], entering_var)

        self.basic_vars[entering_var] = self.basic_vars[leaving_var]
        self._tight_vars[self.basic_vars[leaving_var]] = entering_var
        self.basic_vars[leaving_var] = 0
     
    def _change_base(self, entering_var, leaving_var):
        assert self.basic_vars[entering_var] == 0, 'entering variable must be non-basic'
        assert self.basic_vars[leaving_var] != 0, 'leaving variable must be basic'
        self._change_base_internal(entering_var, leaving_var)

    def _get_entering_candidates(self):
        '''
        The entering candidtes are the variables' indecies with positive coefficients in the objective function
        '''
        return [i for i in range(self._VARIABLES_COL_START_INDEX, self._variables_count + 1) if self.basic_vars[i] == 0 and self.tableau[0, i] > 0]

    def _optimize_solution(self):
        entering_var = self.startegy.find_entering(self.tableau, self._get_entering_candidates())
        constraint_index = self.startegy.find_leaving_constraint(self.tableau, entering_var)
        if constraint_index is None:
            return Solution(Status.UNBOUNDED, self)

        leaving_var = self._tight_vars[constraint_index]

        self._change_base(entering_var, leaving_var)

    def _solve_phase(self):
        while not self._is_optimal_solution():
            result = self._optimize_solution()
            if result.status != Status.SUCCESS:
                return result

        return Solution(Status.SUCCESS, self)

    def _solve_phase_steps(self):
        while not self._is_optimal_solution():
            self._optimize_solution()
            yield Solution(Status.SUCCESS, self)
    
    def _get_phase1_initial_leaving_var_info(self):
        constraint_index, free_var_value = max(
            [(i + self._CONSTRAINT_ROW_START_INDEX, v) for (i, v) in 
                enumerate(self.tableau[self._CONSTRAINT_ROW_START_INDEX:, self._VARIABLES_FREE_VARIABLE_COL_INDEX])],
            key=lambda x: x[1])
        return constraint_index, free_var_value

    @contextmanager
    def _artificial_argument(self):
        # add artificial column in the end
        self.tableau = np.hstack((self.tableau, -1 * ones((self.constraints_count + 1, 1))))
        self.basic_vars = np.hstack((self.basic_vars, np.zeros(1, dtype='int')))
        self._variables_count += 1

        yield

        # remove artificial column in the end
        ALONG_ROW = 0
        ALONG_COL = 1
        self.tableau = np.delete(self.tableau, -1, ALONG_COL)
        self.basic_vars = np.delete(self.basic_vars, -1, ALONG_ROW)
        self._variables_count -= 1
    
    def _phase1(self):
        if self._artificial_variables_count == 0:
            return Solution(Status.SUCCESS, self)

        constraint_index, free_var_value = self._get_phase1_initial_leaving_var_info()
        if free_var_value <= 0:
            # nothing to do
            return
        
        with self._artificial_argument():
            # perfrom first mandatory pivot
            self._change_base_internal(-1, self._tight_vars[constraint_index])

            # solve single phase normally
            result = self._solve_phase()
            assert result.status !=  Status.UNBOUNDED, 'unbounded Phase One ???'

            if self.tableau[self._OBJECTIVE_ROW_INDEX, self._VARIABLES_FREE_VARIABLE_COL_INDEX] > 0:
                result.status = Status.INFEASIBLE
                return result

        return result

    def _phase1_steps(self):
        if self._artificial_variables_count == 0:
            yield Solution(Status.SUCCESS, self)
            return

        constraint_index, free_var_value = self._get_phase1_initial_leaving_var_info()
        if free_var_value <= 0:
            # nothing to do
            return
        
        with self._artificial_argument():
            # perfrom first mandatory pivot
            self._change_base_internal(-1, self._tight_vars[constraint_index])
            yield Solution(Status.SUCCESS, self)

            # solve single phase normally
            for sol in self._solve_phase_steps():
                yield sol

    def _use_objective_function(self):
        self.tableau[self._OBJECTIVE_ROW_INDEX, self._VARIABLES_COL_START_INDEX: self.real_variables_count + 1] = self._objective_function

        if self._artificial_variables_count == 1:
            for variable in range(1, self._variables_count + 1):
                pivot = self.basic_vars[variable]
                if pivot == 0:
                    continue
                self.tableau[self._OBJECTIVE_ROW_INDEX] -= ((self.tableau[self._OBJECTIVE_ROW_INDEX, variable] / self.tableau[pivot, variable]) *
                                                            self.tableau[pivot])

    def _phase2(self):
        self._use_objective_function()
        return self._solve_phase()

    def _phase2_steps(self):
        self._use_objective_function()
        for sol in self._solve_phase_steps():
            yield sol
        yield Solution(Status.SUCCESS, self)

    def solve(self):
        result = self._phase1()
        if result.status != Status.SUCCESS:
            return result
        
        return self._phase2()

    def solution_steps(self):
        return itertools.chain(self._phase1_steps(), self._phase2_steps())
