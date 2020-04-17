from contextlib import contextmanager
import numpy as np
import exceptions
from utils import zeros, eye, ones

class tableau(object):
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

    def __init__(self, linear_program):
        self._objective_function = linear_program.objective_function
        self._constraints_count = linear_program.constraints_count
        self._real_variables_count = linear_program.variables_count
        self.should_initialize = False
        self._using_artificial_variable = False
        self._variables_count = self._constraints_count + self._real_variables_count
        self.pivots_count = 0

        if min(linear_program.righthand_side) < 0:
            self.should_initialize = True

        # 1 col for free variable, 1 row for objective_function
        self._tableau = zeros((self._constraints_count + 1, self._variables_count + 1))

        # lefthand-side:
        # real variales
        self._tableau[self._CONSTRAINT_ROW_START_INDEX:, self._VARIABLES_COL_START_INDEX: self._real_variables_count + 1] = linear_program.lefthand_side
        # slack variables - we assume every line is <= (LE) so we just need to add a slack variable per constraint
        slack_start_index = self._VARIABLES_COL_START_INDEX + self._real_variables_count
        self._tableau[self._CONSTRAINT_ROW_START_INDEX:, slack_start_index: slack_start_index + self._constraints_count] = eye(self._constraints_count)

        # righthand-side:
        self._tableau[self._CONSTRAINT_ROW_START_INDEX:, self._VARIABLES_FREE_VARIABLE_COL_INDEX] = linear_program.righthand_side * -1

        # initially all slack variables are basic ones
        self._basic_vars = np.zeros((self._variables_count + 1,), dtype='int')  # including free variable, unused and should always be 0
        self._basic_vars[slack_start_index: slack_start_index + self._constraints_count] = range(1, self._constraints_count + 1)
        self._tight_vars = np.array(range(self._real_variables_count, self._variables_count + 1), dtype='int')
        self._tight_vars[0] = 0

        assert len(self._basic_vars) == self._tableau.shape[1], 'basic variables array must be the same size as tablue row size'

    def __getitem__(self, key):
        return self._tableau[key]

    @property
    def constraints_count(self):
        return self._constraints_count

    @property
    def pivots_count(self):
        return self.__pivots_count

    @pivots_count.setter
    def pivots_count(self, pivots_count):
         self.__pivots_count = pivots_count

    @property
    def should_initialize(self):
        return self.__should_initialize

    @should_initialize.setter
    def should_initialize(self, should_initialize):
         self.__should_initialize = should_initialize

    def _perform_pivot(self, pivot_row_index, pivot_col_index):
        # canonize according to pivot_col_index
        self._tableau[pivot_row_index] /= self._tableau[pivot_row_index, pivot_col_index]

        # perform Gauss elimination
        num_rows = self._tableau.shape[0]
        for row_index in range(num_rows):
            if row_index == pivot_row_index:
                continue
            self._tableau[row_index] -= self._tableau[row_index, pivot_col_index] * self._tableau[pivot_row_index]

        self.pivots_count += 1
    
    def _change_base_internal(self, entering_var, leaving_var):
        self._perform_pivot(self._basic_vars[leaving_var], entering_var)

        self._basic_vars[entering_var] = self._basic_vars[leaving_var]
        self._tight_vars[self._basic_vars[leaving_var]] = entering_var
        self._basic_vars[leaving_var] = 0
     
    def change_base(self, entering_var, leaving_var, is_forced_initialize=False):
        if is_forced_initialize and not self._using_artificial_variable:
            raise exceptions.SimplexError("Can't force initialization without artificial variables!")

        # Note: in the first initialization step the artificial variables is entering but do
        #if not is_forced_initialize or not self._using_artificial_variable or (entering_var != -1 and entering_var != ):
        assert self._basic_vars[entering_var] == 0, 'entering variable must be non-basic'
        assert self._basic_vars[leaving_var] != 0, 'leaving variable must be basic'

        self._change_base_internal(entering_var, leaving_var)

    def get_current_solution(self):
        solution = zeros(self._real_variables_count)
        for i in range(1, self._real_variables_count + 1):
            if self._basic_vars[i] == 0:
                continue

            # solution variable indices start from 0
            solution[i - 1] = -self._tableau[self._basic_vars[i], 0] / self._tableau[self._basic_vars[i], i]

        return solution

    def get_objective_function_coefficients(self):
        '''
        Returns the objective function coefficients including the free variable
        '''
        return self._tableau[self._OBJECTIVE_ROW_INDEX]

    def get_entering_candidates(self):
        '''
        The entering candidtes are the variables' indices with positive coefficients in the objective function
        Note: the values are 1-based
        '''
        return [i for i in range(self._VARIABLES_COL_START_INDEX, self._variables_count + 1) if self._basic_vars[i] == 0 and self._tableau[0, i] > 0]

    def get_variable_representing_constraint(self, constraint_index):
        if constraint_index is None:
            return None
        return self._tight_vars[constraint_index]

    @contextmanager
    def use_artificial_argument(self):
        # add artificial column in the end
        self._tableau = np.hstack((self._tableau, -1 * ones((self._constraints_count + 1, 1))))
        self._basic_vars = np.hstack((self._basic_vars, np.zeros(1, dtype='int')))
        self._variables_count += 1
        self._using_artificial_variable = True

        yield

        # remove artificial column in the end
        ALONG_ROW = 0
        ALONG_COL = 1
        self._tableau = np.delete(self._tableau, -1, ALONG_COL)
        self._basic_vars = np.delete(self._basic_vars, -1, ALONG_ROW)
        self._variables_count -= 1
        self._using_artificial_variable = False

    def use_objective_function(self):
        self._tableau[self._OBJECTIVE_ROW_INDEX, self._VARIABLES_COL_START_INDEX: self._real_variables_count + 1] = self._objective_function

        for variable in range(1, self._variables_count + 1):
            pivot = self._basic_vars[variable]
            if pivot == 0:
                continue
            self._tableau[self._OBJECTIVE_ROW_INDEX] -= (
                (self._tableau[self._OBJECTIVE_ROW_INDEX, variable] / self._tableau[pivot, variable]) * self._tableau[pivot])

    def get_most_infeasible_basic_variable_info(self):
        constraint_index, free_var_value = max(
            [(i + self._CONSTRAINT_ROW_START_INDEX, v) for (i, v) in 
                enumerate(self._tableau[self._CONSTRAINT_ROW_START_INDEX:, self._VARIABLES_FREE_VARIABLE_COL_INDEX])],
            key=lambda x: x[1])
        
        basic_var_index = self.get_variable_representing_constraint(constraint_index)

        return basic_var_index, free_var_value
