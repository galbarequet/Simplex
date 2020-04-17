from utils import Status, zeros

class Solution(object):
    _ERRORS = {
        Status.SUCCESS: 'The linear program is bounded',
        Status.INFEASIBLE: 'The linear program is INFEASIBLE',
        Status.UNBOUNDED: 'The linear program is UNBOUNDED',
        Status.ITERATION_LIMIT: 'simplex iteration bounds reached',
    }

    def __init__(self, status, simplex):
        self.status = status
        self.solution = zeros(simplex.real_variables_count)
        self.objective_value = simplex.tableau[0, 0]
        self.startegy = simplex.startegy
        if self.status == Status.INFEASIBLE:
            return
        
        for i in range(1, simplex.real_variables_count + 1):
            if simplex.basic_vars[i] != 0:
                # solution variable indices start from 0
                self.solution[i - 1] = -simplex.tableau[simplex.basic_vars[i], 0] / simplex.tableau[simplex.basic_vars[i], i]

    def __str__(self):
        if self.status != Status.SUCCESS:
            return(self._ERRORS[self.status])

        data = f'''Possible optimal solution is: {', '.join(('x_{} = {}'.format(index, value) for index, value in enumerate(self.solution, 1)))}
The objective value for this solution is: {self.objective_value}
Total pivots count: self.stats
The pivot rule used: {self.startegy.__class__.__name__}
'''
        return data