from utils import zeros
from enum import IntEnum


class Solution(object):
    def __init__(self, simplex):
        self.solution = zeros(simplex.real_variables_count)
        self.objective_value = simplex.tableau[0, 0]
        self.startegy = simplex.startegy
        self.iterations_count = simplex.iterations_count
        
        for i in range(1, simplex.real_variables_count + 1):
            if simplex.basic_vars[i] != 0:
                # solution variable indices start from 0
                self.solution[i - 1] = -simplex.tableau[simplex.basic_vars[i], 0] / simplex.tableau[simplex.basic_vars[i], i]

    def __str__(self):
        data = f'''Possible optimal solution is: {', '.join(('x_{} = {}'.format(index, value) for index, value in enumerate(self.solution, 1)))}
The objective value for this solution is: {self.objective_value}
Total pivots count: {self.iterations_count}
The pivot rule used: {self.startegy.__class__.__name__}
'''
        return data
