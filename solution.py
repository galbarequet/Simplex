from utils import zeros
from enum import IntEnum


class Solution(object):
    def __init__(self, tableau, pivot_strategy):
        self.solution = tableau.get_current_solution()
        self.objective_value = tableau[0, 0]
        self.pivot_strategy = pivot_strategy
        self.iterations_count = tableau.pivots_count

    def __str__(self):
        data = f'''Possible optimal solution is: {', '.join(('x_{} = {}'.format(index, value) for index, value in enumerate(self.solution, 1)))}
The objective value for this solution is: {self.objective_value}
Total pivots count: {self.iterations_count}
The pivot rule used: {self.pivot_strategy.__class__.__name__}
'''
        return data
