class Strategy(object):
    def find_entering(self, tableau):
        raise NotImplementedError()
    def find_leaving(self, tableau, entering_variable):
        raise NotImplementedError()

class RatioTestStrategy(Strategy):
    def _find_leaving_constraint(self, tableau, entering_variable):
        """
        Return the smallest constraint index (related to basic variable) according to the standard ratio test on the entering variable.
        That is the leaving variable has the smallest b_i / a_ik ratio.
        Also we apply Bland's rule to ensure the algorithm terminates. 
        Return None if unbounded.
        """

        min_ratio_index = None
        min_ratio_value = float("inf")
        for i in range(1, tableau.constraints_count + 1):
            if tableau[i, entering_variable] <= 0:
                continue

            # looking at b_i / a_ik
            ratio = -1 * tableau[i, 0] / tableau[i, entering_variable]
            # < (strict LESS) because of Bland's rule guaranty that the algorithm will terminate eventually
            if ratio < min_ratio_value:
                min_ratio_index = i
                min_ratio_value = ratio

        return min_ratio_index

    def find_leaving(self, tableau, entering_variable):
        constraint_index = self._find_leaving_constraint(tableau, entering_variable)
        return tableau.get_variable_representing_constraint(constraint_index)


class MaxCoefficientStrategy(RatioTestStrategy):
    """
    Maximum coefficient Strategy chooses the entering variable which has the biggest coefficient in objective function from relevant candidates (positive)
    """
    def find_entering(self, tableau):
        candidates = tableau.get_entering_candidates()
        coefficients = tableau.get_objective_function_coefficients()
        return max(candidates, key=lambda i: coefficients[i])


class MinCoefficientStrategy(RatioTestStrategy):
    """
    Minimum coefficient Strategy chooses the entering variable which has the smallest coefficient in objective function from relevant candidates (positive)
    """
    def find_entering(self, tableau):
        candidates = tableau.get_entering_candidates()
        coefficients = tableau.get_objective_function_coefficients()
        return min(candidates, key=lambda i: coefficients[i])
