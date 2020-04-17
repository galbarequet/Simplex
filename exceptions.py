class SimplexError(Exception):
    def __init__(self, message):
        super().__init__(message)

class SimplexIterationsLimitExceedError(SimplexError):
    def __init__(self):
        super().__init__('Simplex max iterations limit reached!')

class SimplexProblemInfeasibleError(SimplexError):
    def __init__(self):
        super().__init__('The linear program is INFEASIBLE')

class SimplexProblemUnboundedError(SimplexError):
    def __init__(self):
        super().__init__('The linear program is UNBOUNDED')
