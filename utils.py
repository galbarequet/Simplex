import numpy as np
from enum import IntEnum

def fractionize(initilalizer):
    return lambda x: initilalizer(x, dtype=np.float64)

zeros = fractionize(np.zeros)
ones = fractionize(np.ones)
eye = fractionize(np.eye)
array = fractionize(np.array)

class Status(IntEnum):
    SUCCESS = 0
    INFEASIBLE = 1
    UNBOUNDED = 2
    ITERATION_LIMIT = 3
