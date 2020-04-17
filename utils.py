import numpy as np

def fractionize(initilalizer):
    return lambda x: initilalizer(x, dtype=np.float64)

zeros = fractionize(np.zeros)
ones = fractionize(np.ones)
eye = fractionize(np.eye)
array = fractionize(np.array)
