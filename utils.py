import numpy as np

def fractionize(initilalizer):
    return lambda x: initilalizer(x, dtype=np.float64)

zeros = fractionize(np.zeros)
ones = fractionize(np.ones)
eye = fractionize(np.eye)
array = fractionize(np.array)

def items_final_indicator(iterable):
    '''
    Iterates through items in iterable yielding the item and an indicator if it's the last item
    '''
    it = iter(iterable)
    last = next(it)
    for val in it:
        yield last, False
        last = val
    yield last, True
