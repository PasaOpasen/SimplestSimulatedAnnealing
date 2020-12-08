import numpy as np



def simple_continual_mutation(std = 0.1):

    if type(std) in (int, float):
        def func(arr):
            return np.array([np.random.normal(x, std) for x in arr])
    else:
        def func(arr):
            return np.array([np.random.normal(x, sd) for x, sd in zip(arr, std)])
    return func
