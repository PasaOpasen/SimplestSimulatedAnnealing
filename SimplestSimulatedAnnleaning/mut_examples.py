import numpy as np



def simple_continual_mutation(std = 0.1):

    if type(std) in (int, float):
        def func(arr, temperature):
            return np.array([np.random.normal(x, std) for x in arr])
    else:
        def func(arr, temperature):
            return np.array([np.random.normal(x, sd) for x, sd in zip(arr, std)])
    return func


def continual_mutation_with_temperature(coef = 0.05):
    
    def func(arr, temperature):
        return np.array([np.random.normal(x, temperature*coef) for x in arr]) if type(temperature) in (int, float) else np.array([np.random.normal(x, t) for x, t in zip(arr, temperature * coef)])
    return func

