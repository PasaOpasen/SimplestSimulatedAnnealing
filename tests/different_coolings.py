# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:32:52 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import math
import numpy as np

from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling, simple_continual_mutation


def Rastrigin(arr):
    return 10*arr.size+np.sum(arr**2) - 10*np.sum(np.cos(2*math.pi*arr))

dim = 5



model = SimulatedAnnealing(Rastrigin, dim)

best_solution, best_val = model.run(
    start_solution = np.random.uniform(-5, 5, dim),
    mutation = simple_continual_mutation(std = 1),
    cooling = [ # different cooling for each dimention
        Cooling.exponential(0.8),
        Cooling.exponential(0.9),
        Cooling.reverse(beta = 0.0005),
        Cooling.linear_reverse(),
        Cooling.reverse(beta = 0.001)
        ], 
    start_temperature = 100, 
    max_function_evals = 1000, 
    max_iterations_without_progress = 250, 
    step_for_reinit_temperature = 90,
    reinit_from_best = False
    )

print(best_val)

model.plot_report(save_as = 'different_coolings.png')