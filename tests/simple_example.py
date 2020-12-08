# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:06:30 2020

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

mut = simple_continual_mutation(std = 0.5)


model = SimulatedAnnealing(Rastrigin, dim)

best_solution, best_val = model.run(
    start_solution = np.random.uniform(-5, 5, dim),
    mutation = mut,
    cooling = Cooling.exponential(0.9), 
    start_temperature = 100, 
    max_function_evals = 1000, 
    max_iterations_without_progress = 100, 
    step_for_reinit_temperature = 80
    )

print(best_val)

model.plot_report(save_as = 'simple_example.png')



