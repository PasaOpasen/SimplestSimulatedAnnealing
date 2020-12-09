# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:25:56 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
#import matplotlib.pyplot as plt

from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling

SEED = 3

np.random.seed(SEED)

Set = np.random.uniform(low = -15, high=5, size = 100) # all set

dim = 10 # how many objects should we choose

indexes = np.arange(Set.size)
# minimized function -- subset with best |median|
def min_func(arr):
    return abs(np.median(Set[indexes[arr.astype(bool)]]))

# zero vectors with 'dim' ones at random positions 
start_solution = np.zeros(Set.size)
start_solution[np.random.choice(indexes, dim, replace=False)] = 1

# mutation function
# temperature is the number cuz we will use only 1 cooling, but it's not necessary to use it)
def mut(x_as_array, temperature_as_array_or_one_number):
    mask_one = x_as_array == 1
    mask_zero = np.logical_not(mask_one)

    new_x_as_array = x_as_array.copy()
    # replace some zeros with ones
    new_x_as_array[np.random.choice(indexes[mask_one], 1, replace=False)] = 0
    new_x_as_array[np.random.choice(indexes[mask_zero], 1, replace=False)] = 1

    return new_x_as_array

# creating a model
model = SimulatedAnnealing(min_func, dim)

# run search
best_solution, best_val = model.run(
    start_solution = start_solution,
    mutation = mut,
    cooling = Cooling.exponential(0.9), 
    start_temperature = 100, 
    max_function_evals = 1000, 
    max_iterations_without_progress = 100, 
    step_for_reinit_temperature = 80,
    seed = SEED
    )

model.plot_report(save_as = 'best_subset.png')

