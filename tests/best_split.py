# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:44:42 2020

@author: qtckp
"""


import sys
sys.path.append('..')

from collections import defaultdict
import numpy as np

from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling

################### useful methods

def counts_to_vec(dic_count):
    """
    converts dictionary like {1: 3, 2: 4}
    to array [1, 1, 1, 2, 2, 2, 2]
    """
    arrs = [np.full(val, fill_value=key) for key, val in dic_count.items()]

    return np.concatenate(tuple(arrs))

def vec_to_indexes_dict(vector):
    """
    converts vector like [1, 0, 1, 2, 2]
    to dictionary with indexes {1: [0, 2], 2: [3, 4]}
    """

    res = defaultdict(list)

    for i, v in enumerate(vector):
        res[v].append(i)
    
    return {int(key): np.array(val) for key, val in res.items() if key != 0}


#################### START PARAMS

SEED = 3

np.random.seed(SEED)

Set = np.random.uniform(low = -15, high=5, size = 100) # all set
Set_indexes = np.arange(Set.size)

# how many objects should be in each set
dim_dict = {
    1: 10,
    2: 10,
    3: 7,
    4: 14
}


# minimized function: sum of means vy each split set
def min_func(arr):

    indexes_dict = vec_to_indexes_dict(arr)

    means = [np.mean(Set[val]) for val in indexes_dict.values()]

    return sum(means)

# zero vector with available set labels at random positions 
start_solution = np.zeros(Set.size, dtype = np.int8)
labels_vec = counts_to_vec(dim_dict)
start_solution[np.random.choice(Set_indexes, labels_vec.size, replace=False)] = labels_vec


def choice(count = 3):
    return np.random.choice(Set_indexes, count, replace=False)

# mutation function
# temperature is the number cuz we will use only 1 cooling, but it's not necessary to use it)
def mut(x_as_array, temperature_as_array_or_one_number):

    new_x_as_array = x_as_array.copy()
    # replace some values

    while True:
        inds = choice()

        if np.unique(new_x_as_array[inds]).size == 1: # there is no sense to replace same values
            continue
        new_x_as_array[inds] = new_x_as_array[np.random.permutation(inds)]

        return new_x_as_array

# creating a model
model = SimulatedAnnealing(min_func, Set_indexes.size)

# run search
best_solution, best_val = model.run(
    start_solution = start_solution,
    mutation = mut,
    cooling = Cooling.exponential(0.9), 
    start_temperature = 100, 
    max_function_evals = 1000, 
    max_iterations_without_progress = 100, 
    step_for_reinit_temperature = 80,
    seed = SEED,
    reinit_from_best=True
    )

print(best_solution)
print(best_val)

model.plot_report(save_as = 'best_split.png')

