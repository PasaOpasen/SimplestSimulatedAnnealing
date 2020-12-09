# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:40:06 2020

@author: qtckp
"""

import sys
sys.path.append('..')


#
# source of files:
# http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/berlin52.tsp
# http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/berlin52.opt.tour
#

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling

SEED = 1
np.random.seed(SEED)

# read coordinates
coords = pd.read_csv('berlin52_coords.txt', sep=' ', header= None, names = ['index', 'x', 'y'])

# dim is equal to count of cities
dim = coords.shape[0]

# distance matrix
distances = np.empty((dim, dim))

for i in range(dim):
    distances[i,i] = 0
    for j in range(i+1, dim):
        d = math.sqrt(np.sum((coords.iloc[i, 1:]-coords.iloc[j, 1:])**2))
        distances[i,j] = d
        distances[j,i] = d


indexes = np.arange(dim)
# some start solution (indexes shuffle)
start_solution = np.random.choice(indexes, dim, replace = False)


# minized function
def way_length(arr):
    s = 0
    for i in range(1, dim):
        s += distances[arr[i-1], arr[i]]
    # also we should end the way in the beggining
    s += distances[arr[-1], arr[1]]

    return s

def mut(x_as_array, temperature_as_array_or_one_number):
    # random indexes
    rand_inds = np.random.choice(indexes, 3, replace = False)
    # shuffled indexes
    goes_to = np.random.permutation(rand_inds)

    # just replace some positions in the array
    new_x_as_array = x_as_array.copy()
    new_x_as_array[rand_inds] = new_x_as_array[goes_to]

    return new_x_as_array



def plotData(indices, title, save_as = None):
    
    # create a list of the corresponding city locations:
    locs = [coords.iloc[i, 1:] for i in indices]
    locs.append(coords.iloc[indices[0], 1:])

    # plot a line between each pair of consequtive cities:
    plt.plot(*zip(*locs), linestyle='-', color='blue')
    
    # plot the dots representing the cities:
    plt.scatter(coords.iloc[:, 1], coords.iloc[:, 2], marker='o', s = 40, color='red')    
    plt.title(title)
    
    if not (save_as is None):  plt.savefig(save_as, dpi = 300)

    plt.show()


# let's plot start solution
plotData(start_solution, f'start random solution (score = {round(way_length(start_solution), 2)})','salesman_start.png')



# creating a model
model = SimulatedAnnealing(way_length, dim)

# run search
best_solution, best_val = model.run(
    start_solution = start_solution,
    mutation = mut,
    cooling = Cooling.exponential(0.9), 
    start_temperature = 100, 
    max_function_evals = 15000, 
    max_iterations_without_progress = 2000, 
    step_for_reinit_temperature = 80,
    reinit_from_best = True,
    seed = SEED
    )

model.plot_report(save_as = 'best_salesman.png')


plotData(best_solution, f'result solution (score = {round(best_val, 2)})','salesman_result.png')


