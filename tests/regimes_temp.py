# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:12:21 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import math
import numpy as np
import matplotlib.pyplot as plt

from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling, simple_continual_mutation


def Rastrigin(arr):
    return 10*arr.size+np.sum(arr**2) - 10*np.sum(np.cos(2*math.pi*arr))

dim = 5

model = SimulatedAnnealing(Rastrigin, dim)

temperatures = np.arange(300, 50-1, -10)
coolings = [ 
        (Cooling.exponential(0.8), "exponential(0.8)"),
        (Cooling.exponential(0.9), "exponential(0.9)"),
        (Cooling.reverse(beta = 0.0005), "reverse(beta = 0.0005)"),
        (Cooling.linear_reverse(), "linear_reverse()"),
        (Cooling.reverse(beta = 0.001), "reverse(beta = 0.001)")
        ]

count = 50
start_solutions = np.random.uniform(-5, 5, (count, dim))


for cool, desc in coolings:
    
    t = []
    for temp in temperatures[::-1]:
        s = 0
        for i in range(count):
            _, best_val = model.run(
                start_solution = start_solutions[i, :],
                mutation = simple_continual_mutation(std = 1),
                cooling = cool, 
                start_temperature = float(temp), 
                max_function_evals = 1000, 
                max_iterations_without_progress = 250, 
                step_for_reinit_temperature = 90,
                reinit_from_best = False
                )
            s += best_val
        s /= count
        t.append(s)
    plt.plot(temperatures[::-1], t, label = f"{desc} avg. results", marker = '.', markersize = 10)
    
    
    
plt.xlabel('Temperature')
plt.ylabel('Minimized function')
plt.title('Different coolings and temperatures params')
plt.legend()

plt.savefig('regimes_temp.png', dpi = 300)

plt.show()