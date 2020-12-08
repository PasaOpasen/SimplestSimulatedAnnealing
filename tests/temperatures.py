# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:46:22 2020

@author: qtckp
"""

import sys
sys.path.append('..')


from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling


# simplest way to set cooling regime
temperature = 100
cooling = Cooling.reverse(beta = 0.001)
# we can temperature behaviour using this code
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'reverse.png')


# we can set several temparatures (for each dimention)
temperature = [150, 100, 50]
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'reverse_diff_temp.png')


# or several coolings (for each dimention)
temperature = 100
cooling = [
    Cooling.reverse(beta = 0.0001),
    Cooling.reverse(beta = 0.0005),
    Cooling.reverse(beta = 0.001)
    ]
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'reverse_diff_beta.png')


# all supported coolling regimes

temperature = 100
cooling = [
    Cooling.linear(mu = 1),
    Cooling.reverse(beta = 0.0007),
    Cooling.exponential(alpha = 0.85),
    Cooling.linear_reverse(),
    Cooling.logarithmic(c = 100, d = 1)
    ]
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'diff_temp.png')



# and we can set own temperature and cooling for each dimention

temperature = [100, 125, 150]
cooling = [
    Cooling.exponential(alpha = 0.85),
    Cooling.exponential(alpha = 0.9),
    Cooling.exponential(alpha = 0.95),
    ]
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'diff_temp_and_cool.png')





