# Simplest simulated annealing

Simplest implementation of simulated annealing method

- [Simplest simulated annealing](#simplest-simulated-annealing)
  - [Idea of method](#idea-of-method)
  - [Simple usage](#simple-usage)
  - [Parameters of method](#parameters-of-method)
  - [Temperature regimes](#temperature-regimes)
  - [About mutation](#about-mutation)
- [Examples](#examples)

## Idea of method

This is the evolutionary algorithm for *function minimization*. Steps of algorithm:
1. We should determine function `f` must be minimized
2. Determine start solution `x0` (can be random)
3. Determine mutation function `mut`. This function should give new (can be random) `x1` solution using information about `x0` and temperature `T`.
4. Select or create `cooling` regime(s) (temperature behavior)
5. Set start temperature(s) `T`
6. Run searching:
   * at start we have `x0` solution and `f(x0)` best score
   * let's create mutant `x1 = mut(x0)` and calculate `f(x1)`
   * if `f(x1) < f(x0)`, we found better solution `x0 = x1`. Otherwise we can replace `x1` with `x0` with probability equals `exp((f(x0) - f(x1)) / T)`
   * decrease `T` using `cooling` function: `T = cooling(T)`
   * repeat last 3 steps until stop criterion 

## Simple usage

Import packages:

```python
import math
import numpy as np

from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling, simple_continual_mutation
```

Determine minimized function (Rastrigin):

```python
def Rastrigin(arr):
    return 10*arr.size+np.sum(arr**2) - 10*np.sum(np.cos(2*math.pi*arr))

dim = 5
```
We will use simplest gauss mutation:

```python
mut = simple_continual_mutation(std = 0.5)
```

Create model object (set function and dimension):

```python
model = SimulatedAnnealing(Rastrigin, dim)
```

Start searching and see report:

```python
best_solution, best_val = model.run(
    start_solution = np.random.uniform(-5, 5, dim),
    mutation = mut,
    cooling = Cooling.exponential(0.9), 
    start_temperature = 100, 
    max_function_evals = 1000, 
    max_iterations_without_progress = 100, 
    step_for_reinit_temperature = 80
    )

model.plot_report(save_as = 'simple_example.png')
```

![](tests/simple_example.png)

## Parameters of method

## Temperature regimes

## About mutation

# Examples
