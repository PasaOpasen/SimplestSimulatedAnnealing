# Simplest simulated annealing

Simplest implementation of simulated annealing method

- [Simplest simulated annealing](#simplest-simulated-annealing)
  - [Idea of method](#idea-of-method)
  - [Simple usage](#simple-usage)
  - [Parameters of method](#parameters-of-method)
  - [Temperature regimes](#temperature-regimes)
    - [Pattern](#pattern)
    - [Available functions](#available-functions)
    - [Plot temperature](#plot-temperature)
    - [Multiple coolings](#multiple-coolings)
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

Main method of the package is `run()`. Let's check it's arguments:

```python
model. run( start_solution, 
            mutation, 
            cooling, 
            start_temperature, 
            max_function_evals = 1000, 
            max_iterations_without_progress = 250, 
            step_for_reinit_temperature = 90,
            reinit_from_best = False)
```

Where:
* `start_solution` : numpy array; solution from which it should start.
* `mutation` : function (array, array/number). Function like
  ```python
  def mut(x_as_array, temperature_as_array_or_one_number):
      # some code
      return new_x_as_array
  ```
    This function will create new solutions from existing. [See also](#about-mutation)

* `cooling` : cooling function / functions list. Cooling function or a list of ones. [See](#temperature-regimes)
* `start_temperature` : number or number array (list/tuple). Start temperatures. Can be one number or an array of numbers.

* `max_function_evals` : int, optional. Maximum number of function evaluations. The default is 1000.
* `max_iterations_without_progress` : int, optional. Maximum number of iterations without global progress. The default is 250.
* `step_for_reinit_temperature` : int, optional. After this number of iterations without progress temperatures will be initialized as like start. The default is 90.
* `reinit_from_best` : boolean, optional. Start algorithm from best solution after reinit temperatures (or from last current solution). The default is False.

## Temperature regimes

### Pattern

The important part of algorithm is **cooling function**. This function controls temperature value depended on current iteration number, current temperature and start temperature. U can create your own cooling function using pattern:

```python
def func(T_last, T0, k):
    # some code
    return T_new
```
where:

* `T_last` - last temperature (int/float)
* `T0` - start temperature (int/float)
* `k` - iteration number (int > 0)

It's highly recommended to build your function to create only positive temperature. 

### Available functions

In `Cooling` class there are several cooling functions:
* `Cooling.linear(mu, Tmin = 0.01)`
* `Cooling.exponential(alpha = 0.9)`
* `Cooling.reverse(beta = 0.0005)`
* `Cooling.logarithmic(c, d = 1)` - not recommended
* `Cooling.linear_reverse()`

### Plot temperature

U can see the behavior of cooling function using `SimulatedAnnealing.plot_temperature` method. Let's see several examples: 

```python
from SimplestSimulatedAnnleaning import SimulatedAnnealing, Cooling


# simplest way to set cooling regime
temperature = 100
cooling = Cooling.reverse(beta = 0.001)
# we can temperature behaviour using this code
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'reverse.png')
```
![](tests/reverse.png)


```python
# we can set several temparatures (for each dimention)
temperature = [150, 100, 50]
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'reverse_diff_temp.png')
```
![](tests/reverse_diff_temp.png)


```python
# or several coolings (for each dimention)
temperature = 100
cooling = [
    Cooling.reverse(beta = 0.0001),
    Cooling.reverse(beta = 0.0005),
    Cooling.reverse(beta = 0.001)
    ]
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'reverse_diff_beta.png')
```
![](tests/reverse_diff_beta.png)


```python
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
```
![](tests/diff_temp.png)


```python
# and we can set own temperature and cooling for each dimention!

temperature = [100, 125, 150]
cooling = [
    Cooling.exponential(alpha = 0.85),
    Cooling.exponential(alpha = 0.9),
    Cooling.exponential(alpha = 0.95),
    ]
SimulatedAnnealing.plot_temperature(cooling, temperature, iterations = 100, save_as = 'diff_temp_and_cool.png')
```
![](tests/diff_temp_and_cool.png)


### Multiple coolings

It's amazing feature to **use different cooling and start temperature for each dimension**:

```python
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
```
![](tests/different_coolings.png)

## About mutation

# Examples
