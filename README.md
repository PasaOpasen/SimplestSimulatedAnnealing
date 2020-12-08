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

## Parameters of method

## Temperature regimes

## About mutation

## Examples
