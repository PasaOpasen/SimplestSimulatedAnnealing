
import warnings
import math

class Cooling:
    @staticmethod
    def linear(mu, Tmin = 0.01):

        def func(T, T0, k):
            return max(T0 - mu*k, Tmin)
        return func

    @staticmethod
    def exponential(alpha = 0.9):

        if alpha < 0.8 or alpha > 1:
            warnings.warn(f"for exponential cooling it's recommented to use alpha from [0.8, 1], not {alpha}")
        
        def func(T, T0, k):
            return T*alpha # T0*pow(alpha, k)
        return func

    @staticmethod
    def reverse(beta = 0.0005):

        if beta < 0.0001 or beta > 0.002:
            warnings.warn(f"it can be not good to use beta not from [0.0001, 0.0005]")

        def func(T, T0, k):
            return T/(1+beta*T)
        return func

    @staticmethod
    def logarithmic(c, d = 1):

        assert (d > 0), f"d param should be more than 0, not {d}"

        def func(T, T0, k):
            return c/math.log(k + d)
        return func
    
    @staticmethod
    def linear_reverse():

        def func(T, T0, k):
            return T0/k
        return func




