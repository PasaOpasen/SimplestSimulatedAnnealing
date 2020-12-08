
import random
import math

import numpy as np
import matplotlib.pyplot as plt

class SimulatedAnnealing:
    
    def __init__(self, minimized_function, dimention):
        self.f = minimized_function
        self.dim = dimention

    def __start_coolings(self, cooling, start_temperature):
        temp_type = type(start_temperature)
        if callable(cooling):
            if temp_type == int or temp_type == float:
                return cooling, start_temperature, False#[cooling]*self.dim, np.full(self.dim, start_temperature)
            if len(start_temperature) == self.dim:
                return [cooling]*self.dim, np.array(start_temperature), True
            raise Exception(f"invalid start_temperature ({start_temperature}) type or length")
        
        if len(cooling) == len(start_temperature) and len(cooling) == self.dim:
            return cooling, np.array(start_temperature), True

        raise Exception(f"invalid type or length of cooling ({cooling}) and start_temperature ({start_temperature})")


    def run(self, start_solution, 
            mutation, 
            cooling, 
            start_temperature, 
            max_function_evals = 1000, 
            max_iterations_without_progress = 100, 
            step_for_reinit_temperature = 50):

        cooling, start_temperature, make_decision_about_each_dim = self.__start_coolings(cooling, start_temperature) 


        x0 = start_solution
        fx0 = self.f(start_solution)
        temp = start_temperature.copy() if make_decision_about_each_dim else start_temperature

        best = (x0, fx0)

        current_score = [fx0]
        best_score = [fx0]
        k = 1
        func_evals = 1
        it = 0

        if make_decision_about_each_dim:
            def algol(iter_k):
                nonlocal current_score, best_score, x0, fx0, temp, best

                x1 = mutation(x0)
                fx1 = self.f(x1)

                if fx1 < fx0:
                    x0 = x1
                    fx0 = fx1
                    best = (x0, fx0)
                else:
                    f_diff = fx0 - fx1
                    exps = np.exp(f_diff / temp)
                    rands = np.random.random(self.dim)

                    mask = rands < exps
                    x0[mask] = x1[mask]
                    fx0 = self.f(x0)
                
                temp = np.array([cooling[i](temp[i], start_temperature[i], iter_k) for i in range (self.dim)])
                    
                current_score.append(fx0)
                best_score.append(best[1])

                return current_score[-1] < current_score[-2]
        else:
            def algol(iter_k):
                nonlocal current_score, best_score, x0, fx0, temp, best

                x1 = mutation(x0)
                fx1 = self.f(x1)

                if fx1 < fx0:
                    x0 = x1
                    fx0 = fx1
                    best = (x0, fx0)
                else:
                    if random.random() < math.exp((fx0-fx1)/temp):
                        x0 = x1
                        fx0 = fx1
                
                temp = cooling(temp, start_temperature, iter_k)
                    
                current_score.append(fx0)
                best_score.append(best[1])

                return current_score[-1] < current_score[-2]




        func_evals_step = 2 if make_decision_about_each_dim else 1 # because of different func evals

        while func_evals_step < max_function_evals and it < max_iterations_without_progress:
            progress = algol(k)
            k += 1
            func_evals += func_evals_step
            if progress:
                it = 0
            else:
                it += 1
                if it % step_for_reinit_temperature == 0:
                    temp = start_temperature.copy() if make_decision_about_each_dim else start_temperature

        self.best = best
        self.report_best = np.array(best_score)
        self.report = np.array(current_score)

        return self.best

    def plot_report(self):
            
        plt.plot(self.report, label = "used score")
        plt.plot(self.report_best, label = "best total score")


        plt.xlabel('Iteration')
        plt.ylabel('Minimized function')
        plt.title(f'Simulated annealing report')
        plt.legend()

        plt.show()

    @staticmethod
    def plot_temperature(cooling, start_temperature, iterations = 100):

        assert((callable(cooling) or hasattr(cooling, '__len__')) and (type(start_temperature) in (int, float) or hasattr(start_temperature, '__len__'))), f"invalid arguments types!"

        def construct_arr(func, start):
            res = [start]
            for k in range(iterations - 1):
                res.append(func(res[-1], start, k+2))
            return np.array([res])
        
        t = type(start_temperature)
        
        if callable(cooling):
            
            if t == int or t == float:
                plt.plot(construct_arr(cooling, start_temperature), label = f"Temperature")
                plt.title(f'Temperature regime. T0 = {start_temperature}')
            else:
                
                for temp in start_temperature:
                    plt.plot(construct_arr(cooling, temp), label = f"Temperature by time. T0 = {temp}")

                plt.title(f'Temperature regime')
        else:
            if t == int or t == float:
                for i, cool in enumerate(cooling):
                    plt.plot(construct_arr(cool, start_temperature), label = f"Temperature from cooling[{i}]")
                plt.title(f'Temperature regime. T0 = {start_temperature}')
            else:
                
                for i, cool in enumerate(cooling):
                    plt.plot(construct_arr(cool, start_temperature[i]), label = f"Temperature from cooling[{i}]. T0 = {temp}")
                plt.title(f'Temperature regime')


        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        
        plt.legend()

        plt.show()











