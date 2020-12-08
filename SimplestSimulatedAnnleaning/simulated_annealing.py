
import random
import math

import numpy as np
import matplotlib.pyplot as plt

class SimulatedAnnealing:
    
    def __init__(self, minimized_function, dimention):
        """
        Parameters
        ----------
        minimized_function : function np.ndarray -> float/int
            function must be minimized.
        dimention : int>0
            dimention of solution.

        Returns
        -------
        None.

        """
        assert (dimention>0 and type(dimention) == int), f"dimention should be int and more than 0, not {dimention}"

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
        
        if len(cooling) == self.dim:
            if temp_type == int or temp_type == float:
                return cooling, np.full(self.dim, start_temperature), True
            elif len(cooling) == len(start_temperature):
                return cooling, np.array(start_temperature), True

        raise Exception(f"invalid type or length of cooling ({cooling}) and start_temperature ({start_temperature})")


    def run(self, 
            start_solution, 
            mutation, 
            cooling, 
            start_temperature, 
            max_function_evals = 1000, 
            max_iterations_without_progress = 250, 
            step_for_reinit_temperature = 90,
            reinit_from_best = False):
        """
        Apply similated annealing method to minimized function

        Parameters
        ----------
        start_solution : numpy array
            solution from which it should start.
        mutation : function (array, array/number)
            function like
            def mut(x_as_array, temperature_as_array_or_one_number):
                # some code
                return new_x_as_array
        This function will create new solutions from existing

        cooling : cooling function / functions list
            cooling function or a list of them.
        start_temperature : number or number array (list/tuple)
            start temperatures. Can be one number or an array of numbers.

        max_function_evals : int, optional
            maximum number of function evaluations. The default is 1000.
        max_iterations_without_progress : int, optional
            maximum number of iterations without global progress. The default is 250.
        step_for_reinit_temperature : int, optional
            after this number of iterations without progress temperatures will be initialized as like start. The default is 90.
        reinit_from_best : boolean, optional
            Start algorithm from best solution after reinit temperatures (or from last current solution). The default is False.

        Returns
        -------
        tuple(np.ndarray, number)
            pair of best solution and best score.

        """

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

                x1 = mutation(x0, temp)
                fx1 = self.f(x1)
                progress = False

                if fx1 < fx0:
                    x0 = x1
                    fx0 = fx1

                    if fx0 < best[1]:
                        best = (x0.copy(), fx0)
                        progress = True
                else:
                    f_diff = fx0 - fx1
                    exps = np.exp(f_diff / temp)
                    rands = np.random.random(self.dim)

                    mask = rands < exps
                    x0[mask] = x1[mask]
                    fx0 = self.f(x0)

                    if fx0 < best[1]:
                        best = (x0.copy(), fx0)
                        progress = True
                
                temp = np.array([cooling[i](temp[i], start_temperature[i], iter_k) for i in range (self.dim)])
                    
                current_score.append(fx0)
                best_score.append(best[1])

                return progress
        else:
            def algol(iter_k):
                nonlocal current_score, best_score, x0, fx0, temp, best

                x1 = mutation(x0, temp)
                fx1 = self.f(x1)
                progress = False

                if fx1 < fx0:
                    x0 = x1
                    fx0 = fx1
                    if fx0 < best[1]:
                        best = (x0.copy(), fx0)
                        progress = True
                else:
                    if random.random() < math.exp((fx0-fx1)/temp):
                        x0 = x1
                        fx0 = fx1
                
                temp = cooling(temp, start_temperature, iter_k)
                    
                current_score.append(fx0)
                best_score.append(best[1])

                return progress




        func_evals_step = 2 if make_decision_about_each_dim else 1 # because of different func evals

        while func_evals < max_function_evals and it < max_iterations_without_progress:
            progress = algol(k)
            k += 1
            func_evals += func_evals_step
            if progress:
                it = 0
            else:
                it += 1
                # should reinit
                if it % step_for_reinit_temperature == 0:
                    temp = start_temperature.copy() if make_decision_about_each_dim else start_temperature
                    if reinit_from_best:
                        x0, fx0 = best

        self.best = best
        self.report_best = np.array(best_score)
        self.report = np.array(current_score)

        return self.best

    def plot_report(self, save_as = None):
        """
        plot report of algorithm working

        Parameters
        ----------
        save_as : str/None, optional
            file path to save image. The default is None.

        Returns
        -------
        None.

        """
            
        plt.plot(self.report, label = "used score")
        plt.plot(self.report_best, label = "best total score")


        plt.xlabel('Iteration')
        plt.ylabel('Minimized function')
        plt.title(f'Simulated annealing report')
        plt.legend()

        if not (save_as is None):
            plt.savefig(save_as, dpi = 300)

        plt.show()

    @staticmethod
    def plot_temperature(cooling, start_temperature, iterations = 100, save_as = None):
        """
        plot different temperature regimes depended on start temperatures and iterations count

        Parameters
        ----------
        cooling : cooling function / functions list
            cooling function or a list of them.
        start_temperature : number or number array (list/tuple)
            start temperatures. Can be one number or an array of numbers.
        iterations : int, optional
            count of plotted iterations. The default is 100.
        save_as : str or None, optional
            file path to save image. The default is None.

        """
        assert (iterations > 1 and type(iterations) == int), f"iterations param should be int type and more than 1, not {iterations}"
        assert((callable(cooling) or hasattr(cooling, '__len__')) and (type(start_temperature) in (int, float) or hasattr(start_temperature, '__len__'))), f"invalid arguments types!"

        def construct_arr(func, start):
            res = [start]
            for k in range(iterations - 1):
                res.append(func(res[-1], start, k+2))
            return np.array(res)
        
        t = type(start_temperature)
        
        if callable(cooling):
            
            if t == int or t == float:
                plt.plot(construct_arr(cooling, start_temperature), label = f"Temperature")
                plt.title(f'Cooling regime. T0 = {start_temperature}')
            else:
                
                for temp in start_temperature:
                    plt.plot(construct_arr(cooling, temp), label = f"Temperature by time. T0 = {temp}")

                plt.title(f'Cooling regime')
        else:
            if t == int or t == float:
                for i, cool in enumerate(cooling):
                    plt.plot(construct_arr(cool, start_temperature), label = f"Temperature from cooling[{i}]")
                plt.title(f'Cooling regime. T0 = {start_temperature}')
            else:
                
                for i, cool in enumerate(cooling):
                    plt.plot(construct_arr(cool, start_temperature[i]), label = f"Temperature from cooling[{i}]. T0 = {start_temperature[i]}")
                plt.title(f'Cooling regime')


        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        
        plt.legend()

        if not (save_as is None):
            plt.savefig(save_as, dpi = 300)

        plt.show()











