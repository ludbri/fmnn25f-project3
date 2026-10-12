
import numpy as np
from time import perf_counter_ns

from optproblem_class import OptProblem, numerical_multivariate_hessian


class OptMethod:
    '''
    General class for optmization method, this class is inherited from by all optimizing methods

    Parameters
    ----------    
        prob: OptProblem
            The Optimization Problem to solve
    '''
    def __init__(self, prob: OptProblem):
        self.prob = prob
        self.gradient = self.prob.grad
        self.hessian = numerical_multivariate_hessian(self.prob.f)

    def specific_solve(self):
        """
        Defined in each subclass
        """
        raise NotImplementedError()

    def solve(self, x0, residual_tol, cauchy_tol, max_steps) -> np.array:
        '''
        Solve the optimization problem. This function handles stopping criteria

        x0: float
            The starting solution

        residual tol is that the change in function value between iterations should be sufficiently small.
        cauchy tol is that the change in solution value between iterations should be sufficiently small.
        '''
        fdiff = float('inf')
        xdiff = float('inf')
        self.x = np.asarray(x0, dtype=float)  # current solution
        self.steps = 0
        self.x_history = [self.x.copy()]  # history of x values
        self.f_history = [self.prob(self.x)]
        self.exec_time_history = [0]

        while fdiff > residual_tol and xdiff > cauchy_tol and self.steps < max_steps:
            # print("iteration", self.steps)
            self.steps += 1
            
            # take a step using subclass method
            start = perf_counter_ns()
            self.x = self.specific_solve()
            end = perf_counter_ns()

            self.x_history.append(self.x)
            self.f_history.append(self.prob(self.x))
            self.exec_time_history.append(end - start)
            
            # compute residual and cauchy tolerance
            fdiff = np.abs(self.f_history[-1] - self.f_history[-2])
            xdiff = np.linalg.norm(self.x - self.x_history[-1])

            if fdiff < residual_tol:    print("fdiff > residual_tol", fdiff)
            if xdiff < cauchy_tol:      print("xdiff > cauchy_tol", xdiff)
            if self.steps > max_steps:       print("steps < max_steps", self.steps)

        return self.x
