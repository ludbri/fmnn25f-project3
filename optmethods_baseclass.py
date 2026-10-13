
import numpy as np
from time import perf_counter_ns

from optproblem_class import OptProblem, numerical_multivariate_hessian


class OptMethod:
    '''
    Base class for optimization method, this class is inherited by all optimizing methods

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
        Helper method for solve, defined in each subclass.
        """
        raise NotImplementedError()

    def solve(self, x0, residual_tol, cauchy_tol, max_steps) -> np.array:
        '''
        Solve the optimization problem. This function handles stopping criteria

        x0: float
            The starting solution

        residual_tol: float
            is that the change in function value between iterations should be sufficiently small.
        cauchy_tol: float 
            is that the change in solution value between iterations should be sufficiently small.
        '''
        fdiff = float('inf')
        xdiff = float('inf')
        self.x = np.asarray(x0, dtype=float)  # current solution
        f_prev = self.prob(self.x)
        self.steps = 0
        self.x_history = [self.x.copy()]  # history of x values
        self.f_history = [f_prev.copy()]
        self.exec_time_history = [0]

        while fdiff > residual_tol and xdiff > cauchy_tol and self.steps < max_steps:
            print("iteration", self.steps)
            self.steps += 1
            
            # take a step using subclass method
            start = perf_counter_ns()
            x_new = self.specific_solve()
            f_new = self.prob(x_new)
            end = perf_counter_ns()

            self.x_history.append(x_new)
            self.f_history.append(f_new)
            self.exec_time_history.append(self.exec_time_history[-1] + end - start)
            
            # compute residual and cauchy tolerance
            fdiff = np.abs(f_new - f_prev)
            xdiff = np.linalg.norm(x_new - self.x)
            
            # Update state for the next iteration
            f_prev = f_new
            self.x = x_new.copy()


            if fdiff < residual_tol:    print("fdiff > residual_tol", fdiff)
            if xdiff < cauchy_tol:      print("xdiff > cauchy_tol", xdiff)
            if self.steps > max_steps:       print("steps < max_steps", self.steps)

        return self.x




######################################################################
#                   Example sub classing below
######################################################################
"""
class Base:
    def __init__(self, x):
        self.x = x

    def square(self):
        return self.x*self.x

    def greet(self):
        print(self.x, self.square())


class Sub(Base):
    def __init__(self, x, y):
        super().__init__(x)
        self.y = y

    def greet(self):
        print(self.x, self.y, self.square())


if __name__=="__main__":
    b = Base(1)
    s = Sub(1,2)

    b.greet()
    s.greet()
"""