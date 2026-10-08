
from problem import OptProblem
import numpy as np


class OptMethod:
    '''
    General class for optmiziation method, this class is inherited by all optimizing methods

    Parameters
    ----------    
        prob: OptProblem
            The Optimization Problem to solve
        x0: float, optional
            The current solution
    '''
    def __init__(self, prob: OptProblem, x0, *args):
        ...
        # prob(x)  = self.prob.f(x)
        self.prob = prob
        self.x = x0  # current solution

        # values of previous solution
        self.x_prev = self.x.copy()
        self._func = getattr(self.prob, 'f', self.prob)
        self.f_prev = self._func(self.x)
        self.grad_prev = None

    def specific_solve(self):
        """
        Defined in each subclass
        """
        raise NotImplementedError()

    def solve(self, residual_tol, cauchy_tol, max_steps):
        '''
        Solve the optimization problem. This function handles stopping criteria

        residual tol is that the change in function value between iterations should be sufficiently small.
        cauchy tol is that the change in solution value between iterations should be sufficiently small.
        '''
        fdiff = float('inf')
        xdiff = float('inf')
        steps = 0

        while fdiff > residual_tol and xdiff > cauchy_tol and steps < max_steps:
            print("iteration", steps)
            steps += 1
            
            # take a step using subclass method
            x_new = self.specific_solve()
            print()
            f_new = self._func(x_new)
            
            # compute residual and cauchy tolerance
            fdiff = np.abs(f_new - self.f_prev)
            xdiff = np.linalg.norm(x_new - self.x)
            
            # Update state for the next iteration
            self.x_prev = self.x.copy()
            self.f_prev = f_new
            self.x = x_new.copy()


            if fdiff < residual_tol:    print("fdiff > residual_tol", fdiff)
            if xdiff < cauchy_tol:      print("xdiff > cauchy_tol", xdiff)
            if steps > max_steps:       print("steps < max_steps", steps)

        return self.x
    
    '''
    def hessian(self):
        """
        Estimate the hessian of the function according to the current method.

        NOTE: this is moved to the Newtons method class and the line search methods.
        """
    '''




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