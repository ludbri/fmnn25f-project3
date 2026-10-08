

from problem import OptProblem, _numerical_multivariate_hessian, _numerical_multivariate_gradient
from methods import OptMethod
import numpy as np




class NewtonsMethod(OptMethod):
    def __init__(self, prob, x0, *args):
        super().__init__(prob, x0, *args)

        # the function used to estimate the hessian at some point
        self.hessian = _numerical_multivariate_hessian(self._func)
        
        if hasattr(self.prob, 'grad'):
            self.gradient = self.prob.grad
        else:
            self.gradient = _numerical_multivariate_gradient(self._func)

    def specific_solve(self, x):
        # Evaluate gradient and Hessian at x
        g = self.gradient(x)
        G = self.hessian(x)
        
        # solve linear equation G * s = -g for the step direction 's'
        step = np.linalg.solve(G, -g)
        
        return self.x + step
        



class NewtonWithLineSearchMethod(NewtonsMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def find_descent_direction(self):
        pass

    def linesearch(self, dir):
        pass

    def specific_solve(self):
        # find hessian and gradient

        # find descent direction

        # line search
        ...


