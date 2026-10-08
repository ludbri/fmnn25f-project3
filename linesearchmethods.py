

from problem import OptProblem, _numerical_multivariate_hessian
from methods import OptMethod





class NewtonsMethod(OptMethod):
    def __init__(self, prob, *args):
        super.__init__(self, prob, args)

        # the function used to estimate the hessian at some point
        self.hessian = _numerical_multivariate_hessian(self.prob.f)

    def specific_solve(self, x):
        # newton method
        ...
        hess = self.hessian(x)



class NewtonWithLineSearchMethod(NewtonsMethod):
    def __init__(self, f, *args):
        super.__init__(self, f, args)

    def find_descent_direction(self):
        pass

    def linesearch(self, dir):
        pass

    def specific_solve(self):
        # find hessian and gradient

        # find descent direction

        # line search
        ...


