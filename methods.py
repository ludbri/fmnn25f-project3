
from problem import OptProblem


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
        self.x_prev = x0
        self.f_prev = self.prob(x0)

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
        fdiff = 0
        xdiff = 0
        steps = 0
        while fdiff <= residual_tol and xdiff <= cauchy_tol and steps < max_steps:
            steps += 1
            ... = self.specific_solve()

        return None
    
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