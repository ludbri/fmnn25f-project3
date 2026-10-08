
from problem import OptProblem



class OptMethod:
    '''
    
    '''
    def __init__(self, prob: OptProblem, x0, *args):
        ...
        # prob(x)  = self.prob.f(x)
        self.prob = prob
        self.x = x0  # current solution

    def solve(self):
        raise NotImplementedError()
        ...

    def num_hessian(self, x):
        """
        Numerically estimate the hessian of the function.
        """
        ...



class Newtons(OptMethod):
    def __init__(self, f, *args):
        super.__init__(self, f, args)

    def solve(self):
        # newton method
        ...



class NewtonWithLineSearch(OptMethod):
    def __init__(self, f, *args):
        super.__init__(self, f, args)

    def find_descent_direction(self):
        pass

    def linesearch(self, dir):
        pass

    def solve(self):
        # find hessian and gradient

        # find descent direction

        # line search
        ...


class QuasiNewtonMethod(OptMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    




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