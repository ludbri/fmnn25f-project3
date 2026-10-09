
import numpy as np

from problem import OptProblem

def rosenbrock(x):
    """
    the rosenbrock function from R^2 -> R:
    f(x) = 100(x_2 - x_1^2)^2 + (1-x_1)^2.
    """
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def rosenbrock_grad(x):
    """
    The gradient of the rosenbrock function.
    """
    return np.array([-400* x[0] * (x[1 ]- x[0]**2) - 2 * (1-x[0]),
                     200* (x[1] - x[0]**2)])


rosenbrock_problem = OptProblem(rosenbrock, input_shape=2, grad=rosenbrock_grad)
