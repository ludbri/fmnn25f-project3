
import numpy as np

from problem import OptProblem
from methods import OptMethod
from quasinewtonmethod import *

# TODO: use the following imports to have one shared definition?
#  - I also believe the implementation here may be the wrong version of Rosenbrock.
# from rosenbrock import rosenbrock, rosenbrock_grad

A = np.eye(2)
b = np.array([1,1])

def samplefunc(x: np.array):
    x = np.asarray(x)
    return x@A@x.T + b@x 

def rosenbrock(x: np.array):
    x = np.asarray(x)
    return x[0]**4 + x[0]**2 + x[0]*x[1] + x[1]**2 + x[0]

print(A)
print(b)
prob = OptProblem(rosenbrock, input_shape=2)

good_broyden = GoodBroyden(prob)
bad_broyden = BadBroyden(prob)
sym_broyde = SymmetricBroyden(prob)
dfp = DFP(prob)
bfgs = BFGS(prob)

kwargs = {
    "x0" : np.array([-1.2, 1.0]),
    "residual_tol" : 1e-8,
    "cauchy_tol" : 1e-5,
    "max_steps" : 20
}
print("Good Broyden")
good_broyden.solve(**kwargs)
print("Bad Broyden")
bad_broyden.solve(**kwargs)
print("Symmetric Broyden")
sym_broyde.solve(**kwargs)
print("DFP")
dfp.solve(**kwargs)
print("BFGS")
bfgs.solve(**kwargs)
# Task 12
print("Approx Error", bfgs.approx_error)