
import numpy as np

from problem import OptProblem
from methods import OptMethod
from quasinewtonmethod import *

# TODO: use following imports to have one definition?
#  - I believe the implementation here may be the wrong version of Rosenbrock.
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

good_broyden = GoodBroyden(prob, np.array([1.0, 1.0]))
bad_broyden = BadBroyden(prob, np.array([1.0, 1.0]))
sym_broyde = SymmetricBroyden(prob, np.array([1.0, 1.0]))
dfp = DFP(prob, np.array([1.0, 1.0]))
bfgs = BFGS(prob, np.array([1.0, 1.0]))

n = 20
residual = 1e-8
cauchy = 1e-5
print("Good Broyden")
good_broyden.solve(residual,cauchy,n)
print("Bad Broyden")
bad_broyden.solve(residual,cauchy,n)
print("Symmetric Broyden")
sym_broyde.solve(residual,cauchy,n)
print("DFP")
dfp.solve(residual,cauchy,n)
print("BFGS")
bfgs.solve(residual,cauchy,n)
# Task 12
print("Approx Error", bfgs.approx_error)