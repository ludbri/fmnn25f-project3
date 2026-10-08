
import numpy as np

from problem import OptProblem
from methods import OptMethod
from quasinewtonmethod import *

A = np.eye(2)
b = np.array([1,1])

def samplefunc(x: np.array):
    x = np.asarray(x)
    return x@A@x.T + b@x 

print(A)
print(b)
prob = OptProblem(samplefunc, input_shape=2)

good_broyden = GoodBroyden(prob, np.array([1.0, 1.0]))
bad_broyden = BadBroyden(prob, np.array([1.0, 1.0]))
sym_broyde = SymmetricBroyden(prob, np.array([1.0, 1.0]))
dfp = DFP(prob, np.array([1.0, 1.0]))
# bfgs = BFGS(prob, np.array([1.0, 1.0]))

n = 20
residual = 1e-18
cauchy = 1e-18
print("Good Broyden", good_broyden.solve(residual,cauchy,n))
print("Bad Broyden", bad_broyden.solve(residual,cauchy,n))
print("Symmetric Broyden", sym_broyde.solve(residual,cauchy,n))
print("DFP", dfp.solve(residual,cauchy,n))
# print("BFGS", bfgs.solve(residual,cauchy,n))