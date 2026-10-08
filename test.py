
import numpy as np

from problem import OptProblem
from methods import OptMethod
from quasinewtonmethod import GoodBroyden

def samplefunc(x: np.array):
    x = np.asarray(x)
    A = np.arange(4).reshape(2,2)
    b = np.array([1,1])

    return x@A@x.T + b@x 

prob = OptProblem(samplefunc, input_shape=2)
gb = GoodBroyden(prob, np.array([1.0, 1.0]))

print(gb.specific_solve())
