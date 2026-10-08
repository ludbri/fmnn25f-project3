
import numpy as np

from problem import OptProblem
from methods import OptMethod


def samplefunc(x: np.array):
    x = np.asarray(x)
    A = np.arange(4).reshape(2,2)
    A[0, 0] = 1 # without this there is no minimum
    b = np.array([1,1])

    return x@A@x.T + b@x 

prob = OptProblem(samplefunc, input_shape=2)

print(prob([0,1]))
