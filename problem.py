
import numpy as np
from typing import Callable


Func = Callable[[np.array], float]
FuncGrad = Callable[[np.array], float]


def _numerical_multivariate_gradient(func: Func, eps: float = 1e-6) -> FuncGrad:
    """
    Computes the gradient of a scalar-valued multivariate function f: R^n -> R
    using the central difference formula: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2 * h).

    # TODO: vectorize with np.eye?

    Parameters
    ----------
    func : Func
        The scalar-valued function f(x) where x is a 1D numpy array.
    eps : float, optional
        Step size for numerical differentiation. Default is 1e-6 for float64 precision.

    Returns
    -------
    FuncGrad
        A callable function that takes a 1D-numpy array `x` of shape (n,)
        and returns its gradient vector ∇f(x) of shape (n,).
    """
    def func_grad(x: np.ndarray) -> np.ndarray:
        # assert x.ndim == 1 or (x.ndim==2 and x.shape[1]==1)
        x = np.asarray(x, dtype=np.float64)
        grad = np.zeros_like(x)
        e = np.zeros_like(x)
        for i in range(x.size):
            e[i] = eps
            grad[i] = (func(x + e) - func(x - e)) / (2.0*eps)
            e[i] = 0
        return grad

    return func_grad


def _numerical_multivariate_hessian(func: Func, eps: float = 1e-4) -> FuncGrad:
    """
    Computes the Hessian matrix of a scalar-valued multivariate function f: R^n -> R
    using central finite differences and applies a symmetrizing step G := 0.5 * (G + G^T).
    # TODO, make use of existing grad? - look at Tommaso's implementation below.

    Parameters
    ----------
    func : Func
        The scalar-valued function f(x) where x is an 1D numpy array.
    eps : float, optional
        Step size for numerical differentiation. Default is 1e-4, optimal for 
        second-derivative floating-point precision.

    Returns
    -------
    FuncGrad
        A callable function that takes a 1D-numpy array `x` of shape (n,)
        and returns its (n, n) Hessian matrix H.
    """
    # if grad is None:
    #     grad = _numerical_multivariate_gradient(func, eps)
    def func_hess(x: np.ndarray) -> np.ndarray:
        n = x.size
        hess = np.zeros((n, n), dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        
        f0 = func(x)
        eps_sq = eps ** 2

        # Diagonal elements: d^2f / dx_i^2
        # second order central differences
        for i in range(n):
            orig = x[i]

            x[i] = orig + eps
            f_plus = func(x)

            x[i] = orig - eps
            f_minus = func(x)

            x[i] = orig
            hess[i, i] = (f_plus - 2.0 * f0 + f_minus) / eps_sq

        # Off-diagonal elements: d^2f / (dx_i dx_j)
        # uses 4-point central differences
        for i in range(n):
            for j in range(i + 1, n):
                orig_i, orig_j = x[i], x[j]

                x[i], x[j] = orig_i + eps, orig_j + eps
                f_pp = func(x)

                x[j] = orig_j - eps
                f_pm = func(x)

                x[i], x[j] = orig_i - eps, orig_j + eps
                f_mp = func(x)

                x[j] = orig_j - eps
                f_mm = func(x)

                x[i], x[j] = orig_i, orig_j

                val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps_sq)
                hess[i, j] = val
                hess[j, i] = val

        # Check symmetry and apply symmetrizing step: G := 0.5 * (G + G^T)
        # TODO: Why do we check for symmetry, and then apply symmetrization either way?
        #       Also, with the way hess is given values above, it is always symmetric.
        if not np.allclose(hess, hess.T, rtol=1e-5, atol=1e-8):
            hess = 0.5 * (hess + hess.T)
        else:
            hess = 0.5 * (hess + hess.T)

        return hess

    return func_hess


"""  Tommaso's:
def numerical_hessian(func, h=1e-4):
    grad = numerical_gradient(func, h)
    def hessian(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        G = np.zeros((n,n))
        for j in range(n):
            e = np.zeros_like(x); e[j] = h
            G[:,j] = (grad(x + e) - grad(x - e)) / (2*h)
        return 0.5 * (G + G.T)
    return hessian
"""


class OptProblem:
    def __init__(self,
                 f: Func,
                 input_shape: int,
                 grad = None,
                 ):
        self.f = f
        self.input_shape = input_shape
        self.grad = grad if grad is not None else _numerical_multivariate_gradient(f)

    def __call__(self, x):
        return self.f(x)



