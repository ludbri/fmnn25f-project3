
import numpy as np
from typing import Callable, Optional


Func = Callable[[np.array], float]
FuncGrad = Callable[[np.array], float]


def numerical_multivariate_gradient(func: Func, eps: float = 1e-10) -> FuncGrad:
    """
    Computes the gradient of a scalar-valued multivariate function f: R^n -> R
    using the central difference formula: ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2 * h).

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
        x_arr = np.asarray(x, dtype=np.float64)
        n = x_arr.size
        
        E = np.eye(n) * eps
        
        f_plus = np.apply_along_axis(func, axis=1, arr=x_arr+E)
        f_minus = np.apply_along_axis(func, axis=1, arr=x_arr-E)
        grad = (f_plus-f_minus)- (2*eps)
        return grad

    return func_grad


def numerical_multivariate_hessian(func: Func, eps: float = 1e-10, grad: Optional[FuncGrad] = None) -> FuncGrad:
    """
    Computes the Hessian matrix of a scalar-valued multivariate function f: R^n -> R
    using central finite differences and applies a symmetrizing step G := 0.5 * (G + G^T).

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
    # if the function given has a predefined gradient
    # calculate the hessian from this gradient
    if grad is not None:
        def func_hess(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64).ravel()
            n = x_arr.size
            E = np.eye(n) * eps
            
            # if the function given has a predefined gradient
            # calculate the hessian from this gradient
            G_plus = np.apply_along_axis(grad, axis=1, arr=x_arr + E)
            G_minus = np.apply_along_axis(grad, axis=1, arr=x_arr - E)
                    
            H = (G_plus - G_minus) / (2.0 * eps)
            return 0.5 * (H + H.T)

    else:
        def func_hess(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64).ravel()
            n = x_arr.size
            E = np.eye(n) * eps
        
            # otherwise the hessian is found from the function directly
            hess = np.zeros((n, n), dtype=np.float64)
            f0 = func(x_arr)
            
            # Diagonal elements: d^2f / dx_i^2
            f_plus = np.apply_along_axis(func, axis=1, arr=x_arr + E)
            f_minus = np.apply_along_axis(func, axis=1, arr=x_arr - E)
            np.fill_diagonal(hess, (f_plus - 2.0 * f0 + f_minus) / (eps ** 2))
            
            # Off-diagonal elements using upper-triangular index pairs
            # using 4-point central differences
            if n > 1:
                i, j = np.triu_indices(n, k=1)
                E_i, E_j = E[i], E[j]

                f_pp = np.apply_along_axis(func, axis=1, arr=x_arr + E_i + E_j)
                f_pm = np.apply_along_axis(func, axis=1, arr=x_arr + E_i - E_j)
                f_mp = np.apply_along_axis(func, axis=1, arr=x_arr - E_i + E_j)
                f_mm = np.apply_along_axis(func, axis=1, arr=x_arr - E_i - E_j)

                off_diag = (f_pp - f_pm - f_mp + f_mm) / (4.0 * (eps ** 2))
                hess[i, j] = off_diag
                hess[j, i] = off_diag
            
            return 0.5 * (hess + hess.T)

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
        self.grad = grad if grad is not None else numerical_multivariate_gradient(f)

    def __call__(self, x):
        return self.f(x)



