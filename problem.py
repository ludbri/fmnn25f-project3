
import numpy as np
from typing import Callable


Func = Callable[[np.array], float]
FuncGrad = Callable[[np.array], float]


def _numerical_scalar_gradient(func, eps: float = 1e-06) -> FuncGrad:
    '''
    Computes the elementwise derivative of an activation function
    using the central difference formula: f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h). (2nd order)
    
    Parameters
    ----------
    func : Func
        The function to numerically differentiate f(x).
    eps : float, optional
        Step size for numerical differentiation. Default is 1e-6 for float64 precision.

    Returns
    -------
    FuncGrad
        A callable function that takes a 1D-numpy array of inputs `x` 
        and returns the elementwise numerical derivative.
    '''
    def func_grad(input: np.ndarray) -> np.ndarray:
        return (func(input + eps) - func(input - eps)) / (2.0 * eps)
    return func_grad


def _numerical_multivariate_gradient(func: Func, eps: float = 1e-6) -> FuncGrad:
    """
    Numerically computes the gradient of a function with respect to y_pred
    using element-wise central difference.
    TODO: update.
    """
    def func_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(y_pred)
        rows, cols = y_pred.shape
        
        # Perturb each element in y_pred individually
        for i in range(rows):
            for j in range(cols):
                orig_val = y_pred[i, j]
                
                # Perturb +eps
                y_pred[i, j] = orig_val + eps
                loss_plus = func(y_pred, y_true)
                
                # Perturb -eps
                y_pred[i, j] = orig_val - eps
                loss_minus = func(y_pred, y_true)
                
                # Restore original value
                y_pred[i, j] = orig_val
                
                # Central difference derivative
                grad[i, j] = (loss_plus - loss_minus) / (2.0 * eps)
                
        # Scaling adjustment: square_loss computes the AVERAGE loss across batch size N (via np.mean)
        # Because learn_batch divides by batch_size again during weight update,
        # multiplying by batch_size aligns the numerical gradient scale with square_loss_gradient.
        batch_size = y_pred.shape[1]
        return grad * batch_size

    return func_grad


def _numerical_multivariate_hessian(func: Func, eps: float = 1e-6) -> FuncGrad:
    """
    Numerically computes the gradient of a function with respect to y_pred
    using element-wise central difference.
    # TODO: update.
    """
    def func_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(y_pred)
        rows, cols = y_pred.shape
        
        # Perturb each element in y_pred individually
        for i in range(rows):
            for j in range(cols):
                orig_val = y_pred[i, j]
                
                # Perturb +eps
                y_pred[i, j] = orig_val + eps
                loss_plus = func(y_pred, y_true)
                
                # Perturb -eps
                y_pred[i, j] = orig_val - eps
                loss_minus = func(y_pred, y_true)
                
                # Restore original value
                y_pred[i, j] = orig_val
                
                # Central difference derivative
                grad[i, j] = (loss_plus - loss_minus) / (2.0 * eps)
                
        # Scaling adjustment: square_loss computes the AVERAGE loss across batch size N (via np.mean)
        # Because learn_batch divides by batch_size again during weight update,
        # multiplying by batch_size aligns the numerical gradient scale with square_loss_gradient.
        batch_size = y_pred.shape[1]

        # TODO: check that the hessian is symmetric and make symmetric if not.

        return grad * batch_size

    return func_grad


class OptProblem:
    def __init__(self,
                 f: Func,
                 input_shape: int,
                 grad = None):
        self.f = f
        self.input_shape = input_shape
        self.grad = grad if grad is not None else _numerical_multivariate_gradient(f)

    def __call__(self, x):
        return self.f(x)
