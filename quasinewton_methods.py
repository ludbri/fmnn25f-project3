
from newtons_and_linesearch_methods import NewtonWithLineSearchMethod
import numpy as np
import scipy

class QuasiNewtonMethod(NewtonWithLineSearchMethod):
    """Base class for quasi-Newton methods with line search.

    Instead of computing the Hessian, these methods maintain ``self.H``, an
    approximation of the *inverse* Hessian (initialized to the identity),
    and update it every iteration from the observed change in x and in the
    gradient. Subclasses implement :meth:`update_hessian`.

    Parameters:
        f: The optimization problem.
        *args: Passed on to the parent class.
    """
    def __init__(self, f, verbose=True):
        super().__init__(f, verbose)
        self.H = np.eye(self.prob.input_shape)

    def specific_solve(self):
        """Take one quasi-Newton step and update ``self.H``.

        Searches along ``-H @ g`` with a line search, then calls
        :meth:`update_hessian` with the new point. Returns the new iterate.
        """ 
        
        g = self.gradient(self.x)
        
        direction = self.find_descent_direction()
        
        # broken or non-descent direction: reset to steepest descent
        if not np.all(np.isfinite(direction)) or g @ direction >= 0:
            self.H = np.eye(self.prob.input_shape)
            direction = -g
        
        
        alfa_k = self.linesearch(direction)
        
        # line search failed with a non-trivial H: retry once with steepest descent
        if alfa_k == 0.0 and not np.allclose(self.H, np.eye(self.prob.input_shape)):
            self.H = np.eye(self.prob.input_shape)
            direction = -g
            alfa_k = self.linesearch(direction)
            
        
        x_new = self.x + alfa_k*direction
        
        self.update_hessian(x_new)
        
        if not np.all(np.isfinite(self.H)):
            self.H = np.eye(self.prob.input_shape)
        
        if self.verbose:
            print("step direction: ", direction)
            print("alpha: ", alfa_k)
            print("x: ", self.x)
            print("H: ", self.H)
        
        return x_new

    
    def update_hessian(self):
        """Update the inverse Hessian approximation ``self.H``.

        Parameters:
            x_new: The next iterate, x_{k+1}.

        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def find_descent_direction(self):
        return -self.H @ self.gradient(self.x)





class GoodBroyden(QuasiNewtonMethod):
    """Broyden's "good" method.

    Rank-one update of the inverse Hessian approximation derived from
    Broyden's secant update. The update is not symmetric, so ``H`` 
    is not guaranteed to stay symmetric or positive definite.
    """
    def __init__(self, f, verbose=True):
            super().__init__(f, verbose)

    def update_hessian(self, x_new):
        """Apply the good Broyden rank-one update to ``self.H``."""
        delta_k = x_new - self.x # x_(k+1) - x_k
        gamma_k = self.gradient(x_new) - self.gradient(self.x)
        numerator = delta_k@(self.H@gamma_k)
        self.H = self.H + (np.outer((delta_k - self.H@gamma_k), (delta_k@self.H))/(numerator))
        return self


class BadBroyden(QuasiNewtonMethod):
    """Broyden's "bad" method.

    Rank-one update that satisfies the secant condition ``H @ gamma = delta``
    by correcting along ``gamma``. Usually performs worse than
    :class:`GoodBroyden`; not symmetric.
    """
    def __init__(self, f, verbose=True):
        super().__init__(f, verbose)

    def update_hessian(self, x_new):
            """Apply the bad Broyden rank-one update to ``self.H``."""
            delta_k = x_new - self.x 
            gamma_k = self.gradient(x_new) - self.gradient(self.x)
    
            self.H = self.H + (np.outer(delta_k - self.H @ gamma_k, gamma_k)/(gamma_k@gamma_k))


class SymmetricBroyden(QuasiNewtonMethod):
    """Symmetric Broyden (SR1) method.

    Symmetric rank-one update satisfying the secant condition. ``H`` stays
    symmetric but is not guaranteed to be positive definite.
    """
    def __init__(self, f, verbose=True):
        super().__init__(f, verbose)

    def update_hessian(self, x_new):
        """Apply the symmetric rank-one update to ``self.H``."""
        delta_k = x_new - self.x 
        gamma_k = self.gradient(x_new) - self.gradient(self.x)

        u_k = delta_k - self.H@gamma_k
        a_k = 1/(u_k@gamma_k)
        
        self.H = self.H + a_k*np.outer(u_k, u_k)


class DFP(QuasiNewtonMethod):
    """Davidon-Fletcher-Powell (DFP) method.

    Symmetric rank-two update of the inverse Hessian approximation. ``H``
    stays positive definite as long as ``delta @ gamma > 0``, which an
    appropriate line search guarantees.
    """
    def __init__(self, f, verbose=True):
        super().__init__(f, verbose)

    def update_hessian(self, x_new):
        """Apply the DFP rank-two update to ``self.H``."""
        delta_k = x_new - self.x 
        gamma_k = self.gradient(x_new) - self.gradient(self.x)

        self.H = (self.H + 
                    np.outer(delta_k, delta_k)/(delta_k@gamma_k) - 
                    np.outer(self.H @ gamma_k, gamma_k @ self.H)/(gamma_k@self.H@gamma_k))


class BFGS(QuasiNewtonMethod):
    """Broyden-Fletcher-Goldfarb-Shanno (BFGS) method.

    Symmetric rank-two update of the inverse Hessian approximation. The
    most widely used quasi-Newton method: it keeps ``H`` positive definite
    when ``delta @ gamma > 0`` and is more tolerant of inexact line
    searches than :class:`DFP`."""

    def __init__(self, f, verbose=True):
        super().__init__(f, verbose)
        self.approx_error = []

    def update_hessian(self, x_new):
        """Apply the BFGS rank-two update to ``self.H``."""
        delta_k = x_new - self.x 
        gamma_k = self.gradient(x_new) - self.gradient(self.x)

        term_1 = ((1 + (gamma_k@self.H@gamma_k)/(delta_k@gamma_k))/(delta_k@gamma_k))*np.outer(delta_k, delta_k)
        term_2 = (np.outer(delta_k,gamma_k@self.H) + np.outer(self.H@gamma_k, delta_k))/(delta_k@gamma_k)
    
        self.H = self.H + term_1 - term_2 
        # add an optional thing here
        #self.approx_error.append(self.test_BFGS_approx(x_new))

    def test_BFGS_approx(self, x_new):
        """
        Return the Frobenius-norm error of ``H`` versus the exact inverse Hessian at ``x_new``.
        """
        exact_H = scipy.differentiate.hessian(self.prob.f, x_new)
        exact_H = scipy.linalg.inv(exact_H.ddf)

        approx_error = np.linalg.norm(exact_H - self.H, 'fro')
        return approx_error


        

        