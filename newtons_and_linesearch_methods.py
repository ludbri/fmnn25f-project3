
from optmethods_baseclass import OptMethod
import numpy as np


class NewtonsMethod(OptMethod):
    """
    Newton's method for unconstrained optimization

    Every iteration solves ``G(x) * s = - g(x)`` and moves to ``x + s``.
    Has no step length control.

    """
    def specific_solve(self):

        # Evaluate gradient and Hessian at current x
        g = self.gradient(self.x)
        G = self.hessian(self.x)
        
        # solve linear equation G * s = -g to find the descent direction and step 's'
        step = np.linalg.solve(G, -g)
        
        return self.x + step
        

class NewtonWithLineSearchMethod(NewtonsMethod):
    """Newton's method with an exact (golden-section) line search.

    Uses the Newton direction if it is a descent direction, otherwise it
    falls back to steepest descent.
    """

    def find_descent_direction(self):
        """
        Try finding the Newton direction, if unsolvable return the
        negative gradient as direction.
        """
        grad = self.gradient(self.x)
        hess = self.hessian(self.x)
        try:
            dir = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            # if unsolvable, take negative gradient as direction.
            dir = -grad
        # if ascent direction, flip
        if np.dot(grad, dir) >= 0:
            dir = -grad
        return dir

    def linesearch(self, dir):
        """Return a step length ``alpha`` minimizing ``prob(x + alpha*dir)``
        
        Uses Golden-section minimization to find alpha, return 0.0 if no
        decreasing step is found.
        """
        phi = lambda a: self.prob(self.x + a*dir)
        # Bracket the 1-D minimum.
        b = 1.0
        while phi(b) >= phi(0.0) and b > 1e-12:
            b *= 0.5
        if b <= 1e-12:
            return 0.0
        c = 2*b
        while phi(c) < phi(b) and c < 1e6:
            b, c = c, 2*c

        # Golden-section minimization.
        left, right = 0.0, c
        r = (np.sqrt(5.0)-1.0)/2.0
        x1 = right-r*(right-left)
        x2 = left+r*(right-left)
        f1, f2 = phi(x1), phi(x2)
        for _ in range(200):
            if right-left < 1e-10:
                break
            if f1 < f2:
                right, x2, f2 = x2, x1, f1
                x1 = right-r*(right-left)
                f1 = phi(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left+r*(right-left)
                f2 = phi(x2)
        return (left+right)/2

    def specific_solve(self):
        """Return ``x + alpha * dir`` with ``alpha`` from the line search."""
        dir = self.find_descent_direction()
        alpha = self.linesearch(dir)
        return self.x + alpha*dir


class NewtonWithPowellWolfe(NewtonWithLineSearchMethod):
    """Netwon's method with Powell-Wollfe conditions
        
        Parameters:
            prob: OptProblem
                The optimization problem (objective with gradient/Hessian).\n
            sigma: float
                Armijo parameter, must satisfy ``0 < sigma < 0.5``.\n
            rho: float
                Curvature parameter, must satisfy ``sigma < rho < 1``.\n
            alpha_start: float
                 Initial trial step length, must be positive.\n
    """
    def __init__(self, prob, sigma, rho, alpha_start):
        super().__init__(prob)
        assert 0<sigma<0.5
        assert sigma<rho<1
        assert alpha_start > 0
        self.sigma = sigma
        self.rho = rho
        self.alpha_start = alpha_start
    
    def linesearch(self, dir):
        """
        Implementing a Powell-Wolfe line search based on the lecture notes.
        Note: the book by Fletcher swaps the use of sigma and rho.

        """
        phi = lambda a: self.prob(self.x + a*dir)
        phi_grad = lambda a: np.dot(self.gradient(self.x + a*dir), dir)
        f_0 = phi(0)
        f_grad_0 = phi_grad(0)

        sigma = self.sigma
        rho = self.rho
        am = self.alpha_start

        def armijo(a):
            return phi(a) <= f_0 + sigma * a * f_grad_0

        # Algorithm from lecture slides on optimization:
        while not armijo(am):
            am /= 2

        ap=am
        while armijo(ap):
            ap *= 2

        def powell_wolfe(a):
            return phi_grad(a) >= rho * f_grad_0

        while not powell_wolfe(am):
            a0 = (ap+am)/2
            if armijo(a0):
                am = a0
            else:
                ap = a0
        return am


def test_exact_linesearch(exact=True):
    problem = rosenbrock_problem
    if exact:
        method = NewtonWithLineSearchMethod(problem)
    else:
        # Implementation here uses lecture's notation.
        # Note, book uses swapped sigma and rho compared to the optimization lecture.
        method = NewtonWithPowellWolfe(problem, sigma=0.01, rho=0.1, alpha_start=0.1)

    kwargs = {
        "x0" : np.array([-1.2, 1.0]),
        "residual_tol" : 1e-8,
        "cauchy_tol" : 1e-5,
        "max_steps" : 20
    }
    xmin = method.solve(**kwargs)
    print("Computed minimum:", xmin)
    print("f(x):", rosenbrock(xmin))
    print("Iterations:", method.steps)

    h = np.asarray(method.x_history)
    xx = np.linspace(-1.5, 2, 400)
    yy = np.linspace(-1, 3, 400)
    XY = np.meshgrid(xx, yy)
    X,Y = XY
    Z = rosenbrock(XY)

    plt.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 18))
    plt.plot(h[:,0], h[:,1], "o-")
    plt.plot(1, 1, "*", markersize=14)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(f"Rosenbrock - Newton with {"exact" if exact else "Powell-Wolfe"} line search")
    plt.tight_layout()
    plt.savefig("rosenbrock_newton_linesearch.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    from rosenbrock import rosenbrock, rosenbrock_problem
    import matplotlib.pyplot as plt

    test_exact_linesearch(exact=True)
    test_exact_linesearch(exact=False)