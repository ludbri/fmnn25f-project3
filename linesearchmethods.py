
from problem import _numerical_multivariate_hessian, _numerical_multivariate_gradient
from methods import OptMethod
import numpy as np


class NewtonsMethod(OptMethod):
    def __init__(self, prob, x0):
        super().__init__(prob, x0) 

        # the function used to estimate the hessian at some point
        self.hessian = _numerical_multivariate_hessian(self._func)
        self.gradient = self.prob.grad
        
        # if hasattr(self.prob, 'grad'):
        #     self.gradient = self.prob.grad
        # else:
        #     self.gradient = _numerical_multivariate_gradient(self._func)

    def specific_solve(self):
        # Evaluate gradient and Hessian at current x
        g = self.gradient(self.x)
        G = self.hessian(self.x)
        
        # solve linear equation G * s = -g to find the descent direction and step 's'
        step = np.linalg.solve(G, -g)
        
        return self.x + step
        

class NewtonWithLineSearchMethod(NewtonsMethod):
    def __init__(self, prob, x0):
        super().__init__(prob, x0)

    def find_descent_direction(self):
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
        phi = lambda a: self._func(self.x + a*dir)
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
        # TODO: other stopping criteria!
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
        dir = self.find_descent_direction()
        alpha = self.linesearch(dir)
        return self.x + alpha*dir


class NewtonWithPowellWolfe(NewtonWithLineSearchMethod):
    def __init__(self, prob, x0, sigma, rho, alpha_start):
        super().__init__(prob, x0)
        assert 0<sigma<0.5
        assert sigma<rho<1
        assert alpha_start > 0
        self.sigma = sigma
        self.rho = rho
        self.alpha_start = alpha_start
    
    def linesearch(self, dir):
        """
        Following the lecture slides on Powell-Wolfe.
        TODO: this may not be quite right. The assignment asks for:
         Write an inexact line search method based on the Goldstein/Wolfe conditions
         (Algorithm: Fletcher pp.34ff in the additional course materia
        """
        phi = lambda a: self._func(self.x + a*dir)
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
        method = NewtonWithLineSearchMethod(problem, x0=[-1.2, 1.0])
    else:
        method = NewtonWithPowellWolfe(problem, x0=[-1.2, 1.0], sigma=0.1, rho=0.5, alpha_start=1)

    n = 20
    residual = 1e-8
    cauchy = 1e-5
    xmin = method.solve(residual,cauchy,n)
    print("Computed minimum:", xmin)
    print("f(x):", rosenbrock(xmin))
    print("Iterations:", method.steps)

    h = np.asarray(method.history)
    xx = np.linspace(-1.5, 2, 400)
    yy = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(xx, yy)
    Z = 100*(Y-X**2)**2 + (1-X)**2

    plt.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 18))
    plt.plot(h[:,0], h[:,1], "o-")
    plt.plot(1, 1, "*", markersize=14)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(f"Rosenbrock - Newton with {"exact" if exact else "Powell-Wolfe"} line search")
    plt.tight_layout()
    plt.savefig("rosenbrock_newton_linesearch.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    from rosenbrock import rosenbrock, rosenbrock_grad, rosenbrock_problem
    import matplotlib.pyplot as plt

    test_exact_linesearch(exact=False)