from methods import OptMethod 
from linesearchmethods import NewtonWithLineSearchMethod
import numpy as np
import scipy

class QuasiNewtonMethod(NewtonWithLineSearchMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)
        self.H = np.eye(self.prob.input_shape)

    def specific_solve(self):
        """
        Defined in each subclass
        """
        #TODO: Stopping criterion
        
        s_k = -1*(self.H@self.gradient(self.x))
        alfa_k = 0.1 # finns det verkligen linesearch??
        print("s_k", s_k)
        x_new = self.x + alfa_k*s_k
        print("x", self.x)
        self.update_hessian()
        print("H:", self.H)

        return x_new

    
    def update_hessian(self):
        raise NotImplementedError()

    def find_descent_direction(self):
        pass

    def linesearch(self, dir):
        pass



class GoodBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
            super().__init__(f, *args)

    def update_hessian(self):
        delta_k = self.x - self.x_prev # x_(k+1) - x_k
        gamma_k = self.gradient(self.x) - self.gradient(self.x_prev)
        numerator = delta_k@(self.H@gamma_k)
        self.H = self.H + (np.outer((delta_k - self.H@gamma_k), (delta_k@self.H))/(numerator))
        return self

class BadBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def update_hessian(self):
            delta_k = self.x - self.x_prev # x_(k+1) - x_k
            gamma_k = self.gradient(self.x) - self.gradient(self.x_prev)
    
            self.H = self.H + (np.outer(delta_k - self.H @ gamma_k, gamma_k)/(gamma_k@gamma_k))


class SymmetricBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def update_hessian(self):
        delta_k = self.x - self.x_prev # x_(k+1) - x_k
        gamma_k = self.gradient(self.x) - self.gradient(self.x_prev)

        u_k = delta_k - self.H@gamma_k
        a_k = 1/(u_k@gamma_k)
        
        self.H = self.H + a_k*np.outer(u_k, u_k)


class DFP(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def update_hessian(self):
        delta_k = self.x - self.x_prev # x_(k+1) - x_k
        gamma_k = self.gradient(self.x) - self.gradient(self.x_prev)

        self.H = (self.H + 
                    np.outer(delta_k, delta_k)/(delta_k@gamma_k) - 
                    np.outer(self.H @ gamma_k, gamma_k @ self.H)/(gamma_k@self.H@gamma_k))


class BFGS(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)
        self.approx_error = np.array([])

    def update_hessian(self):

        delta_k = self.x - self.x_prev # x_(k+1) - x_k
        gamma_k = self.gradient(self.x) - self.gradient(self.x_prev)

        term_1 = ((1 + (gamma_k@self.H@gamma_k)/(delta_k@gamma_k))/(delta_k@gamma_k))*np.outer(delta_k, delta_k)
        term_2 = (np.outer(delta_k,gamma_k@self.H) + np.outer(self.H@gamma_k, delta_k))/(delta_k@gamma_k)
    
        self.H = self.H + term_1 - term_2 
        np.append(self.approx_error, self.test_BFGS_approx())

    def test_BFGS_approx(self):
        exact_H =scipy.linalg.inv(scipy.differentiate.hessian(self.prob.f, self.x))
        approx_error = np.norm(exact_H - self.H, 'fro')
        return approx_error


        

        