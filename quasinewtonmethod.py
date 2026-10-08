from methods import OptMethod 
from linesearchmethods import NewtonWithLineSearchMethod
import numpy as np


class QuasiNewtonMethod(NewtonWithLineSearchMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)
        self.H = np.ones(self.prob.input_shape)

    def specific_solve(self):
        """
        Defined in each subclass
        """
        #TODO: Stopping criterion
        max_iterations = 10
        i = 0
        while(i<max_iterations):
            s_k = -1*self.H*self.prob.grad(self.x)
            alfa_k = self.linesearch() # finns det verkligen linesearch??
            self.x = self.x + alfa_k*s_k
            self.H = self.update_hessian()
            i+=1

    
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
        gamma_k = self.prob.grad(self.x) - self.prob.grad(self.x_prev)

        self.H = self.H + (np.outer((delta_k - self.H@gamma_k), (delta_k@self.H))/(delta_k@self.H@gamma_k))



class BadBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def update_hessian(self):
            delta_k = self.x - self.x_prev # x_(k+1) - x_k
            gamma_k = self.prob.grad(self.x) - self.prob.grad(self.x_prev)
    
            self.H = self.H + (np.outer(delta_k - self.H @ gamma_k, gamma_k)/(gamma_k@gamma_k))


class SymmetricBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def update_hessian(self):
        delta_k = self.x - self.x_prev # x_(k+1) - x_k
        gamma_k = self.prob.grad(self.x) - self.prob.grad(self.x_prev)

        u_k = delta_k - self.H@gamma_k
        a_k = 1/(u_k@gamma_k)
        
        self.H = self.H + a_k*np.outer(u_k, u_k)


class DFP(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def update_hessian(self):
        delta_k = self.x - self.x_prev # x_(k+1) - x_k
        gamma_k = self.prob.grad(self.x) - self.prob.grad(self.x_prev)

        self.H = (self.H + 
                    np.outer(delta_k, delta_k)/(delta_k@gamma_k) - 
                    np.outer(self.H @ gamma_k, gamma_k @ self.H)/(gamma_k@self.H@gamma_k))


class BFGS(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def update_hessian(self):
        delta_k = self.x - self.x_prev # x_(k+1) - x_k
        gamma_k = self.prob.grad(self.x) - self.prob.grad(self.x_prev)

        term_1 = ((1 + (gamma_k@self.H@gamma_k)/(delta_k@gamma_k))/(delta_k@gamma_k))*np.outer(delta_k, delta_k)
        term_2 = (np.outer(delta_k,gamma_k@self.H) + np.outer(self.H@gamma_k, delta_k))/(delta_k@gamma_k)
    
        self.H = self.H + term_1 - term_2 
        

        