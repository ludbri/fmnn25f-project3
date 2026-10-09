from linesearchmethods import NewtonWithLineSearchMethod
import numpy as np
import scipy

class QuasiNewtonMethod(NewtonWithLineSearchMethod):
    def __init__(self, prob):
        super().__init__(prob)
        self.H = np.eye(self.prob.input_shape)

    def specific_solve(self):
        """
        Defined in each subclass
        """        
        def rosenbrock_grad(x):
            x = np.asarray(x)
            return np.asarray([4*x[0]**3 + 2*x[0] + x[1] + 1, x[0] + 2*x[1]])

        s_k = -1*(self.H@self.gradient(self.x))

        #TODO Remove scipy.optmizie.line_search when the line search is done
        print(np.shape(s_k), np.shape(self.x), type(self.prob.f), type(rosenbrock_grad))
        alfa_k = scipy.optimize.line_search(self.prob.f, rosenbrock_grad, self.x, s_k)[0]
        print("s_k", s_k)
        print(alfa_k)
        x_new = self.x + alfa_k*s_k
        print("x", self.x)
        self.update_hessian(x_new)
        print("H:", self.H)

        return x_new

    
    def update_hessian(self):
        raise NotImplementedError()

    def find_descent_direction(self):
        pass

    def linesearch(self, dir):
        pass



class GoodBroyden(QuasiNewtonMethod):
    def update_hessian(self, x_new):
        delta_k = x_new - self.x # x_(k+1) - x_k
        gamma_k = self.gradient(x_new) - self.gradient(self.x)
        numerator = delta_k@(self.H@gamma_k)
        self.H = self.H + (np.outer((delta_k - self.H@gamma_k), (delta_k@self.H))/(numerator))
        return self

class BadBroyden(QuasiNewtonMethod):
    def update_hessian(self, x_new):
            delta_k = x_new - self.x 
            gamma_k = self.gradient(x_new) - self.gradient(self.x)
    
            self.H = self.H + (np.outer(delta_k - self.H @ gamma_k, gamma_k)/(gamma_k@gamma_k))


class SymmetricBroyden(QuasiNewtonMethod):
    def update_hessian(self, x_new):
        delta_k = x_new - self.x 
        gamma_k = self.gradient(x_new) - self.gradient(self.x)

        u_k = delta_k - self.H@gamma_k
        a_k = 1/(u_k@gamma_k)
        
        self.H = self.H + a_k*np.outer(u_k, u_k)


class DFP(QuasiNewtonMethod):
    def update_hessian(self, x_new):
        delta_k = x_new - self.x 
        gamma_k = self.gradient(x_new) - self.gradient(self.x)

        self.H = (self.H + 
                    np.outer(delta_k, delta_k)/(delta_k@gamma_k) - 
                    np.outer(self.H @ gamma_k, gamma_k @ self.H)/(gamma_k@self.H@gamma_k))


class BFGS(QuasiNewtonMethod):
    def __init__(self, prob):
        super().__init__(prob)
        self.approx_error = []

    def update_hessian(self, x_new):
        delta_k = x_new - self.x 
        gamma_k = self.gradient(x_new) - self.gradient(self.x)

        term_1 = ((1 + (gamma_k@self.H@gamma_k)/(delta_k@gamma_k))/(delta_k@gamma_k))*np.outer(delta_k, delta_k)
        term_2 = (np.outer(delta_k,gamma_k@self.H) + np.outer(self.H@gamma_k, delta_k))/(delta_k@gamma_k)
    
        self.H = self.H + term_1 - term_2 
        self.approx_error.append(self.test_BFGS_approx(x_new))

    def test_BFGS_approx(self, x_new):
        exact_H =scipy.differentiate.hessian(self.prob.f, x_new)
        exact_H = scipy.linalg.inv(exact_H.ddf)

        approx_error = np.linalg.norm(exact_H - self.H, 'fro')
        return approx_error


        

        