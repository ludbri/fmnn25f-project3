
import scipy
import numpy as np
from optproblem_class import OptProblem
from chebyquad_problem import chebyquad, gradchebyquad
from quasinewton_methods import BFGS, DFP, SymmetricBroyden, BadBroyden, GoodBroyden
from newtons_and_linesearch_methods import NewtonsMethod, NewtonWithLineSearchMethod, NewtonWithPowellWolfe



# ---------------- Tasks 10-11: supplied Chebyquad ----------------

def task11():
    print('\nTASK 11: Chebyquad, our BFGS vs scipy.optimize.fmin_bfgs')

    for n in (4,8,11):
        prob=OptProblem(
            chebyquad,
            input_shape=n,
            grad=gradchebyquad
        )

        methods = {
            "Good Broyden" : GoodBroyden(prob),
            "Bad Broyden" : BadBroyden(prob),
            "Symmetric Broyden" : SymmetricBroyden(prob),
            "DFP" : DFP(prob),
            "BFGS" : BFGS(prob),
            "Newton" : NewtonsMethod(prob),
            "Newton exact ls" : NewtonWithLineSearchMethod(prob),
            "Newton P-W ls": NewtonWithPowellWolfe(prob, 
                                                sigma=0.01,
                                                rho=0.1,
                                                alpha_start=1.0)
        }

        x0 = np.linspace(0,1,n)

        solve_kwargs = {
            "x0" : x0,
            "residual_tol" : 1e-8,
            "cauchy_tol" : 1e-5,
            "max_steps" : 20
        }

        solutions = {k:m.solve(**solve_kwargs) for k,m in methods.items()}

        # include scipy bfgs:
        solutions["scipy BFGS"] = scipy.optimize.fmin_bfgs(
            prob.f,
            x0,
            fprime=prob.grad,
            disp=False,
            maxiter=500
            )
        
        print("\n"*3 + "*"*40)
        print("*"*10 + f"{' Done!':^20}" + "*"*10)
        print("*"*40 + "\n"*2)
        
        print("\n\n".join((
                        f"{k}," +\
                        (f"\nIterations: {methods[k].steps}" if k in methods else "") +\
                        (f"\nExecution Time (s): {sum(methods[k].exec_time_history)*10**-9:.5f}" if k in methods else "") +\
                        f"\nf: {prob(x)}," +\
                        f"\nx: {x}"
                        ) for k,x in solutions.items()))

if __name__=="__main__":
    task11()
