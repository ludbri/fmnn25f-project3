
import scipy
import numpy as np
from optproblem_class import OptProblem
from chebyquad_problem import chebyquad, gradchebyquad
from quasinewton_methods import BFGS, DFP, SymmetricBroyden, BadBroyden, GoodBroyden
from newtons_and_linesearch_methods import NewtonsMethod
from rosenbrock import rosenbrock, rosenbrock_grad



# ---------------- Tasks 10-11: supplied Chebyquad ----------------

def task11():

    print('\nTASK 11: Chebyquad, our BFGS vs scipy.optimize.fmin_bfgs')

    for n in (4,8,11):
        prob=OptProblem(
            chebyquad,
            input_shape=n,
            grad=gradchebyquad
        )

        x0 = np.linspace(0,1,n)

        methods = {
            "Good Broyden" : GoodBroyden(prob),
            "Bad Broyden" : BadBroyden(prob),
            "Symmetric Broyden" : SymmetricBroyden(prob),
            "DFP" : DFP(prob),
            "BFGS" : BFGS(prob),
            "Newton" : NewtonsMethod(prob)
        }

        solve_kwargs = {
            "x0" : x0,
            "residual_tol" : 1e-8,
            "cauchy_tol" : 1e-5,
            "max_steps" : 20
        }

        solutions = {k:m.solve(**solve_kwargs) for k,m in methods.items()}
        solutions["scipy BFGS"] = scipy.optimize.fmin_bfgs(
            prob.f,
            x0,
            fprime=prob.grad,
            disp=False,
            maxiter=500
            )

        print("\n\n".join(f"{k},\nf: {prob(x)},\nx: {x}" for k,x in solutions.items()))

task11()

