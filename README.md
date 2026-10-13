# fmnn25f-project3
Project files for the third project in the course in advanced numerical algorithms. The purpose is to implement a set of optimization methods.

# File Structure
- `chebyquad_problem` - defines the chebyquad problem and its gradient.
- `newtons_and_linesearch_methods` - defines the `NewtonsMethod` class and classes for newtons  method with exact line search and inexact Powell-Wolfe line search.
- `optmethods_baseclass` - defines the `OptMethod` base class that is inherited by all further optimization classes.
- `optproblem_class` -  functions to find the numerical gradient and Hessian of a problem. Defines the `OptProblem` class to store optimization problems.
- `quasinewton_methods` - sub-classes of `OptMethod` that use a quasi-Newton method. The following quasi-Newton methods are implemented: Good Broyden, Bad Broyden, Symmetric Broyden, DFP, BFGS.
- `rosenbrock` - creates rosenbrock function as an `OptProblem` with a pre-defined gradient.
- `task11` - function to run results for task 11 testing all of the quasi-Newton methods and Newton method (with and without line search).
- `test` - tests the functionality of the quasi-Newton methods.

# Contributions
- Anna: created functions for the quasi-newton methods, as well as 11, 12
- Annelies: Created functions for numerical gradient, Hessian, and classical Newton method. Helped with the README
- Juan: Helped with tasks 9 and 11 as well as the documentation for the code.
- Ludwig
- Tommaso


# TODO:
- readme!
- in Optproblem TODO (Annelies)
- finishing task 12 ie plotting (Tommaso)
- documentation (Juan)
- look over the Broyden methods (Anna)
- finish task 11
- readme (Juan and Annelies)
- ipy notebook to compile answers to questions (Ludwig)
