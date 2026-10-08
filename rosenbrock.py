

def rosenbrock_function(x):
    """
    the rosenbrock function from R^2 -> R
    f(x) = 100(x_2 - x_1^2) + (1-x_1)^2
    """
    return 100*(x[1] - x[0]^2) + (1-x[0])^2


# TODO: maybe define the Opt problem here already?