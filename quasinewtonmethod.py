from methods import OptMethod 

class QuasiNewtonMethod(OptMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)

    def specific_solve(self):
            """
            Defined in each subclass
            """
            raise NotImplementedError()


class GoodBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
            super().__init__(f, *args)


class BadBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)


class SymmetricBroyden(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)


class DFP(QuasiNewtonMethod):
    def __init__(self, f, *args):
        super().__init__(f, *args)


class BFGS(QuasiNewtonMethod):
     def __init__(self, f, *args):
        super().__init__(f, *args)

        