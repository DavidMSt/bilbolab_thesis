class Callback:
    parameters: dict
    lambdas: dict
    function: callable

    def __init__(self, function: callable, parameters: dict = None, lambdas: dict = None, discard_inputs: bool = False):
        self.function = function

        if parameters is None:
            parameters = {}
        self.parameters = parameters

        if lambdas is None:
            lambdas = {}
        self.lambdas = lambdas

        self.discard_inputs = discard_inputs

    def __call__(self, *args, **kwargs):
        lambdas_exec = {key: value() for (key, value) in self.lambdas.items()}

        if self.discard_inputs:
            ret = self.function(**{**self.parameters, **lambdas_exec})
        else:
            ret = self.function(*args, **{**self.parameters, **kwargs, **lambdas_exec})

        return ret

