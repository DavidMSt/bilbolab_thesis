from __future__ import annotations  # Optional, works either way
import enum
import typing

# Constants to indicate whether a parameter is required or optional.
REQUIRED = True
OPTIONAL = False


class Callback:
    inputs: dict
    lambdas: dict
    parameters: dict
    function: typing.Callable
    once: bool

    container: CallbackContainer

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 function: typing.Callable,
                 inputs: dict = None,
                 lambdas: dict = None,
                 parameters: dict = None,
                 discard_inputs: bool = False,
                 once=False,
                 *args, **kwargs):
        self.function = function

        self.inputs = inputs or {}
        self.lambdas = lambdas or {}
        self.parameters = parameters or {}
        self.discard_inputs = discard_inputs
        self.once = once

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        lambdas_exec = {key: value() for (key, value) in self.lambdas.items()}

        if self.once and self.container is not None:
            self.container.remove(self)

        if self.discard_inputs:
            ret = self.function(**{**self.inputs, **lambdas_exec})
        else:
            ret = self.function(*args, **{**self.inputs, **kwargs, **lambdas_exec})
        return ret


class CallbackContainer:
    callbacks: list[Callback]

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parameters=None):
        self.callbacks = []
        self.parameters = parameters

        if parameters is not None:
            if not isinstance(parameters, list):
                raise ValueError("Parameters must be provided as a list of tuples.")
            expected = {}
            for param_spec in parameters:
                if not (isinstance(param_spec, tuple) and len(param_spec) == 3):
                    raise ValueError("Each parameter specification must be a tuple of (str, type, bool).")
                name, expected_type, required_flag = param_spec
                if not isinstance(name, str):
                    raise ValueError("Parameter name must be a string.")
                if not isinstance(required_flag, bool):
                    raise ValueError("Parameter required flag must be a boolean.")
                if not isinstance(expected_type, type):
                    raise ValueError("Parameter expected type must be a type.")
                expected[name] = (expected_type, required_flag)
            self.expected_parameters = expected
        else:
            self.expected_parameters = {}

    # ------------------------------------------------------------------------------------------------------------------
    def register(self,
                 function: typing.Callable,
                 inputs: dict = None,
                 parameters: dict = None,
                 lambdas: dict = None,
                 discard_inputs=False,
                 once=False,
                 *args, **kwargs):
        if self.expected_parameters:
            if parameters is None:
                parameters = {}
            for param_name, (expected_type, required_flag) in self.expected_parameters.items():
                if param_name not in parameters:
                    if required_flag:
                        raise RuntimeError(f"Missing required parameter: {param_name}")
                    else:
                        parameters[param_name] = None
                else:
                    if parameters[param_name] is not None and not isinstance(parameters[param_name], expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' is expected to be of type {expected_type.__name__}, "
                            f"got {type(parameters[param_name]).__name__}."
                        )

        callback = Callback(function, inputs=inputs, lambdas=lambdas, parameters=parameters,
                            discard_inputs=discard_inputs, once=once, *args, **kwargs)
        self.callbacks.append(callback)
        callback.container = self
        return callback

    # ------------------------------------------------------------------------------------------------------------------
    def remove(self, callback):
        if isinstance(callback, Callback):
            self.callbacks.remove(callback)
        elif callable(callback):
            cb = next((cb for cb in self.callbacks if cb.function == callback), None)
            if cb is not None:
                self.callbacks.remove(cb)

    # ------------------------------------------------------------------------------------------------------------------
    def call(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def __iter__(self):
        return iter(self.callbacks)

    # ------------------------------------------------------------------------------------------------------------------
    def clear_callbacks(self):
        self.callbacks.clear()


def callback_definition(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        resolved_annotations = typing.get_type_hints(cls)
        for name, annotation in resolved_annotations.items():
            if getattr(self, name, None) is None:
                if isinstance(annotation, type) and issubclass(annotation, CallbackContainer):
                    setattr(self, name, CallbackContainer())
        if original_init:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


class CallbackGroup:
    def clearAllCallbacks(self):
        resolved_annotations = typing.get_type_hints(self.__class__)
        for name, annotation in resolved_annotations.items():
            attr = getattr(self, name, None)
            if isinstance(attr, CallbackContainer):
                attr.clear_callbacks()


@callback_definition
class CallbacksA(CallbackGroup):
    callback1: CallbackContainer
    callback2: CallbackContainer


if __name__ == "__main__":
    instance = CallbacksA()

    instance.callback1.register(lambda: print("Callback 1 executed"))
    instance.callback2.register(lambda: print("Callback 2 executed"))

    print("Number of callbacks before clearing:")
    print(len(instance.callback1.callbacks), len(instance.callback2.callbacks))

    instance.clearAllCallbacks()

    print("Number of callbacks after clearing:")
    print(len(instance.callback1.callbacks), len(instance.callback2.callbacks))
