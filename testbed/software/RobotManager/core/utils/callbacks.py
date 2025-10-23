# import enum
#
# # Constants to indicate whether a parameter is required or optional.
# REQUIRED = True
# OPTIONAL = False
#
#
# class Callback:
#     """
#     Represents a callback function along with its associated inputs, lambda functions, and parameters.
#
#     Attributes:
#         inputs (dict): Static inputs to be passed to the callback.
#         lambdas (dict): Callable values that are evaluated at the time of invocation.
#         parameters (dict): Additional parameters associated with the callback.
#         function (callable): The callback function to be executed.
#         discard_inputs (bool): If True, ignores any extra positional or keyword arguments when calling the function.
#     """
#     inputs: dict
#     lambdas: dict
#     parameters: dict
#     function: callable
#
#     def __init__(self, function: callable, inputs: dict = None, lambdas: dict = None, parameters: dict = None,
#                  discard_inputs: bool = False, *args, **kwargs):
#         """
#         Initializes a new Callback instance.
#
#         Args:
#             function (callable): The function to be used as callback.
#             inputs (dict, optional): Fixed inputs to pass to the function.
#             lambdas (dict, optional): A dictionary of callables whose results will be passed as inputs.
#             parameters (dict, optional): Additional parameters for the callback.
#             discard_inputs (bool, optional): If True, extra inputs provided during the call are ignored.
#         """
#         self.function = function
#
#         if inputs is None:
#             inputs = {}
#         self.inputs = inputs
#
#         if lambdas is None:
#             lambdas = {}
#         self.lambdas = lambdas
#
#         if parameters is None:
#             parameters = {}
#         self.parameters = parameters
#
#         self.discard_inputs = discard_inputs
#
#     def __call__(self, *args, **kwargs):
#         """
#         Calls the callback function, merging inputs from stored values, evaluated lambdas, and the call arguments.
#
#         Returns:
#             The return value of the callback function.
#         """
#         # Evaluate any lambda functions.
#         lambdas_exec = {key: value() for (key, value) in self.lambdas.items()}
#         # Depending on discard_inputs, call with only the stored inputs or merge with provided args.
#         if self.discard_inputs:
#             ret = self.function(**{**self.inputs, **lambdas_exec})
#         else:
#             ret = self.function(*args, **{**self.inputs, **kwargs, **lambdas_exec})
#         return ret
#
#
# class CallbackContainer:
#     """
#     A container for managing callbacks. It allows registration, removal, and invocation of callbacks.
#     It can also be parameterized with expected parameters; required parameters must be provided when registering a callback.
#
#     Attributes:
#         callbacks (list[Callback]): The list of registered Callback objects.
#         expected_parameters (dict): Mapping of parameter names to a tuple (expected type, required flag).
#     """
#     callbacks: list[Callback]
#
#     def __init__(self, parameters=None):
#         """
#         Initializes a new CallbackContainer.
#
#         Args:
#             parameters (list of tuple, optional): A list of tuples (param_name, expected_type, required_flag)
#                 that define the expected parameters. For example:
#                 [('param1', int,  OPTIONAL), ('param2', float, REQUIRED)]
#         """
#         self.callbacks = []
#         self.parameters = parameters  # original input
#
#         # Validate and convert the parameter specifications into a dict.
#         if parameters is not None:
#             if not isinstance(parameters, list):
#                 raise ValueError("Parameters must be provided as a list of tuples.")
#             expected = {}
#             for param_spec in parameters:
#                 if not (isinstance(param_spec, tuple) and len(param_spec) == 3):
#                     raise ValueError("Each parameter specification must be a tuple of (str, type, bool).")
#                 name, expected_type, required_flag = param_spec
#                 if not isinstance(name, str):
#                     raise ValueError("Parameter name must be a string.")
#                 if not isinstance(required_flag, bool):
#                     raise ValueError("Parameter required flag must be a boolean.")
#                 if not isinstance(expected_type, type):
#                     raise ValueError("Parameter expected type must be a type.")
#                 expected[name] = (expected_type, required_flag)
#             self.expected_parameters = expected
#         else:
#             self.expected_parameters = {}
#
#     def register(self, function: callable, inputs: dict = None, parameters: dict = None, lambdas: dict = None,
#                  discard_inputs=False, *args, **kwargs):
#         """
#         Registers a callback function. Checks that all required parameters (if any) are provided.
#
#         Args:
#             function (callable): The function to register as a callback.
#             inputs (dict, optional): Dictionary of fixed inputs.
#             parameters (dict, optional): Dictionary of parameters to associate with the callback.
#             lambdas (dict, optional): Dictionary of lambdas to evaluate at call time.
#             discard_inputs (bool, optional): Whether to discard extra call inputs.
#             *args, **kwargs: Additional arguments passed to the Callback constructor.
#
#         Raises:
#             RuntimeError: If any required parameter (as specified by expected_parameters) is missing.
#             TypeError: If a parameter value does not match its expected type.
#         """
#         # If expected parameters have been set, validate the provided parameters.
#         if self.expected_parameters:
#             if parameters is None:
#                 parameters = {}
#             for param_name, (expected_type, required_flag) in self.expected_parameters.items():
#                 if param_name not in parameters:
#                     if required_flag:
#                         raise RuntimeError(f"Missing required parameter: {param_name}")
#                     else:
#                         parameters[param_name] = None
#                 else:
#                     if parameters[param_name] is not None and not isinstance(parameters[param_name], expected_type):
#                         raise TypeError(
#                             f"Parameter '{param_name}' is expected to be of type {expected_type.__name__}, "
#                             f"got {type(parameters[param_name]).__name__}."
#                         )
#
#         # Create a new Callback with an empty inputs dict.
#         callback = Callback(function, inputs=inputs, lambdas=lambdas, parameters=parameters,
#                             discard_inputs=discard_inputs, *args, **kwargs)
#         self.callbacks.append(callback)
#
#     def remove(self, callback):
#         """
#         Removes a callback from the container.
#
#         Args:
#             callback (Callback or callable): The callback instance or function to remove.
#         """
#         if isinstance(callback, Callback):
#             self.callbacks.remove(callback)
#         elif callable(callback):
#             cb = next((cb for cb in self.callbacks if cb.function == callback), None)
#             if cb is not None:
#                 self.callbacks.remove(cb)
#
#     def call(self, *args, **kwargs):
#         """
#         Invokes all registered callbacks with the given arguments.
#
#         Args:
#             *args: Positional arguments to pass to each callback.
#             **kwargs: Keyword arguments to pass to each callback.
#         """
#         for callback in self.callbacks:
#             callback(*args, **kwargs)
#
#     def __iter__(self):
#         """
#         Returns:
#             An iterator over the registered callbacks.
#         """
#         return iter(self.callbacks)
#
#     def clear_callbacks(self):
#         """
#         Clears all registered callbacks in this container.
#         """
#         self.callbacks.clear()
#
#
# def callback_definition(cls):
#     """
#     Class decorator to automatically instantiate CallbackContainer attributes declared in the class annotations.
#     This decorator is intended for classes that (directly or indirectly) subclass CallbackGroup.
#     """
#     original_init = cls.__init__
#
#     def new_init(self, *args, **kwargs):
#         # Iterate over all annotations. If an attribute is annotated as CallbackContainer (or a subclass), auto-instantiate it.
#         for name, annotation in cls.__annotations__.items():
#             # Only auto-instantiate if the attribute hasn't been set yet.
#             if getattr(self, name, None) is None:
#                 # If annotation is CallbackContainer or a subclass of it, create a new instance.
#                 try:
#                     if annotation == CallbackContainer or issubclass(annotation, CallbackContainer):
#                         setattr(self, name, CallbackContainer())
#                 except TypeError:
#                     # In case the annotation is not a type, skip auto-instantiation.
#                     pass
#         if original_init:
#             original_init(self, *args, **kwargs)
#
#     cls.__init__ = new_init
#     return cls
#
#
# class CallbackGroup:
#     """
#     Base class for grouping multiple CallbackContainer instances.
#     When subclassed, type annotations can be used for IDE autocomplete.
#
#     The method `clearAllCallbacks()` iterates over annotated attributes and clears each CallbackContainer.
#     """
#
#     def clearAllCallbacks(self):
#         """
#         Clears all callbacks from each CallbackContainer attribute of this instance.
#         """
#         for name in self.__class__.__annotations__:
#             attr = getattr(self, name, None)
#             if isinstance(attr, CallbackContainer):
#                 attr.clear_callbacks()
#
#
# # Example usage:
# # Now you can define your custom callback class by subclassing CallbackGroup.
# @callback_definition
# class CallbacksA(CallbackGroup):
#     callback1: CallbackContainer
#     callback2: CallbackContainer
#
#
# # Demonstration in a __main__ block.
# if __name__ == "__main__":
#     instance = CallbacksA()
#
#     # Registering some dummy callbacks for demonstration.
#     instance.callback1.register(lambda: print("Callback 1 executed"))
#     instance.callback2.register(lambda: print("Callback 2 executed"))
#
#     print("Number of callbacks before clearing:")
#     print(len(instance.callback1.callbacks), len(instance.callback2.callbacks))
#
#     # Now you can clear callbacks using the method from the base class.
#     instance.clearAllCallbacks()
#
#     print("Number of callbacks after clearing:")
#     print(len(instance.callback1.callbacks), len(instance.callback2.callbacks))

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
    function: callable

    def __init__(self, function: callable, inputs: dict = None, lambdas: dict = None, parameters: dict = None,
                 discard_inputs: bool = False, *args, **kwargs):
        self.function = function

        self.inputs = inputs or {}
        self.lambdas = lambdas or {}
        self.parameters = parameters or {}
        self.discard_inputs = discard_inputs

    def __call__(self, *args, **kwargs):
        lambdas_exec = {key: value() for (key, value) in self.lambdas.items()}
        if self.discard_inputs:
            ret = self.function(**{**self.inputs, **lambdas_exec})
        else:
            ret = self.function(*args, **{**self.inputs, **kwargs, **lambdas_exec})
        return ret


class CallbackContainer:
    callbacks: list[Callback]

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

    def register(self, function: typing.Callable, inputs: dict = None, parameters: dict = None, lambdas: dict = None,
                 discard_inputs=False, *args, **kwargs):
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
                            discard_inputs=discard_inputs, *args, **kwargs)
        self.callbacks.append(callback)

    def remove(self, callback):
        if isinstance(callback, Callback):
            self.callbacks.remove(callback)
        elif callable(callback):
            cb = next((cb for cb in self.callbacks if cb.function == callback), None)
            if cb is not None:
                self.callbacks.remove(cb)

    def call(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)

    def __iter__(self):
        return iter(self.callbacks)

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
