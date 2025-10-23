import logging

from utils.callbacks import Callback


# ======================================================================================================================
class DataLink:
    identifier: str  # Descriptor of the parameter which can be accessed from remote
    description: str  # Description of the parameter which will be shown when generating a list of parameters
    datatype: (tuple, type)  # Allowed datatypes for the parameter
    limits: list  # Allowed values
    limits_mode: str  # Can be either 'range' or 'explicit' for the values given in limits
    writable: bool  # If 'false', this parameter cannot be written
    write_function: Callback
    read_function: Callback
    index: int
    obj: object  # Object which the parameter is associated to
    name: str  # Name of the parameter in the object obj

    # === INIT =========================================================================================================
    def __init__(self, identifier, description, datatype, limits: list = None, limits_mode: str = 'range',
                 write_function: (Callback, callable) = None, read_function: (Callback, callable) = None,
                 obj: object = None, name: str = None, writable: bool = True, index: int = None,
                 ):
        self.identifier = identifier
        self.description = description
        self.datatype = datatype
        self.limits = limits
        self.limits_mode = limits_mode
        self.write_function = write_function
        self.read_function = read_function
        self.obj = obj
        self.name = name
        self.writable = writable
        self.index = index

    # ------------------------------------------------------------------------------------------------------------------
    def get(self):
        if self.read_function is not None:
            return self.read_function()
        elif self.obj is not None and self.name is not None:
            if isinstance(self.obj, dict):
                return self.obj[self.name]
            else:
                return getattr(self.obj, self.name)

    # ------------------------------------------------------------------------------------------------------------------
    def set(self, value) -> bool:

        if not self.writable:
            return False

        # Check if the datatype is ok
        if not isinstance(value, self.datatype):
            # Exception for float and int:
            if self.datatype == float and isinstance(value, int):
                value = float(value)
            else:
                return False

        # Check if the limits are ok
        if self.limits is not None:
            if self.limits_mode == 'explicit':
                if value not in self.limits:
                    return False
            elif self.limits_mode == 'range':
                if value < self.limits[0] or value > self.limits[1]:
                    return False

        if self.obj is not None and self.name is not None:
            if isinstance(self.obj, dict):
                self.obj[self.name] = value
            else:
                setattr(self.obj, self.name, value)
        elif self.obj is not None and self.index is not None:
            self.obj[self.index] = value

        if self.write_function is not None:
            if not isinstance(self.write_function, Callback) and not hasattr(self.write_function, '__self__'):
                return self.write_function(self.obj, value)
            return self.write_function(value)

        return True

    # ------------------------------------------------------------------------------------------------------------------
    def generateDescription(self):
        out = {
            'identifier': self.identifier,
            'description': self.description,
            'datatype': str(self.datatype),
            'limits': self.limits,
            'writable': self.writable,
            'value': self.get()
        }
        return out


# ======================================================================================================================
class Command:
    identifier: str
    callback: (callable, Callback)
    arguments: list[str]
    description: str

    def __init__(self, identifier: str, callback: (callable, Callback), arguments: list, description: str):
        self.identifier = identifier
        self.callback = callback
        self.arguments = arguments
        self.description = description

    def execute(self, arguments: dict = None):

        if arguments is None:
            arguments = {}

        # Check if all arguments are accounted for
        if self.arguments is not None:
            for arg in self.arguments:
                if arg not in arguments:
                    logging.error(f"Missing argument {arg} for command {self.identifier}")
                    return

        # Execute the command
        self.callback(**arguments)

    # ------------------------------------------------------------------------------------------------------------------
    def generateDescription(self):
        out = {
            'identifier': self.identifier,
            'description': self.description,
            'arguments': self.arguments
        }
        return out


# ======================================================================================================================
def generateDataDict(data: dict[str, DataLink]):
    out = {}
    for name, value in data.items():
        if isinstance(value, DataLink):
            out[name] = value.generateDescription()
        elif isinstance(value, dict):
            out[name] = generateDataDict(value)

    return out


# ======================================================================================================================
def generateCommandDict(commands: dict[str, Command]):
    out = {}
    for name, command in commands.items():
        out[name] = command.generateDescription()
    return out
