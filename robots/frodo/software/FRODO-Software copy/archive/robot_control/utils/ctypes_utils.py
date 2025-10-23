import ctypes

def struct_to_dict(struct: ctypes.Structure):
    # return dict((f, getattr(struct, f)) for f, _ in struct._fields_)
    result = {}
    for field, _ in struct._fields_:
        value = getattr(struct, field)
        # if the type is not a primitive and it evaluates to False ...
        if (type(value) not in [int, float, bool]) and not bool(value):
            # it's a null pointer
            value = None
        elif hasattr(value, "_length_") and hasattr(value, "_type_"):
            # Probably an array
            value = list(value)
        elif hasattr(value, "_fields_"):
            # Probably another struct
            value = struct_to_dict(value)
        result[field] = value
    return result

