def flatten_dict(data: dict, indent: int = 0) -> list[tuple[str, str]]:
    """
    Recursively flatten a dictionary into a list of (key, value) tuples.
    The key is indented by two spaces per level. If a value is a dict, the key
    is shown with an empty value, and its contents are flattened below.
    Lists are displayed as a comma-separated list inside square brackets.
    """
    rows = []
    for key, value in data.items():
        prefix = "  " * indent + str(key)
        if isinstance(value, dict):
            rows.append((prefix, ""))
            rows.extend(flatten_dict(value, indent=indent + 1))
        elif isinstance(value, list):
            rows.append((prefix, "[" + ", ".join(str(x) for x in value) + "]"))
        else:
            rows.append((prefix, str(value)))
    return rows


def replaceField(data, expected_type, key, new_value):
    """
    Recursively replaces the value of the specified key(s) with a new value
    if the existing value matches the given type.

    Parameters:
        data (dict or list): The input dictionary (or list of dictionaries) to process.
        expected_type (type): The type to match before replacing the value.
        key (str or list): The key or list of keys to search for.
        new_value: The value to replace with.

    Returns:
        None: Modifies the input dictionary in place.
    """
    if isinstance(key, str):
        keys = [key]
    else:
        keys = key

    if isinstance(data, dict):
        for k in data:
            if k in keys and isinstance(data[k], expected_type):
                data[k] = new_value
            else:
                replaceField(data[k], expected_type, keys, new_value)
    elif isinstance(data, list):
        for item in data:
            replaceField(item, expected_type, keys, new_value)


# ======================================================================================================================
class ObservableDict(dict):
    def __init__(self, *args, on_change=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_change = on_change

    def _notify(self):
        if self._on_change:
            self._on_change()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._notify()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._notify()

    def clear(self):
        super().clear()
        self._notify()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._notify()

    def pop(self, *args):
        result = super().pop(*args)
        self._notify()
        return result

    def popitem(self):
        result = super().popitem()
        self._notify()
        return result