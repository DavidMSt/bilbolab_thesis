import warnings

from core.utils.files import fileExists
import json


def readJSON(file) -> dict | None:
    if not fileExists(file):
        warnings.warn(f"File {file} does not exist", UserWarning)
        return None
    with open(file) as f:
        data = json.load(f)
    return data


def writeJSON(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=1)
