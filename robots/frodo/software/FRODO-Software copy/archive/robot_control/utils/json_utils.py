import json

def readJSON(file):
    with open(file) as f:
        data = json.load(f)
    return data


def writeJSON(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)
