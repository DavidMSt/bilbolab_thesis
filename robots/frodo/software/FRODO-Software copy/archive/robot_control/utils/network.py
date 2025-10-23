import re


def splitServerAddress(address: str):
    server_address = None
    server_port = None
    try:
        server_address = re.search(r'[0-9.]*(?=:)', address)[0]
    except:
        ...
    try:
        server_port = int(re.search(r'(?<=:)[0-9]*', address)[0])
    except:
        ...

    return server_address, server_port
