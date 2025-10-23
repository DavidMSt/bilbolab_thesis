def bytes_(val: int):
    assert (val < 2e8)
    return bytes([val])


def byteArrayToInt(b: (list, bytearray, bytes)):
    if isinstance(b, list):
        b = bytes(b)
    assert (isinstance(b, (list, bytes, bytearray)))
    return int.from_bytes(b, 'big')


def intToByte(i: int, num_bytes):
    return i.to_bytes(length=num_bytes, byteorder='big')


def intToByteList(i: int, num_bytes):
    return list(intToByte(i, num_bytes))


def setBit(number, bit):
    assert (number < 2e8)
    number = number | 1 << bit
    return number


def clearBit(number, bit):
    assert (number < 2e8)
    number = number & ~(1 << bit)
    return number


def toggleBit(number, bit):
    assert (number < 2e8)
    number = number ^ 1 << bit
    return number


def checkBit(number, bit):
    assert (number < 2e8)
    bit = (number >> bit) & 1
    return bit


def changeBit(number, bit, val):
    assert (number < 2e8)
    number = number ^ (-val ^ number) & (1 << bit)
    return number


def bytearray_to_string(data, pos=False):
    """

    :param data:
    :param pos:
    :return:
    """
    if isinstance(data, int):
        data = bytes([data])

    if pos:
        data = " ".join("0x{:02X}({:d})".format(b, i) for (i, b) in enumerate(data))
    else:
        data = " ".join("0x{:02X}".format(b) for b in data)

    return data
