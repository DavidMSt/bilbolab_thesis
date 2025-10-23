import time


def time_ms():
    return int(time.time_ns() / 1000)
