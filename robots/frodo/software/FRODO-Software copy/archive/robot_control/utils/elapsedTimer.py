import time


class ElapsedTimer:
    _reset_time: float

    def __init__(self):
        self._reset_time = time.time()


    @property
    def time(self):
        return time.time() - self._reset_time

    def reset(self):
        self._reset_time = time.time()

    def set(self, value):
        self._reset_time = time.time() - value

    def __gt__(self, other):
        return self.time > other

    def __lt__(self, other):
        return self.time < other