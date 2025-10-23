import dataclasses
from abc import ABC, abstractmethod


@dataclasses.dataclass
class DataPacket:
    pass


class Message:
    _protocol: 'Protocol' = None

    def encode(self):
        return self._protocol.encode(self)


class Protocol(ABC):
    Message: type
    base: 'Protocol'
    identifier: int

    def __init__(self):
        pass

    @abstractmethod
    def decode(self, data):
        pass

    @abstractmethod
    def encode(self, msg: Message):
        pass

    @abstractmethod
    def check(self, data):
        pass
