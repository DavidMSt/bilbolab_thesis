import dataclasses
import logging
import time
from typing import Union
import orjson

from core.communication.protocol import Protocol, Message
from .tcp_base_protocol import TCP_Base_Protocol

time_offset = 100


# ======================================================================================================================
@dataclasses.dataclass
class TCP_JSON_Message(Message):
    data: dict = dataclasses.field(default_factory=dict)
    address: (str, dict) = None
    source: str = None
    type: str = None
    meta: dict = dataclasses.field(default_factory=dict)
    event: str = None

    def __init__(self):
        self.meta = {
            'time':  time.time(),
            'id': id(self)
        }

    # TODO
    def createMetaData(self):
        self.meta['time'] = time.time() + time_offset
        self.meta['id'] = id(self)


# ======================================================================================================================
class TCP_JSON_Protocol(Protocol):
    base = TCP_Base_Protocol
    Message = TCP_JSON_Message
    identifier = 0x02
    allowed_types = ['write', 'read', 'answer', 'command', 'event', 'stream']
    meta_fields = ['time', 'id']

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def decode(cls, data: bytes):
        assert (isinstance(data, bytes))
        msg = cls.Message()
        msg_content = orjson.loads(data)

        if 'data' in msg_content:
            msg.data = msg_content['data']
        if 'address' in msg_content:
            msg.address = msg_content['address']
        if 'source' in msg_content:
            msg.source = msg_content['source']
        if 'type' in msg_content:
            msg.type = msg_content['type']
        if 'event' in msg_content:
            msg.event = msg_content['event']
        return msg

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def encode(cls, msg: TCP_JSON_Message):
        """
        :param msg:
        :return:
        """
        # Check if the command is allowed
        if msg.type not in cls.allowed_types:
            logging.error(f"Command not allowed!")
            return

        # Check if the metadata is present
        for f in cls.meta_fields:
            if f not in msg.meta:
                logging.error(f'Meta data {f} missing. Message not encoded.')
                return

        # Check if it correctly set up for a command
        data = orjson.dumps(msg)

        return data

    @classmethod
    def check(cls, data):
        return 1

    @classmethod
    def setTimeOffset(cls, server_time):
        global time_offset
        time_offset = server_time - time.time()


TCP_JSON_Message._protocol = TCP_JSON_Protocol
