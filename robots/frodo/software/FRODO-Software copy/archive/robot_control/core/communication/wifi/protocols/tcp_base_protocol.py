from typing import Union
import logging

from core.communication.protocol import Protocol, Message


class TCP_Base_Message(Message):
    data_protocol_id: int = None
    source: list = None
    address: list = None
    data: list = None

    def __repr__(self):
        return f"(TCP Base Message) Source: {self.source}, Address: {self.address}, " \
               f"Protocol: {self.data_protocol_id}, Data: {self.data}"


class TCP_Base_Protocol(Protocol):
    """
    |   BYTE    |   NAME            |   DESCRIPTION                 |   VALUE
    |   0       |   HEADER[0]       |   first header byte           |   0x55
    |   1       |   HEADER[1]       |   second header byte          |   0x55
    |   2       |   SRC[0]          |   Source ID                   |
    |   3       |   SRC[1]          |   Source ID                   |
    |   4       |   ADD[0]          |   Address                     |
    |   5       |   ADD[1]          |   Address                     |
    |   6       |   PROTOCOL        |   Protocol ID                 |
    |   7       |   LEN[0]          |   Length of the payload       |
    |   8       |   LEN[1]          |   Length of the payload       |
    |   9       |   PAYLOAD[0]      |   Payload                     |
    |   9+N-1   |   PAYLOAD[N-1]    |   Payload                     |
    |   9+N     |   CRC8            |   CRC8 of the Payload         |
    |   10+N     |   FOOTER          |   Footer                      |   0x5D
    """
    base = None
    identifier = 0
    Message = TCP_Base_Message

    idx_protocol = 6
    idx_src = slice(2, 4)
    idx_add = slice(4, 6)
    idx_len = slice(7, 9)
    idx_payload = 9
    offset_crc = 9
    offset_footer = 10

    header_0 = 0x55
    header_1 = 0x55
    footer = 0x5D

    protocol_overhead = 11

    def __init__(self):
        super().__init__()

    @classmethod
    def decode(cls, data: Union[list, bytes, bytearray]) -> TCP_Base_Message:
        check = cls.check(data)

        if not check:
            return None

        msg = TCP_Base_Message()
        msg.data_protocol_id = data[cls.idx_protocol]
        msg.source = list(data[cls.idx_src])
        msg.address = list(data[cls.idx_add])
        payload_len = int.from_bytes(data[cls.idx_len], byteorder="little")
        msg.data = data[cls.idx_payload:cls.idx_payload + payload_len]
        return msg

    @classmethod
    def encode(cls, msg: TCP_Base_Message, base_encode=False):
        """
        - Encode a TCP message from a given TCP_Message
        :param msg: TCP_Message object
        :param base_encode: Since the TCP_Protocol has no base protocol, this is not used
        :return: byte buffer of the message
        """
        assert (isinstance(msg, TCP_Base_Message))
        buffer = [0] * (len(msg.data) + cls.protocol_overhead)
        buffer[0] = cls.header_0
        buffer[1] = cls.header_1
        buffer[cls.idx_src] = msg.source
        buffer[cls.idx_add] = msg.address

        if hasattr(msg, 'data_protocol_id'):
            buffer[cls.idx_protocol] = msg.data_protocol_id
        else:
            buffer[cls.idx_protocol] = 0

        buffer[cls.idx_len] = len(msg.data).to_bytes(length=2, byteorder="little")
        buffer[cls.idx_payload: cls.idx_payload + len(msg.data)] = msg.data
        buffer[cls.offset_crc + len(msg.data)] = 0x00  # TODO
        buffer[cls.offset_footer + len(msg.data)] = cls.footer
        buffer = bytes(buffer)
        return buffer

    @classmethod
    def check(cls, data):
        if not data[0] == cls.header_0:
            return 0
        if not data[1] == cls.header_1:
            return 0

        payload_len = int.from_bytes(data[cls.idx_len], byteorder="little")
        if not len(data) == payload_len + cls.protocol_overhead:
            return 0

        if not data[payload_len + cls.offset_footer] == cls.footer:
            return 0

        return 1


TCP_Base_Message._protocol = TCP_Base_Protocol
