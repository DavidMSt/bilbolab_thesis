from .tcp_base_protocol import TCP_Base_Message, TCP_Base_Protocol
from core.communication.protocol import Protocol, Message


class TCP_Handshake_Message(Message):
    uid: list
    protocols: list
    name: str


class TCP_Handshake_Protocol(Protocol):
    """
    |   BYTE    |   NAME            |   DESCRIPTION                     |   VALUE
    |   0       |   HEADER          |   Header Byte                     |   0x7F
    |   1       |   UID[0]          |   UID                             |
    |   2       |   UID[1]          |   UID                             |
    |   3       |   NUM_PROTOCOLS   |   number of protocols N           |
    |   4       |   PROTOCOL[0]     |   first protocol                  |
    |   5       |   PROTOCOL[1]     |   second protocol                 |
    |   6+N-1   |   PROTOCOL[N]     |   last protocol                   |
    |   6+N     |   LEN_NAME        |   number of bytes M in the name   |
    |   7+N     |   NAME[0]         |   name                            |
    |   8+N     |   NAME[1]         |   name                            |
    |   7+M+N   |   NAME[M]         |   name                            |

    """
    Message = TCP_Handshake_Message

    idx_header = 0
    header = 0x7F
    idx_uid = 1
    idx_num_protocols = 3
    idx_protocols = 4

    offset_len_name = 4
    offset_name = 5

    protocol_overhead = 5

    uid_len = 2

    identifier = 0x01

    base = TCP_Base_Protocol

    def __init__(self):
        super().__init__()

    @classmethod
    def decode(cls, data):
        check = cls.check(data)

        if not check:
            return None

        msg = cls.Message()
        msg.uid = list(data[cls.idx_uid:cls.idx_uid + 2])
        num_protocols = data[cls.idx_num_protocols]
        msg.protocols = list(data[cls.idx_protocols:cls.idx_protocols + num_protocols])
        len_name = data[cls.offset_len_name + num_protocols]
        msg.name = (data[cls.offset_name + num_protocols:cls.offset_name + num_protocols + len_name]).decode("utf-8")
        return msg

    @classmethod
    def encode(cls, msg: Message):
        buffer = [0] * (cls.protocol_overhead + len(msg.protocols) + len(msg.name))
        buffer[cls.idx_header] = cls.header
        buffer[cls.idx_uid] = msg.uid[0]
        buffer[cls.idx_uid + 1] = msg.uid[1]
        buffer[cls.idx_num_protocols] = len(msg.protocols)
        buffer[cls.idx_protocols:cls.idx_protocols + len(msg.protocols)] = msg.protocols
        buffer[cls.offset_len_name + len(msg.protocols)] = len(msg.name)
        buffer[cls.offset_name + len(msg.protocols):cls.offset_name + len(msg.protocols) + len(msg.name)] = list(
            msg.name.encode("utf-8"))
        buffer = bytes(buffer)
        return buffer

    @classmethod
    def check(cls, data):
        if not data[cls.idx_header] == cls.header:
            return 0

        num_protocols = data[cls.idx_num_protocols]
        len_name = data[cls.offset_len_name + num_protocols]

        if not len(data) == num_protocols + len_name + cls.protocol_overhead:
            return 0

        return 1


TCP_Handshake_Message._protocol = TCP_Handshake_Protocol
