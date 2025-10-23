import cobs.cobs as cobs

from core.communication.wifi.tcp.protocols.tcp_base_protocol import TCP_Base_Message, TCP_Base_Protocol
from core.communication.wifi.tcp_json_protocol import TCP_JSON_Message, TCP_JSON_Protocol


def example_tcp_message():

    data_list = []

    for i in range(30000):
        data_list.append(i)

    msg = TCP_JSON_Message()
    msg.data = {
        'x': 3,
        'y': "HELLO",
        'data': data_list
    }
    msg.type = 'event'
    msg.event = 'test'

    payload = msg.encode()

    print(len(payload))

    base_message = TCP_Base_Message()
    base_message.source = '192.168.2.1'
    base_message.address = '192.168.2.2'
    base_message.data = payload
    base_message.data_protocol_id = 6

    buffer = base_message.encode()

    buffer_cobs = cobs.encode(buffer)

    print(buffer_cobs)

    buffer_decobs = cobs.decode(buffer_cobs)

    base_message_decoded = TCP_Base_Protocol.decode(buffer_decobs)

    data_from_base_message = base_message_decoded.data

    tcp_json_message_decoded = TCP_JSON_Protocol.decode(data_from_base_message)

    print(tcp_json_message_decoded)

    data_decoded = tcp_json_message_decoded.data['data']
    # print(tcp_json_message_decoded.data['x'])


if __name__ == '__main__':
    example_tcp_message()
