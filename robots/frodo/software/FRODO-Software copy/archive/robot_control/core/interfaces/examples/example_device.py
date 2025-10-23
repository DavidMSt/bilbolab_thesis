import logging
import time

from core.communication.wifi.protocols.tcp_json_protocol import TCP_JSON_Message
from core.interfaces.wifi_interface import WIFI_Interface

logging.basicConfig(level='DEBUG')


def example_device():
    d = Interface()
    d.device_class = 'board'
    d.device_type = 'board'
    d.name = 'c4-02'
    d.id = 'rc_c4-02'
    d.parent = None
    d.children = []

    d.start()

    while True:
        time.sleep(1)

        if d.connection.connected:
            msg = TCP_JSON_Message()
            msg.source = ''
            msg.address = ''
            msg.type = 'event'
            msg.data = {
                'hello': 1234
            }
            # d._wifi_send(msg)


if __name__ == '__main__':
    example_device()
