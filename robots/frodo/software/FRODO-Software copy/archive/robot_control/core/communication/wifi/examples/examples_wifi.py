import logging
import time

from core.communication.wifi.wifi_connection import WIFI_Connection

logging.basicConfig(level='DEBUG')


def example_wifi_1():
    wifi = WIFI_Connection(name='c4-02', address=b'\x01\x01', config={'rx_queue': True})

    def testfunction(message):
        nonlocal wifi
        wifi.send(message)
        print(wifi.rx_queue.qsize())

    def connectedfunction():
        print("CONNECTED")

    wifi.registerCallback('rx', testfunction)
    wifi.registerCallback('connected', connectedfunction)

    wifi.start()
    time.sleep(100)


if __name__ == '__main__':
    example_wifi_1()
