import time
from typing import Any

import board
from RPi import GPIO

from core.utils.time import precise_sleep


def test_spi():
    spi = board.SPI()
    while not spi.try_lock():
        pass
    spi.configure(baudrate=20_000_000, phase=0, polarity=0)

    rx_buffer = bytearray(100)

    while True:
        print("Sending data...")
        data = bytearray([0x66, 0x01, 0x00, 0x00])
        spi.write(data, start=0, end=len(data))
        precise_sleep(0.005)

        spi.readinto(rx_buffer, 0, 100, 0x05)
        data_list = list(rx_buffer)
        print(rx_buffer)
        print(data_list)

        time.sleep(2)


if __name__ == '__main__':
    test_spi()
