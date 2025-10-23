import logging
import os
import queue
import threading
import time

import cobs.cobs as cobs
import serial

SERIAL_BUFFER_SIZE = 8192


class UART_Socket:
    """

    """
    device: str  # name of the device or port, such as "COMx" oder "/dev/ttyAMAx"
    baudrate: int
    timeout: int

    _device: serial.Serial

    config: dict

    callbacks: dict[str, list]

    events: dict[str, threading.Event]

    rx_queue: queue.Queue
    tx_queue: queue.Queue

    _thread: threading.Thread

    _exit: bool

    def __init__(self, device: str = None, baudrate: int = 115200, timeout: int = 0, config: dict = None):
        """

        :param device:
        :param baudrate:
        :param timeout:
        :param config:
        """

        default_config = {
            'delimiter': b'\x00',
            'cobs': True
        }

        self.device = device
        self.baudrate = baudrate
        self.timeout = timeout

        if config is None:
            config = {}

        self.config = {**default_config, **config}

        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()

        # Test if the device exists:
        try:
            os.stat(self.device)
        except OSError:
            raise Exception(f"UART Device {self.device} does not exist!")

        self._device = serial.Serial(self.device, baudrate=self.baudrate, timeout=self.timeout)

        self.callbacks = {
            'rx': [],
            'connected': [],
            'error': []
        }

        self.events = {
            'rx': threading.Event(),
            'connected': threading.Event(),
            'error': threading.Event()
        }

        self._exit = False

    # === METHODS ======================================================================================================
    def start(self):

        # Open the device
        try:
            self._device.open()
        except serial.SerialException:
            try:  # Restart the device
                self._device.close()
                self._device.open()
            except serial.SerialException:
                raise Exception("Cannot open the serial device")

        self._thread = threading.Thread(target=self._thread_fun, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        self._exit = True
        self._thread.join()
        self._device.close()

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, data):
        if self.config['cobs']:
            data = cobs.encode(data)
        if self.config['delimiter'] is not None:
            data = data + self.config['delimiter']

        self.tx_queue.put_nowait(data)

    # ------------------------------------------------------------------------------------------------------------------
    def registerCallback(self, type, callback):
        if type in self.callbacks.keys():
            self.callbacks[type].append(callback)
        else:
            raise Exception("Callback not in list of callbacks")

    # === PRIVATE METHODS ==============================================================================================
    def _thread_fun(self):
        """

        :return:
        """
        # TODO: This might not be the best way to do it
        buffer = [b'\x00'] * SERIAL_BUFFER_SIZE
        idx_write = 0

        while not self._exit:
            # Read data
            data = self._device.read(size=1)

            # Put the data into the reception buffer
            if len(data) > 0:
                buffer[idx_write] = data

                # Check if the delimiter is reached
                if buffer[idx_write] == self.config['delimiter']:
                    self._rx_handling(buffer[0:idx_write + 1])
                    idx_write = 0
                else:
                    idx_write += 1

            # TODO: Might not be the best way to do it. Could I just get data from the queue and test if it is None?
            if self.tx_queue.qsize() > 0:
                if self._device.writable():
                    data_out = self.tx_queue.get_nowait()
                    self._write(data_out)

            time.sleep(0.0001)

    # ------------------------------------------------------------------------------------------------------------------
    def _rx_handling(self, buffer):
        if isinstance(buffer, list):
            buffer = b''.join(buffer)

        if self.config['cobs']:
            # Remove the last zero
            buffer = buffer[0:-1]
            try:
                buffer = cobs.decode(buffer)
            except:
                logging.warning(f"Cannot COBS decode buffer:{buffer} ")
                return

        self.rx_queue.put_nowait(buffer)
        for cb in self.callbacks['rx']:
            cb(buffer)

        self.events['rx'].set()

    # ------------------------------------------------------------------------------------------------------------------
    def _write(self, data):
        self._device.write(data)
