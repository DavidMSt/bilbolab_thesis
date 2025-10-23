#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : David Stoll
# Supervisor  : Dustin Lehmann
# Created Date: date/month/time ..etc
# version ='1.0'
# ---------------------------------------------------------------------------
""" This Module is responsible for receiving the Host-Server Ip via UDP """
# ---------------------------------------------------------------------------
# Module Imports
import errno
import logging
import queue
import select
import socket
import threading
import time
import cobs.cobs as cobs

from utils.time import time_ms
from utils.callbacks import Callback
from core.communication.wifi.protocols.tcp_base_protocol import TCP_Base_Protocol
from core.communication.protocol import Protocol

MAX_SOCKET_READ = 8192

logger = logging.getLogger('tcp')
logger.setLevel('DEBUG')


# TODO:
# - give the user the possibility to decide whether he wants to use the queue or callbacks or both
# ---------------------------------------------------------------------------
class TCP_Socket:
    """

    """
    server_address: str
    server_port: int

    rx_queue: queue.Queue
    tx_queue: queue.Queue

    config: dict
    connected: bool
    protocol: Protocol = TCP_Base_Protocol

    _socket: socket.socket
    _input_connections: list
    _output_connections: list
    _exit: bool

    _thread: threading.Thread
    _tx_thread: threading.Thread

    _close_check_time: int
    _close_check_interval_ms = 1000

    callbacks: dict[str, list[Callback]]
    events: dict[str, threading.Event]

    # === INIT =========================================================================================================
    def __init__(self, server_address: str = None, server_port: int = 6666, config: dict = None):
        """

        :param server_address:
        :param server_port:
        """
        self.server_address = server_address
        self.server_port = server_port

        self.connected = False

        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()

        self._socket = None
        self._exit = False

        self._input_connections = []
        self._output_connections = []

        if config is None:
            config = {}

        default_config = {
            'delimiter': b'\x00',
            'cobs': True,
            'rx_queue': False,
        }
        self.config = {**default_config, **config}

        self.callbacks = {
            'connected': [],
            'disconnected': [],
            'error': [],
            'rx': [],
        }

        self.events = {
            'connected': threading.Event(),
            'disconnected': threading.Event(),
            'error': threading.Event(),
            'rx': threading.Event()
        }

        self._close_check_time = 0

    # === METHODS ======================================================================================================
    def start(self):
        """

        :return:
        """
        if self.server_address is None or self.server_port is None:
            raise Exception("Cannot start socket without address")

        # logger.info(f"Starting TCP connection with server {self.server_address}:{self.server_port}")
        self._thread = threading.Thread(target=self._threadFunction)
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def connect(self, address=None, port=None):

        if address is not None:
            self.server_address = address
        if port is not None:
            self.server_port = port

        assert (self.server_address is not None and self.server_port is not None)

        self.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        """

        :return:
        """
        self._exit = True

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, data):
        """
        :param data:
        :return:
        """

        if self.connected:
            buffer = self._prepareTxData(data)
            self.tx_queue.put_nowait(buffer)
            return True
        else:
            return False

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None):
        callback = Callback(function, parameters, lambdas)

        if callback_id in self.callbacks:
            self.callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    # === PRIVATE METHODS ==============================================================================================
    def _connect(self, server_address, server_port):
        """
        connect to client if server_address AND server Port are defined
        :param server_address: address of Host-Server
        :param server_port: port of Host-Server
        :return: nothing
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info(f"Connect to server {self.server_address}:{server_port} ...")
        self._socket.connect((server_address, server_port))
        logger.info(f"Success! Connected to server {server_address}:{server_port}!")
        self._socket.setblocking(False)
        self._output_connections.append(self._socket)
        self._input_connections.append(self._socket)
        self.connected = True

        for cb in self.callbacks['connected']:
            cb(self, server_address)

        self.events['connected'].set()

    # ------------------------------------------------------------------------------------------------------------------
    def _threadFunction(self):
        """

        :return:
        """
        while not self._exit:
            self._updateFunction()
            time.sleep(0.0001)  # TODO: Magic Number

        self._socket.close()
        logger.info("Close TCP Socket")

    # ------------------------------------------------------------------------------------------------------------------
    def _updateFunction(self):
        """

        :return:
        """

        if not self.connected:
            self._connect(self.server_address, self.server_port)

        readable, writable, exceptional = select.select(self._input_connections, self._output_connections,
                                                        self._input_connections, 0)

        # Check if the connection was closed:
        if time_ms() > (self._close_check_time + self._close_check_interval_ms):
            closed = self._remote_connection_closed()
            self._close_check_time = time_ms()

            if closed:
                logger.warning(f"Socket: Lost connection to server on {self.server_address}")
                self.connected = False
                self.events['connected'].clear()
                for callback in self.callbacks['disconnected']:
                    callback(self)
                return

        # rx data
        for connection in readable:
            try:
                # save received data
                data = []
                try:
                    data = connection.recv(MAX_SOCKET_READ)
                except:
                    pass
                if len(data) > 0:
                    self._rxFunction(data)

            except (ConnectionResetError, ConnectionAbortedError, InterruptedError):
                logger.warning(f"Lost connection with TCP server {self.server_address}")
                self._output_connections.remove(connection)
                self._input_connections.remove(connection)
                connection.close()
                self.connected = False
                self.events['connected'].clear()
                for callback in self.callbacks['disconnected']:
                    callback(self)
                return
        #
        if writable:
            self._sendTxData()

        for connection in exceptional:
            logger.error(f"Server exception with {connection.getpeername()}")
            self._input_connections.remove(connection)
            if connection in self._output_connections:
                self._output_connections.remove(connection)
            connection.close()

            for callback in self.callbacks['error']:
                callback(self)

    # ------------------------------------------------------------------------------------------------------------------
    # def _tx_thread_fun(self):
    #     """
    #     - this thread handles the sending of the data over the TCP socket
    #     - it polls the tx_queue and if data is available it encodes the data, adds the delimiter and sends it over the
    #     TCP socket
    #     """
    #     while not self._exit:
    #         # Get the latest data from the queue:
    #         try:
    #             data = self.tx_queue.get(block=True, timeout=1)
    #         except queue.Empty:
    #             continue
    #
    #         # Encode the data to utf-8 if it's a string
    #         if isinstance(data, str):
    #             data = data.encode('utf-8')
    #         elif isinstance(data, list):
    #             data = bytes(data)
    #
    #         assert (isinstance(data, (bytes, bytearray)))
    #
    #         # Encode the data and add a delimiter of those options are set
    #         if self.config['cobs']:
    #             data = cobs.encode(data)
    #         if self.config['delimiter'] is not None:
    #             data = data + self.config['delimiter']
    #
    #         logging.debug(f"Sending: {data}")
    #
    #         # Check if the output connection is writable
    #         writable = False
    #         while not writable:
    #             readable, writable, exceptional = select.select(self._input_connections, self._output_connections,
    #                                                             self._input_connections, 0)
    #
    #         self._socket.send(data)  # Send the data over the socket

    # ------------------------------------------------------------------------------------------------------------------
    def _prepareTxData(self, data: (str, list)):
        # Encode the data to utf-8 if it's a string
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, list):
            data = bytes(data)

        assert (isinstance(data, (bytes, bytearray)))

        # Encode the data and add a delimiter of those options are set
        if self.config['cobs']:
            data = cobs.encode(data)

        if self.config['delimiter'] is not None:
            data = data + self.config['delimiter']

        return data

    # ------------------------------------------------------------------------------------------------------------------
    def _sendTxData(self):

        try:
            data = self.tx_queue.get_nowait()
        except queue.Empty:
            return

        self._socket.send(data)

    # ------------------------------------------------------------------------------------------------------------------
    def _rxFunction(self, data: bytes):

        # Split the data by the delimiter
        data_packets = data.split(self.config["delimiter"])

        if not data_packets[-1] == b'':
            pass  # TODO: here i need some handling of incomplete packets

        data_packets = data_packets[0:-1]

        # COBS-decode the data packets if configured
        if self.config["cobs"]:
            for i, packet in enumerate(data_packets):
                try:
                    data_packets[i] = cobs.decode(packet)
                except:
                    pass  # TODO: This means we have cut a package in the middle

        # Process the individual packets
        for packet in data_packets:

            if self.config['rx_queue']:
                self.rx_queue.put_nowait(packet)
            for callback in self.callbacks['rx']:
                callback(packet)
            self.events['rx'].set()

    # ------------------------------------------------------------------------------------------------------------------
    def _remote_connection_closed(self) -> bool:
        """
        Returns True if the remote side did close the connection

        """
        try:
            buf = self._socket.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            if buf == b'':
                return True
        except BlockingIOError as exc:
            if exc.errno != errno.EAGAIN:
                # Raise on unknown exception
                pass
        except:
            pass
        return False
