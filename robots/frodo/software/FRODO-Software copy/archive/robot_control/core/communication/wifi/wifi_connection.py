import dataclasses
import logging
import queue
import threading
import time

import core.communication.wifi.core.tcp as tcp
import core.communication.wifi.core.udp as udp
import utils as network
from core.communication.protocol import Message
from core.communication.wifi.protocols.tcp_handshake_protocol import TCP_Handshake_Message, TCP_Handshake_Protocol
from core.communication.wifi.protocols.tcp_base_protocol import TCP_Base_Message, TCP_Base_Protocol
from core.communication.wifi.protocols.tcp_json_protocol import TCP_JSON_Protocol
from utils.callbacks import Callback
import utils.network

import core.communication.wifi.addresses as addresses
from utils.elapsedTimer import ElapsedTimer

logger = logging.getLogger('wifi')
logger.setLevel('DEBUG')
logging.getLogger('udp').setLevel('WARNING')

@dataclasses.dataclass
class CorruptPackage:
    timestamp: float

# ======================================================================================================================
@dataclasses.dataclass
class ServerData:
    address: str = None
    port: int = None
    uid: list = None


# ======================================================================================================================
class WIFI_Connection:
    callbacks: dict[str, list[Callback]]
    events: dict[str, threading.Event]

    name: str
    address: bytes
    config: dict
    rx_queue: queue.Queue
    connected: bool = False
    registered: bool = False

    base_protocol = TCP_Base_Protocol
    handshake_protocol = TCP_Handshake_Protocol
    protocols: dict = {'json': TCP_JSON_Protocol, 'handshake': TCP_Handshake_Protocol}

    _server_data: ServerData
    _thread: threading.Thread
    _udp_socket: udp.UDP_Socket
    _tcp_socket: tcp.TCP_Socket
    _exit: bool = False

    _corruptPackages: list[CorruptPackage]

    _connectionTimer: ElapsedTimer
    _faultyPackagesCheckTimer: ElapsedTimer

    # === INIT =========================================================================================================
    def __init__(self, name: str = '', address: (list, bytes) = None, config: dict = None):

        # Config
        default_config = {
            'rx_queue': False
        }

        if config is None:
            config = {}

        self.config = {**default_config, **config}

        self.name = name

        self.address = bytes(address)

        self._corruptPackages = []

        self._connectionTimer = ElapsedTimer()
        self._faultyPackagesCheckTimer = ElapsedTimer()

        self._server_data = ServerData()
        self._udp_socket = udp.UDP_Socket()
        self._udp_socket.registerCallback('rx', self._udp_rxCallback)

        tcp_socket_config = {
            'rx_queue': False
        }

        self._tcp_socket = tcp.TCP_Socket(config=tcp_socket_config)
        self._tcp_socket.registerCallback('rx', self._tcp_rxCallback)
        self._tcp_socket.registerCallback('disconnected', self._tcp_disconnectedCallback)

        self.callbacks = {
            'rx': [],
            'connected': [],
            'disconnected': []
        }

        self.events = {
            'rx': threading.Event()
        }

        self.rx_queue = queue.Queue()
        self._thread = threading.Thread(target=self._threadFunction, daemon=True)

    # === METHODS ======================================================================================================
    def start(self):
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        self._exit = True
        self._thread.join()

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, message: Message, address=None):
        self._send(message, address)

    # ------------------------------------------------------------------------------------------------------------------
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None):
        callback = Callback(function, parameters, lambdas)

        if callback_id in self.callbacks:
            self.callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    # === PRIVATE METHODS ==============================================================================================
    def _threadFunction(self):
        self._connectionTimer.set(5) # Set this so that the connection attempt starts immediately

        while not self._exit:

            # Check the connection status every 5 seconds
            if self._connectionTimer > 5:
                self._connectionTimer.reset()

                if not self.connected:
                    success = self._connect()
                    if not success:
                        logging.warning(f'Cannot connect to server. Retry in 5s ...')

            # Remove the corrupted packages
            if self._faultyPackagesCheckTimer > 1:
                self._faultyPackagesCheckTimer.reset()

                cutoff_time = time.time() - 5
                self._corruptPackages = [package for package in self._corruptPackages if package.timestamp > cutoff_time]

                if len(self._corruptPackages) > 10:
                    logger.warning("TCP Connection unreliable")

            time.sleep(0.0001)

    # ------------------------------------------------------------------------------------------------------------------
    def _connect(self) -> bool:
        logger.info("Starting to connect to Server")
        success = self._listenForServerData(timeout=5)

        if not success:
            return False

        self._tcp_socket.connect(address=self._server_data.address, port=self._server_data.port)

        connected = self._tcp_socket.events['connected'].wait(timeout=1)

        if connected:
            self.connected = True
            # Send the handshake message
            logger.info(f"Sending handshake to server {self._server_data.address}:{self._server_data.port} ...")
            self._sendHandshake()
        else:
            ...

        for callback in self.callbacks['connected']:
            callback()

        return self.connected

    # ------------------------------------------------------------------------------------------------------------------
    def _listenForServerData(self, timeout=2):
        self._server_data = ServerData()
        # Listen for the server address on UDP
        logger.info(f"Listening for server data on UDP port {self._udp_socket.port}...")
        self._udp_socket.start()
        start_time = time.time()

        while self._server_data.address is None and time.time() < (start_time + timeout):
            time.sleep(0.1)

        if self._server_data.address is not None:
            logger.info(f"Server data received: {self._server_data.address}:{self._server_data.port}")
            self._udp_socket.close()
            return True
        else:
            logger.warning(f"Timeout occurred when receiving server data")
            self._udp_socket.close()
            return False

    # ------------------------------------------------------------------------------------------------------------------
    def _send(self, message: Message, address=None):

        # Check if the tcp socket is connected
        if not self._tcp_socket.connected:
            logger.error(f"Cannot send message over TCP: Not connected")
            return

        # Check if the protocol of the message is supported
        if next((protocol for name, protocol in self.protocols.items() if protocol == message._protocol), None) is None:
            logger.warning("Protocol not supported")
            return

        # Generate the payload buffer from the message
        payload = message.encode()

        # Generate a base message for the tcp socket
        tcp_msg = self._tcp_socket.protocol.Message()
        tcp_msg.data = payload
        tcp_msg.data_protocol_id = message._protocol.identifier
        tcp_msg.source = self.address

        # Send the message to the server if not specified differently
        if address is None:
            address = addresses.server

        tcp_msg.address = address

        # Generate the buffer from the base message
        buffer = tcp_msg.encode()
        # Send the buffer over the tcp socket
        self._tcp_socket.send(buffer)

    # ------------------------------------------------------------------------------------------------------------------
    def _sendHandshake(self):

        # Generate the handshake message
        handshake_message = TCP_Handshake_Message()
        handshake_message.protocols = [protocol.identifier for name, protocol in self.protocols.items()]
        handshake_message.uid = self.address
        handshake_message.name = self.name

        self._send(handshake_message, address=addresses.server)

    # ------------------------------------------------------------------------------------------------------------------
    def _receiveServerHandshake(self, message):
        logger.info(f"Received server data. Name: {message.name}, UID: {message.uid}.")

    # ------------------------------------------------------------------------------------------------------------------
    def _tcp_rxCallback(self, message: bytes):
        """
        This function is called if a message is received from the server
        Args:
            message:
        """

        message = self._tcp_decodeMessage(message)

        if message is not None:
            if self.config['rx_queue']:
                self.rx_queue.put_nowait(message)

            for callback in self.callbacks['rx']:
                callback(message)

            self.events['rx'].set()

    # ------------------------------------------------------------------------------------------------------------------
    def _tcp_disconnectedCallback(self, tcp_socket, *args, **kwargs):
        self.connected = False
        for callback in self.callbacks['disconnected']:
            callback()

    # ------------------------------------------------------------------------------------------------------------------
    def _udp_rxCallback(self, message: udp.UDP_Message):

        # Try to decode the UDP message and look for address and port of the server
        address, port = utils.network.splitServerAddress(message.data.decode('utf-8'))
        if address is not None and port is not None:
            self._server_data.address = address
            self._server_data.port = port

    # ------------------------------------------------------------------------------------------------------------------
    def _tcp_decodeMessage(self, data):

        base_message: TCP_Base_Message = TCP_Base_Protocol.decode(data)

        if base_message is None:
            self._corruptPackages.append(CorruptPackage(timestamp=time.time()))
            logging.info(f"Received faulty TCP message")
            return

        # Protocol 1 is the handshake protocol
        if base_message.data_protocol_id == self.handshake_protocol.identifier:
            handshake_message = self.handshake_protocol.decode(base_message.data)
            self._receiveServerHandshake(handshake_message)
            return

        protocol = next(
            (protocol for name, protocol in self.protocols.items() if
             protocol.identifier == base_message.data_protocol_id),
            None)

        if protocol is not None:
            message = protocol.decode(base_message.data)
        else:
            logger.warning(f"Received message with unknown protocol ID {base_message.data_protocol_id}")
            return
        # logging.debug(f" TCP RX: Protocol: [{message.data_protocol_id}]  data: {message.data}")
        return message
