import threading
import queue
import time
import logging

from core.communication.wifi.tcp.protocols.tcp_base_protocol import TCP_Base_Protocol, TCP_Base_Message
# from core.communication.wifi.tcp.protocols.tcp_handshake_protocol import TCP_Handshake_Protocol, \
#     TCP_Handshake_Message
from core.communication.wifi.tcp_json_protocol import TCP_JSON_Protocol, TCP_JSON_Message
from core.communication.protocol import Message
from core.communication.wifi.tcp.tcp_socket import TCP_Socket
from core.utils.callbacks import callback_definition, CallbackContainer

logger = logging.getLogger('tcp_c')
logger.setLevel('INFO')


@callback_definition
class TCPConnectionCallback:
    disconnected: CallbackContainer
    handshake: CallbackContainer
    rx: CallbackContainer


########################################################################################################################
class TCP_Connection:
    rx_queue: queue.Queue
    client: TCP_Socket
    config: dict
    callbacks: TCPConnectionCallback
    base_protocol = TCP_Base_Protocol
    protocol = TCP_JSON_Protocol

    _events: dict[str, threading.Event]
    _thread: threading.Thread
    sent: int
    received: int
    error_packets: int

    # === INIT =========================================================================================================
    def __init__(self, client: TCP_Socket = None, config: dict = None):
        super().__init__()

        # Config for the TCP Device
        default_config = {
            'rx_queue': False,
        }
        if config is None:
            config = {}

        self.config = {**default_config, **config}

        self.client = client
        self.rx_queue = queue.Queue()

        self.callbacks = TCPConnectionCallback()

        self.sent = 0
        self.received = 0
        self.error_packets = 0

        self.events = {
            'handshake': threading.Event(),
            'rx': threading.Event()
        }

    # === PROPERTIES ===================================================================================================
    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, client: TCP_Socket):
        self._client = client
        if client is not None:
            self.connected = True
            self.client.callbacks.rx.register(self._clientRx_callback)
            self.client.callbacks.disconnected.register(self._clientDisconnect_callback)

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def ip(self):
        return self.client.address

    # === METHODS ======================================================================================================
    def close(self):
        self.client.close()

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, msg):

        # Encode the message
        data = self._encodeMessage(msg)
        self.client.send(data)
        self.sent += 1

    # ------------------------------------------------------------------------------------------------------------------
    def sendRaw(self, buffer):
        """
        - sends raw data over the TCP Client. For debug purposes only
        :param buffer: Byte array to be sent
        :return:
        """
        self.client.send(buffer)
        self.sent += 1

    # === PRIVATE METHODS ==============================================================================================
    def _encodeMessage(self, msg: Message):

        if msg._protocol is not self.protocol:
            print("OH NOO")
            return

        payload = msg.encode()

        base_message = self.base_protocol.Message()
        base_message.source = self.address
        base_message.address = self.address
        base_message.data = payload
        base_message.data_protocol_id = msg._protocol.identifier

        buffer = base_message.encode()

        return buffer

    # ------------------------------------------------------------------------------------------------------------------
    def _processIncomingHandshake(self, message: TCP_JSON_Message):
        """

        :param message:
        """
        self.address = message.data['address']
        self.name = message.data['name']

        for callback in self.callbacks.handshake:
            callback(self, message)

    # ------------------------------------------------------------------------------------------------------------------
    def _processDataPacket(self, data):
        """

        :param data:
        :return:
        """

        # Decode the data into a message
        base_msg: TCP_Base_Message = self.base_protocol.decode(data)

        if base_msg is None:
            # the received data package is not a valid TCP message
            self.error_packets += 1
            return None

        # The received message is valid
        self.received += 1
        self.last_contact = time.time()

        # Check if the protocol ID uses a protocol known to the device
        if base_msg.data_protocol_id is not self.protocol.identifier:
            return

        # Decode the message
        message = self.protocol.decode(base_msg.data)  # Type: Ignore

        # Check if the message is a handshake event
        if message.type == 'event' and message.event == 'handshake':
            self._processIncomingHandshake(message)
            return

        # logger.debug(
        #     f" (TCP RX) Device: \"{self.name}\", Protocol: {base_msg.data_protocol_id}, data: {base_msg.data}")

        if self.config['rx_queue']:
            self.rx_queue.put_nowait(message)

        for callback in self.callbacks.rx:
            callback(message)

        self.events['rx'].set()

    # ------------------------------------------------------------------------------------------------------------------
    def _clientRx_callback(self, *args, **kwargs):
        """
        - callback called when the socket receives data
        :param args:
        :param kwargs:
        :return:
        """
        # TODO: this is bad and blocks all other receiving things. This
        #  should be in a separate thread
        while self.client.rx_queue.qsize() > 0:
            buffer = self.client.rx_queue.get_nowait()
            self._processDataPacket(buffer)

    # ------------------------------------------------------------------------------------------------------------------
    def _clientDisconnect_callback(self, client):
        self.connected = False

        for callback in self.callbacks.disconnected:
            callback(self)
