import time

import core.archive.settings as settings
from core.communication.wifi.tcp.tcp_socket import TCP_SocketsHandler, TCP_Socket
from core.communication.wifi.udp.protocols.udp_json_protocol import UDP_JSON_Message
from core.communication.wifi.udp.udp import UDP, UDP_Broadcast
import core.communication.addresses as addresses
from core.communication.wifi.tcp.tcp_connection import TCP_Connection
from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.logging_utils import Logger
from core.communication.wifi.tcp.protocols.tcp_base_protocol import TCP_Base_Protocol
from core.communication.wifi.tcp_json_protocol import TCP_JSON_Protocol

logger = Logger('server')
logger.setLevel('INFO')


@callback_definition
class TCPServerCallbacks:
    connected: CallbackContainer
    disconnected: CallbackContainer


# ======================================================================================================================
class TCP_Server:
    connections: list[TCP_Connection]

    base_protocol = TCP_Base_Protocol
    protocol = TCP_JSON_Protocol

    callbacks: TCPServerCallbacks
    address: str

    _unregistered_connections: list[TCP_Connection]
    _tcp: TCP_SocketsHandler
    _udp: UDP

    # === INIT =========================================================================================================
    def __init__(self, address):

        self.address = address

        self._tcp = TCP_SocketsHandler(address=self.address)
        self._udp = UDP(address=self.address, port=settings.UDP_PORT_ADDRESS_STREAM)

        self.connections = []
        self._unregistered_connections = []

        self._configureServer()

        self.callbacks = TCPServerCallbacks()

    # === METHODS ======================================================================================================
    def init(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def start(self):

        # If no specific address is specified, look for a local IP to host the server
        logger.info(f"Starting server on {self.address}")
        self._startServer()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        self._tcp.close()
        self._udp.close()
        logger.info("TCP Server closed")
        time.sleep(0.01)

    # ------------------------------------------------------------------------------------------------------------------
    def getDevice(self, name=None, address=None):
        """

        :param name:
        :param address:
        :return:
        """
        assert (name is None or address is None)
        if name is not None:
            device = next((device for device in self.connections if device.name == name), None)
            return device

        if address is not None:
            device = next((device for device in self.connections if device.address == address), None)
            return device

    # ------------------------------------------------------------------------------------------------------------------
    def send(self, message, address):
        self._send(message, address)

    # ------------------------------------------------------------------------------------------------------------------
    def broadcast(self, message):
        for device in self.connections:
            device.send(message)

    # === PRIVATE METHODS ==============================================================================================
    def _configureServer(self):
        self._tcp.callbacks.client_connected.register(self._deviceConnected_callback)
        self._tcp.callbacks.client_disconnected.register(self._deviceDisconnected_callback)

        broadcast_message = UDP_JSON_Message()
        broadcast_message.type = None
        broadcast_message.data = {
            'address': self.address,
            'port': self._tcp.port
        }

        udp_broadcast = UDP_Broadcast()
        udp_broadcast.message = broadcast_message
        udp_broadcast.time = 1

        self._udp.addBroadcast(udp_broadcast)

    # ------------------------------------------------------------------------------------------------------------------
    def _startServer(self):
        self._udp.start()
        self._tcp.start()

    # ------------------------------------------------------------------------------------------------------------------
    def _send(self, message, address):
        if isinstance(address, list) and all(isinstance(elem, int) for elem in address):
            address = bytes(address)

        if isinstance(address, bytes):

            # TODO: Also consider groups
            if address == addresses.broadcast:  # Send to all devices
                self.broadcast(message)
            else:
                device = self.getDevice(address=address)
                if device is not None:
                    device.send(message)

        elif isinstance(address, str):
            if address == 'broadcast':
                self.broadcast(message)
            else:
                device = self.getDevice(name=address)
                if device is not None:
                    device.send(message)

        elif isinstance(address, TCP_Connection):
            address.send(message)

    # ------------------------------------------------------------------------------------------------------------------
    def _deviceConnected_callback(self, socket: TCP_Socket, *args, **kwargs):
        # put the client into the list of unregistered tcp devices
        unregistered_device = TCP_Connection(client=socket)
        self._unregistered_connections.append(unregistered_device)
        unregistered_device.callbacks.handshake.register(self._deviceHandshake_callback)

    # ------------------------------------------------------------------------------------------------------------------
    def _deviceDisconnected_callback(self, socket: TCP_Socket, *args, **kwargs):

        # Check if the socket belongs to one of the unregistered devices
        if any(device for device in self._unregistered_connections if device.client == socket):
            unregistered_device = next((device for device in self._unregistered_connections
                                        if device.client == socket), None)
            if unregistered_device is not None:
                self._unregistered_connections.remove(unregistered_device)
                logger.info(f"Unregistered TCP device with address {unregistered_device.client.address} disconnected")

        # Check if the client belongs to a registered device
        if any(device for device in self.connections if device.client == socket):
            registered_device = next((device for device in self.connections if device.client == socket), None)
            if registered_device is not None:
                logger.info(
                    f"TCP Connection closed. [Name: \"{registered_device.name}\", Adress: {registered_device.address}]")
                self.connections.remove(registered_device)
                for callback in self.callbacks.disconnected:
                    callback(registered_device)

    # ------------------------------------------------------------------------------------------------------------------
    def _deviceHandshake_callback(self, device: TCP_Connection, handshake_msg):
        # Check all protocol ids that the device supports and check if they are in the supported
        # data protocols of the server

        # Check if the device is in the list of unregistered devices
        if device not in self._unregistered_connections:
            # there might be a problem here. Better raise an Exception for now
            logger.error(f"Device ({device.address}) tried to register, even though it is not in the list "
                         f"of unregistered devices")
            return

        # Put the device into the list of registered devices
        self.connections.append(device)
        self._unregistered_connections.remove(device)
        device.registered = True

        logger.info(f"New TCP connection. Name: \"{device.name}\", Address: {device.address}")

        for callback in self.callbacks.connected:
            callback(device)

        # # Send the handshake back
        # server_handshake_msg = TCP_Handshake_Message()
        # server_handshake_msg.name = 'server'
        # server_handshake_msg.protocols = [protocol.identifier for name, protocol in self.data_protocols.items()]
        # server_handshake_msg.address = addresses.server
        # device.send(server_handshake_msg, source=addresses.server)
