import threading
import time
import socket
import queue
import dataclasses

import cobs.cobs as cobs

from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.logging_utils import Logger

# Initialize logger for TCP communications.
logger = Logger('tcp')
logger.setLevel('INFO')

# Constants for package management.
PACKAGE_TIMEOUT_TIME = 5
FAULTY_PACKAGES_MAX_NUMBER = 10

# Default minimum delay (in seconds) between consecutive TX writes.
DEFAULT_MIN_TX_DELAY = 0.001


@dataclasses.dataclass
class FaultyPackage:
    """
    Data class for storing information about a faulty package.
    """
    timestamp: float


@callback_definition
class TCPSocketCallbacks:
    """
    Callbacks for events related to a TCP_Socket instance.
    """
    rx: CallbackContainer
    disconnected: CallbackContainer


class TCP_Socket:
    """
    TCP socket wrapper that handles asynchronous reads and writes.
    Incoming messages are buffered and processed, while outgoing messages
    are queued and sent with a minimum delay between writes.
    """
    address: str  # IP address of the client
    rx_queue: queue.Queue  # Queue for incoming messages
    tx_queue: queue.Queue  # Queue for outgoing messages
    config: dict  # Configuration parameters

    rx_callback: callable  # Callback function for received messages
    rx_event: threading.Event  # Event signaling new data reception

    _connection: socket.socket
    _rxThread: threading.Thread
    _txThread: threading.Thread  # Thread handling the TX queue
    _faultyPackages: list
    _rx_buffer: bytes  # Buffer for accumulating partial data
    _last_faulty_cleanup: float
    _exit: bool  # Flag to signal shutdown

    def __init__(self, connection: socket.socket, address: str):
        """
        Initialize the TCP_Socket instance with the provided socket connection and address.
        Starts both the RX and TX threads.
        """
        self._connection = connection
        self.address = address

        # Default configuration: delimiter for packets, whether to use COBS encoding,
        # and minimum TX delay between consecutive sends.
        self.config = {
            'delimiter': b'\x00',
            'cobs': True,
            'min_tx_delay': DEFAULT_MIN_TX_DELAY
        }

        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()

        self._exit = False

        self.callbacks = TCPSocketCallbacks()

        self.rx_event = threading.Event()

        self._faultyPackages = []
        self._rx_buffer = b''
        self._last_faulty_cleanup = time.time()

        # Start the RX thread to handle incoming data.
        self._rxThread = threading.Thread(target=self._rx_thread_fun, daemon=True)
        self._rxThread.start()

        # Start the TX thread to process outgoing messages from the tx_queue.
        self._txThread = threading.Thread(target=self._tx_thread_fun, daemon=True)
        self._txThread.start()

    # -------------------------------------------------------------------------
    def send(self, data):
        """
        Encode and queue data to be sent over the socket.

        Instead of sending data immediately, the data is encoded and then put
        into a transmit queue. The TX thread processes this queue and ensures
        that there is a minimum delay between consecutive writes.
        """
        data = self._prepareTxData(data)
        self.tx_queue.put(data)

    # -------------------------------------------------------------------------
    def rxAvailable(self):
        """
        Check how many received messages are available in the rx_queue.
        """
        return self.rx_queue.qsize()

    # -------------------------------------------------------------------------
    def close(self):
        """
        Close the socket connection and signal threads to exit.
        Calls the disconnected callbacks.
        """
        try:
            self._connection.close()
        except Exception:
            pass
        self._exit = True
        try:
            self._rxThread.join()
        except RuntimeError:
            ...
        logger.info("TCP socket %s closed", self.address)
        for callback in self.callbacks.disconnected:
            callback(self)

    # -------------------------------------------------------------------------
    def setConfig(self, config):
        """
        Merge new configuration parameters with the existing configuration.
        """
        self.config = {**self.config, **config}

    # -------------------------------------------------------------------------
    def _rx_thread_fun(self):
        """
        Thread function to continuously receive data from the socket.
        Received data is accumulated into an internal buffer and processed to extract
        complete packets. This implementation properly handles partial packets
        that may exceed the recv() call's size.

        Note: Even if a packet is larger than the recv() size (8192 bytes),
        TCP (being a stream protocol) will deliver it in parts, which are then
        reassembled by the buffer.
        """
        while not self._exit:
            try:
                # Increase recv buffer size to 8192 bytes.
                data = self._connection.recv(8192)
            except Exception as e:
                logger.warning("Error in TCP connection: %s. Closing connection.", e)
                self.close()
                return

            # If no data is received, assume the client closed the connection.
            if not data:
                self.close()
                break

            # Process the received data, accumulating partial packets if needed.
            self._processRxData(data)

            # Clean up old faulty packages approximately once per second.
            now = time.time()
            if int(now - self._last_faulty_cleanup) > 1:
                self._faultyPackages = [
                    p for p in self._faultyPackages if now < (p.timestamp + PACKAGE_TIMEOUT_TIME)
                ]
                if len(self._faultyPackages) > FAULTY_PACKAGES_MAX_NUMBER:
                    logger.warning("Received %d faulty TCP packages in the last %d seconds",
                                   FAULTY_PACKAGES_MAX_NUMBER, PACKAGE_TIMEOUT_TIME)
                self._last_faulty_cleanup = now

    # -------------------------------------------------------------------------
    def _tx_thread_fun(self):
        """
        TX thread function that continuously sends messages from the tx_queue.
        Ensures that a defined minimum delay is observed between consecutive writes.
        """
        min_tx_delay = self.config.get("min_tx_delay", DEFAULT_MIN_TX_DELAY)
        while not self._exit:
            try:
                # Use a timeout to periodically check the _exit flag.
                data = self.tx_queue.get(timeout=1)
            except queue.Empty:
                continue

            self._write(data)
            time.sleep(min_tx_delay)

    # -------------------------------------------------------------------------
    def _prepareTxData(self, data):
        """
        Prepare data for transmission.

        The data is converted to bytes (if needed), optionally encoded using COBS,
        and appended with a delimiter.
        """
        if isinstance(data, list):
            data = bytes(data)
        if self.config.get('cobs', False):
            data = cobs.encode(data)
        if self.config.get('delimiter') is not None:
            data += self.config['delimiter']
        return data

    # -------------------------------------------------------------------------
    def _write(self, data):
        """
        Write data immediately to the socket using sendall.
        If an error occurs during the send, the connection is closed.
        """
        try:
            self._connection.sendall(data)
        except Exception as e:
            logger.warning("Error sending data: %s", e)
            self.close()

    # -------------------------------------------------------------------------
    def _processRxData(self, data):
        """
        Append new data to the internal buffer and extract complete packets.
        Incomplete data remains in the buffer until more data arrives.

        If COBS encoding is enabled, the packet is decoded before being added
        to the receive queue.
        """
        # Append newly received data to the persistent buffer.
        self._rx_buffer += data
        delimiter = self.config.get('delimiter')

        while True:
            index = self._rx_buffer.find(delimiter)
            if index == -1:
                # No complete packet found yet; wait for more data.
                break

            # Extract one complete packet from the buffer.
            packet = self._rx_buffer[:index]
            # Remove the processed packet and delimiter from the buffer.
            self._rx_buffer = self._rx_buffer[index + len(delimiter):]

            # If COBS encoding is enabled, attempt to decode the packet.
            if self.config.get('cobs', False):
                try:
                    packet = cobs.decode(packet)
                except Exception:
                    # If decoding fails, log a faulty package and skip this packet.
                    self._faultyPackages.append(FaultyPackage(timestamp=time.time()))
                    continue

            self.rx_queue.put(packet)

        # Signal and invoke receive callbacks if any packets have been queued.
        if not self.rx_queue.empty():
            self.rx_event.set()
            for callback in self.callbacks.rx:
                callback(self)


@callback_definition
class TCPSocketsHandlerCallbacks:
    """
    Callbacks for events related to the TCP_SocketsHandler.
    """
    client_connected: CallbackContainer
    client_disconnected: CallbackContainer
    server_error: CallbackContainer


class TCP_SocketsHandler:
    """
    Handles a TCP server that accepts multiple client connections.
    Manages the lifecycle of connected TCP_Socket instances.
    """
    address: str
    port: int
    sockets: list  # List of connected client TCP_Socket instances
    _thread: threading.Thread
    _server: socket.socket
    config: dict
    callbacks: TCPSocketsHandlerCallbacks
    _exit: bool

    def __init__(self, address, hostname: bool = False, config: dict = None):
        """
        Initialize the TCP_SocketsHandler with the given address and configuration.
        """
        default_config = {
            'max_clients': 100,
            'port': 6666,
        }
        if config is None:
            config = {}
        self.config = {**default_config, **config}

        self.sockets = []
        self.address = address
        self.port = self.config['port']
        self.callbacks = TCPSocketsHandlerCallbacks()
        self._exit = False

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # -------------------------------------------------------------------------
    def init(self):
        """
        Placeholder for any additional initialization steps.
        """
        pass

    # -------------------------------------------------------------------------
    def start(self):
        """
        Start the server thread that accepts new client connections.
        """
        self._thread = threading.Thread(target=self._threadFunction, daemon=True)
        self._thread.start()

    # -------------------------------------------------------------------------
    def close(self):
        """
        Close the server socket and stop the server thread.
        Also, logs the closure event.
        """
        logger.info("TCP host closed on %s:%d", self.address, self.port)
        self._exit = True
        try:
            self._server.close()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join()

    # -------------------------------------------------------------------------
    def send(self):
        """
        Placeholder method for sending data to clients.
        Implement sending to all or a particular client as needed.
        """
        pass

    # -------------------------------------------------------------------------
    def _threadFunction(self):
        """
        Main server thread function that binds the socket, listens for new
        client connections, and accepts them as they arrive.
        """
        server_address = (self.address, self.port)
        try:
            self._server.bind(server_address)
        except OSError as e:
            raise Exception("Address already in use. Please wait until the address is released") from e

        self._server.listen(self.config['max_clients'])
        logger.info("Starting TCP host on %s:%d", self.address, self.port)

        while not self._exit:
            try:
                connection, client_address = self._server.accept()
                self._acceptNewClient(connection, client_address)
            except Exception as e:
                if not self._exit:
                    logger.warning("Error accepting new client: %s", e)
                    for callback in self.callbacks.server_error:
                        callback(e)

    # -------------------------------------------------------------------------
    def _acceptNewClient(self, connection, address):
        """
        Accept a new client connection and instantiate a TCP_Socket for it.
        Registers disconnection callbacks.
        """
        client = TCP_Socket(connection, address)
        self.sockets.append(client)
        logger.info("New client connected: %s", client.address)
        client.callbacks.disconnected.register(self._clientClosed_callback)
        for callback in self.callbacks.client_connected:
            callback(client)

    # -------------------------------------------------------------------------
    def _clientClosed_callback(self, client: TCP_Socket):
        """
        Callback that is called when a client disconnects.
        Removes the client from the active sockets list and triggers further callbacks.
        """
        if client in self.sockets:
            self.sockets.remove(client)
        for cb in self.callbacks.client_disconnected:
            cb(client)
