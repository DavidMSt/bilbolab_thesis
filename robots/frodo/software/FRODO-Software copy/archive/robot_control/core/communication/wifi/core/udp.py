import dataclasses
import logging
import queue
import socket
import threading
import time

from utils.callbacks import Callback

logger = logging.getLogger('udp')
logger.setLevel('DEBUG')


@dataclasses.dataclass
class UDP_Message:
    data: bytes = None
    address: str = None


# ======================================================================================================================
class UDP_Socket:
    rx_queue: queue.Queue
    port: int
    open: bool

    _socket: socket.socket
    _thread: threading.Thread
    _callbacks: dict[str, list[Callback]]
    _exit: bool
    _socket_timeout = 1
    _thread_timeout = 0.001

    # === INIT =========================================================================================================
    def __init__(self, port: int = 37020):
        self.port = port

        self._callbacks = {'rx': []}
        self._thread = threading.Thread(target=self._threadFunction, daemon=True)
        self.open = False

    # === METHODS ======================================================================================================
    def start(self):
        logger.info(f"Starting UDP socket on port {self.port}")
        self._thread = threading.Thread(target=self._threadFunction, daemon=True)
        self._exit = False

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket.bind(("", self.port))
        self._socket.settimeout(self._socket_timeout)
        self._thread.start()
        self.open = True

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        self._exit = True
        try:
            self._thread.join()
        except RuntimeError:
            ...
        self._socket.close()
        logger.info(f"Closed UDP socket on port {self.port}")
        self.open = False

    # ------------------------------------------------------------------------------------------------------------------
    def pauseReceive(self):
        ...

    def startReceive(self):
        ...

    # ------------------------------------------------------------------------------------------------------------------
    def registerCallback(self, callback_id, function: callable, parameters: dict = None, lambdas: dict = None):
        callback = Callback(function, parameters, lambdas)

        if callback_id in self._callbacks:
            self._callbacks[callback_id].append(callback)
        else:
            raise Exception("Invalid Callback type")

    # === PRIVATE METHODS ==============================================================================================
    def _threadFunction(self):
        while not self._exit:
            try:
                data, addr = self._socket.recvfrom(1024)
                msg = UDP_Message(data=data, address=addr)

                for callback in self._callbacks['rx']:
                    callback(message=msg)
            except socket.timeout:
                ...

            time.sleep(self._thread_timeout)
